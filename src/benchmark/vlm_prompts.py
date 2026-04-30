"""Image+text prompt construction for the VLM context sweep.

The VLM rows in the result file should be apples-to-apples with the text-only
rows: when we say "context_length = 16384" we want exactly 16384 tokens going
into the model. A VLM expands an image into N image tokens that count against
the budget, so we have to subtract those from the text portion.

Strategy:
  1. Build a tiny probe prompt with image + empty text. Read its token count
     to learn how many tokens the image consumes for THIS processor at THIS
     image resolution.
  2. Tile the seed text to fill the remainder (target_tokens - image_tokens),
     then truncate to the exact remainder using the tokenizer.
  3. Build the real prompt and verify the final length lands on target.

A fixed 224x224 PIL image is used so image_token_count is constant across the
sweep; this is recorded in every result row so it can be reproduced later.
"""

from __future__ import annotations

from PIL import Image

# Same seed as the text sweep so the two prompt formats stay comparable.
SEED = "The quick brown fox jumps over the lazy dog. "

# Fixed-resolution probe image. 224x224 is small enough to keep image-token
# overhead modest on Qwen2-VL (typically a few hundred tokens) and big enough
# that the processor doesn't scale-up unexpectedly.
_IMAGE_SIZE = (224, 224)


def make_probe_image():
    """A neutral grey PIL image at the fixed sweep resolution."""
    return Image.new("RGB", _IMAGE_SIZE, color=(128, 128, 128))


def _build_messages(text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def _process(processor, image, text):
    """Run the processor end-to-end and return (inputs, total_tokens)."""
    chat_text = processor.apply_chat_template(
        _build_messages(text), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt"
    )
    total_tokens = int(inputs["input_ids"].shape[1])
    return inputs, total_tokens


def measure_image_token_count(processor):
    """Build a probe prompt with empty text to learn how many tokens the
    image AND the chat-template scaffolding (role markers, vision-start/end,
    bos/eos, generation prompt) consume for this processor.

    Note: the returned count is *not* "image-only" — it's the floor that the
    full prompt has to clear before we can add any user text. We bundle them
    because they're both fixed costs across the sweep and the only knob is
    text length. CLAUDE.md's schema section documents this so the JSONL
    `image_token_count` field is interpreted correctly.

    Returns (image_plus_scaffolding_tokens, image, same_count_for_clarity).
    """
    image = make_probe_image()
    _, total = _process(processor, image, "")
    return total, image, total


def build_vlm_prompt(processor, target_tokens):
    """
    Build a (image, text) prompt whose tokenized total equals target_tokens.

    Returns:
        inputs: BatchFeature ready to pass to model.generate(**inputs)
        actual_total_tokens: int (always equals target_tokens after fitting)
        image_token_count: int — the floor consumed by image + chat scaffolding
        text_token_count: int — actual_total_tokens - image_token_count
    """
    base_total, image, _ = measure_image_token_count(processor)
    if target_tokens <= base_total:
        raise ValueError(
            f"Target {target_tokens} tokens is at or below the image+scaffolding "
            f"floor of {base_total} for this processor. Pick a longer target."
        )

    text_target = target_tokens - base_total

    # Overshoot, then let the tokenizer truncate. Tokens-per-SEED is ~10
    # for typical BPE tokenizers; multiply with a comfortable safety margin.
    text = SEED * (text_target // 5 + 32)

    # Truncate text to roughly text_target tokens via the tokenizer alone, then
    # iterate against the full processor (which adds chat template + image
    # tokens) until total == target_tokens.
    tok = processor.tokenizer
    text_ids = tok(
        text, add_special_tokens=False, truncation=True, max_length=text_target,
    )["input_ids"]
    text = tok.decode(text_ids)

    # The chat template can add or remove a token or two when re-encoding the
    # truncated text. Adjust by trimming/extending tokens until the total lands
    # on the target. Cap iterations so a pathological tokenizer can't loop.
    total = -1
    for _ in range(8):
        inputs, total = _process(processor, image, text)
        if total == target_tokens:
            return inputs, total, base_total, total - base_total
        diff = total - target_tokens
        if diff > 0:
            # Too long: drop `diff` tokens from the end of the text portion.
            text_ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(text_ids) <= diff:
                break  # can't go shorter without losing the text entirely
            text = tok.decode(text_ids[: -diff])
        else:
            # Too short: append more seed and let the next loop trim the excess.
            text = text + SEED * (abs(diff) // 5 + 4)

    raise RuntimeError(
        f"Could not converge VLM prompt to exactly {target_tokens} tokens; "
        f"last total was {total}. This is a tokenizer/template edge case."
    )
