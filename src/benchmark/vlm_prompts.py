"""Build exact-length image+text prompts for VLM sweeps.

Design: measure image+template token floor first, then fit text to hit the
requested total context length exactly.
"""

from __future__ import annotations

from PIL import Image

# Same seed as text sweep to keep prompt style consistent.
SEED = "The quick brown fox jumps over the lazy dog. "

# Fixed probe image so image-token cost stays stable across runs.
_IMAGE_SIZE = (224, 224)


def make_probe_image():
    """Return a neutral fixed-size probe image."""
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
    """Return `(processor_inputs, total_tokens)`."""
    chat_text = processor.apply_chat_template(
        _build_messages(text), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt"
    )
    total_tokens = int(inputs["input_ids"].shape[1])
    return inputs, total_tokens


def measure_image_token_count(processor):
    """Measure fixed token floor from image + chat scaffolding."""
    image = make_probe_image()
    _, total = _process(processor, image, "")
    return total, image, total


def build_vlm_prompt(processor, target_tokens):
    """Build `(inputs, total, image_tokens, text_tokens)` at exact length."""
    base_total, image, _ = measure_image_token_count(processor)
    if target_tokens <= base_total:
        raise ValueError(
            f"Target {target_tokens} tokens is at or below the image+scaffolding "
            f"floor of {base_total} for this processor. Pick a longer target."
        )

    text_target = target_tokens - base_total

    # Overshoot first, then trim.
    text = SEED * (text_target // 5 + 32)

    # First pass: approximate text length with tokenizer-only truncation.
    tok = processor.tokenizer
    text_ids = tok(
        text, add_special_tokens=False, truncation=True, max_length=text_target,
    )["input_ids"]
    text = tok.decode(text_ids)

    # Final fit loop on full processor because template can shift token count.
    total = -1
    for _ in range(8):
        inputs, total = _process(processor, image, text)
        if total == target_tokens:
            return inputs, total, base_total, total - base_total
        diff = total - target_tokens
        if diff > 0:
            # Too long: drop tail tokens.
            text_ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(text_ids) <= diff:
                break
            text = tok.decode(text_ids[: -diff])
        else:
            # Too short: append seed and trim on next pass.
            text = text + SEED * (abs(diff) // 5 + 4)

    raise RuntimeError(
        f"Could not converge VLM prompt to exactly {target_tokens} tokens; "
        f"last total was {total}. This is a tokenizer/template edge case."
    )
