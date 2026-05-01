"""Build exact-length synthetic prompts for text sweeps."""

# Repeated seed text used to fill long contexts.
SEED = "The quick brown fox jumps over the lazy dog. "


def build_synthetic_prompt(tokenizer, target_tokens):
    """Return `(input_ids, actual_token_count)` at the target token length."""
    # Overshoot and truncate so token count lands exactly on target.
    text = SEED * (target_tokens // 5 + 16)
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"]
    actual = input_ids.shape[1]
    if actual < target_tokens:
        raise ValueError(
            f"Prompt undershot target: got {actual} tokens, wanted {target_tokens}. "
            "Increase the SEED multiplier in prompts.py."
        )
    return input_ids, actual
