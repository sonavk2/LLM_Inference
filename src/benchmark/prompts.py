"""Synthetic prompt construction with token-length verification.

Long-context experiments are meaningless if the actual token count drifts from
the requested target, so prompts are built by tiling a seed string and then
truncated to the exact target length using the tokenizer.
"""

# A short seed sentence; tiles and truncates to whatever target length is asked.
SEED = "The quick brown fox jumps over the lazy dog. "


def build_synthetic_prompt(tokenizer, target_tokens):
    """
    Build a prompt of exactly `target_tokens` tokens.

    Returns:
        input_ids: torch.LongTensor of shape (1, target_tokens)
        actual_token_count: int (always equals target_tokens after truncation)
    """
    # Overshoot first so we can truncate down to the exact target.
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
