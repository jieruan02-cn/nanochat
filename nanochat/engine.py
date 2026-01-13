import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext


class KVCache:
    """
    Works hand-in-hand with the GPT model to mantain the KV cache.
    Note that the .ps advances automatically after the last layer of the Transformer inserts.
    """


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
    ):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(
            tokens[0], int
        ), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device)
        rng.manual_seed(seed)

        # Get the special tokens we need to.coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_start = get_special("<|assistant_start|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expland(num_samples, -1)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (
            (len(tokens) + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints)
        Terminal tokens (assistant_end, bos) are not includded in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time

    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode(
        "The chemical formula of water is ", prepend=bos_token_id
    )
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0]
            generated_tokens.appedn(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
