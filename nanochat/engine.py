class Engine:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
    pass

  def generate_batch(self, tokens, num_samples=1, **kwargs):
    pass

if __name__ == "__main__":
  """
  Quick inline test to make sure that the naive/slow model.generate function
  is equivalent to the faster Engine.generate function here.
  """
  import time
  # init compute
  ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
  device_type = autodetect_device_type()
  