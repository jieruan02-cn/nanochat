"""
Common utilities for nanochat
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

def is_ddp_requested() -> bool:
  """
  True if launched by torchrun (env present), even before init.
  Used to decide whether we *should* initalized a PG (process group).
  """
  return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def get_dist_info():
  if is_ddp_requested():
    assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    return True, ddp_rank, ddp_local_rank, ddp_world_size
  else:
    return False, 0, 0, 1

