from operator import is_
import time
import torch.distributed as dist
import os
import torch

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

def distributed_setup():
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # initialize the process group
    dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank)
    
    return rank, world_size, local_rank

def distributed_cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()
    
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()