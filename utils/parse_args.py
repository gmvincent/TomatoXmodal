import argparse
import os
import random
import numpy as np
import torch
from warnings import warn

import torch
import torch.distributed as dist

from .yaml_config_hook import yaml_config_hook


def parse_args(config=None, desc="Multi-Task", **kwargs):
    parser = argparse.ArgumentParser(description=desc)

    # parse config file first, then add arguments from config file
    config = "./configs/default_config.yaml" if config is None else config
    parser.add_argument("--config", default=config)
    args, unknown = parser.parse_known_args()
    config = yaml_config_hook(args.config)

    # add arguments from `config` dictionary into parser, handling boolean args too
    bool_configs = [
        "pretrained",
        "dataparallel",
        "ddp",
    ]
    for k, v in config.items():
        if k == "config":  # already added config earlier, so skip
            continue
        v = kwargs.get(k, v)
        if k in bool_configs:
            parser.add_argument(f"--{k}", default=v, type=str)
        elif k == "hidden_layers":
            parser.add_argument(f"--{k}", default=v, type=list)
        elif k == "gpu":
            # Avoid duplicate addition of the --gpu argument
            if not any(arg.dest == "gpu" for arg in parser._actions):
                parser.add_argument(f"--{k}", default=v, type=list,  help="Comma-separated list of GPU IDs to use, e.g., '0,1,2'")
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    for k, v in kwargs.items():
        if k not in config:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # ddp related arguments
    parser.add_argument("--rank", default=0, type=int, help="Rank of the current process (for DDP).")
    parser.add_argument(
        "--world_size", type=int, help="Number of total processes (for DDP). Defaults to the number of GPUs."
    )
    parser.add_argument(
        "--master_addr", default="localhost", type=str, help="Address for the master process."
    )
    parser.add_argument(
        "--master_port", default="12355", type=str, help="Port for the master process."
    )

    # parse added arguments
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if k in bool_configs and isinstance(v, str):
            if v.lower() in ["yes", "no", "true", "false", "none"]:
                exec(f'args.{k} = v.lower() in ["yes", "true"]')
    

    # Ensure output path exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    if isinstance(args.gpu, str):
        if args.gpu.strip() == "":
            args.gpu = []
        else:
            args.gpu = [int(x) for x in args.gpu.split(",") if x.strip().isdigit()]
    elif isinstance(args.gpu, int):
        args.gpu = [args.gpu]
    elif not isinstance(args.gpu, list):
        args.gpu = []

    # Set world size
    if not args.world_size:
        args.world_size = len(args.gpu)
            
    if args.ddp:
        parser.add_argument("--port", default=str(find_free_port()))
        args.master_port = find_free_port()
        
    # Validate dataparallel or ddp and GPU arguments
    if args.dataparallel and len(args.gpu) < 2:
        raise ValueError(
            "DataParallel requires multiple GPUs. Specify at least two GPUs using --gpu, e.g., --gpu 0,1"
        )
    if len(args.gpu) > 1 and not (args.dataparallel or args.ddp):
        raise ValueError(
            "Multiple GPUs specified, but DataParallel or DistibutedDataParallel is not enabled. Use --dataparallel True or --ddp True to enable it."
        )
    if args.ddp and len(args.gpu) != args.world_size:
        raise ValueError(
            f"DDP requires the number of GPUs ({len(args.gpu)}) to match the world size ({args.world_size})."
        )
    if args.ddp and len(args.gpu) < 1:
        raise ValueError("DDP requires at least one GPU.")
    
    return args

def find_free_port():
    # taken from https://github.com/ShigekiKarita/pytorch-distributed-slurm-example/blob/master/main_distributed.py
    import socket

    s = socket.socket()
    s.bind(("", 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def setup_ddp(args, rank):
    """
    Initializes the process group for Distributed Data Parallel (DDP).
    """
    
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    # Initialize the process group
    dist.init_process_group(
            backend="nccl",  # Use NCCL for GPUs; use "gloo" for CPU
            init_method="env://",  # Initialize via environment variables
            world_size=args.world_size,
            rank=rank,
        )

    # Set the GPU device for this process
    torch.device(f"cuda:{args.gpu[rank]}")
    
def cleanup_ddp():
    """
    Cleans up the process group for Distributed Data Parallel (DDP).
    """
    dist.barrier()
    dist.destroy_process_group()