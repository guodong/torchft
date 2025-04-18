#!/usr/bin/env python3

"""
Example of using ManagerFFT with error bus for enhanced fault tolerance.

This example demonstrates how to use the ManagerFFT class, which extends the 
standard Manager with error bus functionality for improved fault detection and recovery.

Usage:
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 manager_fft_example.py
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Dict, Any

from torchft.manager_fft import ManagerFFT
from torchft.process_group import TorchProcessGroup


def setup_process_group():
    """Set up the PyTorch distributed process group."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group("gloo")
    
    print(f"Initialized process group: rank {rank}/{world_size}")
    return TorchProcessGroup(dist.group.WORLD)


def dummy_model_state() -> Dict[str, Any]:
    """Return a dummy model state for demonstration purposes."""
    return {
        "model_weights": torch.randn(10, 10),
        "optimizer_state": {"step": 0, "lr": 0.01},
        "epoch": 0
    }


def load_state_dict(state_dict: Dict[str, Any]) -> None:
    """Load state dict function for ManagerFFT."""
    print(f"Loading state dict: {state_dict}")
    # In a real scenario, you would apply this state to your model and optimizer


def main():
    # Set up the process environment 
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create process group
    pg = setup_process_group()
    
    # Create ManagerFFT instance
    manager = ManagerFFT(
        pg=pg,
        load_state_dict=load_state_dict,
        state_dict=dummy_model_state,
        min_replica_size=1,
        rank=rank,
        world_size=world_size,
        # Error bus configuration
        error_bus_host='127.0.0.1',
        error_bus_port=22223,
        error_bus_debug=True
    )
    
    try:
        # Simulate training loop
        for step in range(5):
            print(f"[Rank {rank}] Starting step {step}")
            
            # Start quorum at the beginning of each step
            manager.start_quorum()
            
            # Simulate training step
            time.sleep(0.5)
            
            # Simulate computing gradients
            gradient = torch.randn(10)
            
            # Perform fault-tolerant allreduce
            print(f"[Rank {rank}] Performing allreduce")
            result_future = manager.allreduce(gradient)
            result = result_future.wait()
            
            # Check if we should commit this step
            if manager.should_commit():
                print(f"[Rank {rank}] Step {step} succeeded with {manager.num_participants()} participants")
            else:
                print(f"[Rank {rank}] Step {step} failed or not enough participants")
            
            # Simulate an error on rank 0 at step 3
            if rank == 0 and step == 3:
                print(f"[Rank {rank}] Simulating an error")
                manager.report_error(Exception("Simulated error"))
        
        print(f"[Rank {rank}] Training completed successfully")
        print(f"[Rank {rank}] Total batches committed: {manager.batches_committed()}")
    
    finally:
        # Clean up
        manager.shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 