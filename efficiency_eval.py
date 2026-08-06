import os
import sys
import argparse
import torch
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from utils.reproducibility import set_seed
from utils.config import load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters.
    
    Returns:
        Dict with total, trainable, and non-trainable parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "total_params_m": round(total / 1e6, 2),
        "trainable_params": trainable,
        "trainable_params_m": round(trainable / 1e6, 2),
        "non_trainable_params": total - trainable,
    }


def measure_latency(model: torch.nn.Module, device: torch.device, n_runs: int = 100) -> dict:
    """Measure inference latency.
    
    Returns:
        Dict with mean and std latency in milliseconds.
    """
    model.eval()
    model.to(device)
    
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_ehr = torch.randn(1, 12).to(device)
    dummy_ehr_mask = torch.ones(1).to(device)
    dummy_img_mask = torch.ones(1).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            model(dummy_img, dummy_ehr, dummy_ehr_mask, dummy_img_mask)
    
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_img, dummy_ehr, dummy_ehr_mask, dummy_img_mask)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
    }


def estimate_flops(model: torch.nn.Module) -> dict:
    """Estimate FLOPs using fvcore.
    
    Returns:
        Dict with FLOPs in GFLOPs.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        model.eval()
        dummy_img = torch.randn(1, 3, 224, 224).to(device)
        dummy_ehr = torch.randn(1, 12).to(device)
        dummy_ehr_mask = torch.ones(1).to(device)
        dummy_img_mask = torch.ones(1).to(device)
        
        flop_counter = FlopCountAnalysis(
            model, (dummy_img, dummy_ehr, dummy_ehr_mask, dummy_img_mask)
        )
        total_flops = flop_counter.total()
        return {
            "flops": total_flops,
            "flops_g": round(total_flops / 1e9, 2),
        }
    except Exception as e:
        print(f"FLOPs estimation failed: {e}")
        return {"flops": 0, "flops_g": 0.0}


def main():
    parser = argparse.ArgumentParser(description="HEAL-Net Efficiency Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()
    
    config_path = os.path.join(_SCRIPT_DIR, args.config)
    config = load_config(config_path)
    
    set_seed(config.get("seed", 42))
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    model = build_model(config)
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model.to(device)
    
    print(f"\n{'='*60}")
    print("HEAL-Net Efficiency Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total:     {params['total_params_m']:.2f}M ({params['total_params']:,})")
    print(f"  Trainable: {params['trainable_params_m']:.2f}M")
    
    flops = estimate_flops(model)
    print(f"\nFLOPs: {flops['flops_g']:.2f}G")
    
    latency = measure_latency(model, device, n_runs=args.n_runs)
    print(f"\nInference Latency ({args.n_runs} runs):")
    print(f"  Mean:   {latency['mean_latency_ms']:.2f} ms")
    print(f"  Std:    {latency['std_latency_ms']:.2f} ms")
    print(f"  Min:    {latency['min_latency_ms']:.2f} ms")
    print(f"  Max:    {latency['max_latency_ms']:.2f} ms")
    
    results = {**params, **flops, **latency}
    
    log_dir = config.get("experiment", {}).get("log_dir", "./results")
    os.makedirs(log_dir, exist_ok=True)
    
    import json
    output_path = os.path.join(log_dir, "efficiency.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
