import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MIMICDataset, get_dataloader
from data.transforms import get_val_transforms
from data.synthetic import create_synthetic_dataset
from models import build_model
from engines.evaluator import Evaluator
from utils.reproducibility import set_seed
from utils.config import load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="HEAL-Net Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    config_path = os.path.join(_SCRIPT_DIR, args.config)
    config = load_config(config_path)
    
    set_seed(config.get("seed", 42))
    
    model = build_model(config)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    
    test_transform = get_val_transforms(config.get("data", {}).get("image_size", 224))
    
    data_cfg = config.get("data", {})
    test_dir = data_cfg.get("test_dir", "./data/mimic/test")
    
    if os.path.exists(test_dir):
        test_dataset = MIMICDataset(test_dir, split="test", transform=test_transform)
    else:
        print(f"Warning: {test_dir} not found. Using synthetic data.")
        test_dataset = create_synthetic_dataset(20, config)
    
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    
    evaluator = Evaluator(model, test_loader, config)
    results = evaluator.evaluate()
    
    efficiency = evaluator.compute_efficiency()
    results.update(efficiency)
    
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"Params:    {results['params_m']:.2f}M")
    print(f"FLOPs:     {results['flops_g']:.2f}G")
    
    evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
