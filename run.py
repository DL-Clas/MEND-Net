import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MIMICDataset, get_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from data.synthetic import create_synthetic_dataset
from models import build_model, build_teacher
from engines.trainer import Trainer
from engines.evaluator import Evaluator
from losses import MENDNetLoss
from utils.reproducibility import set_seed
from utils.config import load_config
from utils.experiment_recorder import ExperimentRecorder

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

TASK_PRIMARY_METRIC = {
    "segmentation": "dice",
    "classification": "accuracy",
    "detection": "map",
}

MODULE_NAMES = ["EA-NAS", "HKD", "DFC"]


def train(config: dict, teacher_path: str = None, stage: str = "all") -> None:
    """Training pipeline with three progressive stages.
    
    Args:
        config: Configuration dictionary.
        teacher_path: Path to pretrained teacher model.
        stage: Training stage ('search', 'distill', 'compensate', 'all').
    """
    set_seed(config.get("seed", 42))
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    
    train_transform = get_train_transforms(config.get("data", {}).get("image_size", 224))
    val_transform = get_val_transforms(config.get("data", {}).get("image_size", 224))
    
    data_cfg = config.get("data", {})
    train_dir = data_cfg.get("train_dir", "./data/mimic/train")
    val_dir = data_cfg.get("val_dir", "./data/mimic/val")
    dfc_cfg = config.get("dfc", {})
    modality_drop_prob = dfc_cfg.get("modality_drop_prob", 0.5)
    
    if os.path.exists(train_dir):
        train_dataset = MIMICDataset(
            train_dir, split="train", transform=train_transform,
            modality_drop_prob=modality_drop_prob,
        )
        val_dataset = MIMICDataset(val_dir, split="val", transform=val_transform)
    else:
        print(f"Warning: Data directory {train_dir} not found. Using synthetic data for testing.")
        train_dataset = create_synthetic_dataset(100, config)
        val_dataset = create_synthetic_dataset(20, config)
    
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    
    teacher = None
    if teacher_path and os.path.exists(teacher_path):
        teacher = build_teacher(config)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.to(device)
        teacher.eval()
        print(f"Loaded teacher from {teacher_path}")
    elif config.get("teacher", {}).get("checkpoint") and os.path.exists(
        config["teacher"]["checkpoint"]
    ):
        teacher = build_teacher(config)
        teacher.load_state_dict(
            torch.load(config["teacher"]["checkpoint"], map_location=device)
        )
        teacher.to(device)
        teacher.eval()
        print(f"Loaded teacher from {config['teacher']['checkpoint']}")
    elif stage in ("all", "distill", "compensate"):
        print("Warning: no teacher provided; distillation terms will be skipped.")
    
    recorder = ExperimentRecorder(
        innovation_names=config.get("ablation", {}).get("innovation_names", []),
        sota_baselines=config.get("sota", {}),
        baseline_metrics={},
        iteration_round=1,
        dataset_name=data_cfg.get("dataset_name", "MIMIC"),
    )
    
    stages = ["search", "distill", "compensate"] if stage == "all" else [stage]
    prev_checkpoint = None
    
    for current_stage in stages:
        print(f"\n{'='*60}")
        print(f"Stage: {current_stage.upper()}")
        print(f"{'='*60}")
        
        model = build_model(config)
        if prev_checkpoint is None and current_stage in ("distill", "compensate"):
            prev_name = "search" if current_stage == "distill" else "distill"
            prev_checkpoint = os.path.join(
                config.get("experiment", {}).get("save_dir", "./weights"),
                f"mend_net_{prev_name}.pth",
            )
        if prev_checkpoint and os.path.exists(prev_checkpoint):
            model.load_state_dict(torch.load(prev_checkpoint, map_location=device))
            print(f"Loaded weights from previous stage: {prev_checkpoint}")
        
        hkd_cfg = config.get("hkd", {})
        nas_cfg = config.get("ea_nas", {})
        
        loss_fn = MENDNetLoss(
            gamma=hkd_cfg.get("gamma", 1.0),
            beta=hkd_cfg.get("beta", 0.5),
            lambda_flops=nas_cfg.get("lambda_flops", 0.05),
            rho=dfc_cfg.get("rho", 0.1),
            temperature=hkd_cfg.get("temperature", 3.0),
            target_flops=nas_cfg.get("target_flops"),
        )
        
        trainer = Trainer(
            model, train_loader, val_loader, config,
            recorder=recorder, stage=current_stage,
            teacher=teacher, loss_fn=loss_fn,
        )
        
        stage_epochs = {
            "search": nas_cfg.get("epochs", 50),
            "distill": config.get("training", {}).get("epochs", 200),
            "compensate": config.get("training", {}).get("epochs", 200),
        }
        epochs = stage_epochs.get(current_stage, 200)
        
        trainer.fit(epochs=epochs)
        
        if current_stage == "search":
            model.search_space.discretize()
            print("Search space discretized via argmax.")
        
        save_dir = config.get("experiment", {}).get("save_dir", "./weights")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"mend_net_{current_stage}.pth")
        torch.save(model.state_dict(), save_path)
        prev_checkpoint = save_path
        print(f"Saved {current_stage} model to {save_path}")
    
    print("\nTraining pipeline complete.")


def test(config: dict, model_path: str) -> None:
    """Evaluate a trained model.
    
    Args:
        config: Configuration dictionary.
        model_path: Path to trained model checkpoint.
    """
    set_seed(config.get("seed", 42))
    
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    test_transform = get_val_transforms(config.get("data", {}).get("image_size", 224))
    
    data_cfg = config.get("data", {})
    test_dir = data_cfg.get("test_dir", "./data/mimic/test")
    
    if os.path.exists(test_dir):
        test_dataset = MIMICDataset(test_dir, split="test", transform=test_transform)
    else:
        print(f"Warning: Test directory {test_dir} not found. Using synthetic data.")
        test_dataset = create_synthetic_dataset(20, config)
    
    batch_size = config.get("training", {}).get("batch_size", 64)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    
    evaluator = Evaluator(model, test_loader, config)
    results = evaluator.evaluate()
    
    efficiency = evaluator.compute_efficiency()
    results.update(efficiency)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"Params:    {results['params_m']:.2f}M")
    print(f"FLOPs:     {results['flops_g']:.2f}G")
    
    log_dir = config.get("experiment", {}).get("log_dir", "./results")
    evaluator.save_results(results, log_dir)


def run_ablation(
    config: dict,
    num_seeds: int = 5,
    teacher_path: str = None,
) -> None:
    """Run leave-one-out ablation for all innovation modules.

    For each random seed, trains the full model and three variants that
    remove exactly one module (EA-NAS / HKD / DFC), then evaluates every
    variant under both complete input and 50% EHR-missing conditions.
    This mirrors the full-vs-V4/V5/V6 contrasts reported in the paper.

    Args:
        config: Configuration dictionary.
        num_seeds: Number of random seeds for statistical significance.
        teacher_path: Optional pretrained teacher checkpoint.
    """
    metric_name = TASK_PRIMARY_METRIC.get(
        config.get("task_type", "classification"), "accuracy"
    )
    
    with_scores = {name: {"ideal": [], "missing": []} for name in MODULE_NAMES}
    without_scores = {name: {"ideal": [], "missing": []} for name in MODULE_NAMES}
    full_scores = {"ideal": [], "missing": []}
    
    train_dir = config.get("data", {}).get("train_dir", "./data/mimic/train")
    val_dir = config.get("data", {}).get("val_dir", "./data/mimic/val")
    dfc_cfg = config.get("dfc", {})
    modality_drop_prob = dfc_cfg.get("modality_drop_prob", 0.5)
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    if os.path.exists(train_dir):
        train_dataset = MIMICDataset(
            train_dir, split="train", transform=train_transform,
            modality_drop_prob=modality_drop_prob,
        )
        val_dataset = MIMICDataset(val_dir, split="val", transform=val_transform)
    else:
        train_dataset = create_synthetic_dataset(100, config)
        val_dataset = create_synthetic_dataset(20, config)
    
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=config.get("training", {}).get("num_workers", 0))
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    teacher = None
    if teacher_path and os.path.exists(teacher_path):
        teacher = build_teacher(config)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.to(device)
        teacher.eval()
    else:
        teacher = build_teacher(config)
        teacher.to(device)
        teacher.eval()
    
    epochs = config.get("training", {}).get("epochs", 200)
    
    for seed in range(num_seeds):
        seed_val = config.get("seed", 42) + seed
        print(f"\n{'='*60}")
        print(f"Ablation Seed {seed + 1}/{num_seeds} (seed={seed_val})")
        print(f"{'='*60}")
        
        combos = {
            "full": None,
            "w/o EA-NAS": ["EA-NAS"],
            "w/o HKD": ["HKD"],
            "w/o DFC": ["DFC"],
        }
        trained = {}
        
        for combo_name, disabled in combos.items():
            set_seed(seed_val)
            model = build_model(config, disabled_modules=disabled)
            trainer = Trainer(
                model, train_loader, val_loader, config,
                stage="compensate", teacher=teacher,
            )
            trainer.fit(epochs=epochs)
            
            evaluator = Evaluator(model, val_loader, config)
            ideal_acc = evaluator.evaluate(missing_rate=0.0)["accuracy"]
            missing_acc = evaluator.evaluate(missing_rate=0.5)["accuracy"]
            trained[combo_name] = {"ideal": ideal_acc, "missing": missing_acc}
            print(
                f"  [{combo_name}] ideal={ideal_acc:.4f} | "
                f"50% missing={missing_acc:.4f}"
            )
        
        full_scores["ideal"].append(trained["full"]["ideal"])
        full_scores["missing"].append(trained["full"]["missing"])
        
        without_map = {
            "EA-NAS": "w/o EA-NAS",
            "HKD": "w/o HKD",
            "DFC": "w/o DFC",
        }
        for name in MODULE_NAMES:
            wout = trained[without_map[name]]
            with_scores[name]["ideal"].append(trained["full"]["ideal"])
            with_scores[name]["missing"].append(trained["full"]["missing"])
            without_scores[name]["ideal"].append(wout["ideal"])
            without_scores[name]["missing"].append(wout["missing"])
    
    recorder = ExperimentRecorder(
        innovation_names=MODULE_NAMES,
        sota_baselines=config.get("sota", {}),
        baseline_metrics={},
        iteration_round=1,
        dataset_name=config.get("data", {}).get("dataset_name", "MIMIC"),
    )
    
    for name in MODULE_NAMES:
        recorder.record_ablation_batch(
            f"{name} (ideal)",
            with_scores[name]["ideal"], without_scores[name]["ideal"],
            metric_name=metric_name,
            claimed_improvement=f"{name} contribution under complete input",
        )
        recorder.record_ablation_batch(
            f"{name} (50% missing)",
            with_scores[name]["missing"], without_scores[name]["missing"],
            metric_name=metric_name,
            claimed_improvement=f"{name} contribution under 50% EHR missing",
        )
    
    recorder.record_metric(
        f"best_{metric_name}",
        {"mean": float(np.mean(full_scores["ideal"])),
         "std": float(np.std(full_scores["ideal"], ddof=1)) if num_seeds > 1 else 0.0},
    )
    
    log_dir = config.get("experiment", {}).get("log_dir", "./results")
    os.makedirs(log_dir, exist_ok=True)
    recorder.save(os.path.join(log_dir, "experiment_summary.json"))
    print(f"\nAblation results saved to {log_dir}/experiment_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="MEND-Net: Hierarchical Knowledge Distillation and NAS Framework"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "ablation"],
        default="train",
        help="Running mode: train, test, or ablation"
    )
    parser.add_argument(
        "--stage", type=str,
        choices=["search", "distill", "compensate", "all"],
        default="all",
        help="Training stage (default: all)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--teacher_path", type=str, default=None,
        help="Path to pretrained teacher model"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained student model (for test mode)"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=None,
        help="Number of seeds for ablation (default: from config)"
    )
    args = parser.parse_args()
    
    config_path = os.path.join(_SCRIPT_DIR, args.config)
    config = load_config(config_path)
    
    if args.mode == "train":
        train(config, teacher_path=args.teacher_path, stage=args.stage)
    elif args.mode == "test":
        if args.model_path is None:
            save_dir = config.get("experiment", {}).get("save_dir", "./weights")
            args.model_path = os.path.join(save_dir, "mend_net_compensate.pth")
        test(config, args.model_path)
    elif args.mode == "ablation":
        num_seeds = args.num_seeds or config.get("experiment", {}).get("num_seeds", 5)
        run_ablation(config, num_seeds=num_seeds, teacher_path=args.teacher_path)


if __name__ == "__main__":
    main()
