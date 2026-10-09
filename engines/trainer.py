import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.early_stopping import EarlyStopping
from losses import MENDNetLoss


class Trainer:
    """Stage-aware training engine for MEND-Net.

    Supports the three-stage progressive pipeline described in the paper:
    - search:       alternating weight / architecture optimization with a
                    differentiable hardware (FLOPs) penalty, followed by
                    argmax discretization.
    - distill:      cross-modal covariance relation alignment + KL soft-label
                    distillation driven by the teacher model.
    - compensate:   adds the DFC reconstruction loss (Eq.19) on top of
                    distillation.

    Args:
        model: MENDNet model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration dictionary.
        recorder: Optional ExperimentRecorder instance.
        stage: Training stage ('search', 'distill', 'compensate').
        teacher: Optional teacher model (required for distillation).
        loss_fn: Optional MENDNetLoss instance; built from config if None.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        recorder=None,
        stage: str = "distill",
        teacher=None,
        loss_fn=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.recorder = recorder
        self.stage = stage
        self.teacher = teacher

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if loss_fn is None:
            hkd_cfg = config.get("hkd", {})
            nas_cfg = config.get("ea_nas", {})
            dfc_cfg = config.get("dfc", {})
            loss_fn = MENDNetLoss(
                gamma=hkd_cfg.get("gamma", 1.0),
                beta=hkd_cfg.get("beta", 0.5),
                lambda_flops=nas_cfg.get("lambda_flops", 0.05),
                rho=dfc_cfg.get("rho", 0.1),
                temperature=hkd_cfg.get("temperature", 3.0),
                target_flops=nas_cfg.get("target_flops"),
            )
        self.loss_fn = loss_fn.to(self.device)

        train_cfg = config.get("training", {})
        nas_cfg = config.get("ea_nas", {})
        self.epochs = train_cfg.get("epochs", 200)
        self.grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
        self.amp_enabled = train_cfg.get("amp", True)
        self.log_interval = config.get("experiment", {}).get("log_interval", 10)

        self.model.to(self.device)
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()

        self.is_search = self.stage == "search"
        self.arch_optimizer = None

        weight_lr = (
            nas_cfg.get("search_lr", train_cfg.get("learning_rate", 0.001))
            if self.is_search
            else train_cfg.get("learning_rate", 0.001)
        )
        weight_params = self._weight_parameters()
        self.optimizer = optim.AdamW(
            weight_params,
            lr=weight_lr,
            weight_decay=train_cfg.get("weight_decay", 0.0001),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=train_cfg.get("min_lr", 1e-6),
        )

        if self.is_search:
            nas_cfg = config.get("ea_nas", {})
            self.arch_optimizer = optim.AdamW(
                [self.model.search_space.arch_params],
                lr=nas_cfg.get("arch_lr", 0.0003),
                weight_decay=nas_cfg.get("arch_weight_decay", 0.0001),
            )
            self.arch_scheduler = CosineAnnealingLR(
                self.arch_optimizer,
                T_max=max(self.epochs, 1),
                eta_min=nas_cfg.get("arch_min_lr", 1e-6),
            )

        self.scaler = GradScaler(enabled=self.amp_enabled)

        save_dir = config.get("experiment", {}).get("save_dir", "./weights")
        os.makedirs(save_dir, exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 20),
            path=os.path.join(save_dir, "mend_net_best.pth"),
        )

        self.best_metric = 0.0
        self.loss_history = []

    def _weight_parameters(self):
        """Return network-weight parameters, excluding architecture params.

        During the search stage the architecture parameters alpha are
        optimized by a dedicated optimizer, so they must be excluded from
        the network-weight optimizer.
        """
        if not self.is_search or not hasattr(self.model, "search_space"):
            return list(self.model.parameters())
        arch_ids = {id(self.model.search_space.arch_params)}
        return [p for p in self.model.parameters() if id(p) not in arch_ids]

    def train_epoch(self) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary with 'loss' and optional stage component losses.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        component_sum = {}

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)

            self.optimizer.zero_grad()
            with autocast(enabled=self.amp_enabled):
                loss_dict = self.loss_fn(self.model, self.teacher, batch, self.stage)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            for key, val in loss_dict.items():
                if key == "loss":
                    continue
                if isinstance(val, torch.Tensor):
                    if val.dim() != 0:
                        continue
                    val = val.detach().item()
                if isinstance(val, (int, float)):
                    component_sum[key] = component_sum.get(key, 0.0) + float(val)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        result = {"loss": total_loss / max(num_batches, 1)}
        for key, val in component_sum.items():
            result[key] = val / max(num_batches, 1)
        return result

    def update_arch(self) -> dict:
        """Update architecture parameters on the validation split.

        Implements the outer loop of the alternating bilevel optimization
        (Eq.7): fixed network weights, the architecture parameters alpha are
        updated with gradients computed on the validation split D_val.
        """
        if not self.is_search or self.arch_optimizer is None:
            return {"arch_loss": 0.0}

        self.model.train()
        arch_loss_sum = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            self.arch_optimizer.zero_grad()
            with autocast(enabled=self.amp_enabled):
                loss_dict = self.loss_fn(self.model, self.teacher, batch, self.stage)
                arch_loss = loss_dict["loss"]
            self.scaler.scale(arch_loss).backward()
            self.scaler.step(self.arch_optimizer)
            self.scaler.update()
            arch_loss_sum += arch_loss.item()
            num_batches += 1

        return {"arch_loss": arch_loss_sum / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model.

        Returns:
            Dictionary with 'loss' and 'accuracy'.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            logits = self.model(batch["image"], batch["ehr"], batch["ehr_mask"], batch["img_mask"])
            loss = nn.CrossEntropyLoss()(logits, batch["label"])
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch["label"].size(0)
            correct += predicted.eq(batch["label"]).sum().item()

        return {
            "loss": total_loss / max(len(self.val_loader), 1),
            "accuracy": correct / max(total, 1),
        }

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.to(self.device)
            else:
                out[key] = val
        return out

    def fit(self, epochs: int = None) -> None:
        """Full training loop.

        Args:
            epochs: Override number of epochs.
        """
        if epochs is not None:
            self.epochs = epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.epochs, 1),
                eta_min=self.config.get("training", {}).get("min_lr", 1e-6),
            )
            if self.arch_optimizer is not None:
                nas_cfg = self.config.get("ea_nas", {})
                self.arch_scheduler = CosineAnnealingLR(
                    self.arch_optimizer,
                    T_max=max(self.epochs, 1),
                    eta_min=nas_cfg.get("arch_min_lr", 1e-6),
                )

        log_dir = self.config.get("experiment", {}).get("log_dir", "./results")
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            if self.arch_optimizer is not None:
                arch_metrics = self.update_arch()
                self.arch_scheduler.step()
                train_metrics.update(arch_metrics)
            val_metrics = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            log_entry = {
                "epoch": epoch,
                "stage": self.stage,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": current_lr,
                **{k: v for k, v in train_metrics.items() if k != "loss"},
            }
            self.loss_history.append(log_entry)

            if epoch % self.log_interval == 0 or epoch == 1:
                print(
                    f"[{self.stage}] Epoch [{epoch}/{self.epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            if val_metrics["accuracy"] > self.best_metric:
                self.best_metric = val_metrics["accuracy"]
                self.early_stopping(val_metrics["loss"], self.model)

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        losses_path = os.path.join(log_dir, f"losses_{self.stage}.json")
        with open(losses_path, "w") as f:
            json.dump(self.loss_history, f, indent=2)

        if self.recorder is not None:
            metric_name = self.config.get("task_primary_metric", "accuracy")
            self.recorder.record_metric(f"best_{metric_name}", self.best_metric)
            self.recorder.save()
