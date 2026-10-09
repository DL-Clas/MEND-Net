import torch
import torch.nn as nn
import torch.nn.functional as F


# Candidate operations for the search space
class SepConv3x3(nn.Module):
    """Depthwise separable 3x3 convolution. Lightweight operator for NAS search space."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SepConv5x5(nn.Module):
    """Depthwise separable 5x5 convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv3x3(nn.Module):
    """Dilated 3x3 convolution with rate 2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class AvgPool(nn.Module):
    """Average pooling 3x3."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class MaxPool(nn.Module):
    """Max pooling 3x3."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SkipConnect(nn.Module):
    """Identity skip connection with optional 1x1 projection."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels != out_channels:
            self.op = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.op = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SparseAttention(nn.Module):
    """Lightweight sparse channel attention.

    Computes a channel-wise gating vector and re-weights the input
    feature map. The gating path outputs a 1D vector, so no
    spatial BatchNorm2d is applied (BatchNorm2d requires 4D input).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(out_channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(out_channels // 4, 1), out_channels),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.op(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# Operation registry
OPERATIONS = {
    "sep_conv_3x3": SepConv3x3,
    "sep_conv_5x5": SepConv5x5,
    "dil_conv_3x3": DilConv3x3,
    "avg_pool": AvgPool,
    "max_pool": MaxPool,
    "skip_connect": SkipConnect,
    "sparse_attention": SparseAttention,
}


class MixedOp(nn.Module):
    """Mixed operation: weighted combination of candidate ops.
    
    Implements Eq.1-2 from MEND-Net paper:
    f_{i,j}(x_i) = sum_o softmax(alpha_{i,j}^o) * o(x_i)
    """
    def __init__(self, in_channels: int, out_channels: int, op_names: list):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in op_names:
            op_cls = OPERATIONS[name]
            self.ops.append(op_cls(in_channels, out_channels))
        self.num_ops = len(op_names)
        self.discrete = False
        self.best_idx = 0
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """Forward with architecture weights.
        
        Args:
            x: Input tensor (B, C, H, W).
            weights: Architecture weights (num_ops,) or None when the edge
                has been discretized (only the selected op is applied).
        """
        if self.discrete:
            return self.ops[self.best_idx](x)
        return sum(w * op(x) for w, op in zip(weights, self.ops))
    
    def discretize(self, best_idx: int) -> None:
        """Keep only the single best operation (argmax discretization).

        Implements Eq.8 from the paper:
        o*_{i,j} = argmax_o alpha^o_{i,j}
        The candidate modules are kept (so checkpoints stay loadable) but only
        the selected operation is executed.
        
        Args:
            best_idx: Index of the operation with the largest weight.
        """
        self.discrete = True
        self.best_idx = int(best_idx)


class DifferentiableSearchSpace(nn.Module):
    """Continuous search space for EA-NAS.
    
    Constructs a DAG with mixed operations on each edge.
    Architecture parameters alpha control operation selection.
    
    Args:
        in_channels: Number of input channels.
        hidden_channels: Hidden channel dimension.
        num_nodes: Number of intermediate nodes in the DAG.
        op_names: List of candidate operation names.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int = 4,
        op_names: list = None,
    ):
        super().__init__()
        if op_names is None:
            op_names = list(OPERATIONS.keys())
        
        self.num_nodes = num_nodes
        self.op_names = op_names
        self.num_ops = len(op_names)
        
        self.ops = nn.ModuleList()
        self._edge_indices = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                in_ch = in_channels if i == 0 else hidden_channels
                self.ops.append(MixedOp(in_ch, hidden_channels, op_names))
                self._edge_indices.append((i, j))
        
        self._edge_to_idx = {edge: k for k, edge in enumerate(self._edge_indices)}
        
        self.discrete = False
        
        num_edges = len(self.ops)
        self.arch_params = nn.Parameter(
            1e-3 * torch.randn(num_edges, self.num_ops)
        )
        
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with continuous relaxation.
        
        Implements Eq.2 from the paper:
        x_j = sum_{i < j} f_{i,j}(x_i)
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Output tensor (B, hidden_channels, H, W).
        """
        x = self.proj_in(x)
        
        nodes = [x]
        weights = None if self.discrete else F.softmax(self.arch_params, dim=-1)
        spatial = x.shape[2:]
        
        for j in range(1, self.num_nodes):
            acc = None
            for i in range(j):
                key = (i, j)
                if key not in self._edge_to_idx:
                    continue
                idx = self._edge_to_idx[key]
                node_i = nodes[i]
                if node_i.shape[2:] != spatial:
                    node_i = F.adaptive_avg_pool2d(node_i, spatial)
                out = self.ops[idx](
                    node_i, None if weights is None else weights[idx]
                )
                acc = out if acc is None else acc + out
            if acc is None:
                acc = nodes[-1]
            nodes.append(acc)
        
        return nodes[-1]
    
    def discretize(self) -> None:
        """Discretize the architecture via argmax on architecture weights.
        
        Implements Eq.8 from the paper:
        o*_{i,j} = argmax_o alpha^o_{i,j}
        Each mixed edge selects the operation with the largest weight and the
        architecture parameters are frozen afterwards. Candidate modules are
        retained so that saved checkpoints remain loadable by downstream
        training stages.
        """
        chosen = F.softmax(self.arch_params, dim=-1).argmax(dim=-1)
        for idx, op in enumerate(self.ops):
            op.discretize(int(chosen[idx].item()))
        self.discrete = True
        self.arch_params.requires_grad = False
    
    def get_arch_weights(self) -> torch.Tensor:
        """Get current architecture weights (after softmax)."""
        return F.softmax(self.arch_params, dim=-1)
    
    @staticmethod
    def _op_flops(op_module: nn.Module, input_size: int) -> float:
        """Estimate FLOPs of a single candidate operation.
        
        Counts multiply-accumulate FLOPs of Conv2d and Linear layers.
        
        Args:
            op_module: Candidate operation module.
            input_size: Spatial feature map size (H == W).
            
        Returns:
            Estimated FLOPs.
        """
        h = w = input_size
        flops = 0.0
        if isinstance(op_module, nn.Sequential):
            layers = list(op_module)
        elif hasattr(op_module, "op") and isinstance(op_module.op, nn.Sequential):
            layers = list(op_module.op)
        else:
            layers = [op_module]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                kh, kw = layer.kernel_size
                if layer.groups == layer.in_channels and layer.groups > 1:
                    flops += layer.in_channels * kh * kw * h * w
                else:
                    flops += layer.in_channels * layer.out_channels * kh * kw * h * w
            elif isinstance(layer, nn.Linear):
                flops += layer.in_features * layer.out_features
        return flops
    
    def expected_flops(self, input_size: int = 7) -> torch.Tensor:
        """Differentiable expected FLOPs of the search space.
        
        Implements Eq.5 from the paper:
        FLOPs(alpha) = sum_{i<j} sum_o softmax(alpha^o_{i,j}) * FLOPs(o)
        The result is a differentiable function of the architecture
        parameters, providing a computational cost penalty gradient to alpha.
        
        Args:
            input_size: Spatial feature map size fed into the search space.
            
        Returns:
            Scalar tensor with expected FLOPs.
        """
        weights = F.softmax(self.arch_params, dim=-1)
        device = weights.device
        dtype = weights.dtype
        
        total = torch.zeros((), dtype=dtype, device=device)
        for idx, op in enumerate(self.ops):
            op_flops = torch.tensor(
                [self._op_flops(m, input_size) for m in op.ops],
                dtype=dtype, device=device,
            )
            total = total + torch.dot(weights[idx], op_flops)
        return total
    
    def get_flops(self, input_size: int = 7) -> float:
        """Get scalar FLOPs estimate (non-differentiable).
        
        Args:
            input_size: Spatial feature map size.
            
        Returns:
            Estimated FLOPs as a float.
        """
        return float(self.expected_flops(input_size).detach().item())


class EfficiencyAwareLoss(nn.Module):
    """Efficiency-aware loss with FLOPs penalty.
    
    Implements Eq.3-6 from MEND-Net paper:
    L_NAS = L_CE + lambda * log(FLOPs(alpha))
    """
    def __init__(self, lambda_flops: float = 0.05, target_flops: float = 2.5e9):
        super().__init__()
        self.lambda_flops = lambda_flops
        self.target_flops = target_flops
    
    def forward(
        self,
        ce_loss: torch.Tensor,
        current_flops: float,
    ) -> torch.Tensor:
        """Compute efficiency-aware loss.
        
        Args:
            ce_loss: Cross-entropy loss tensor.
            current_flops: Current architecture FLOPs estimate.
            
        Returns:
            Combined loss with computational cost penalty.
        """
        if current_flops <= 0:
            current_flops = 1.0
        flops_penalty = torch.log(torch.tensor(current_flops, dtype=torch.float32))
        return ce_loss + self.lambda_flops * flops_penalty
