"""The three pruning strategies compared in the demo.

1. `global_unstructured_prune` -- zero out individual weights (fine-grained).
   Produces a sparse tensor of the *same shape*, so nothing gets smaller or
   faster on a dense runtime.

2. `naive_structured_prune` -- physically remove channels by hand. Safe for a
   channel *inside* a residual block, broken for a channel on the block's
   *output*, because the residual add requires both branches to agree on the
   channel count. This is the "channel dependency" problem.

3. `structured_prune` -- let torch_pruning's DepGraph discover the full group of
   coupled layers and cut all of them consistently.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import torch_pruning as tp


# ---------------------------------------------------------------------------
# 0. Evidence of parameter redundancy
# ---------------------------------------------------------------------------
def _device_of(model: nn.Module) -> torch.device:
    """Where this model currently lives.

    Several helpers here take a model and an example input. Those inputs are
    built on CPU, while the model may have been left on an accelerator by an
    earlier `evaluate()` call, which moves models in place. Tracing the graph
    then fails with a device mismatch, so every entry point aligns the two.
    """
    return next(model.parameters()).device


def channel_l2_norms(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Per-output-channel L2 norm of every Conv2d filter bank.

    A channel whose norm is near zero contributes almost nothing to the next
    layer: that is the redundancy structured pruning exploits.
    """
    norms = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach().flatten(1)          # [out_channels, -1]
            norms[name] = w.norm(p=2, dim=1).cpu()
    return norms


def prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# ---------------------------------------------------------------------------
# 1. Unstructured pruning
# ---------------------------------------------------------------------------
def global_unstructured_prune(
    model: nn.Module,
    amount: float,
    permanent: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """Zero the `amount` fraction of smallest-|w| weights across the whole net.

    Args:
        permanent: when True, `torch_prune.remove` folds the binary mask into
            the weight tensor and deletes the mask, leaving an ordinary module
            that holds real zeros. When False the mask stays attached and is
            re-applied on every forward pass -- which is what keeps the model
            sparse while it is being fine-tuned. Call `make_masks_permanent`
            afterwards to bake the zeros in.
    """
    model = model if inplace else copy.deepcopy(model)
    params = [(m, "weight") for _, m in prunable_layers(model)]
    torch_prune.global_unstructured(params, pruning_method=torch_prune.L1Unstructured, amount=amount)
    if permanent:
        for m, name in params:
            torch_prune.remove(m, name)
    return model


def make_masks_permanent(model: nn.Module) -> nn.Module:
    """Fold any remaining `torch.nn.utils.prune` masks into the weights."""
    for _, m in prunable_layers(model):
        if torch_prune.is_pruned(m) and hasattr(m, "weight_mask"):
            torch_prune.remove(m, "weight")
    return model


# ---------------------------------------------------------------------------
# 2. Naive (dependency-unaware) structured pruning
# ---------------------------------------------------------------------------
def _slice_conv_out(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    new = nn.Conv2d(
        conv.in_channels, len(keep), conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
        bias=conv.bias is not None,
    )
    new.weight.data = conv.weight.data[keep].clone()
    if conv.bias is not None:
        new.bias.data = conv.bias.data[keep].clone()
    return new


def _slice_conv_in(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    new = nn.Conv2d(
        len(keep), conv.out_channels, conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
        bias=conv.bias is not None,
    )
    new.weight.data = conv.weight.data[:, keep].clone()
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def _slice_bn(bn: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
                         track_running_stats=bn.track_running_stats)
    if bn.affine:
        new.weight.data = bn.weight.data[keep].clone()
        new.bias.data = bn.bias.data[keep].clone()
    if bn.track_running_stats:
        new.running_mean.data = bn.running_mean.data[keep].clone()
        new.running_var.data = bn.running_var.data[keep].clone()
        new.num_batches_tracked.data = bn.num_batches_tracked.data.clone()
    return new


def _keep_idx(conv: nn.Conv2d, ratio: float) -> torch.Tensor:
    """Indices of the channels with the largest L2 norm."""
    n_keep = max(1, int(round(conv.out_channels * (1.0 - ratio))))
    norms = conv.weight.detach().flatten(1).norm(p=2, dim=1)
    return torch.sort(torch.topk(norms, n_keep).indices).values


def naive_prune_internal(model: nn.Module, block: str = "layer1.0", ratio: float = 0.5) -> nn.Module:
    """Case A -- prune the channel between conv1 and conv2 of a residual block.

    conv1's outputs are consumed only by bn1 -> relu -> conv2, all inside the
    block. Cutting them touches nothing else, so the forward pass still runs.
    """
    model = copy.deepcopy(model)
    blk = model.get_submodule(block)
    keep = _keep_idx(blk.conv1, ratio)
    blk.conv1 = _slice_conv_out(blk.conv1, keep)
    blk.bn1 = _slice_bn(blk.bn1, keep)
    blk.conv2 = _slice_conv_in(blk.conv2, keep)
    return model


def naive_prune_block_output(model: nn.Module, block: str = "layer1.0", ratio: float = 0.5) -> nn.Module:
    """Case B -- prune the *output* channel of the same residual block.

    conv2's outputs are added to the identity shortcut, so cutting them without
    also cutting everything else on that residual stream leaves the add with
    mismatched channel counts. The forward pass raises RuntimeError. This
    function deliberately produces a broken model.
    """
    model = copy.deepcopy(model)
    blk = model.get_submodule(block)
    keep = _keep_idx(blk.conv2, ratio)
    blk.conv2 = _slice_conv_out(blk.conv2, keep)
    blk.bn2 = _slice_bn(blk.bn2, keep)
    return model


# ---------------------------------------------------------------------------
# 3. Dependency-aware structured pruning (torch_pruning)
# ---------------------------------------------------------------------------
def dependency_group(
    model: nn.Module,
    example_inputs: torch.Tensor,
    layer: nn.Module,
    idxs: Optional[Sequence[int]] = None,
):
    """The full set of (layer, channel-index) pairs coupled to `layer`'s outputs.

    DepGraph traces the autograd graph once, then walks it to find every layer
    that must be cut together with this one. Printing the returned group is the
    clearest way to show what naive pruning missed.
    """
    # Callers hand us a CPU tensor while the model may still be on the
    # accelerator from a previous evaluate() call, which moves models in place.
    example_inputs = example_inputs.to(_device_of(model))
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    if idxs is None:
        idxs = list(range(layer.out_channels // 2))
    return dg.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=list(idxs))


def group_module_names(group) -> List[Tuple[str, str]]:
    """Readable (module_name, action) pairs for a DepGraph pruning group.

    `dep.target.name` embeds the module's repr, e.g.
    ``"conv1 (Conv2d(3, 64, ...))"``. We keep only the dotted module path, and
    drop the synthetic `_ElementWiseOp_*` nodes that stand for the add/relu ops
    in the graph rather than for parameters we can cut.
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for dep, _idxs in group:
        name = dep.target.name.split(" (")[0]
        if name.startswith("_ElementWiseOp"):
            continue
        action = getattr(dep.handler, "__name__", str(dep.handler))
        if "out_channel" in action:
            action = "prune_out_channels"
        elif "in_channel" in action:
            action = "prune_in_channels"
        key = (name, action)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def structured_prune(
    model: nn.Module,
    example_inputs: torch.Tensor,
    ratio: float = 0.5,
    global_pruning: bool = True,
    importance_p: int = 2,
    iterative_steps: int = 1,
    max_pruning_ratio: float = 0.9,
    ignored_layers: Optional[List[nn.Module]] = None,
    inplace: bool = False,
) -> nn.Module:
    """Remove `ratio` of convolution channels, dependency-consistently.

    Args:
        global_pruning: rank channels across the whole network (layers that
            carry more redundancy lose more) instead of applying the same ratio
            to every layer.
        max_pruning_ratio: per-group ceiling, so no layer is emptied out.
        ignored_layers: kept intact. The classifier must be here, otherwise the
            10 output logits would be pruned too.
    """
    # `ignored_layers` holds modules of the *caller's* model. Deep-copying first
    # would leave those references pointing at the original, the pruner would not
    # recognise them, and it would happily prune the classifier down to one
    # output. Resolve them to names before copying, then look them up again in
    # the copy.
    ignored_names: Optional[List[str]] = None
    if ignored_layers is not None:
        by_id = {id(m): n for n, m in model.named_modules()}
        ignored_names = [by_id[id(m)] for m in ignored_layers if id(m) in by_id]

    model = model if inplace else copy.deepcopy(model)
    model.eval()
    example_inputs = example_inputs.to(_device_of(model))

    if ignored_names is not None:
        ignored_layers = [model.get_submodule(n) for n in ignored_names]
    else:
        # Default: keep every classifier head intact.
        ignored_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    importance = tp.importance.GroupMagnitudeImportance(p=importance_p)
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=ratio,
        global_pruning=global_pruning,
        max_pruning_ratio=max_pruning_ratio,
        iterative_steps=iterative_steps,
        ignored_layers=ignored_layers,
        round_to=None,
    )
    for _ in range(iterative_steps):
        pruner.step()
    return model


def channel_counts(model: nn.Module) -> Dict[str, int]:
    """out_channels per Conv2d -- used to show which layers lost the most."""
    return {n: m.out_channels for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
