"""Smoke tests: the pruning pipeline must work without touching CIFAR-10.

Everything here runs on random tensors and a freshly initialised network, so
`pytest -q` finishes in a few seconds and needs no dataset download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import example_inputs, stratified_indices
from src.metrics import (activation_memory_mb, count_nonzero_params, count_params,
                         latency_ms, macs_and_params, model_size_mb, weight_sparsity)
from src.model import resnet18_cifar
from src.pruning import (channel_l2_norms, dependency_group, global_unstructured_prune,
                         group_module_names, make_masks_permanent,
                         naive_prune_block_output, naive_prune_internal, structured_prune)

RATIO = 0.5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return resnet18_cifar().eval()


@pytest.fixture(scope="module")
def ex():
    return example_inputs()


def test_baseline_shape(model):
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_unstructured_zeros_weights_but_keeps_shape(model, ex):
    pruned = global_unstructured_prune(model, RATIO, permanent=True)
    # Sparsity reaches the requested level ...
    assert weight_sparsity(pruned) == pytest.approx(100 * RATIO, abs=0.5)
    assert count_nonzero_params(pruned) < count_nonzero_params(model)
    # ... but the tensors are the same size, so nothing gets cheaper.
    assert count_params(pruned) == count_params(model)
    assert macs_and_params(pruned, ex)[0] == macs_and_params(model, ex)[0]


def test_unstructured_masks_survive_an_optimizer_step(model):
    pruned = global_unstructured_prune(model, RATIO, permanent=False)
    opt = torch.optim.SGD(pruned.parameters(), lr=0.1)
    pruned.train()
    pruned(torch.randn(2, 3, 32, 32)).sum().backward()
    opt.step()
    make_masks_permanent(pruned)
    assert weight_sparsity(pruned) == pytest.approx(100 * RATIO, abs=0.5)


def test_naive_pruning_inside_a_block_is_safe(model):
    pruned = naive_prune_internal(model, "layer1.0", RATIO)
    assert pruned.layer1[0].conv1.out_channels == 32
    assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_naive_pruning_a_block_output_breaks_the_residual_add(model):
    broken = naive_prune_block_output(model, "layer1.0", RATIO)
    with pytest.raises(RuntimeError, match="must match the size"):
        broken(torch.randn(2, 3, 32, 32))


def test_dependency_group_spans_more_than_the_pruned_layer(model, ex):
    group = dependency_group(model, ex, model.layer1[0].conv2, idxs=list(range(32)))
    touched = {name for name, _action in group_module_names(group)}
    # naive_prune_block_output edits only conv2 and bn2. DepGraph finds eight more
    # layers on the same residual stream, including the stem convolution and the
    # downsample shortcut of the next stage.
    assert {"layer1.0.conv2", "layer1.0.bn2"} <= touched
    assert "layer2.0.downsample.0" in touched
    assert {"conv1", "bn1"} <= touched
    assert len(touched) >= 10


@pytest.mark.parametrize("ratio", [0.4, 0.5, 0.6])
def test_structured_pruning_shrinks_the_network_and_still_runs(model, ex, ratio):
    pruned = structured_prune(model, ex, ratio=ratio, global_pruning=True)
    assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    base_macs, base_params = macs_and_params(model, ex)
    macs, params = macs_and_params(pruned, ex)
    assert params < base_params
    assert macs < base_macs
    assert model_size_mb(pruned) < model_size_mb(model)
    assert activation_memory_mb(pruned, ex) < activation_memory_mb(model, ex)
    # More pruning must not produce a bigger network.
    assert params / base_params < 1.0 - ratio / 2


def test_classifier_is_not_pruned(model, ex):
    pruned = structured_prune(model, ex, ratio=0.7, global_pruning=True)
    assert pruned.fc.out_features == 10


def test_explicitly_ignored_layers_survive_the_deepcopy(model, ex):
    """`ignored_layers` names modules of the caller's model, not of the copy.

    Regression test: resolving them after the deep copy instead of before left
    the pruner with stale references and it pruned the classifier to a single
    output, which only showed up as a cross-entropy IndexError during fine-tuning.
    """
    pruned = structured_prune(
        model, ex, ratio=0.7, global_pruning=True,
        ignored_layers=[model.fc, model.layer4[1].conv2],
    )
    assert pruned.fc.out_features == 10
    assert pruned.layer4[1].conv2.out_channels == model.layer4[1].conv2.out_channels
    assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_latency_is_measurable(model, ex):
    lat = latency_ms(model, ex, torch.device("cpu"), runs=5, warmup=2)
    assert lat["median"] > 0 and lat["p90"] >= lat["median"]


def test_channel_norms_cover_every_conv(model):
    norms = channel_l2_norms(model)
    convs = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    assert set(norms) == set(convs)
    assert all(v.ndim == 1 for v in norms.values())


def test_stratified_subset_is_class_balanced():
    targets = [i % 10 for i in range(10_000)]
    idx = stratified_indices(targets, per_class=200)
    assert len(idx) == 2000
    counts = {}
    for i in idx:
        counts[targets[i]] = counts.get(targets[i], 0) + 1
    assert set(counts.values()) == {200}


ACCEL = ("cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available() else None)


@pytest.mark.skipif(ACCEL is None, reason="no accelerator on this machine")
def test_pruning_works_on_a_model_left_on_the_accelerator(ex):
    """Regression: evaluate() moves models in place, example inputs stay on CPU.

    The benchmark sweep crashed with `input(device='cpu') and weight(device=mps)`
    because the baseline was still on the accelerator when the dependency graph
    was traced with a CPU input.
    """
    model = resnet18_cifar().eval().to(ACCEL)
    group = dependency_group(model, ex, model.layer1[0].conv2, idxs=list(range(8)))
    assert len(group) > 0

    pruned = structured_prune(model, ex, ratio=0.5, ignored_layers=[model.fc])
    assert pruned(torch.randn(2, 3, 32, 32, device=ACCEL)).shape == (2, 10)


@pytest.mark.skipif(ACCEL is None, reason="no accelerator on this machine")
def test_benchmark_leaves_the_model_where_it_found_it(ex):
    from src.metrics import benchmark

    model = resnet18_cifar().eval().to(ACCEL)
    benchmark(model, "probe", ex, latency_runs=2)
    assert next(model.parameters()).device.type == ACCEL
