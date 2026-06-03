"""
Quantization Aware Training (QAT) with PyTorch native API
==========================================================
Key idea
--------
Instead of quantizing a finished FP32 model (post-training), QAT inserts
*FakeQuantize* modules into the graph *before* training starts.  During each
forward pass those modules:
  1. Observe the running min/max of the activation they wrap.
  2. Round the tensor to the nearest INT8 grid point.
  3. Immediately dequantize back to FP32.

This round-trip deliberately injects quantization noise into the loss, so the
optimizer learns weights that are robust to INT8 rounding.  Gradients still
flow in FP32 through the Straight-Through Estimator (STE), which treats the
rounding step as an identity for the backward pass.

Pipeline
--------
  FP32 weights
      │
      ▼
  fuse_modules()          – merge Conv+ReLU / Linear+ReLU
      │
      ▼
  prepare_qat()           – insert FakeQuantize stubs (observers ON, fake-quant ON)
      │
      ▼
  train() x N epochs      – quantization noise is part of the loss every step
      │
      ▼
  convert()               – freeze scale/zero_point, swap to real INT8 kernels
      │
      ▼
  INT8 model  (CPU, torchscript-ready)
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    fuse_modules,
    get_default_qat_qconfig,   # ← QAT-specific qconfig (uses FakeQuantize)
    prepare_qat,               # ← inserts FakeQuantize modules
    convert,
)
from torchvision import datasets, transforms


# ── Model ─────────────────────────────────────────────────────────────────────

class QuantizableCNN(nn.Module):
    """CNN with QuantStub / DeQuantStub boundaries and explicit ReLU layers.

    Explicit ReLU submodules (not the functional form) are required so that
    fuse_modules() can locate them by name and merge them with the preceding
    Conv / Linear layer.
    """

    def __init__(self):
        super().__init__()
        # --- quantization boundary stubs -----------------------------------
        # After prepare_qat() these will each contain a FakeQuantize child
        # that runs every forward pass during training.
        self.quant   = QuantStub()    # FP32 → fake-INT8 at the model input
        self.dequant = DeQuantStub()  # fake-INT8 → FP32 at the model output

        # --- layers (kept separate from activations so fuse_modules works) -
        self.conv1 = nn.Conv2d(1,  32,  3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64,  3, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.relu3 = nn.ReLU()
        self.fc1   = nn.Linear(5 * 5 * 128, 512)
        self.relu4 = nn.ReLU()
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.quant(x)               # ← FakeQuantize: input gets quantised here
        x = self.relu1(self.conv1(x))   # after fusion this pair becomes ConvReLU2d
        x = self.relu2(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.relu3(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)             # ← FakeQuantize: output gets dequantised here
        return x

    def fuse(self):
        """Fuse Conv/Linear + ReLU before inserting FakeQuantize nodes.

        Fusion is strongly recommended before QAT: it avoids placing a
        FakeQuantize between a Conv and its paired ReLU, which would
        double-count quantization noise on that activation.
        """
        fuse_modules(
            self,
            [["conv1", "relu1"],
             ["conv2", "relu2"],
             ["conv3", "relu3"],
             ["fc1",   "relu4"]],
            inplace=True,
        )


# ── Data ──────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,   shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1000, shuffle=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()   # STE: gradients flow through FakeQuantize as-if identity
        optimizer.step()
        if batch_idx % 150 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data):>5}/{len(loader.dataset)}"
                  f" ({100. * batch_idx / len(loader):.0f}%)]  loss: {loss.item():.6f}")


def evaluate(model, device, loader, label="Model"):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            total_loss += nn.functional.cross_entropy(out, target, reduction='sum').item()
            correct    += out.argmax(dim=1).eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"[{label}]  loss: {avg_loss:.4f}  acc: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)")
    return avg_loss, accuracy


def measure_inference_time(model, device, loader):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for data, _ in loader:
            model(data.to(device))
    return time.time() - t0


def inspect_fake_quant(model, max_modules=6):
    """Print the scale, zero_point and observer state of every FakeQuantize
    module in the model.  Call this after calibration and again after training
    to see how the ranges shift as the model adapts.
    """
    print("\n  name                                 | dtype  | scale      | zp  | obs_en | fq_en")
    print("  " + "-" * 86)
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
            scale = mod.scale.item()   if mod.scale.numel()      == 1 else f"[{mod.scale.numel()} ch]"
            zp    = mod.zero_point.item() if mod.zero_point.numel() == 1 else f"[{mod.zero_point.numel()} ch]"
            print(f"  {name:<40} | {str(mod.dtype):<6} | {str(scale):<10} | {str(zp):<3} "
                  f"| {int(mod.observer_enabled.item())}      | {int(mod.fake_quant_enabled.item())}")
            count += 1
            if count >= max_modules:
                remaining = sum(1 for _, m in model.named_modules()
                                if isinstance(m, torch.ao.quantization.FakeQuantizeBase)) - count
                if remaining:
                    print(f"  … and {remaining} more FakeQuantize modules …")
                break
    print()


# ── Step 1 – FP32 baseline ────────────────────────────────────────────────────

device = torch.device("cpu")   # QAT + convert() must run on CPU
print("=" * 60)
print("Step 1 – FP32 baseline")
print("=" * 60)

fp32_model = QuantizableCNN()

if os.path.exists("fp32_model.pth"):
    print("  Loading pre-trained FP32 weights …")
    fp32_model.load_state_dict(torch.load("fp32_model.pth", map_location="cpu"))
else:
    print("  Training FP32 model from scratch …")
    optimizer = optim.Adam(fp32_model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):
        train_one_epoch(fp32_model, device, train_loader, optimizer, criterion, epoch)
    torch.save(fp32_model.state_dict(), "fp32_model.pth")

fp32_loss, fp32_acc = evaluate(fp32_model, device, test_loader, label="FP32 baseline")
fp32_size = os.path.getsize("fp32_model.pth") / (1024 ** 2)
fp32_time = measure_inference_time(fp32_model, device, test_loader)


# ── Step 2 – Fuse and attach QAT qconfig ─────────────────────────────────────
# get_default_qat_qconfig() returns a qconfig whose activation and weight
# factories produce *FakeQuantize* objects (not plain Observers).
# Internally each FakeQuantize wraps:
#   • an Observer (e.g. MovingAverageMinMaxObserver) that tracks running stats
#   • a quantize_per_tensor / quantize_per_channel call that does the rounding

print("\n" + "=" * 60)
print("Step 2 – Fuse layers + attach QAT qconfig")
print("=" * 60)

qat_model = QuantizableCNN()
qat_model.load_state_dict(torch.load("fp32_model.pth", map_location="cpu"))
qat_model.eval()

print("  Fusing Conv/Linear + ReLU pairs …")
qat_model.fuse()

# 'x86' → per-channel weight quantization, suited for SSE4/AVX2 CPUs.
# Use 'qnnpack' for ARM / mobile targets.
qat_model.qconfig = get_default_qat_qconfig("x86")
print(f"\n  qconfig: {qat_model.qconfig}")


# ── Step 3 – prepare_qat(): insert FakeQuantize modules ──────────────────────
# prepare_qat() does two things:
#   1. For every leaf module whose type appears in the default mapping
#      (Conv2d, Linear, …) it wraps the weights with a FakeQuantize.
#   2. For every activation boundary it inserts a FakeQuantize stub.
# After this call the model is still FP32 internally – FakeQuantize just
# quantises-then-dequantises (round-trip) so computation stays in FP32
# while quantization error is injected.

print("\n" + "=" * 60)
print("Step 3 – prepare_qat(): inserting FakeQuantize modules")
print("=" * 60)

qat_model.train()
prepare_qat(qat_model, inplace=True)

print("\n  Model after prepare_qat() (FakeQuantize nodes visible):")
print(qat_model)

print("\n  FakeQuantize state immediately after prepare_qat():")
inspect_fake_quant(qat_model)
# At this point:
#   observer_enabled = 1  → observers are collecting statistics
#   fake_quant_enabled = 1 → fake-quantize rounding is also active


# ── Step 4 – QAT training loop ────────────────────────────────────────────────
# The model trains normally.  The only difference from FP32 training is that
# every activation and weight tensor is round-tripped through INT8 on every
# forward pass, so the loss includes quantization noise from the very first
# step.  Gradients flow through the rounding via STE.
#
# Common schedule tip: start with observers ON + fake-quant ON (default).
# Optionally freeze BN stats and observer stats in the last few epochs.

print("\n" + "=" * 60)
print("Step 4 – QAT fine-tuning")
print("=" * 60)

QAT_EPOCHS = 3
optimizer  = optim.SGD(qat_model.parameters(), lr=1e-3, momentum=0.9)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
criterion  = nn.CrossEntropyLoss()

for epoch in range(QAT_EPOCHS):
    print(f"\n  --- QAT epoch {epoch + 1}/{QAT_EPOCHS} "
          f"(lr={scheduler.get_last_lr()[0]:.2e}) ---")

    # ── optional: freeze observer stats in the last epoch ─────────────────
    # Once activation ranges are stable, disable the observer so scale and
    # zero_point stop moving and the optimizer can fine-tune weights more
    # cleanly.  Comment this block out to let observers run for all epochs.
    if epoch == QAT_EPOCHS - 1:
        print("  Freezing observer statistics (last epoch) …")
        qat_model.apply(torch.ao.quantization.disable_observer)

    train_one_epoch(qat_model, device, train_loader, optimizer, criterion, epoch + 1)
    evaluate(qat_model, device, test_loader, label=f"QAT (fake-quant, epoch {epoch+1})")
    scheduler.step()

print("\n  FakeQuantize state after QAT training:")
inspect_fake_quant(qat_model)
# scale/zero_point values should now be stable and tuned to this model's
# activations rather than the initial calibration from prepare_qat().


# ── Step 5 – convert(): freeze fake-quant → real INT8 kernels ────────────────
# convert() does a final pass over the graph:
#   • Reads the learned scale and zero_point from every FakeQuantize module.
#   • Replaces QuantStub → torch.quantize_per_tensor
#   • Replaces DeQuantStub → dequantize
#   • Replaces nn.Conv2d / nn.Linear → QuantizedConv2d / QuantizedLinear
#     with weights permanently packed as INT8.
# After convert() the model no longer contains any FakeQuantize objects.

print("\n" + "=" * 60)
print("Step 5 – convert() to real INT8")
print("=" * 60)

qat_model.eval()
convert(qat_model, inplace=True)

print("\n  Model after convert() (pure INT8 ops, no more FakeQuantize):")
print(qat_model)


# ── Step 6 – Evaluate final INT8 model ───────────────────────────────────────

print("\n" + "=" * 60)
print("Step 6 – Final evaluation")
print("=" * 60)

qat_loss, qat_acc = evaluate(qat_model, device, test_loader, label="QAT INT8 final")
qat_time = measure_inference_time(qat_model, device, test_loader)

torch.save(qat_model.state_dict(), "qat_model.pth")
qat_size = os.path.getsize("qat_model.pth") / (1024 ** 2)


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"{'Metric':<28} {'FP32':>10} {'QAT INT8':>12}")
print("-" * 60)
print(f"{'Model size (MB)':<28} {fp32_size:>10.2f} {qat_size:>12.2f}")
print(f"{'Accuracy (%)':<28} {fp32_acc:>10.2f} {qat_acc:>12.2f}")
print(f"{'Inference time (s)':<28} {fp32_time:>10.4f} {qat_time:>12.4f}")
print("=" * 60)
print(f"Size reduction : {(1 - qat_size / fp32_size) * 100:.1f}%")
print(f"Speed-up       : {fp32_time / qat_time:.2f}x")
print(f"Accuracy delta : {qat_acc - fp32_acc:+.2f} pp  "
      f"({'better' if qat_acc >= fp32_acc else 'worse'} than FP32)")
