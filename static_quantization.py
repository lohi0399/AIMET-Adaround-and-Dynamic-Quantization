import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    fuse_modules,
    get_default_qconfig,
    prepare,
    convert,
)
from torchvision import datasets, transforms


# ── Model Definition ──────────────────────────────────────────────────────────
# Static quantization requires QuantStub / DeQuantStub to mark the boundaries
# where tensors enter and leave the quantized domain, and the forward pass must
# be fully traceable on CPU.

class QuantizableCNN(nn.Module):
    """Same architecture as the base CNN but wrapped with quant/dequant stubs."""

    def __init__(self):
        super().__init__()
        # Stubs: QuantStub inserts a fake-quantize op at the input; DeQuantStub
        # converts the INT8 output back to FP32 for the loss / post-processing.
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(1,  32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.relu3 = nn.ReLU()
        self.fc1   = nn.Linear(5 * 5 * 128, 512)
        self.relu4 = nn.ReLU()
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.quant(x)                        # enter quantized domain
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.relu3(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)                      # exit quantized domain
        return x

    def fuse(self):
        """Fuse Conv+ReLU (and Linear+ReLU) pairs before inserting observers.

        Fusion merges the layers into a single op so the quantisation is done
        on the combined kernel, which reduces quantisation error and speeds up
        inference.
        """
        fuse_modules(self, [["conv1", "relu1"],
                             ["conv2", "relu2"],
                             ["conv3", "relu3"],
                             ["fc1",   "relu4"]], inplace=True)


# ── Data Loaders ──────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,   shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1000, shuffle=False)
# Calibration loader – a small, fixed subset of training data is sufficient
# (typically a few hundred to a few thousand samples).
calib_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}"
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
    print(f"[{label}] Avg loss: {avg_loss:.4f}  Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)")
    return avg_loss, accuracy


def measure_inference_time(model, device, loader):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for data, _ in loader:
            model(data.to(device))
    return time.time() - t0


# ── Step 1 – Train (or load) the FP32 baseline ───────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fp32_model = QuantizableCNN().to(device)

if os.path.exists("fp32_model.pth"):
    print("Loading pre-trained FP32 weights …")
    fp32_model.load_state_dict(torch.load("fp32_model.pth", map_location=device))
else:
    print("Training FP32 model …")
    optimizer = optim.Adam(fp32_model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):
        train_one_epoch(fp32_model, device, train_loader, optimizer, criterion, epoch)
    torch.save(fp32_model.state_dict(), "fp32_model.pth")

fp32_loss, fp32_acc = evaluate(fp32_model, device, test_loader, label="FP32 baseline")
fp32_size = os.path.getsize("fp32_model.pth") / (1024 ** 2)
fp32_time = measure_inference_time(fp32_model, device, test_loader)


# ── Step 2 – Prepare the model for static quantization ───────────────────────
# Static quantization must run on CPU.

print("\n--- Static Quantization ---")
print("Step 2a: Fusing Conv/Linear + ReLU pairs …")

# Work on a fresh CPU copy so the original FP32 model is untouched.
quant_model = QuantizableCNN().to("cpu")
quant_model.load_state_dict(torch.load("fp32_model.pth", map_location="cpu"))
quant_model.eval()

# Fuse layers before setting the qconfig and inserting observers.
quant_model.fuse()

print("Step 2b: Attaching qconfig and inserting observers …")
# 'x86' backend targets SSE4/AVX2 CPUs. Use 'qnnpack' for ARM / mobile.
quant_model.qconfig = get_default_qconfig("x86")

# prepare() walks the module graph and replaces each activation with the
# corresponding Observer (e.g. MinMaxObserver, HistogramObserver).  The
# weights get a PerChannelMinMaxObserver by default.
prepare(quant_model, inplace=True)
print(quant_model)   # show the model with observers inserted


# ── Step 3 – Calibration phase ────────────────────────────────────────────────
# Run representative data through the *observed* model (no gradients needed).
# Observers collect min/max (or histogram) statistics of every activation.
# After this loop every Observer holds the scale + zero_point it will use.

print("\nStep 3: Calibrating with representative data …")
NUM_CALIB_BATCHES = 10   # ~640 images – tune up for better accuracy

quant_model.eval()
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(calib_loader):
        if batch_idx >= NUM_CALIB_BATCHES:
            break
        quant_model(data)   # observers record statistics silently
        if (batch_idx + 1) % 5 == 0:
            print(f"  Calibrated {(batch_idx + 1) * calib_loader.batch_size} samples …")

print("Calibration complete – observers have collected activation statistics.")


# ── Step 4 – Convert: replace observers with quantized kernels ────────────────
# convert() replaces:
#   • Observer-wrapped activations  →  quantize / dequantize nodes
#   • nn.Conv2d / nn.Linear          →  QuantizedConv2d / QuantizedLinear
#     (weights are statically quantised to INT8 at this point)
# The result is a fully INT8 model that can be serialised and deployed.

print("\nStep 4: Converting observed model to INT8 …")
convert(quant_model, inplace=True)
print(quant_model)   # show the converted INT8 model


# ── Step 5 – Evaluate and compare ────────────────────────────────────────────

print("\nEvaluating static-quantized model …")
sq_loss, sq_acc = evaluate(quant_model, torch.device("cpu"), test_loader, label="Static INT8")
sq_time = measure_inference_time(quant_model, torch.device("cpu"), test_loader)

# Save the quantized model
torch.save(quant_model.state_dict(), "static_quant_model.pth")
sq_size = os.path.getsize("static_quant_model.pth") / (1024 ** 2)

# ── Results Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"{'Metric':<30} {'FP32':>10} {'INT8 Static':>12}")
print("-" * 55)
print(f"{'Model size (MB)':<30} {fp32_size:>10.2f} {sq_size:>12.2f}")
print(f"{'Accuracy (%)':<30} {fp32_acc:>10.2f} {sq_acc:>12.2f}")
print(f"{'Inference time (s)':<30} {fp32_time:>10.4f} {sq_time:>12.4f}")
print("=" * 55)
print(f"Size reduction : {(1 - sq_size / fp32_size) * 100:.1f}%")
print(f"Speed-up       : {fp32_time / sq_time:.2f}x")
print(f"Accuracy drop  : {fp32_acc - sq_acc:.2f} pp")
