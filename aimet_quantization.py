import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import matplotlib.pyplot as plt

# AIMET imports for AdaRound static quantization
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme


# ── Model Definition (same architecture as quantization.py) ──────────────────

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── Data Loaders ──────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

train_loader       = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader        = torch.utils.data.DataLoader(test_dataset,  batch_size=1000, shuffle=False)
# Smaller loader used for AdaRound calibration (a few hundred samples is enough)
calibration_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')


def evaluate(model, device, loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'Test set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


def measure_inference_time(model, device, loader):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for data, _ in loader:
            model(data.to(device))
    return time.time() - start


# ── AIMET calibration callback ────────────────────────────────────────────────
# AIMET's compute_encodings() calls this function to run representative data
# through the model so it can record min/max activation statistics.

def forward_pass_callback(model, num_batches: int = 4):
    """Run `num_batches` calibration batches through the model (CPU only)."""
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            model(data)          # data stays on CPU — AIMET static quant is CPU-based
            if i + 1 >= num_batches:
                break


# ── Train the FP32 baseline ───────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("Training FP32 baseline model...")
for epoch in range(1):
    train(model, device, train_loader, optimizer, criterion, epoch)

fp32_loss, fp32_accuracy = evaluate(model, device, test_loader)
fp32_inference_time = measure_inference_time(model, device, test_loader)

torch.save(model.state_dict(), 'fp32_model.pth')
fp32_size = os.path.getsize('fp32_model.pth') / (1024 * 1024)

# Move to CPU for AIMET (AIMET static quantization runs on CPU)
model.cpu()


# ── Step 1 — AdaRound (optimise weight rounding) ─────────────────────────────
# AdaRound learns the optimal rounding direction for each weight element by
# minimising the reconstruction error on a small calibration set, rather than
# always rounding to the nearest integer.

print("\nApplying AIMET AdaRound to optimise weight rounding...")

dummy_input = torch.randn(1, 1, 28, 28)   # representative input shape for MNIST

adaround_params = AdaroundParameters(
    data_loader=calibration_loader,
    num_batches=4,                   # number of calibration batches (~256 samples)
    default_num_iterations=500,      # optimisation iterations per layer
)

adarounded_model = Adaround.apply_adaround(
    model=model,
    dummy_input=dummy_input,
    params=adaround_params,
    path='./',                       # where to save the .encodings file
    filename_prefix='adaround',
    default_param_bw=8,              # 8-bit weight quantization
    default_quant_scheme=QuantScheme.post_training_tf_enhanced,
)


# ── Step 2 — Static Quantization Simulation ───────────────────────────────────
# QuantizationSimModel wraps the model and inserts fake-quantization nodes for
# both weights and activations (INT8 for both → true static quantization).

print("\nBuilding AIMET QuantizationSimModel (static, INT8)...")

sim = QuantizationSimModel(
    model=adarounded_model,
    dummy_input=dummy_input,
    default_param_bw=8,              # 8-bit weights
    default_output_bw=8,             # 8-bit activations
    quant_scheme=QuantScheme.post_training_tf_enhanced,
)

# Freeze the weight encodings computed by AdaRound so compute_encodings() only
# calibrates the activation quantizers, not the weights again.
sim.load_encodings('./adaround.encodings', freeze_encodings=True)

# Calibrate activation ranges using representative data
sim.compute_encodings(
    forward_pass_callback=forward_pass_callback,
    forward_pass_callback_args=4,    # passed as `num_batches` to the callback
)


# ── Step 3 — Evaluate the statically quantized model ─────────────────────────

print("\nEvaluating AIMET AdaRound statically quantized model...")
quant_loss, quant_accuracy = evaluate(sim.model, torch.device('cpu'), test_loader)
quant_inference_time = measure_inference_time(sim.model, torch.device('cpu'), test_loader)

# Export to ONNX (INT8 weights are stored in the ONNX graph)
sim.onnx.export(path='./', filename_prefix='aimet_quant_model', dummy_input=dummy_input)
quant_size = os.path.getsize('./aimet_quant_model.onnx') / (1024 * 1024)


# ── Results ───────────────────────────────────────────────────────────────────

print("\n── Results ──────────────────────────────────────────────")
print(f"FP32  model  — Size: {fp32_size:.2f} MB | Accuracy: {fp32_accuracy:.2f}% | Inference: {fp32_inference_time:.4f}s")
print(f"INT8 (AIMET) — Size: {quant_size:.2f} MB | Accuracy: {quant_accuracy:.2f}% | Inference: {quant_inference_time:.4f}s")
print("─────────────────────────────────────────────────────────\n")


# ── Visualisation ─────────────────────────────────────────────────────────────

labels = ['Model Size (MB)', 'Accuracy (%)', 'Inference Time (s)']
fp32_values  = [fp32_size,  fp32_accuracy,  fp32_inference_time]
quant_values = [quant_size, quant_accuracy, quant_inference_time]

x, width = range(len(labels)), 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width / 2 for i in x], fp32_values,  width, label='FP32 (baseline)')
ax.bar([i + width / 2 for i in x], quant_values, width, label='INT8 AdaRound (AIMET)')
ax.set_ylabel('Values')
ax.set_title('FP32 vs AIMET AdaRound Static INT8 Quantization')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
