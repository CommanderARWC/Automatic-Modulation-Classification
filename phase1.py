import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# Load Dataset
# =====================================================

with h5py.File("modulation_dataset.h5", "r") as f:
    X = f["X"][:]
    Y = f["Y"][:]
    SNR = f["SNR"][:]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.long)
SNR = torch.tensor(SNR, dtype=torch.int32)

print("X Shape:", X.shape)
print("Y Shape:", Y.shape)
print("SNR Shape:", SNR.shape)

# =====================================================
# Shuffle Dataset
# =====================================================

indices = torch.randperm(len(X))

X = X[indices]
Y = Y[indices]
SNR = SNR[indices]

# =====================================================
# Train/Test Split
# =====================================================

split = int(0.8 * len(X))

X_train = X[:split].unsqueeze(1)
Y_train = Y[:split]

X_test = X[split:].unsqueeze(1)
Y_test = Y[split:]

SNR_test = SNR[split:]

# =====================================================
# Data Loaders
# =====================================================

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

# =====================================================
# CNN Model
# =====================================================

model = nn.Sequential(

    nn.Conv2d(
        in_channels=1,
        out_channels=128,
        kernel_size=(2, 8)
    ),

    nn.ReLU(),

    nn.MaxPool2d(
        kernel_size=(1, 2)
    ),

    nn.Conv2d(
        in_channels=128,
        out_channels=64,
        kernel_size=(1, 16)
    ),

    nn.ReLU(),

    nn.MaxPool2d(
        kernel_size=(1, 2)
    ),

    nn.Flatten(),

    nn.LazyLinear(128),

    nn.ReLU(),

    nn.Linear(128, 64),

    nn.ReLU(),

    nn.Linear(64, 32),

    nn.ReLU(),

    nn.Linear(32, 9)

)

model = model.to(device)

# =====================================================
# Loss Function & Optimizer
# =====================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# Training
# =====================================================

epochs = 15

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_X, batch_Y in train_loader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = loss_fn(outputs, batch_Y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss = {avg_loss:.4f}"
    )

# =====================================================
# Testing
# =====================================================

model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_Y in test_loader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        outputs = model(batch_X)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == batch_Y).sum().item()

        total += batch_Y.size(0)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_Y.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(all_labels, all_preds)

mod_names = [
    "OOK",
    "ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "BFSK",
    "4FSK",
    "8FSK"
]

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")
plt.colorbar()

plt.xticks(
    np.arange(len(mod_names)),
    mod_names,
    rotation=45
)

plt.yticks(
    np.arange(len(mod_names)),
    mod_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

for i in range(len(mod_names)):
    for j in range(len(mod_names)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

# =====================================================
# Accuracy vs SNR
# =====================================================

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

snr_test_np = SNR_test.numpy()

snr_values = np.unique(snr_test_np)

snr_accuracy = []

for snr in snr_values:

    mask = (snr_test_np == snr)

    correct = np.sum(
        all_preds[mask] == all_labels[mask]
    )

    total = np.sum(mask)

    accuracy_snr = 100 * correct / total

    snr_accuracy.append(accuracy_snr)

# =====================================================
# Plot Accuracy vs SNR
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    snr_values,
    snr_accuracy,
    marker="o"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy vs SNR")

plt.grid(True)

plt.show()

# =====================================================
# Normalized Confusion Matrix
# =====================================================

cm_norm = confusion_matrix(
    all_labels,
    all_preds,
    normalize="true"
)

plt.figure(figsize=(10, 8))

plt.imshow(
    cm_norm,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    np.arange(len(mod_names)),
    mod_names,
    rotation=45
)

plt.yticks(
    np.arange(len(mod_names)),
    mod_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")

for i in range(len(mod_names)):
    for j in range(len(mod_names)):
        plt.text(
            j,
            i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_norm[i, j] > 0.5 else "black"
        )

plt.tight_layout()
plt.show()
