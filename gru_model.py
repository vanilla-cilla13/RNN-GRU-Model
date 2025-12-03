# ============================
# Imports
# ============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

from synthetic_data import generate_synthetic_data


# ============================
# Hyperparameters
# ============================
input_dim = 3
hidden_size = 32
output_size = 4
num_layers = 1
seq_length = 10
num_sequences = 500
num_epochs = 100
learning_rate = 0.001
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Generate Synthetic Data
# ============================
X, y = generate_synthetic_data(
    num_sequences=num_sequences,
    seq_length=seq_length,
    input_dim=input_dim
)

# Train/Test Split
split = int(0.8 * num_sequences)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# ============================
# GRU Model Definition
# ============================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


model = GRUClassifier(
    input_dim, hidden_size, output_size, num_layers
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ============================
# DataLoader
# ============================
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# ============================
# Training Loop
# ============================
def train_model():
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()

            optimizer.zero_grad()
            output = model(x_batch)

            # Flatten for CrossEntropyLoss
            output = output.reshape(-1, output_size)
            y_batch = y_batch.reshape(-1)

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_losses


# ============================
# Evaluation
# ============================
def evaluate_model():
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        predicted = torch.argmax(outputs, dim=2)

        true_flat = y_test.cpu().numpy().reshape(-1)
        pred_flat = predicted.cpu().numpy().reshape(-1)

        acc = accuracy_score(true_flat, pred_flat)
        bal_acc = balanced_accuracy_score(true_flat, pred_flat)
        f1 = f1_score(true_flat, pred_flat, average="weighted")

        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"F1 Score:          {f1:.4f}")

        return predicted.cpu()


# ============================
# Visualization – True vs Predicted
# ============================
def plot_true_vs_pred(true_labels, pred_labels):
    plt.figure(figsize=(10, 5))
    plt.plot(true_labels.flatten(), label="True", linewidth=1.5)
    plt.plot(pred_labels.flatten(), label="Predicted", linewidth=1)
    plt.xlabel("Time Index")
    plt.ylabel("Class")
    plt.title("True vs Predicted Classes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================
# Visualization – XYZ + Predictions
# ============================
def plot_xyz_with_predictions(X_test, pred_labels, window=200):
    XYZ = X_test.reshape(-1, 3).cpu().numpy()

    x = XYZ[:window, 0]
    y = XYZ[:window, 1]
    z = XYZ[:window, 2]
    preds = pred_labels.flatten()[:window]

    plt.figure(figsize=(14, 6))
    plt.plot(x, label="X")
    plt.plot(y, label="Y")
    plt.plot(z, label="Z")

    scale = max(abs(x).max(), abs(y).max(), abs(z).max()) + 1
    plt.plot(preds * scale, label="Predicted Class", linestyle="--")

    plt.title("XYZ Signals with Predicted Classes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================
# Visualization – Per-Sequence Examples
# ============================
def visualize_examples(preds, num_examples=3):
    y_true = y_test.cpu().numpy()
    y_pred = preds.numpy()

    for i in range(num_examples):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[i], "o-", label="True")
        plt.plot(y_pred[i], "x--", label="Predicted")
        plt.title(f"Sequence {i} – True vs Predicted")
        plt.xlabel("Time Step")
        plt.ylabel("Class")
        plt.grid(True)
        plt.legend()
        plt.show()

        print(f"\nSequence {i}:")
        print("True:", y_true[i])
        print("Pred:", y_pred[i])


# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("Training GRU model...\n")
    train_losses = train_model()

    print("\nEvaluating model...")
    preds = evaluate_model()

    plot_true_vs_pred(y_test.cpu().numpy(), preds.numpy())
    plot_xyz_with_predictions(X_test, preds)

    # Training Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.title("GRU Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    visualize_examples(preds, num_examples=3)
