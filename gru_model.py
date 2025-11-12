#TODO make a virtual env on your local machine <-- done
# pip install in your virtual env <-- done


import torch
import torch.nn as nn
import torch.optim as optim
from synthetic_data import generate_synthetic_data
import matplotlib.pyplot as plt


# 1. Hyperparameters
input_dim = 3
hidden_size = 32
output_size = 4      # TODO: many to many --> input size == output size <-- done
num_layers = 1
seq_length = 10
num_sequences = 500
num_epochs = 20
learning_rate = 0.001
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Generate Synthetic Data
X, y = generate_synthetic_data(num_sequences=num_sequences, seq_length=seq_length, input_dim=input_dim)

# Split into train/test
split = int(0.8 * num_sequences)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# 3. Define the GRU Model
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)  
        out = self.fc(out)    
        return out

model = GRUClassifier(input_dim, hidden_size, output_size, num_layers).to(device)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Training Loop
def train_model():
    model.train()
    train_losses = []  # store loss values 
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            x_batch, y_batch = X_train[indices], y_train[indices]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.view(-1, output_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X_train.size(0) / batch_size)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return train_losses  

# 6. Evaluation
def evaluate_model():
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        predicted = torch.argmax(outputs, dim=2)
        correct = (predicted == y_test.to(device)).float().mean()
        print(f"Test Accuracy: {correct.item() * 100:.2f}%")
    return correct.item()

# 7. Run Training and Evaluation
if __name__ == "__main__":
    print("Training GRU model on synthetic data...\n")
    train_losses = train_model()  # <— capture losses here

    print("\nEvaluating model on test data...")
    test_acc = evaluate_model()

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.title('GRU Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 8. Visualize Example Predictions vs True Labels
def visualize_examples(num_examples=3):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))                   
        y_pred = torch.argmax(outputs, dim=2).cpu().numpy()  
        y_true = y_test.cpu().numpy()

    # plot  random examples
    for i in range(num_examples):
        seq_idx = i 
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[seq_idx], 'o-', label='True Classes', linewidth=2)
        plt.plot(y_pred[seq_idx], 'x--', label='Predicted Classes', linewidth=2)
        plt.title(f"Sequence {seq_idx} - True vs Predicted Classes")
        plt.xlabel("Time Step")
        plt.ylabel("Class ID")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"\nExample {i+1}:")
        print("True y:", y_true[seq_idx])
        print("Pred y:", y_pred[seq_idx])

if __name__ == "__main__":
    print("Training GRU model on synthetic data...\n")
    train_losses = train_model()

    print("\nEvaluating model on test data...")
    test_acc = evaluate_model()

    # plot for loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.title('GRU Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    visualize_examples(num_examples=3)
