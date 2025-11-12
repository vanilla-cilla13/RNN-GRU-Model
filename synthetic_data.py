import torch

def generate_synthetic_data(
    num_sequences=1000, seq_length=20, input_dim=3
):
    """Generate synthetic sequence data for classification"""
    # Random input sequences
    X = torch.randn(num_sequences, seq_length, input_dim)

    # Create patterns in the sequences that determine the classes
    y = torch.zeros(num_sequences, seq_length, dtype=torch.long)

    for i in range(num_sequences):
        # Create a simple pattern: class depends on the sign and magnitude of features
        for t in range(seq_length):
            if X[i, t, 0] > 0.5 and X[i, t, 1] > 0:
                y[i, t] = 0
            elif X[i, t, 0] < -0.5 and X[i, t, 2] > 0:
                y[i, t] = 1
            elif X[i, t, 1] < -0.5:
                y[i, t] = 2
            else:
                y[i, t] = 3

    return X, y

# Parameters
input_dim = 3
seq_length = 10
num_sequences = 500

# Generate data
X, y = generate_synthetic_data(num_sequences=num_sequences, seq_length=seq_length, input_dim=input_dim)
print(X.shape)
print(y.shape)