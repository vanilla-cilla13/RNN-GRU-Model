#python3 -m venv venv
#source venv/bin/activate

import torch
print(torch.__version__)
print(torch.backends.mps.is_available()) # For Apple Silicon Macs


'''import torch
import torch.nn as nn

# RNN GRU model
class BasicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BasicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the last time step
        out = self.fc(out[:, -1, :])
        return out


# Model parameters
if __name__ == "__main__":
    # Sample: sequence length = 5, batch = 3, features = 10
    input_size = 10
    hidden_size = 16
    output_size = 2   
    num_layers = 1
    
    #print("output_size type:", type(output_size))

    model = BasicGRU(input_size, hidden_size, output_size, num_layers)
    
    # Fake data
    x = torch.randn(3, 5, input_size)
    
    y = model(x)
    print("Model output shape:", y.shape)
    print("Output:", y)
'''