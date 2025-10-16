import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratedModel(nn.Module):
    """Auto-generated neural network architecture"""
    
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        
        # Layer definitions
        self.layer_0 = nn.LayerNorm(input_dim)
        self.layer_1 = nn.Dropout(0.1)
        self.layer_2 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, input_dim)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, input_dim)
        )
        self.layer_4 = nn.GRU(input_dim, input_dim, batch_first=True)
        self.layer_5 = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.layer_6 = nn.Dropout(0.1)
        self.layer_7 = nn.Conv1d(input_dim, input_dim, 3, padding=1)
        self.layer_8 = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.layer_9 = nn.GRU(input_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        outputs = []
        
        x = self.layer_0(x)
        outputs.append(x)

        # Get input from layer 0
        x = outputs[0] if outputs else x
        x = self.layer_1(x)
        outputs.append(x)

        # Combine inputs from layers [3, 1]
        if len(outputs) > max([3, 1]):
            x = sum([outputs[idx] for idx in [3, 1]])
        x = self.layer_2(x)
        outputs.append(x)

        x = self.layer_3(x)
        outputs.append(x)

        # Combine inputs from layers [0, 1]
        if len(outputs) > max([0, 1]):
            x = sum([outputs[idx] for idx in [0, 1]])
        x, _ = self.layer_4(x)
        outputs.append(x)

        # Combine inputs from layers [7, 2, 7]
        if len(outputs) > max([7, 2, 7]):
            x = sum([outputs[idx] for idx in [7, 2, 7]])
        x, _ = self.layer_5(x, x, x)
        outputs.append(x)

        x = self.layer_6(x)
        outputs.append(x)

        x = self.layer_7(x)
        outputs.append(x)

        x, _ = self.layer_8(x)
        outputs.append(x)

        # Get input from layer 0
        x = outputs[0] if outputs else x
        x, _ = self.layer_9(x)
        outputs.append(x)

        return outputs[-1] if outputs else x

# Usage example
if __name__ == "__main__":
    model = GeneratedModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(1, 10, 512)  # batch_size=1, seq_len=10, dim=512
    output = model(x)
    print(f"Output shape: {output.shape}")
