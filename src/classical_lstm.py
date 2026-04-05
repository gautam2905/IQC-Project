"""
Classical LSTM Baseline
Matched parameter count with QLSTM for fair comparison.
"""

import torch
import torch.nn as nn


class ClassicalLSTM(nn.Module):
    """Standard PyTorch LSTM for baseline comparison."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: input sequence (batch, seq_len, input_size)
        Returns:
            output: prediction (batch, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc_out(lstm_out[:, -1, :])  # use last time step
        return out


if __name__ == "__main__":
    model = ClassicalLSTM(input_size=1, hidden_size=8, output_size=1)
    x = torch.randn(2, 10, 1)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
