"""
Quantum Long Short-Term Memory (QLSTM) Implementation
Based on: Chen et al. (2020) - "Quantum Long Short-Term Memory" (arXiv:2009.01783)

Each LSTM gate's linear transformation is replaced by a Variational Quantum Circuit (VQC):
  1. Angle embedding encodes classical input into qubit rotations
  2. Strongly entangling layers apply parameterized rotations + entanglement
  3. Measurements extract classical output
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class VQC(nn.Module):
    """Variational Quantum Circuit that replaces a linear layer in LSTM."""

    def __init__(self, n_qubits, n_layers, input_size, output_size):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size

        # Classical pre-processing: project input to n_qubits dimensions for angle embedding
        self.pre_net = nn.Linear(input_size, n_qubits)

        # Quantum device
        dev = qml.device("default.qubit", wires=n_qubits)

        # Define quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle embedding: encode classical data into qubit rotations
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Strongly entangling layers: parameterized rotations + CNOT entanglement
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

        # Trainable quantum parameters: shape (n_layers, n_qubits, 3) for StronglyEntanglingLayers
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Classical post-processing: project n_qubits measurements to desired output size
        self.post_net = nn.Linear(n_qubits, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        # Pre-process: project to n_qubits dimensions
        x = self.pre_net(x)  # (batch, n_qubits)
        x = torch.tanh(x)  # bound inputs to [-1, 1] for angle embedding

        # Run each sample through the quantum circuit
        q_out = torch.zeros(batch_size, self.n_qubits, device=x.device)
        for i in range(batch_size):
            result = self.circuit(x[i], self.q_params)
            q_out[i] = torch.stack(result)

        # Post-process: project measurements to output size
        out = self.post_net(q_out)  # (batch, output_size)
        return out


class QLSTMCell(nn.Module):
    """QLSTM Cell: replaces all four LSTM gate linear transforms with VQCs."""

    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        combined_size = input_size + hidden_size

        # Four VQCs, one per gate (forget, input, cell candidate, output)
        self.vqc_forget = VQC(n_qubits, n_layers, combined_size, hidden_size)
        self.vqc_input = VQC(n_qubits, n_layers, combined_size, hidden_size)
        self.vqc_cell = VQC(n_qubits, n_layers, combined_size, hidden_size)
        self.vqc_output = VQC(n_qubits, n_layers, combined_size, hidden_size)

    def forward(self, x, hidden_state):
        """
        Args:
            x: input tensor (batch, input_size)
            hidden_state: tuple (h, c) each of shape (batch, hidden_size)
        Returns:
            (h_new, c_new)
        """
        h_prev, c_prev = hidden_state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_size + hidden_size)

        # Gate computations using VQCs
        f_t = torch.sigmoid(self.vqc_forget(combined))   # forget gate
        i_t = torch.sigmoid(self.vqc_input(combined))     # input gate
        c_tilde = torch.tanh(self.vqc_cell(combined))     # cell candidate
        o_t = torch.sigmoid(self.vqc_output(combined))    # output gate

        # Cell state and hidden state update (same as classical LSTM)
        c_new = f_t * c_prev + i_t * c_tilde
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new


class QLSTM(nn.Module):
    """Full QLSTM model for sequence prediction."""

    def __init__(self, input_size, hidden_size, output_size, n_qubits=4, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.qlstm_cell = QLSTMCell(input_size, hidden_size, n_qubits, n_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: input sequence (batch, seq_len, input_size)
        Returns:
            output: prediction (batch, output_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        # Process sequence step by step
        for t in range(seq_len):
            h, c = self.qlstm_cell(x[:, t, :], (h, c))

        # Final prediction from last hidden state
        out = self.fc_out(h)
        return out


if __name__ == "__main__":
    # Quick test
    model = QLSTM(input_size=1, hidden_size=4, output_size=1, n_qubits=2, n_layers=1)
    x = torch.randn(2, 10, 1)  # batch=2, seq_len=10, features=1
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y.detach().numpy()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
