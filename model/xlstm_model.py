import torch
import torch.nn as nn


class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Gates
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)

        # Exponential gating
        self.exp_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # LSTM gates
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h_prev))
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h_prev))
        g = torch.tanh(self.W_xc(x) + self.W_hc(h_prev))
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h_prev))

        # Cell state
        c = f * c_prev + i * g

        # Exponential gating
        combined = torch.cat([x, h_prev], dim=-1)
        exp_factor = torch.exp(self.exp_gate(combined))

        # Output
        h = o * torch.tanh(c) * exp_factor

        return h, c

class xLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # xLSTM layers
        self.lstm_layers = nn.ModuleList()

        # First layer: embed_dim -> hidden_dim
        self.lstm_layers.append(xLSTMCell(embed_dim, hidden_dim))

        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(1, num_layers):
            self.lstm_layers.append(xLSTMCell(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)  # FIXED: hidden_dim, not embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()

        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        embedded = self.embedding(x)
        outputs = []

        for t in range(seq_len):
            layer_input = embedded[:, t, :]
            new_hidden = []

            for layer_idx, lstm_cell in enumerate(self.lstm_layers):
                h, c = lstm_cell(layer_input, hidden[layer_idx])
                new_hidden.append((h, c))
                layer_input = h  # Output becomes input for next layer

            hidden = new_hidden
            outputs.append(hidden[-1][0])  # Last layer hidden state

        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        logits = self.output_layer(outputs)

        return logits, hidden

    def _init_hidden(self, batch_size, device):
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            hidden.append((h, c))
        return hidden


