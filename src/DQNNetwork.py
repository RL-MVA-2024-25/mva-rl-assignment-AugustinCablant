import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(nb_neurons, nb_neurons*2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(nb_neurons*2, nb_neurons)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(nb_neurons, nb_neurons)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(nb_neurons, n_action)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x
    
class DQN(nn.Module):
    def __init__(self, n_action, in_channels = 6):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, activation=torch.nn.SiLU(), normalization=None):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.ReLU()
        if normalization == 'batch':
            self.normalization = torch.nn.BatchNorm1d(hidden_dim)
        elif normalization == 'layer':
            self.normalization = torch.nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        return self.output_layer(x)

    
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, rnn_type='LSTM', activation=torch.nn.SiLU(), normalization=None):
        """
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden states.
            output_dim (int): Dimension of the output.
            depth (int): Number of RNN layers.
            rnn_type (str): Type of RNN ('LSTM', 'GRU', or 'RNN').
            activation (torch.nn.Module): Activation function.
            normalization (str or None): 'batch', 'layer', or None.
        """
        super(RNNModel, self).__init__()

        # Choose the RNN type
        if rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=depth, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=depth, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=depth, batch_first=True)
        else:
            raise ValueError("rnn_type must be one of 'LSTM', 'GRU', or 'RNN'.")

        # Fully connected output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        # Activation and normalization
        self.activation = activation if activation is not None else torch.nn.ReLU()
        if normalization == 'batch':
            self.normalization = torch.nn.BatchNorm1d(hidden_dim)
        elif normalization == 'layer':
            self.normalization = torch.nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x, hidden_state=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            hidden_state (torch.Tensor or tuple): Initial hidden state(s) for the RNN.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        out, hidden_state = self.rnn(x, hidden_state)  # out shape: (batch_size, seq_length, hidden_dim)
        if out.dim() == 3:
            out = out[:, -1, :]  # Take the last time step's hidden state
        elif out.dim() == 2:
            pass  # Output is already in the right format
        else:
            raise ValueError(f"Unexpected tensor shape: {out.shape}")
        if self.normalization is not None:
            out = self.normalization(out)
        out = self.activation(out)
        out = self.output_layer(out)
        return out