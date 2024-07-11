import torch.nn as nn
import torch
import time

# model for coordinate transformation
class CTModel(nn.Module):
    def __init__(self, layer_widths, nonlinearity="sigmoid", dtype=torch.float32, device="cpu"):
        super(CTModel, self).__init__()

        # Select device for the model to run on
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        print(f"Using device \"{self.device}\"")

        self.dtype = dtype

        # Set up the layers
        self.layer_widths = layer_widths
        layers = []
        lstm_layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1], bias=(i < len(layer_widths) - 2)))
            if i < len(layer_widths) - 2:  # For all but the last layer
                # Add LSTM layer
                lstm_layers.append(nn.LSTM(input_size=layer_widths[i + 1], hidden_size=layer_widths[i + 1], bias=True, batch_first=True))

        self.linear_layers = nn.ModuleList(layers).to(device=self.device, dtype=self.dtype)
        self.lstm_layers = nn.ModuleList(lstm_layers).to(device=self.device, dtype=self.dtype)
        self.lstm_states = None
        # self.reset_state() # Reset states of LSTM layers (not necessary in the current configuration)
        
        if nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "leaky_relu":
            self.nonlinearity = nn.LeakyReLU()
        elif nonlinearity == "softplus":
            self.nonlinearity = nn.Softplus()
        else:
            raise ValueError("Invalid nonlinearity.")

    # Forward pass through the model
    # input shape: sequence length x input width
    def forward(self, x):
        x = x.to(self.device, dtype=self.dtype)
        self.reset_state()

        # Pass through linear layers and LSTM layers
        for i, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if i < len(self.lstm_layers):
                # saving the states is unnecessary at the moment, but could enable the continuation of sequences in the future
                x, self.lstm_states[i] = self.lstm_layers[i](x, self.lstm_states[i])
                x = x.squeeze(0)
                x = self.nonlinearity(x)

        return x
        
    
    def reset_state(self):
        self.lstm_states = []
        for layer_index in range(len(self.lstm_layers)):
            self.lstm_states.append((torch.zeros((1, self.layer_widths[layer_index + 1])),
                                     torch.zeros((1, self.layer_widths[layer_index + 1]))))