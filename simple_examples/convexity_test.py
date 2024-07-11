import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class Linear_positive(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear_positive, self).__init__(*args, **kwargs)

    def forward(self, x):
        weights = torch.exp(self.weight)
        return nn.functional.linear(x, weights, self.bias)

class FFNN(torch.nn.Module):
    def __init__(self, layer_widths, nonlinearity="softplus", dtype=torch.float32):
        super(FFNN, self).__init__()
        
        self.layer_widths = layer_widths
        self.dtype = dtype
        
        self.layers = nn.ParameterDict()
        for i, (layer_i, layer_ip1) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
            if i == 0:
                layer_type = nn.Linear
            else:
                layer_type = Linear_positive
            self.layers[f"layer_{i}"] = layer_type(layer_i, layer_ip1, bias=(i < len(layer_widths) - 2))
        
        if nonlinearity == "softplus":
            self.nonlinearity = nn.Softplus()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "leaky_relu":
            self.nonlinearity = nn.LeakyReLU()
        else:
            raise ValueError(f"unrecognized type of nonlinearity: {nonlinearity}")
    
    def forward(self, x):
        y = x
        for layer_index, layer_name in enumerate(self.layers):
            y = self.layers[layer_name](y)
            if layer_index < len(self.layers) - 1:
                y = self.nonlinearity(y)
        
        return y

# sufficient conditions for convexity
# 1) non-negative weights (except for the first layer))
# 2) convext activation functions
# 3) monotinically increasing activation functions

torch.random.manual_seed(seed=3)
dtype = torch.float32
shape_F = (4, 4)
layer_widths = [shape_F[0]*shape_F[1], 50, 50, 1]
ffnn = FFNN(layer_widths, nonlinearity="relu", dtype=dtype)

# Input matrix
F = torch.rand(shape_F, dtype=dtype)
print(f"number of elements in input matrix: {torch.numel(F)}")

x_plot = np.linspace(-1,1,100)

fig, axs = plt.subplots(*shape_F, figsize=(10,10))

for i in range(shape_F[0]):
    for j in range(shape_F[1]):
        
        det_F_plot = []
        y_plot = []
        
        with torch.no_grad():
            
            for x in x_plot:
                
                F_ij = F
                F_ij[i,j] = x
                det_F_plot.append(torch.det(F).item())
                x = F.flatten()
                
                y_i = ffnn(x)
                
                y_plot.append(y_i.item())
        
        y_plot = np.array(y_plot)
        det_F_plot = np.array(det_F_plot)



        axs[i,j].set_title("y over x and def(F)")
        axs[i,j].plot(x_plot, y_plot)
        axs[i,j].plot(det_F_plot, y_plot)
        axs[i,j].set_xlabel("x, det(F)")
        axs[i,j].set_ylabel("y")
        axs[i,j].grid()
        #axs[i,j].legend(["y over x", "y over def(F)"])

plt.tight_layout()
plt.show()