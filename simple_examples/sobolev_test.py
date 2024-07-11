import torch
import torch.nn as nn

class FFNN(torch.nn.Module):
    def __init__(self, layer_widths):
        super(FFNN, self).__init__()
        
        self.layer_widths = layer_widths
        self.dtype = torch.float32
        
        self.layers = nn.ParameterDict({f"layer_{i}": nn.Linear(layer_i, layer_ip1, bias=(i < len(layer_widths) - 2)) for i, (layer_i, layer_ip1) in enumerate(zip(layer_widths[:-1], layer_widths[1:]))})
    
    def forward(self, x):
        y = x
        for layer_name in self.layers:
            print(layer_name)
            y = self.layers[layer_name](y)
        
        return y
            
        
layers_widths = [1, 10, 10, 1]

ffnn = FFNN(layers_widths)

x = torch.tensor([1], dtype=ffnn.dtype, requires_grad=True)

y = ffnn(x)

print(f"output: {y}")

loss = y - 0.

grad_x = torch.autograd.grad(inputs=x, outputs=y, create_graph=True) # create graph for gradients, so that they can be used in the loss function    

print(f"gradient of x: {grad_x}")

print("done")