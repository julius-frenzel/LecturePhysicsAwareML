import numpy as np
#import torch.nn as nn
from math import pi
from autodiff import AutodiffArray, matmul, sigmoid, matsum, matsquare
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called

# ----------possible improvements------------
# validate the implementation using PyTorch


# set seed to make results reproducible
np.random.seed(seed=0)

# layer sizes for the network
layer_sizes = [30, 30, 1]

# initialize parameters
weights = []
biases = []
for layer_index in range(len(layer_sizes)-1):
    # weights
    s_n = layer_sizes[layer_index]
    s_np1 = layer_sizes[layer_index + 1]
    random_weights = np.random.randn(s_np1, s_n)/np.sqrt((s_n + s_np1)/2)
    weights.append(AutodiffArray(random_weights, requires_grad=True, retain_grad=True))
    
    # biases
    biases.append(AutodiffArray(np.zeros((layer_sizes[layer_index])).reshape(-1, 1), requires_grad=True, retain_grad=True))

# function for performing a forward pass
def forward(x):
    for layer_index in range(len(layer_sizes)-1):

        x = matmul(weights[layer_index], matsum(x, biases[layer_index]))

        if layer_index < len(layer_sizes) - 2:
            x = sigmoid(x)
        
    return x

# function for updating parameters using gradient descent
def sgd(weights, biases, lr=0.001):
    for param in weights + biases:
        param -= lr * param.grad
        param.grad = None

# definition of the optimization problem
func = lambda x: np.sin(2*pi*x)
x = np.linspace(-1, 1, 20)
y = func(x)

# hyperparameters
batch_size = 1 # the best results have been observed with batch size 1
learning_rate = 0.03

fig, axs = plt.subplots(2, 1)

# training loop
epoch_index = 0
losses_plot = []
epochs_plot = []
while True:
    
    # iterate over all samples
    loss_total = AutodiffArray(np.zeros((1,1)))
    loss_batch = AutodiffArray(np.zeros((1,1)))
    for sample_index, (x_i, y_i) in enumerate(zip(x, y)):
        x_i = AutodiffArray(np.array(x_i).reshape(-1, 1))
        y_i = np.array(y_i).reshape(-1, 1)
        
        # perform forward pass
        y_i_pred = forward(x_i)
        
        # compute loss
        loss_sample = matsquare(matsum(y_i_pred, -y_i))
        
        loss_total = matsum(loss_total, loss_sample)
        loss_batch = matsum(loss_batch, loss_sample)
        
        if (sample_index + 1) % batch_size == 0:
            # compute gradients
            loss_batch.backward()
            # perform gradient descent step for the batch (also resets the gradients)
            sgd(weights, biases, lr=learning_rate)
            # reset accumulated loss for the next batch
            loss_batch = AutodiffArray(np.zeros((1,1)))
    
    losses_plot.append(loss_total.item())
    epochs_plot.append(epoch_index + 1)
    
    # visualize progress
    if epoch_index % 100 == 0:
        print(f"epoch {epoch_index+1}: loss={loss_total}")
        
        axs[0].clear()
        x_plot = np.linspace(x.min(), x.max(), 1000)
        axs[0].plot(x_plot, func(x_plot), "b")
        y_plot_pred = [forward(AutodiffArray(np.array(x_i).reshape(-1, 1))).item() for x_i in x_plot]
        axs[0].plot(x_plot, y_plot_pred, "r")
        y_pred = [forward(AutodiffArray(np.array(x_i).reshape(-1, 1))).item() for x_i in x]
        axs[0].scatter(x, y_pred, c="r")
        axs[0].set_title("predicted values")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(True)
        
        axs[1].clear()
        axs[1].plot(epochs_plot, losses_plot)
        axs[1].set_title("loss over epochs")
        axs[1].set_xlabel("epochs")
        axs[1].set_ylabel("loss")
        axs[1].set_yscale("log")
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.pause(1e-2)
    
    epoch_index += 1
        





