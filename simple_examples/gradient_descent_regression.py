import matplotlib.pyplot as plt
import matplotlib
import numpy as np
np.random.seed(seed=0)
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called

data = set([(0, 1), (1, 5), (2,  7), (3, 8), (4, 15), (2.5, 10)])

def grad(params, data):
    a = params[0]
    b = params[1]
    grad = 0.
    for (x_i, y_i) in data:
        grad += (a*x_i + b - y_i)*np.array([x_i, 1])
    grad *= float(2./len(data))
    
    return grad

params = np.array([0., 0.])
lr = 0.01

fig, ax = plt.subplots()

iter_index = 0
while True:
    params = params - lr*grad(params, data)
    
    if iter_index % 10 == 0:
        print(f"iteration {iter_index+1}")
        
        ax.clear()
        x_min = min([sample[0] for sample in data])
        x_max = max([sample[0] for sample in data])
        ax.plot([x_min, x_max], [params[1] + x_min*params[0], params[1] + x_max*params[0]], "b")
        ax.scatter([sample[0] for sample in data], [sample[1] for sample in data], c="r")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        
        plt.pause(1e-2)
    
    iter_index += 1
    




