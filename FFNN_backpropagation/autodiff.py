import numpy as np
#import torch.nn as nn
import time

class AutodiffArray(np.ndarray):
    def __new__(cls, input_array, requires_grad=False, retain_grad=False, origin=None, *args, **kwargs):
        #args = (arg for arg in args if not arg in ["requires_grad", "origin"])
        obj = np.array(input_array, *args, **kwargs).view(cls)
        return obj

    def __init__(self, input_array, requires_grad=False, retain_grad=False, origin=None, *args, **kwargs):
        if retain_grad and not requires_grad:
            print("Argument \"retain_grad\" for contructor of class AutodiffArray may have no effect without \"requires_grad\". Setting \"requires_grad\" to True.")
            requires_grad = True
            
        self.requires_grad = requires_grad  # indicates whether gradients are to be computed for this array
        self.retain_grad = retain_grad # indicates whether the gradients are to be stored for optimization (avoids storing unnecessary intermediate gradients)
        self.origin = origin  # the operation that created the array
        self.grad = None # gradient of the array
    
    def backward(self):
        assert self.requires_grad, f"Can't perform backward pass for {self}, because \"self.requires_grad\"=False."
        
        if self.grad is None:
            self.grad = np.ones_like(self) # set gradients to one, if this array is the output
        
        if self.origin is not None:
            return self.origin.backward(self)
        
        # delete unneeded gradients of input nodes
        if self.origin is None and not self.retain_grad:
            self.grad = None
            return True
        

class DifferentiableFunction:
    def __init__(self):
        self.saved_values = dict()
        self.output_requires_grad = False
        self.parents_with_gradients = []
        self.children = []
        self.children_grads = []
    
    def __call__(self, *args):
        #for arg in args:
        #    if not isinstance(arg, AutodiffArray):
        #        raise TypeError(f"arguments for {self} must be of type {AutodiffArray}")
        return self.forward(*args)
    
    def forward_impl(self, *args):
        pass
    
    def forward(self, *args):
        for arg in args:
            if type(arg) == AutodiffArray and arg.requires_grad:
                self.output_requires_grad = True
        
        results = self.forward_impl(*args)
        self.children = results if isinstance(results, tuple) else [results]
        assert len(self.parents_with_gradients) > 0, "Attribute \"self.parents_with_gradients\" has to be set in \"forward_impl\"."
        
        for child in self.children:
            child.requires_grad = self.output_requires_grad
            child.origin = self
        
        self.children_grads = [None]*len(self.children)
        
        return results
    
    def backward_impl(self):
        pass
    
    def backward(self, child):
        if len(self.saved_values) == 0:
            raise ValueError(f"Stored values of {self} are empty. Perform a forward pass to populate them.")
        
        if any(child is c for c in self.children):
            child_index = self.children.index(child)
            self.children_grads[child_index] = child.grad
        
        if any([child_grad is None for child_grad in self.children_grads]):
            return False
        else:
            grads = self.backward_impl()
            grads = grads if isinstance(grads, tuple) else [grads]
            
            # perform backpropagation for parents
            for grad, parent in zip(grads, self.parents_with_gradients):
                #print(f"setting gradient for parent {parent} to {grad}")
                if type(parent) == AutodiffArray and parent.requires_grad:
                    parent.grad = grad
                    parent.backward()
            
            # cleanup
            self.saved_values = dict()
            self.children_grads = None
            for child in self.children: # reset gradients of childred, which don't need to retain their gradients
                if not child.retain_grad:
                    child.grad = None
            return grads


class Matmul(DifferentiableFunction):
    def __init__(self):
        super().__init__()
    
    def forward_impl(self, A, x):
        self.parents_with_gradients = [A, x]
        x = x.reshape((-1,1))
        
        y = AutodiffArray(A @ x)
        self.saved_values["x"] = x.copy()
        self.saved_values["A"] = A.copy()
        
        return y
    
    def backward_impl(self):
        grad_y = self.children_grads[0]
        grad_y = grad_y.reshape((-1, 1))
        
        dydA = self.saved_values["x"]
        dydx = self.saved_values["A"].transpose()
        grad_A = grad_y @ dydA.transpose()
        grad_x = dydx @ grad_y
        
        return grad_A, grad_x

# convenience function
def matmul(*args):
    return Matmul()(*args)


class Sigmoid(DifferentiableFunction):
    def __init__(self):
        super().__init__()
    
    def forward_impl(self, x):
        self.parents_with_gradients = [x]
        
        sigmoid_value = AutodiffArray(1/(1+np.exp(-x)))
        self.saved_values["sigmoid_value"] = sigmoid_value
        
        return sigmoid_value
    
    def backward_impl(self):
        grad_y = self.children_grads[0]
            
        sigmoid_value = self.saved_values["sigmoid_value"]
        dydx = sigmoid_value * (1 - sigmoid_value)
        grad_x = dydx*grad_y
        
        return grad_x

# convenience function
def sigmoid(*args):
    return Sigmoid()(*args)


class Matsum(DifferentiableFunction):
    def __init__(self):
        super().__init__()

    def forward_impl(self, a, b):
        assert a.ndim == b.ndim, f"Error in {self}: The number of dimensions of the inputs must match."
        self.parents_with_gradients = [a, b]
        
        y = AutodiffArray(a + b)
        self.saved_values["dummy"] = 0 # save a dummy value to pass the check during the backward pass
        
        return y
    
    def backward_impl(self):
        grad_y = self.children_grads[0]
        
        grad_a = grad_y
        grad_b = grad_y
        
        return grad_a, grad_b

# convenience function
def matsum(*args):
    return Matsum()(*args)


class Matsquare(DifferentiableFunction):
    def __init__(self):
        super().__init__()

    def forward_impl(self, x):
        self.parents_with_gradients = [x]
        
        y = AutodiffArray(x**2)
        self.saved_values["x"] = x.copy()
        
        return y
    
    def backward_impl(self):
        grad_y = self.children_grads[0]
        
        dydx = 2*self.saved_values["x"]
        grad_x = dydx*grad_y
        
        return grad_x

# convenience function
def matsquare(*args):
    return Matsquare()(*args)