
"""
Classes:
    Layer: Implementation of a layer of neurons
    Network: Implementation of a feedforward neural network
    
Usage:

    l1 = Layer(2, 3) # Create a layer with 2 inputs and 3 outputs
    n1 = Network([4, 3, 2]) # Create a network with 2 input, 3 hidden, and 1 output neurons
    n2 = Network([100, 10, 5, 10])  # Create a network with 2 hidden layers

    output = n1.ff([0.5, 0.8, 0.1, 0.9])  # Forward pass
    error = n1.bp([0.4, -0.7])            # Backward pass, returns the error at the 4 input neurons
    
    etc. (see the examples at the end of the file)
    
Both contain methods to do a forward pass and a backward pass.
Layers and Networks can be chained together.

"""

__author__ = "Panagiotis Thomaidis"

from math import exp
from random import random

# activation functions
linear  = lambda x: x
binary  = lambda x: 1 if (x>=0) else 0
rectif  = lambda x: x if (x>=0) else 0
sigmoid = lambda x: 1/(1 + exp(-x))
stoch   = lambda x: 1 if ((1/(1 + exp(-x)))>random()) else 0

class Layer:
    """Implementation of a layer of neurons
    
    Params:
        nx:     the number of input neurons
        ny:     the number of output neurons
        w[][]:  the weight of the layer
        b[]:    the bias terms of the output neurons
        dw[][]: the calculated weight change
        db[]:   the calculated bias change
        x[]:    the latest input vector
        y[]:    the latest output vector
        ex[]:   the latest output error vector
        ey[]:   the latest input error vector
    
    """
    
    def __init__(self, inno, outno):
        """Constructor
        
        Args:
            inno: the number of input neurons
            outno: the number of output neurons
            
        """
        self.nx = inno    # number of inputs
        self.ny = outno  # number of outputs
        self.w = [[random()*2-1 for j in range(self.ny)] for i in range(self.nx)]    # weights 
        self.b = [random()*2-1 for j in range(self.ny)]  # bias terms
        self.dw = [[0]*self.ny for i in range(self.nx)]    # weight change 
        self.db = [0]*self.ny  # bias change
        self.x = [0]*self.nx # last input vector
        self.y = [0]*self.ny # last output vector
        self.ex = [0]*self.nx # last error at input
        self.ey = [0]*self.ny # last error at output

    def ff(self, input):
        """Feedforward pass
        
        Args:
            input: activation values of the input neurons
        Returns:
            the activation of the neurons of this layer
            
        """
        self.x = input
        for j in range(self.ny):
            self.y[j] = self.b[j]
            for i in range(self.nx):
                self.y[j] += self.x[i]*self.w[i][j]
            self.y[j] = sigmoid(self.y[j])
        return self.y

    def bp(self, error):
        """Back-propagate the error and update the weights
        
        Args:
            error: the error of the output neurons
        Returns:
            the error of the input neurons
            
        """
        self.ey = error
        self.ex = [0]*self.nx
        for j in range(self.ny):
            dEdz = self.y[j]*(1-self.y[j])*self.ey[j]         # dE/dz
            for i in range(self.nx):
                self.ex[i] += dEdz*self.w[i][j]         # back propagated error derivative
                self.dw[i][j] = dEdz*self.x[i]          # weight update
                self.w[i][j] += self.dw[i][j]
            self.db[j] = dEdz                           #  bias update
            self.b[j] += self.db[j]
            
        return self.ex
    
class Network:
    """Implementation of a feedforward neural network
    
    Params:
        ls[]: the layers that constitute the network
    
    """

    def __init__(self, ns):
        """Constructor
        
        Args:
            ns[]: the number of neurons per layer
        
        """
        self.ls = [0]*(len(ns)-1)
        for i in range(len(ns)-1):
            self.ls[i] = Layer(ns[i], ns[i+1])
    
    # Feed Forward
    def ff(self, input):
        """Feedforward pass
        
        Args:
            input: activation values of the input layer
        Returns:
            the activation of the neurons of the output layer
            
        """
        output = input
        for l in self.ls:
            output = l.ff(output)
        return output

    def bp(self, error):
        """Back-propagate the error and update the weights
        
        Args:
            error: the error of the output layer
        Returns:
            the error of the input layer
            
        """
        errin = error
        for l in reversed(self.ls):
            errin = l.bp(errin)
        return errin
            
if __name__ == "__main__":
    
    # targets and vectors
    targets = [[0], [1], [1], [0]]
    vectors = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    # Using the Layers
    print("=== Layers ===")
    l1 = Layer(2, 3)
    l2 = Layer(3, 1)

    for k in range(2000):
        for i in range(len(vectors)):
            out = l1.ff(vectors[i])
            out2 = l2.ff(out)
            diff = [targets[i][j]-out2[j] for j in range(len(out2))]    # Error derivative: -(target - output)
            err2 = l2.bp(diff)
            err = l1.bp(err2)
            
    # Print the network outputs after training
    for i in range(len(vectors)):
        out = l1.ff(vectors[i])
        out2 = l2.ff(out)
        print(out2)
    
    # Using the Network
    print("=== Network ===")
    n1 = Network([2, 3, 1])
    for k in range(2000):
        for i in range(len(vectors)):
            out = n1.ff(vectors[i])
            diff = [targets[i][j]-out[j] for j in range(len(out))]
            err = n1.bp(diff)

    # Print the network outputs after training            
    for i in range(len(vectors)):
        out = n1.ff(vectors[i])
        print(out)
