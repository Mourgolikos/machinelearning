
"""
Classes:
    Autoencoder: Implementation of an autoencoder (feedforward with backpropagation)
    
Usage:

    a1 = Autoencoder([4, 2, 1], [1, 3, 4]) # The first arguments contains the number of neurons per layer for the encoder
                                                 # The second for the decoder, this would create a network with layers 4-2-1-3-4

    a1.showexample([0.5, 0.8, 0.1, 0.9])  # Shows an input vector to the autoencoder, the autoencoder performs one forward and one backward pass adjusting the weights
    output = n1.ff([0.5, 0.8, 0.1, 0.9])  # Forward pass through the encoder part
    error = n1.bp([0.4])                    # Backward pass through the encoder part    
"""

__author__ = "Panagiotis Thomaidis"

from backpropagation import Layer

class Autoencoder:

    def __init__(self, encoderlayers, decoderlayers):
        if encoderlayers[0] != decoderlayers[-1]:
            raise ValueError('Output of autoencoder must have the same number of units as input')
        if encoderlayers[-1] != decoderlayers[0]:
            raise ValueError('Output of encoder must have the same number of units as input of decoder')
        self.enc = [0]*(len(encoderlayers)-1)
        self.dec = [0]*(len(decoderlayers)-1)
        for i in range(len(encoderlayers)-1):
            self.enc[i] = Layer(encoderlayers[i], encoderlayers[i+1])
        for i in range(len(decoderlayers)-1):
            self.dec[i] = Layer(decoderlayers[i], decoderlayers[i+1])

    def showexample(self, v):
        # Encode
        output = v
        for l in self.enc:
            output = l.ff(output)
        # Decode
        for l in self.dec:
            output = l.ff(output)
        # Compare output with input
        diff = [v[j]-output[j] for j in range(len(output))]
        errin = diff
        # Backpropagate error
        for l in reversed(self.dec):
            errin = l.bp(errin)
        for l in reversed(self.enc):
            errin = l.bp(errin)
        return output

    def ff(self, input):    # propagate until the hidden layer
        output = input
        for l in self.enc:
            output = l.ff(output)
        return output

    def bp(self, error):    # back-propagate from the hidden layer
        errin = error
        for l in reversed(self.enc):
            errin = l.bp(errin)
        return errin

if __name__ == "__main__":

    # vectors
    vectors = [[0, 0], [0, 1], [0, 1], [0, 0]]

    # Using the Autoencoder
    print("=== Autoencoder ===")
    a1 = Autoencoder([2, 1], [1, 2])
    for kk in range(2000):
        for i in range(len(vectors)):
            a1.showexample(vectors[i])

    for i in range(len(vectors)):
        out = a1.showexample(vectors[i])
        print(out)
