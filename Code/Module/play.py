from numpy import *
from tqdm import *
from matplotlib import *
import numpy as np
import os
import pickle

class First_layers: # fct d'activation : Relu
    def __init__(self, weights, biases):
        self.weights = np.array(weights, dtype=np.float64)
        self.biases = np.array(biases, dtype=np.float64)

    def forward(self, inputs):   
        self.inputs = np.array(inputs)
        self.output_not_activated = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output_not_activated)

class Last_layer: # fct d'activation : Softmax
    def __init__(self, weights, biases): 
        self.weights = np.array(weights, dtype = np.float64)
        self.biases = np.array(biases, dtype = np.float64)

    def forward(self, inputs): 
        self.inputs = np.array(inputs)
        output_not_activated = np.dot(inputs, self.weights) + self.biases
        exp_values = np.exp(output_not_activated - np.max(output_not_activated, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True) 

def open():
    global nn
    n = len(os.listdir("Mia/weights"))
    nn = [First_layers(np.load(f"Mia/weights/weights_{j}.npy", [], True), np.load(f"Mia/biases/biases_{j}.npy", [], True)) for j in range(n-1)] + [Last_layer(np.load(f"Mia/weights/weights_{n-1}.npy", [], True), np.load(f"Mia/biases/biases_{n-1}.npy", [], True))]

def forpropagation(input):
    nn[0].forward(input)
    for i in range(1,len(nn)):
        nn[i].forward(nn[i-1].output)
    return nn[-1].output

open()
b2 = np.load("Mia/bool2.npy",allow_pickle=True)
while b2[0]:
    try:
        b1 = np.load("Mia/bool1.npy",allow_pickle=True)
        if b1[0] == 3: 
            inputs = np.load("Mia/input.npy",allow_pickle=True).reshape(1,400)
            np.save("Mia/output.npy",forpropagation(inputs))
            np.save("Mia/bool1.npy",np.array([2]))
        b2 = np.load("Mia/bool2.npy",allow_pickle=True) 
    except (EOFError,ValueError, pickle.UnpicklingError,PermissionError) as e:
        ()
user_input = input("Please enter something: ")
print(f"You entered: {user_input}")
