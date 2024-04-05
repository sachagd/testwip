from numpy import *
from tqdm import *
from random import randint
from matplotlib import *
from tkinter import *
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import re

class First_layers: # fct d'activation : Relu
    def __init__(self, weights, biases):
        self.weights = np.array(weights, dtype=np.float64)
        self.biases = np.array(biases, dtype=np.float64)

    def forward(self, inputs):   
        self.inputs = np.array(inputs)
        self.output_not_activated = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output_not_activated)

    def backward(self, input_gradient, learning_rate):
        input_gradient = input_gradient * deriv_relu(self.output_not_activated)
        self.output_gradient = np.dot(input_gradient, self.weights.T)
        self.weights -= np.dot(self.inputs.T, input_gradient) * learning_rate / np.shape(input_gradient)[0]
        self.biases -= np.mean(input_gradient, axis = 0) * learning_rate 

class Last_layer: # fct d'activation : Softmax
    def __init__(self, weights, biases): 
        self.weights = np.array(weights, dtype = np.float64)
        self.biases = np.array(biases, dtype = np.float64)

    def forward(self, inputs): 
        self.inputs = np.array(inputs)
        output_not_activated = np.dot(inputs, self.weights) + self.biases
        exp_values = np.exp(output_not_activated - np.max(output_not_activated, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True) 

    def backward(self, y_pred, y_true, learning_rate): # fct de perte : Categorical-crossentropy
        input_gradient = y_pred - y_true 
        self.output_gradient = np.dot(input_gradient, self.weights.T)
        self.weights -= np.dot(self.inputs.T, input_gradient) * learning_rate / np.shape(input_gradient)[0]
        self.biases -= np.mean(input_gradient, axis = 0) * learning_rate

def open():
    global nn
    n = len(os.listdir("Mia/weights"))
    nn = [First_layers(np.load(f"Mia/weights/weights_{j}.npy", [], True), np.load(f"Mia/biases/biases_{j}.npy", [], True)) for j in range(n-1)] + [Last_layer(np.load(f"Mia/weights/weights_{n-1}.npy", [], True), np.load(f"Mia/biases/biases_{n-1}.npy", [], True))]

def create(npc):
    n = len(npc)
    os.makedirs("Mia/weights")
    os.makedirs("Mia/biases")
    os.makedirs("Mia/info")    
    for j in range(n-1):
        np.save(f"Mia/weights/weights_{j}", 0.1 * np.random.randn(npc[j],npc[j+1]))
        np.save(f"Mia/biases/biases_{j}", np.random.randn(npc[j+1]))
    np.save("Mia/info/info_npc.npy",npc)
    np.save("Mia/info/info_old_scores.npy",[0])
    np.save("Mia/info/info_actual_score.npy", [0])

# def get_info():
#     info = np.load("info.npy", [], True)
#     print(info[0])
#     plt.plot(range(len(info[2])), info[2]) 
#     plt.title("best accuracy : "+str(info[1]))
#     plt.show()

# def save():
#     try:
#         info = np.load("info.npy", [], True) 
#         folder = f"{len(os.listdir())-8} {round(info[1]*100,2)} {info[0]}"
#         os.makedirs(folder)
#         os.rename("biases.npy",folder+"/biases.npy")
#         os.rename("weights.npy",folder+"/weights.npy")
#         os.rename("info.npy",folder+"/info.npy")
#     except OSError:
#         raise FileExistsError("pas de fichiers à sauvergarder")

# def delete_file():
#     try:
#         os.remove("biases.npy")
#         os.remove("weights.npy")
#         os.remove("info.npy")
#     except OSError:
#         raise FileExistsError("pas de fichiers à supprimer")

# def delete_folder(n):
#     for folder in os.listdir():
#         if os.path.isdir(folder) and folder.startswith(str(n)):
#             b = input(f"le nom du dossier est {folder} êtes-vous sûr de continuer ? (o/n)")
#             if b == "o":
#                 os.remove(folder+"/biases.npy")
#                 os.remove(folder+"/weights.npy")
#                 os.remove(folder+"/info.npy")
#                 os.rmdir(folder)
#             return
#     raise FileExistsError(f"aucun dossier ne commence par {n}")

# def load(n):
#     if "info.npy" in os.listdir():
#         raise FileExistsError("des fichiers sont déjà chargés")
#     else:
#         for folder in os.listdir():
#             if os.path.isdir(folder) and folder.startswith(str(n)):
#                 os.rename(folder+"/biases.npy", "biases.npy")
#                 os.rename(folder+"/weights.npy", "weights.npy")
#                 os.rename(folder+"/info.npy", "info.npy")
#                 return
#         raise FileExistsError(f"aucun dossier ne commence par {n}")
    
# def resave(rename_folder = False):
#     info = np.load("info.npy", [], True)
#     for folder in os.listdir():
        # if os.path.isdir(folder) and len(os.listdir(folder)) == 0:
#             if rename_folder:
#                 id = re.match(r'^\d+\s', folder).group()
#                 new_folder = f"{id} {round(info[1] * 100, 2)} {info[0]}"
#                 os.rename(folder, new_folder)
#             else:
#                 new_folder = folder
#             os.rename("biases.npy",new_folder+"/biases.npy")
#             os.rename("weights.npy",new_folder+"/weights.npy")
#             os.rename("info.npy",new_folder+"/info.npy")
#             return
#     raise FileExistsError("pas de fichiers à sauvergarder")

def ls():
    for file in os.listdir():
        if file[0] in "1234567890":
            print(file)

def train(lr, nb, t_samples, trainmode = -1):
    open()
    n = len(os.listdir("Mia/Saves"))//2
    for _ in tqdm(range(nb)):
        for i in tqdm(range(n), leave=False):
            inputfile = np.load(f"Mia/Saves/ArraySaved_{i}.npy", [], True)
            labelfile = np.load(f"Mia/Saves/InputSaved_{i}.npy", [], True)
            n1 = np.shape(inputfile)[0]
            if trainmode == -1:
                for _ in tqdm(range((n1//t_samples) * 2), leave=False):
                    choix = [randint(0,n1-1) for _ in range(t_samples)]
                    inputs = [np.reshape(inputfile[k],(400)) for k in choix]
                    labels = [convert_input(labelfile[k]) for k in choix]
                    output = forpropagation(inputs)
                    backpropagation(output, labels, lr)
            else:
                for j in tqdm(range(n1), leave = False):
                    if labelfile[j][trainmode]:
                        inputs = np.reshape(inputfile[j],(400))
                        labels = convert_input(labelfile[j])
                        output = forpropagation(inputs)
                        backpropagation(output, labels, lr)
    n = len(nn)
    folder = f"{len(os.listdir())}"
    os.makedirs(f"Mia/{folder}")
    shutil.move("Mia/biases",f"Mia/{folder}/biases")
    shutil.move("Mia/weights",f"Mia/{folder}/weights")
    shutil.copytree("Mia/info",f"Mia/{folder}/info")
    os.makedirs("Mia/weights")
    os.makedirs("Mia/biases")
    for i in range(n):
        np.save(f"Mia/weights/weights_{i}", nn[i].weights)
        np.save(f"Mia/biases/biases_{i}", nn[i].biases)
    
def forpropagation(input):
    nn[0].forward(input)
    for i in range(1,len(nn)):
        nn[i].forward(nn[i-1].output)
    return nn[-1].output

def backpropagation(y_pred, y_true, lr):
    nn[-1].backward(y_pred, y_true, lr)
    for i in range(len(nn)-2,-1,-1):
        nn[i].backward(nn[i+1].output_gradient, lr)

def deriv_relu(x):
    return x>0

def convert_input(label):
    i = 0
    if ((label[4]) and not (label[5]) and not (label[6])):
        i += 9
    elif(not (label[4]) and (label[5]) and not (label[6])):
        i += 18
    elif(not (label[4]) and not (label[5]) and (label[6])):
        i += 27
    elif((label[4]) and (label[5]) and not (label[6])):
        i += 34
    elif((label[4]) and not (label[5]) and (label[6])):
        i += 45
    elif(label[4] or label[5] or label[6]):
        raise ValueError
    if (not (label[1]) and not (label[0]) and not (label[3]) and (label[2])):
        i += 1
    elif (not (label[1]) and (label[0]) and not (label[3]) and not (label[2])):
        i += 2
    elif (not (label[1]) and not (label[0]) and (label[3]) and not (label[2])):
        i += 3
    elif((label[1]) and not (label[0]) and not (label[3]) and not (label[2])):
        i += 4
    elif(not (label[1]) and (label[0]) and not (label[3]) and (label[2])):
        i += 5
    elif((label[1]) and not (label[0]) and not (label[3]) and (label[2])):
        i += 6
    elif(not (label[1]) and (label[0]) and (label[3]) and not (label[2])):
        i += 7
    elif((label[1]) and not (label[0]) and (label[3]) and not (label[2])):
        i += 8
    elif(label[1] or label[0] or label[3] or label[2]):
        raise ValueError
    y_true = [0 for _ in range(54)]
    y_true[i] = 1
    return y_true
