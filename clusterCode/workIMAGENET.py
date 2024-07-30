# -*- coding: utf-8 -*-
"""main_AGD_IHT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bW4Ie7gpvVv5n_BlbMTZUO59GA9fxz5g

### AUTHOR: Dimitri Kachler

# Global Parameters
"""



#User-Dependent Variables
layerByLayer = False
datasetChoice = "MNIST"

# -------------- INACTIVE
#useNeptune = True

"""# Imports"""

# Neural Networks
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer, required
import torch

# Arrays & Mathematics
import math
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import pandas as pd

#System / IO
import abc
import itertools
import importlib

#Data Visualization
#import seaborn as sns

#External Utilities
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# CUDA Check
print(torch.__version__)
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Commented out IPython magic to ensure Python compatibility.
# NOTE: you can still run this and it should still work and send the data to my Neptune.ai project,
# unfortunately you won't be able to see the graph without my account
# Capture makes it so that the cell doesn't output text
# NOTE: %%capture doesn't work on jupyterlab
#%%capture
#
try:
    import neptune
    print('it worked, already imported neptune')
except ImportError as e:
    abort()
#     %pip install -U neptune
    import neptune
#import neptune
from getpass import getpass

project="dimitri-kachler-workspace/sanity-MNIST"
api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQxNDllOS04OGY1LTRjM2EtYTczZi0xNWI0NTRmZTA1OTEifQ=="
#project = neptune.init_project(api_token=api_token, project=project)

"""# Github Imports"""

import subprocess
#subprocess.call(["git", "pull"])
subprocess.call(["git", "clone","https://github.com/NanoNero1/IHT_AGD"])
##!git clone https://github.com/NanoNero1/IHT_AGD

# Commented out IPython magic to ensure Python compatibility.
# NOTE: This might be very expenisive! - only keep active on prototype testing
# %load_ext autoreload
# %autoreload 2

#%cd /content/IHT_AGD/

# NOTE: for whatever reason, this does not actually work locally, you have to manually open terminal to and do a git pull.


# NOTE: for some reason, there is difficulty in what directory git pull should be called
#!git pull

import os

path = os.getcwd()
print(path)

ihtPath = path + "/IHT_AGD"
os.chdir(ihtPath)
path = os.getcwd()
print(path)

subprocess.call(["git", "pull"])

os.chdir("..")
path = os.getcwd()
print(path)

#subprocess.call(["cd", "IHT_AGD"])
#




# Data Collection
import IHT_AGD.data_loaders.dataLoaders as dataLoaders
datasetChoice = dataLoaders.datasetChoice

##############################
from datasets import load_dataset
from torchvision.transforms import Lambda
import time

# If the dataset is gated/private, make sure you have run huggingface-cli login
timeOne = time.time()
dataset = load_dataset("imagenet-1k")


print("yes this did gather imagenet!")

timeTwo = time.time()

print(timeTwo - timeOne)

# Select the first row in the dataset
sample = dataset['train'][0]

# Split up the sample into two variables
datapt, label = sample['image'], sample['label']

print(label)
print(datapt)


### SOURCE: https://medium.com/@ricodedeijn/image-classification-computer-vision-from-scratch-pt-3-9d5fbcf3c363
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # Loads the dataset that needs to be transformed
        self.dataset = load_dataset("imagenet-1k", split=f"{dataset}")

    def __getitem__(self, idx):
        # Sample row idx from the loaded dataset
        sample = self.dataset[idx]
        
        # Split up the sample example into an image and label variable
        data, label = sample['image'], sample['label']
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to size 256x256
            Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),  # Convert all images to RGB format
            transforms.ToTensor(),  # Transform image to Tensor object
        ])
        
        # Returns the transformed images and labels
        return transform(data), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

# Call the class to populate variable train_set with the train data
train_set = MyDataset('train')
test_set = MyDataset('test')

BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

print("so far so good!")


##############################


###train_loader = dataLoaders.train_loader
###test_loader = dataLoaders.test_loader



#abort() # I should have the data loaded locally!!!!
#abort()
# Just for debugging
#import IHT_AGD.architectures.architect
#IHT_AGD.architectures.architect.seeVariable

# Neural Netwok Architecture
from IHT_AGD.architectures.convNets import MNIST_convNet

# Taining and Testing Functions
from IHT_AGD.modelTrainTest.trainingMetrics import getTestAccuracy,getTestLoss
from IHT_AGD.modelTrainTest.trainLoop import train

# Optimizers (base, SGD, AGD, IHT, etc.)
from IHT_AGD.optimizers.baseOptimizer import myOptimizer
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD
from IHT_AGD.optimizers.ihtSGD import ihtSGD
from IHT_AGD.optimizers.vanillaAGD import vanillaAGD
from IHT_AGD.optimizers.ihtAGD import ihtAGD
from IHT_AGD.optimizers.nativePytorchSGD import dimitriPytorchSGD

# Visualization Functions
from IHT_AGD.visualizationGraphs.plotting import plotMetric

# Experiment Functions
from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer
from IHT_AGD.experimentScaffolding.chooseOptimizer import fixedChooseOptimizer
from IHT_AGD.experimentScaffolding.experimentFuncs import runOneExperiment
from IHT_AGD.experimentScaffolding.experimentFuncs import runMainExperiment
from IHT_AGD.experimentScaffolding.experimentFuncs import runPipeline

#To know the sizes
firstInput, firstTarget = next(iter(train_loader))
print(firstInput.size())

"""# Experiment Github Imports?

# Tracking
"""

variablesToTrack = ['sparsity','sparsityBias','lr','iteration','trackSparsity','trackSparsityBias','trackSparsityLinear','testAccuracy','beta']
expensiveVariables = ['testAccuracy']
functionsToHelpTrack = ['trackingSparsity','getTestAccuracy']
expensiveFunctions = ['getTestAccuracy']

print('new setups')

#experimentName = 'differentSparsities'
experimentName = 'gradientClipping'
setups = None
exec(f"import IHT_AGD.setups.setup_{experimentName}")
exec(f"setups = IHT_AGD.setups.setup_{experimentName}.setups")
print(setups)
#abort()

"""# Running the Experiment"""

datasetChoice = "IMAGENET"
print(datasetChoice)

""" MAIN CELL """
#setups = [setup_ihtAGD]#,setup_vanillaSGD]#,setup_ihtAGD]
#setups = [setup_pytorchSGD]
print(setups)


run = neptune.init_run(api_token=api_token, project=project)
runPipeline(setups,
            datasetChoice="IMAGENET",
            epochs=20,trials=1,
            functionsToHelpTrack=functionsToHelpTrack,
            variablesToTrack=variablesToTrack,
            expensiveVariables=expensiveVariables,
            expensiveFunctions=expensiveFunctions,


            device=device,
            run=run,
            test_loader=test_loader,
            train_loader=train_loader)
run.stop()



# Commented out IPython magic to ensure Python compatibility.
# %reload(runPipeline)

import importlib

importlib.reload(IHT_AGD.experimentScaffolding.experimentFuncs)

abort()

"""# -----------------------------------------------------------------------
# END OF THE BASELINE FRAMEWORK, NEXT SECTION DEDICATED TO EXTENSIONS

## Bias Left Untouched
"""

class untouchedIhtAGD(ihtAGD):
  def __init__(self,params,sparsity=0.9,kappa=5.0,beta=50.0):
    super().__init__(params)
    self.methodName = "untouched_iht_AGD"
    self.alpha = beta / kappa
    self.beta = beta
    self.kappa = kappa

  def sparsify(self):
    # TO-DO: remember to remove this zero, it is inconsequential, but still remove it in good practice
    concatWeights = torch.zeros((1)).to(device)
    for group in self.param_groups:
      for p in group['params']:

        #Skip Bias Layers
        if len(p.data.shape) < 2:
          continue

        flatWeights = torch.flatten(p.data)
        concatWeights = torch.cat((concatWeights,flatWeights),0)

    topK = int(len(concatWeights)*(1-self.sparsity))
    vals, bestI = torch.topk(torch.abs(concatWeights),topK,dim=0)
    cutoff = vals[-1]
    for group in self.param_groups:
      for p in group['params']:

        #Skip Bias Layers
        if len(p.data.shape) < 2:
          continue

        p.data[abs(p.data) <= cutoff] = 0.0

setup_untouched_ihtAGD = {
    "scheme":"untouchedIhtAGD",
    "lr":0.1,
    "sparsity":0.90,
    "kappa":10.0,
    "beta":100.0}
setups = [setup_untouched_ihtAGD, setup_ihtAGD]

run = neptune.init_run(api_token=api_token, project=project)
all_models,all_training_losses,all_testing_losses,all_accuracies = runMainExperiment(setups)
run.stop()

"""# Grid Search"""

from os import setgroups

def gridSearch(default,variables,values,metric,epochs=1):
  """ Desc: searches in a grid for the best combination of values of arbitrary dimension,
        we can check for more than 2 variables at a time, but this can be very costly

  default [dictionary]: a dictionary for all the default settings, this is also how one can set the type of algorithm
  variables [array[string]]: the settings to change
  values [2Darray]: what values to take on
  metric [string]: what metric to use for the best value
  """

  # We will not know how to traverse this list easily however
  # TO-DO: find a way to organize, or traverse this list
  setups = []

  # This list has every possible combination of the settings
  valuePermutations = list(itertools.product(*values))

  for permutation in valuePermutations:
    newSetup = default.copy()
    for idx,val in enumerate(permutation):

      # Adjusts the settings one-by-one
      newSetup[variables[idx]] = val

    setups.append(newSetup)

  print(setups)


  all_models,all_training_losses,all_testing_losses,all_accuracies = runMainExperiment(setups,epochs=epochs)

  # NEXT: Interchange with a different metric
  # TO-DO: try "highest loss" over entire dataset using model

  # Right now we use the accuracy in after the last epoch
  # BUG: is the last epoch at 0 or -1 I need to check
  min_accuracies = [accuracies[-1] for accuracies in all_accuracies]
  bestSetupIndex = min_accuracies.index(min(min_accuracies))



  return setups[bestSetupIndex]

default = {
    "scheme":"vanillaAGD",
    "lr":0.1,
    "sparsity":0.90,
    "kappa":15.0,
    "beta":10000.0}
# We set a big value to see if we overwrite it in the Grid Search

gridSearch(default,["kappa","beta"],[[2.0,10.0,100.0],[10.0,100.0,300.0]],"loss",5)

#This works! It recognizes it as a class name
type(eval("ihtAGD"))

"""# **Appendix**

# Saving and Loading Model

SOURCE: https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

def saveModel(model,pathdir):
  torch.save(model.state_dict(), pathdir)

def loadModel(pathdir,modeltype):
  match modeltype:
    case "basicNeuralNet": model = basicNeuralNet(784,10).to(device)
    case "convNet": model = convNet().to(device)

  model.load_state_dict(torch.load(pathdir))
  model.eval()
  return model

saveModel(all_models[0],"testModel")

tryModel = loadModel("testModel","convNet")

"""# Notes

Sparsify Interval
Base case
Fine-Tuning Phase (Freeze weights) , < Sparsify interval
Real-time visualization - add trainin loss per batch and test loss, and test accuracy
Weights and Biases


AC/DC proof 8.1.4,

Make proof on board work for large numbers, i.e.! T:(S* times Kappa^2 * some constant factor)
Want the damage to be 1 + epsilon

make sure you can collect useful information - e.g. things like sparsity

# Empirically Testing the Model
"""

def testModel(model):
  randomExampleInt = np.random.randint(1000)
  exampleX = dataset2.data[randomExampleInt].reshape(28, 28)
  plt.imshow(exampleX)
  print(exampleX.shape)
  exampleX = torch.reshape(exampleX, (1, 1,28,28))
  predicted = model(torch.tensor(exampleX,dtype=torch.float32).to(device))
  print(torch.argmax(predicted))

testModel(tryModel)

"""# TO DO

- check the sparsity of bias persists if we increase sparsity (95% sparsity and 99%)

- compare with and without first phase I added

- try to see if same spike appears with inserting SGD on decompression

- Visualization?
-  Save Model? - MAYBE USEFUL
-  Checkpoints? - YES DO THIS!
"""