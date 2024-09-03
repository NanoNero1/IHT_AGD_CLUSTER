import torch
import numpy as np
from IHT_OPT.baseOptimizer import myOptimizer


###############################################################################################################################################################
# ---------------------------------------------------- VANILLA-SGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class vanillaSGD(myOptimizer):
  def __init__(self,params,**kwargs):
    print(kwargs)
    super().__init__(params,**kwargs)

    # Internal States
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0

    # Internal Variables
    self.methodName = "vanilla_SGD"

  @torch.no_grad()
  def step(self):
    self.updateWeights()
    self.iteration +=1
    return None
  
  def copyGradient(self):
    abort()

  # Regular Gradient Descent
  def updateWeights(self,**kwargs):
    for p in self.paramsIter():
        state = self.state[p]
        p.add_(  (-1.0 / self.beta) * p.grad - state['xt_gradient'] * self.momentum / self.beta )