import torch
from IHT_OPT.vanillaAGD import vanillaAGD
from IHT_OPT.ihtSGD import ihtSGD
import numpy as np

###############################################################################################################################################################
# ---------------------------------------------------- IHT-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class ihtAGD(vanillaAGD,ihtSGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"
    self.alpha = self.beta / self.kappa

    self.specificSteps = 0

    

  def step(self):
    self.specificSteps += 1

    self.compressOrDecompress()

  def decompressed(self):
    self.areWeCompressed = False
    print('decompressed')
    self.updateWeightsTwo()

  def warmup(self):
    self.areWeCompressed = False
    print('warmup')
    self.updateWeightsTwo()

  # I checked this, it seems to work
  def truncateAndFreeze(self):
    
    self.updateWeightsTwo()
    self.areWeCompressed = True

    # Truncate xt
    self.sparsify()

    self.copyXT()


    # Freeze xt
    self.freeze()

    pass

  ##############################################################################

  def updateWeightsTwo(self):

    print("AGD updateWeights")
    # Update z_t the according to the AGD equation in the note

    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]


        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        howFarAlong = ((self.iteration - self.warmupLength) % self.phaseLength) + 1

        if self.areWeCompressed:
          if self.iteration >= self.startFineTune:
            self.refreeze(iterate='zt')
          else:
            self.sparsify(iterate='zt')


        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    self.getNewGrad('zt')

    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad



    # We need to keep a separate storage of xt because we replace the actual network parameters
    self.copyXT()
    pass


  def compressedStep(self):
    self.areWeCompressed = True
    print('compressed step')
    self.updateWeightsTwo()
    self.refreeze()

  def clipGradients(self,clipAmt=0.0001):
    torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'],clip_value=clipAmt)
    pass

  def trackMatchingMasks(self):
    concatMatchMask = torch.zeros((1)).to(self.device)
    for p in self.paramsIter():
      state = self.state[p]

      matchingMask = ((torch.abs(p.data) > 0).type(torch.uint8) == (torch.abs(state['zt'])).type(torch.uint8) > 0 ).type(torch.float)
      
      concatMatchMask = torch.cat((concatMatchMask,matchingMask),0)

    self.run[f"trials/{self.methodName}/matchingMasks"].append(torch.mean(matchingMask))

  def saveOldIterates(self):
    for p in self.paramsIter():
      state = state = self.state[p]
      state['prev_xt'] = p.data.clone().detach()
      state['prev_zt'] = state['zt'].clone().detach()

  # Tracking how much the iterate moves between steps
  def trackIterateMovement(self):
    concat_xt_diff = torch.zeros((1)).to(self.device)
    concat_zt_diff = torch.zeros((1)).to(self.device)

    for p in self.paramsIter():
      state = self.state[p]
      xt_diff = p.data.clone().detach() - state['prev_xt']
      zt_diff = state['zt'] - state['prev_zt']

      concat_xt_diff =  torch.cat((concat_xt_diff,torch.flatten(xt_diff)),0)
      concat_zt_diff =  torch.cat((concat_zt_diff,torch.flatten(zt_diff)),0)
    
    avg_xt_move = torch.sum(torch.abs(concat_xt_diff)) / len(concat_xt_diff)
    avg_zt_move = torch.sum(torch.abs(concat_zt_diff)) / len(concat_zt_diff)

    self.run[f"trials/{self.methodName}/move_xt"].append(avg_xt_move)
    self.run[f"trials/{self.methodName}/move_zt"].append(avg_zt_move)

  # Tracking how much the mask changes
  def trackChangeMask(self):
    concat_xt_diffmask = torch.zeros((1)).to(self.device)

    for p in self.paramsIter():
      state = self.state[p]

      oldMask = (torch.abs(state['prev_xt']) > 0).type(torch.float)
      newMask = (torch.abs(p.data) > 0).type(torch.float)

      diffMask = (oldMask != newMask).type(torch.float)

      concat_xt_diffmask=  torch.cat((concat_xt_diffmask,torch.flatten(diffMask)),0)
    
    avg_xt_moveMask = torch.sum(concat_xt_diffmask) / len(concat_xt_diffmask)

    self.run[f"trials/{self.methodName}/move_xt_mask"].append(avg_xt_moveMask)

  # To make sure we're copying xt correctly
  def checkXTCopy(self):
    for p in self.paramsIter():
      state = self.state[p]

      if (state['xt'] != p.data).any():
        abort()

  def modelSwitchIterate(self,iterate):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        p.data = state[iterate].clone().detach()


  def weightedSparsify(self,iterate):
    weightedWeights = torch.zeros((1)).to(self.device)
    with torch.no_grad():
      for p in self.paramsIter():
        if iterate == None:
          layer = p.data
        else:
          state = self.state[p]
          layer = state[iterate]

          weightedLayer = torch.flatten(torch.abs(layer) * torch.log(layer.size()))
          weightedWeights = torch.cat((weightedWeights,weightedLayer),0)
      
      topK = int(len(weightedWeights)*(1-self.sparsity))

      # All the top-k values are sorted in order, we take the last one as the cutoff
      vals, bestI = torch.topk(torch.abs(weightedWeights),topK,dim=0)
      weightedCutoff = vals[-1]

      for p in self.paramsIter():
        state = self.state[p]
        if iterate == None:
          p.data[torch.abs(p) * torch.log(p.size()) <= weightedCutoff] = 0.0
        else:
          (state[iterate])[torch.abs(state[iterate]) * torch.log(state[iterate].size()) <= weightedCutoff] = 0.0

  ##########################################