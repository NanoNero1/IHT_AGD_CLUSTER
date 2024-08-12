""" Main module for testing optimizers """
# Load libraries and pick the CUDA device if available
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
##from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Custom libraries
from AdaptiveLinearCoupling import *
from AdaACSA import *
from AdaAGDplus import *
from AdaJRGS import *

############################################Dimitri's IMPORTS
#Dimitri's Optimizers
from IHT_OPT.baseOptimizer import myOptimizer
from IHT_OPT.vanillaSGD import vanillaSGD
from IHT_OPT.vanillaAGD import vanillaAGD
from IHT_OPT.ihtAGD import ihtAGD
from IHT_OPT.ihtSGD import ihtSGD
from IHT_OPT.clipGradientIHTAGD import clipGradientIHTAGD

# Imagenet Model
from torchvision.models import resnet18
from torchvision.models import resnet50

#Neptune
withNeptune = True
if withNeptune:
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



################################################################


from loader import *
from models import MODELS_MAP
from misc import *
from misc import progress_bar

def flat_weight_dump(model):
    """ Returns a 1-d tensor containing all the network weights """
    is_empty = True
    for _, param in model.named_parameters():
        if param.requires_grad:
            if is_empty:
                flat_tensor = param.data.flatten()
                is_empty = False
            else:
                flat_tensor = torch.cat([flat_tensor, param.data.flatten()])
    return flat_tensor


# def tb_dump(epoch, net, writer):
#     """ Routine for dumping info on tensor board at the end of an epoch """
#     print('=> eval on test data')
#     (test_loss, test_acc, _) = test(testloader, net, device)
#     writer.add_scalar('Loss/test', test_loss, epoch)
#     writer.add_scalar('Accuracy/test', test_acc, epoch)

#     print('=> eval on train data')
#     (train_loss, train_acc, _) = test(trainloader, net, device)
#     writer.add_scalar('Loss/train', train_loss, epoch)
#     writer.add_scalar('Accuracy/train', train_acc, epoch)
#     print('epoch %d done\n' % (epoch))

def print_neptune_params(run=None,optimizer=None):
    params_to_neptune = {key: config[key] for key in ['beta','kappa','sparsity','lr','epochs','dataset']}
    run[f"trials/{optimizer.methodName}/{"params"}"].append(params_to_neptune)



def test(testloader, net, device):
    """ Routine for evaluating test error """
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):


            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            #huh = criterion(outputs, labels)
            #print(huh)
            #loss += criterion(outputs, labels) * labels.size(0)

            # track total loss until now, not average loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #_, predicted = torch.max(outputs.data, 1)
            #correct = (predicted == labels.to(device)).sum().item()
            #train_acc = 100 * correct / labels.size(0)
            #run[f"trials/{optimizer.methodName}/{"errorAcc"}"].append(train_acc)

            #progress_bar(
            #    batch_idx, len(testloader), 'Loss: %.5f | Acc: %.3f%% (%d/%d)'
            #    % (loss/total, 100.*correct/total, correct, total))
    # Return the average loss (i.e. total loss averaged by number of samples)
    #return (loss.item() / total, 100.0*correct/total, total)
    return (0, 100.0*correct/total, total)

# Trains the network
def train_net(epochs, path_name, net, optimizer,run=None):
    print_neptune_params(run=run,optimizer=optimizer)
    """ Train the network """
    print(optimizer)
    #writer = SummaryWriter(path_name)
    n_iter = 0

    # Dump info on the network before running any training step
    #tb_dump(0, net, writer)
    #last_train_acc = 0.0
    for epoch in range(epochs):  # Loop over the dataset multiple times

        epochStepCount = 0
        for i, data in enumerate(trainloader, 0):

            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Save weights before gradient step, in order to measure movement
            if config_dump_movement and (i % config_batch_statistics_freq == 0):
                old_weights = flat_weight_dump(net)

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            #run[f"trials/{optimizer.trialNumber}/{optimizer.setupID}/loss"].append(loss)
            loss.backward()

            # For AGD
            if config_optimizer == -2 or config_optimizer == -3:
                optimizer.currentDataBatch = (inputs.clone(),labels.clone())

            optimizer.step()

            # Compute statistics
            train_loss = loss.item()


            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(device)).sum().item()
            train_acc = 100 * correct / labels.size(0)

            if withNeptune:
                run[f"trials/{optimizer.methodName}/{"dummyLoss"}"].append(train_loss)
                run[f"trials/{optimizer.methodName}/{"dummyAcc"}"].append(train_acc)

            #last_train_acc = train_acc

            # if i == 0:
            #     print('newline')
            #     progress_bar(
            #         i, len(trainloader), 'Loss: %.5f | Acc: %.3f%%'
            #         % (train_loss, 100.*correct/labels.size(0)))
            #     dummy=0

            epochStepCount = i


            
            # # Print statistics every couple of mini-batches
            # if i % config_batch_statistics_freq == 0:
            #     writer.add_scalar('Loss/batch', train_loss, n_iter)
            #     writer.add_scalar('Accuracy/batch', train_acc, n_iter)

            #     if config_dump_movement:
            #         new_weights = flat_weight_dump(net)
            #         movement = torch.norm(
            #             torch.add(old_weights, new_weights, alpha=-1))
            #         writer.add_scalar('Movement', movement.item(), n_iter)

            #     writer.flush()
            #     n_iter = n_iter + 1

            if i  == 150:
                final_loss,final_accuracy,final_total = test(testloader, net, device)
                print(final_accuracy)
                exit()
            #     #abort()
        
        if epoch == 0:
            run[f"trials/{optimizer.methodName}/{"epochSize"}"].append(epochStepCount)

        if epoch == 5:
            run[f"trials/{optimizer.methodName}/{"lr"}"].append(optimizer.param_groups[0]['lr'])
            for g in optimizer.param_groups:
                 g['lr'] *= 0.100
            run[f"trials/{optimizer.methodName}/{"lr"}"].append(optimizer.param_groups[0]['lr'])

        




    

        #tb_dump(epoch+1, net, writer)
    final_loss,final_accuracy,final_total = test(testloader, net, device)
    #testAccuracy = float()
    run[f"trials/{optimizer.methodName}/{"testAccuracy"}"].append(final_accuracy)
    print('Finished Training')

    #writer.close()


# ################
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
    '--config', default='config.json', type=str, help='config file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)
config_experiment_number = config['experiment_number']
config_dataset = config['dataset']
config_architecture = config['architecture']
config_batch_size = config['batch_size']
config_optimizer = config['optimizer']
config_lr = config['lr']
config_momentum = config['momentum']
config_radius = config['radius']
config_epochs = config['epochs']
config_tb_path = config['tb_path']
config_batch_statistics_freq = config['batch_statistics_freq']
config_dump_movement = bool(config['dump_movement'] == 1)
config_projected = bool(config['projected'] == 1)
config_weight_decay = config['weight_decay']
config_radius = config['radius']
config_random_seed = config['random_seed']
config_gamma0 = config['gamma0']
config_initial_accumulator_value = config['initial_accumulator_value']
config_beta = config['beta']
config_eps = config['eps']

config_kappa = config['kappa']
config_sparsity = config['sparsity']


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Set random seed
torch.manual_seed(config_random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config_random_seed)

# Load data
if config_dataset == 'MNIST':
    trainloader, testloader = mnist_loader(batch_size=config_batch_size)
elif config_dataset == 'CIFAR':
    trainloader, testloader = cifar_loader(batch_size=config_batch_size)
elif config_dataset == 'IMAGENET':
    trainloader, testloader = imagenet_loader(batch_size=config_batch_size)
elif config_dataset == 'CIFAR100':
    trainloader, testloader = cifar100_loader(batch_size=config_batch_size)


if config_architecture == "ImageNetRN":
    model = resnet18().to(device)
    #model = resnet50().to(device)
else:
    model = MODELS_MAP[config_architecture]()
net = model.to(device)
criterion = nn.CrossEntropyLoss()
if config_optimizer == -5:
    optimizer = clipGradientIHTAGD(
      net.parameters(), beta=config_beta,kappa=config_kappa,sparsity=config_sparsity,
      momentum=config_momentum, weight_decay=config_weight_decay,device=device,model=net)
elif config_optimizer == -4:
    optimizer = ihtSGD(
      net.parameters(), beta=config_beta,kappa=config_kappa,sparsity=config_sparsity,
      momentum=config_momentum, weight_decay=config_weight_decay,device=device,model=net)
elif config_optimizer == -3:
    optimizer = ihtAGD(
      net.parameters(), beta=config_beta,kappa=config_kappa,sparsity=config_sparsity,
      momentum=config_momentum, weight_decay=config_weight_decay,device=device,model=net)
elif config_optimizer == -2:
    optimizer = vanillaAGD(
      net.parameters(), beta=300.0,kappa=10.0,
      momentum=config_momentum, weight_decay=config_weight_decay,device=device,model=net)
elif config_optimizer == -1:
    optimizer = vanillaSGD(
      net.parameters(), beta=config_beta,
      momentum=config_momentum, weight_decay=config_weight_decay)
elif config_optimizer == 0:
    optimizer = optim.SGD(
      net.parameters(), lr=config_lr,
      momentum=config_momentum, weight_decay=config_weight_decay)
    setattr(optimizer, 'methodName', 'nativeSGD')
elif config_optimizer == 1:
    optimizer = optim.Adagrad(
      net.parameters(), lr=config_lr, weight_decay=config_weight_decay)
elif config_optimizer == 2:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=0, weight_decay=config_weight_decay)
elif config_optimizer == 3:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=1, weight_decay=config_weight_decay)
elif config_optimizer == 4:
    optimizer = optim.RMSprop(net.parameters(), lr=config_lr)
elif config_optimizer == 5:
    optimizer = AdaptiveLinearCoupling(
        net.parameters(), lr=config_lr,
        weight_decay=config_weight_decay)
elif config_optimizer == 6:
    #optimizer = AdaACSA(
    #    net.parameters(), lr=config_lr, radius=1, projected=config_projected)
    optimizer = AdaACSA(
        net.parameters(), lr=config_lr, radius=config_radius,
        weight_decay=config_weight_decay, projected=config_projected,
        gamma0=config_gamma0, beta=config_beta,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 7:
    optimizer = AdaAGDplus(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 8:
    optimizer = AdaJRGS(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)


# Writer path for display on TensorBoard
if not os.path.exists(config_tb_path):
    os.makedirs(config_tb_path)
path_name = config_tb_path + \
    str(config_experiment_number) + "_" + str(optimizer)

# Initialize weights
net.apply(weights_init_uniform_rule)

if withNeptune:
    run = neptune.init_run(api_token=api_token, project=project)
    setattr(optimizer, 'run', run)
train_net(
    epochs=config_epochs, path_name=path_name, net=net, optimizer=optimizer,run = run if withNeptune else None)

if withNeptune:
    run.stop()

# Dump some info on the range of parameters after training is finished
for param in net.parameters():
    print(str(torch.min(param.data).item()) + " " + str(torch.max(param.data).item()))

exit()
