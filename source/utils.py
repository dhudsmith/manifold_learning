import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from time import time

##
## Setup

class L2Norm(nn.Module):
    """L2Norm model class -- a layer class for normalizing a matrix of vectors
    
    Args:
        dim (int) - which dimension to normalize along. 0 for column vectors, 1 for row vectors
    """
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        
        self.dim=dim
        
    def forward(self,input):
        return nn.functional.normalize(input, p=2, dim=self.dim)
    
class Hamiltonian(nn.Module):
    """Hamiltonian -- a class used to help learn physically relevant sub-manifold
    
    Args:
        H - the hamiltonian matrix in the original basis
        N (int) - the size of the original basis
        N_proj (int) - the size of the projected basis
    """
    
    def __init__(self, H, N, N_proj):

        super(Hamiltonian, self).__init__()
        
        # initialize the class variables
        self.H = H
        self.N = N
        self.N_proj = N_proj
        self.register_buffer('eye', torch.eye(N_proj))
        
        # initialize the encoder and decoder architectures
        self.decoder = nn.Sequential(OrderedDict([('matmul', nn.Linear(self.N_proj, self.N, bias=False)),
                                                  ('normalize', L2Norm())
                                                  ]))
        
    # return the current cost matrix
    def forward(self):
        M = self.decoder(self.eye)
        return M @ self.H @ M.t(), M @ M.t()
    
class Cost(nn.Module):
    """Cost class -- used to calculate goodness of projection map for optimization purposes
    
    Args:
        N_proj - number of basis states in the projected space
        alpha (float) - the weight of the orthogonality cost relative to the energy cost (units of energy)
    """
    
    def __init__(self, N_proj, alpha):
        super(Cost, self).__init__()
        
        self.N_proj = N_proj
        self.alpha = alpha
        self.register_buffer('eye', torch.eye(N_proj))
        self.register_buffer('sorter', torch.tensor([[1/((i+1)*(j+1))**2 for i in range(self.N_proj)] for j in range(self.N_proj)]))
        
    def __call__(self, Hproj, Iproj):
        cost_H = ((Hproj*self.sorter)**2).sum()/self.N_proj**2
        cost_I = (self.alpha * (Iproj - self.eye)**2 * torch.sqrt(self.sorter)).sum()/self.N_proj**2
        return cost_H + cost_I, cost_H, cost_I
    
def early_stop(errs, rel_tol, patience=2):
    stop=True
    for i in range(patience):
        rel_change = np.abs((errs[-(i+1)] - errs[-(i+2)]) / errs[-(i+1)])
        stop = stop and rel_change < rel_tol
    return stop

def optimize(model, optimizer, criterion, n_iter, n_save, log_func, log_wait=200, show_progress=True, stop_early=True, rel_tol=0.01, patience=2):
    model.decoder.matmul.reset_parameters()
    running_loss= 0
    running_loss_h= 0
    running_loss_i= 0
    start_time = time()
    
    # store progress every n_save iters
    j=0
    its = []
    errs = []
    errs_h = []
    errs_i = []
    Hps = []
    Ips = []
    ts = []
    
    # iterate
    for i in range(n_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        Hp, Ip = model()
        loss, loss_h, loss_i = criterion(Hp, Ip)
        loss.backward()
        optimizer.step()

        # print /save statistics
        running_loss += loss.item()
        running_loss_h += loss_h.item()
        running_loss_i += loss_i.item()
        
        # show progress and log ever n_save steps
        # todo: separate logging and progress
        if i % n_save == n_save-1:
            # save progress
            its.append(i)
            errs.append(running_loss)
            errs_h.append(running_loss_h)
            errs_i.append(running_loss_i)
            Hps.append(Hp)
            Ips.append(Ip)
            ts.append(time() - start_time)
            
            loss_ = running_loss / n_save
            loss_h = running_loss_h / n_save
            loss_i = running_loss_i / n_save
            
            # print progress 
            if show_progress:
                change = errs[j]-errs[j-1] if j>0 else 0
                print('[%d] loss: %.8f (%.6f/%.6f). diff: %.8f. time: %.4f' % (i + 1, 
                                                                                 loss_, 
                                                                                 loss_h, 
                                                                                 loss_i, 
                                                                                 change, ts[-1]))
            # log stuff (to wandb)
            if i>=log_wait:
                log_func(loss_, loss_h, loss_i, i)
                
            if j>=patience and stop_early and early_stop(errs, rel_tol, patience):
                print("Early stopping criteria met.")
                break
            
            running_loss = 0.0
            running_loss_h = 0.0
            running_loss_i = 0.0
            j+=1
            

    print('Finished Training')
    return its, errs, Hps, Ips, ts

def plot_rel_err(eigs, eigs_approx):
    plt.plot((eigs_approx - eigs)/eigs, 'o-')
    plt.hlines(0, 0, len(eigs), linestyles='dashed')
    plt.xlabel = 'excitation number'
    plt.ylabel = 'error'