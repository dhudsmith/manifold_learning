import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from time import time

##
## Setup

# use nvidia gpu if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# only float32 tested
dtype = torch.float32

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
        self.eye = torch.eye(N_proj, device=device, dtype=dtype)
        
        # initialize the encoder and decoder architectures
        self.decoder = nn.Sequential(OrderedDict([('matmul', nn.Linear(self.N_proj, self.N, bias=False)),
                                                  ('normalize', L2Norm())
                                                  ]))

    # return the current cost matrix
    def forward(self):
        M = self.decoder(self.eye)
        return M @ self.H @ M.t(), M @ M.t()
    
class Cost():
    """Cost class -- used to calculate goodness of projection map for optimization purposes
    
    Args:
        N_proj - number of basis states in the projected space
        alpha (float) - the weight of the orthogonality cost relative to the energy cost (units of energy)
    """
    
    def __init__(self, N_proj, alpha):
        super(Cost, self).__init__()
        
        self.N_proj = N_proj
        self.alpha = alpha
        self.eye = torch.eye(N_proj, device=device, dtype=dtype)
        self.sorter = torch.tensor([[1/((i+1)*(j+1))**2 for i in range(self.N_proj)] for j in range(self.N_proj)], device=device, dtype=dtype)

        
    def __call__(self, Hproj, Iproj):
        cost_matrix = (Hproj*self.sorter)**2 + self.alpha * (Iproj - self.eye)**2
        return cost_matrix.sum()/self.N_proj**2
    
def early_stop(errs, rel_tol, patience=2):
    stop=True
    for i in range(patience):
        rel_change = np.abs((errs[-(i+1)] - errs[-(i+2)]) / errs[-(i+1)])
        stop = stop and rel_change < rel_tol
    return stop

def optimize(model, optimizer, criterion, n_iter, n_save, show_progress=True, stop_early=True, rel_tol=0.01, patience=2):
    model.decoder.matmul.reset_parameters()
    running_loss= 0
    start_time = time()
    
    # store progress every n_save iters
    j=0
    its = []
    errs = []
    Hps = []
    Ips = []
    ts = []
    
    # iterate
    for i in range(n_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        Hp, Ip = model()
        loss = criterion(Hp, Ip)
        loss.backward()
        optimizer.step()

        # print /save statistics
        running_loss += loss.item()
        if i % n_save == 0:
            # save progress
            its.append(i)
            errs.append(running_loss)
            Hps.append(Hp)
            Ips.append(Ip)
            ts.append(time() - start_time)
            
            # print progress 
            if show_progress:
                change = errs[j]-errs[j-1] if j>0 else 0
                print('[%d] loss: %.8f. diff: %.8f. time: %.4f' % (i + 1, running_loss / 10, change, ts[-1]))
            
            if j>=patience and stop_early and early_stop(errs, rel_tol, patience):
                print("Early stopping criteria met.")
                break
            
            running_loss = 0.0
            j+=1
            

    print('Finished Training')
    return its, errs, Hps, Ips, ts