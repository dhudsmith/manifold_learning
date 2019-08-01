import numpy as np
import math
import torch
from scipy.stats import ortho_group
from scipy.sparse import spdiags

##
## Setup

# use nvidia gpu if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# only float32 tested
dtype = torch.float32
    
##
## Hamiltonian functions

def H_ring_2particles(basis_size, g): # g for interaction strength, it is better if the basis_size is an odd number 
    H_data = g*np.ones((basis_size,basis_size)) # potential energy
    H_diag=np.zeros(basis_size)
    
    for i in range(basis_size):
        H_diag[i]= (2*math.pi*int(((i+1)/2)//1))**2 # kinetic energy E=k^2; k=2 Pi n, n=0,\pm1 , \pm 2,...
    
    H_phys =  torch.tensor(H_data+np.diag(H_diag),device=device, dtype=dtype)
    
    return H_phys

def H_random(basis_size):
    eigs,_ = (torch.rand(basis_size, device=device, dtype=dtype)*basis_size).sort()
    Hgen_proj = torch.diag(eigs)
    Wgen = torch.tensor(ortho_group.rvs(basis_size), device=device, dtype=dtype)
    
    Hgen = Wgen @ Hgen_proj @ Wgen.t()
    
    return Hgen, Wgen, eigs

def H_random_banded(basis_size, n_diags):
    '''
    Generate a random band-diagonal hamiltonian with `n_diags` non-zero diagonals above the center diagonal.
    '''
    diags = []
    for i in range(n_diags):
        diag = np.random.rand(basis_size-i)*2*basis_size/(i+1)**2
        diag = np.sort(diag).tolist()
        diag = [0] * (basis_size - len(diag)) + diag
        diags.append(diag)
        
    H = spdiags(diags, diags=[i for i in range(n_diags)], m=basis_size, n=basis_size)
    
    H = np.sqrt(H @ H.transpose())
    
    return torch.tensor(H.toarray(), dtype=dtype, device=device)

##
## Run tests with `python hamiltonian.py`
if __name__=='__main__':
    ##
    ## Test the hamiltonian functions
    print("Testing Hamiltonians")
    
    # H_ring_2particles
    basis_size=7
    g=2
    print("\nTesting `H_ring_2particles` with parameters: basis_size=%i, g=%0.2f" % (basis_size, g))
    print(H_ring_2particles(basis_size,g))
    
    # H_random
    basis_size=7
    print("\nTesting `H_random` with parameters: basis_size=%i" % basis_size)
    print(H_random(basis_size))
    
    # H_random_banded
    basis_size=7
    n_diags = 4
    print("\nTesting `H_random` with parameters: basis_size=%i, n_diags=%i" % (basis_size, n_diags))
    print(H_random_banded(basis_size, n_diags))