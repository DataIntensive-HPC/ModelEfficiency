import torch
import sys

dtype = torch.float
device = torch.device("cpu")



#Define the sizes


S, D, d = 4, 512, 64
r= 2



#Create Tensors

TX = torch.randn(S,D)

TWq1 = torch.rand(D,r)
TWq2 = torch.rand(r,d)


print("Matrix Contraction of TX and TWq1")

TXWq1= torch.einsum('ij,jm->im', [TX,TWq1])

print("Size of TXWq1")

