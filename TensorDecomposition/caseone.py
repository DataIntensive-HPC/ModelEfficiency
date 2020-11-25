import torch
import sys

dtype = torch.float
device = torch.device("cpu")


#Define the sizes

A_i, A_j, A_k = 512, 16, 16

A1_i, A1_r1   =  512, 2
A2_r1, A2_r2, A2_j = 2, 2, 16
A3_r2, A3_k = 2, 16

B_i, B_j, B_k = 16, 32 , 16
C_i, C_j, C_k = 512, 16, 16
D_i, D_j, D_k = 16, 32, 16

#Create Tensors

TA  =  torch.randn(A_k, A_i, A_j)

TA1 =  torch.randn(A1_i, A1_r1 )
TA2 =  torch.randn(A2_j, A2_r1, A2_r2)
TA3 =  torch.randn(A3_r2, A3_k )


TB  =  torch.randn(B_k, B_i, B_j)
TC  =  torch.randn(C_k, C_i, C_j)
TD  =  torch.randn(D_k, D_i, D_j)


print("Matrix Contraction of A1 and A2")

#Calculate A1A2

TA1TA2 = torch.einsum('ij, kjr -> ikr' , [TA1, TA2])

print("Size of A1A2")
print(TA1TA2.size())

#Calculate A1A2*C

TA1TA2TC = torch.einsum( 'kij, kab -> ijab ' , [TA1TA2 , TC])


print("Size of A1A2C")
print(TA1TA2TC.size())

#Calculate A3B

TA3TB = torch.einsum('ij,kjl -> ikl' , [TA3, TB])

print("Size of TA3TB")
print(TA3TB.size())

#Calculate A3B*D

TA3TBTD = torch.einsum('kij, kab -> ijab' , [TA3TB, TD])

print("Size of TA3TBTD")
print(TA3TBTD.size())

#Calculate  A1A2C*A3BD

TA1TA2TCTA3TBTD = torch.einsum('abcd, cdij -> abij' , [TA1TA2TC , TA3TBTD])

print("Size of TA3TBTD")
print(TA3TBTD.size())