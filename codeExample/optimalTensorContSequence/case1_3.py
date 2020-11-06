import torch


dtype = torch.float
device = torch.device("cpu")


#Define the size
A1_i, A1_j, A1_k = 512, 16, 16
A2_i, A2_j, A2_k = 16, 32, 16

B1_i, B1_j, B1_k = 512, 16, 16
B2_i, B2_j, B2_k = 16, 32, 16

TA1 = torch.randn(A1_k, A1_i, A1_j)
TA2 = torch.randn(A2_k, A2_i, A2_j)

TB1 = torch.randn(B1_k, B1_i, B1_j)
TB2 = torch.randn(B2_k, B2_i, B2_j)


print("Matrix Contraction of A1 and A2 = A1A2 ")
print("Size of A1")
print(TA1.size())
print("Size of A2")
print(TA2.size())


#Calculate A1A2

print("Calculating A1*A2 contradiction")
TA1A2 = torch.einsum('kij,lja -> kila', [TA1,TA2])

print("Size of A1A2")
print(TA1A2.size())

#Calculate B1B2

print("Calculating B1B2 contradiction")
TB1B2 = torch.einsum('abc,icj  -> abij' , [TB1,TB2])

print("Size of B1B2")
print(TB1B2.size())

#Calculate A1A2*B1B2

print("Calculating (A1A2)*(B1B2)")
TA1A2B1B2 = torch.einsum('abcd,aicj  -> bdij' , [TA1A2, TB1B2]) 

print("Size of A1A2B1B2")
print(TA1A2B1B2.size())
















