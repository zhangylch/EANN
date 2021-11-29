import numpy as np
import torch

# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same as the atomtype in the file input_density
atomtype=['O', 'H', 'C', 'N']
mass=[15.999,1.008,12.001,14.007]
mass=torch.from_numpy(np.array(mass)).to(device)

#load the serilizable model
pes=torch.jit.load("EANN_PES_FLOAT.pt")
pes.eval()
pes.to(device).to(torch.double)

#pes=torch.jit.freeze(pes,optimize_numerics=True)
# set the eval mode
# save the lattice parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([0,0,0],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
npoint1=0
rmse=torch.zeros(1,dtype=torch.double,device=device)
cart=[]
#read the data
with open("configuration",'r') as f1:
    while True:
        string=f1.readline() # head
        if not string or npoint==100000: break
		
        string=f1.readline() # lattice
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
		
        string=f1.readline() # pbc
		
        species=[]
        cart.append([])
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart[npoint].append(tmp1[0:3]) # cart coordinates
            species.append(atomtype.index(tmp[0])) # atom-label
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  
        tmass=mass.index_select(0,species)
        npoint+=1
		
cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double) 
for i in range(100000):
    var=pes(period_table,cart[i],tcell,species,tmass)
    print(var[0].item())
