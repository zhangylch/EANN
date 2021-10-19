Embedded Atom Neural Network 
=================================================
**1. Introduction:**
___________________________
    Embedded Atomic Neural Network (EANN) is a physically-inspired neural network framework. The EANN package is implemented using the PyTorch framework. All optimizable parameters can be optimized by making use of the Autograd embedded in PyTorch on multiple GPUs or CPUs. In addition, the EANN package has been interfaced with LAMMPS and is used for high efficiency MD simulation in both the GPU and CPU with high parallel efficiency. Details about the setup of training and inference can be found in the "program/manual" folder.

**2. Requirements:**
___________________________________
* PyTorch 1.9.0
* opt_einsum 3.2.0

**References:**
1. The EANN model: Yaolong Zhang, Ce Hu and Bin Jiang *J. Phys. Chem. Lett.* 10, 4962-4967 (2019).
2. The EANN model for dipole/transition dipole/polarizability: Yaolong Zhang  Sheng Ye, Jinxiao Zhang, Jun Jiang and Bin Jiang *J. Phys. Chem. B*  124, 7284–7290 (2020).
