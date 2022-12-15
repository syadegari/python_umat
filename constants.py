import numpy as np


fl = np.float64

# Reference Temperature of the body and temperature of transformation
Theta0  = 300.e0   # (K)
Theta_T = 633.e0   # (K)

# mass density in reference configuration           
Rho_0 =  78.e+2      # 78.d-16   #  (Kg/um^3)

# slip resistance parameters
W_F=7.d0
Mu_F=55.d0           # GPa

# Hardening parameteres for ferrite
s0_F   = 1.58e-1     # GPa     #1.58d-4   (N/um^2)
sInf_F = 4.12e-1     # GPa     #4.12d-4   (N/um^2)
k0_F   = 1.9e0       # GPa     #1.9d-3    (N/um^2)
w0_F   = 2.8e0
q0_F   = 1.e0
c0_F   = 5.e-1               

# Plastic Kinetic Law ferrite
mExp_F      = 2.e-2
GammaDot0_F = 1.e-3       
Phi_F       = 4.27e-9   # GJ/(kg.K)  #4.27350427d+6   (N.um/kg.K)       
