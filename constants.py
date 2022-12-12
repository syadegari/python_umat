import numpy as np


fl = np.float64


# Stress unit is GPa
# other quantities unit is in accordance with the stress unit
# Thermal exapansion coefficients for Austenite, Martensite and Ferrite
alpha_A = 2.1d-5
alpha_M = 2.1d-5
alpha_F = 1.7d-5
# Reference Temperature of the body and temperature of transformation
Theta0  = 300.d0   # (K)
Theta_T = 633.d0   # (K)
# mass density in reference configuration
Rho_0 =  78.d+2        #78.d-16   #  (Kg/um^3)
# specific heat for Austenite, Martensite and Ferrite
SHeatA=  450.d-9  # GJ/(kg.K)      # 450.d+6     N.um/(kg.K)
SHeatM=  450.d-9  # GJ/(kg.K)      # 450.d+6     N.um/(kg.K)
SHeatF=  450.d-9  # GJ/(kg.K)      # 450.d+6     N.um/(kg.K)
# slip resistance parameters, surface energy, latent heat
W_A=10.d0
W_F=7.d0
SurfEn= 4.d-3     # GPa     #4.d-6     [Khi/L0]=(N/um^2)
Lambda= -50.5d-6  # GJ/kg   #-50.5d+9           (N.um/kg)
Mu_F=55.d0        # GPa
Mu_A=67.5d0       # GPa
Mu_M=98.4d0       # GPa
# heat conduction
HCond=6.d+1     # (N/K.s)
# debug
#       logical debugoption=TRUE
# temperature perturbation parameter
      dtheta = 1.d-8
#
# Hardening parameteres for asutenite
s0_A   = 1.88d-1     # GPa     #1.88d-4   (N/um^2)
sInf_A = 5.79d-1     # GPa     #5.79d-4   (N/um^2)
k0_A   = 3.0d0       # GPa     #3.0d-3    (N/um^2)
w0_A   = 2.8d0
q0_A   = 1.d0
c0_A   = 5.d-1
#
s0_F   = 1.58d-1     # GPa     #1.58d-4   (N/um^2)
sInf_F = 4.12d-1     # GPa     #4.12d-4   (N/um^2)
k0_F   = 1.9d0       # GPa     #1.9d-3    (N/um^2)
w0_F   = 2.8d0
q0_F   = 1.d0
c0_F   = 5.d-1
# Transformation Kinetic law austenite
Nu_A      = 1.7d-1
XiMax_A   = 3.d-3     # (s^-1)
TDFCrit_A = 227.d-3   # GPa     #227.d-6   (N/um^2)
# Plastic Kinetic Law austenite
mExp_A      = 2.d-2
GammaDot0_A = 1.d-3
Phi_A       = 5.13d-9   # GJ/(kg.K)  #5.13d+6  (N.um/kg.K)
# Plastic Kinetic Law ferrite
mExp_F      = 2.d-2
GammaDot0_F = 1.d-3
Phi_F       = 4.27d-9   # GJ/(kg.K)  #4.27350427d+6   (N.um/kg.K)
