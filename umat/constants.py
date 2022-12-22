import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FerriticPhaseConstants:

    # Reference Temperature of the body and temperature of transformation
    Theta0: float  = 300.e0   # (K)
    Theta_T: float = 633.e0   # (K)

    # mass density in reference configuration           
    Rho_0: float =  78.e+2      # 78.d-16   #  (Kg/um^3)

    # slip resistance parameters
    omega_F: float = 7.e0
    mu_F: float    = 55.e0           # GPa

    # Hardening parameteres for ferrite
    s0_F: float   = 1.58e-1     # GPa     #1.58d-4   (N/um^2)
    sInf_F: float = 4.12e-1     # GPa     #4.12d-4   (N/um^2)
    k0_F: float   = 1.9e0       # GPa     #1.9d-3    (N/um^2)
    uExp_F: float = 2.8e0
    q0_F: float   = 1.e0
    c0_F: float   = 5.e-1               

    # Plastic Kinetic Law ferrite
    pExp_F: float      = 2.e-2
    GammaDot0_F: float = 1.e-3       
    phi_F: float       = 4.27e-9   # GJ/(kg.K)  #4.27350427d+6   (N.um/kg.K)       

    N: int = 24 # number of slip systems

    # Thermal driving force is constant for an insothermal process
    g_th: float = 0.0099918 # Rho_0 * Theta0 * phi_F (GPa)

    
consts = FerriticPhaseConstants()
