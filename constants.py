import numpy as np


fl = np.float64


! Stress unit is GPa 
! other quantities unit is in accordance with the stress unit
! Thermal exapansion coefficients for Austenite, Martensite and Ferrite
double precision, parameter::alpha_A = 2.1d-5
double precision, parameter::alpha_M = 2.1d-5
double precision, parameter::alpha_F = 1.7d-5
! Reference Temperature of the body and temperature of transformation
double precision, parameter::Theta0  = 300.d0   ! (K)
double precision, parameter::Theta_T = 633.d0   ! (K)
! mass density in reference configuration           
double precision, parameter::Rho_0 =  78.d+2        !78.d-16   !  (Kg/um^3)
! specific heat for Austenite, Martensite and Ferrite
double precision, parameter::SHeatA=  450.d-9  ! GJ/(kg.K)      ! 450.d+6     N.um/(kg.K)
double precision, parameter::SHeatM=  450.d-9  ! GJ/(kg.K)      ! 450.d+6     N.um/(kg.K)
double precision, parameter::SHeatF=  450.d-9  ! GJ/(kg.K)      ! 450.d+6     N.um/(kg.K)
! slip resistance parameters, surface energy, latent heat
double precision, parameter::W_A=10.d0
double precision, parameter::W_F=7.d0
double precision, parameter::SurfEn= 4.d-3     ! GPa     !4.d-6     [Khi/L0]=(N/um^2)
double precision, parameter::Lambda= -50.5d-6  ! GJ/kg   !-50.5d+9           (N.um/kg)           
double precision, parameter::Mu_F=55.d0        ! GPa  
double precision, parameter::Mu_A=67.5d0       ! GPa  
double precision, parameter::Mu_M=98.4d0       ! GPa  
! heat conduction
double precision, parameter::HCond=6.d+1     ! (N/K.s)       
! debug
!       logical debugoption=TRUE
! temperature perturbation parameter
      double precision, parameter::dtheta = 1.d-8
!
! Hardening parameteres for asutenite
double precision, parameter::s0_A   = 1.88d-1     ! GPa     !1.88d-4   (N/um^2)
double precision, parameter::sInf_A = 5.79d-1     ! GPa     !5.79d-4   (N/um^2)
double precision, parameter::k0_A   = 3.0d0       ! GPa     !3.0d-3    (N/um^2)
double precision, parameter::w0_A   = 2.8d0     
double precision, parameter::q0_A   = 1.d0
double precision, parameter::c0_A   = 5.d-1
!
double precision, parameter::s0_F   = 1.58d-1     ! GPa     !1.58d-4   (N/um^2)
double precision, parameter::sInf_F = 4.12d-1     ! GPa     !4.12d-4   (N/um^2)
double precision, parameter::k0_F   = 1.9d0       ! GPa     !1.9d-3    (N/um^2)
double precision, parameter::w0_F   = 2.8d0
double precision, parameter::q0_F   = 1.d0
double precision, parameter::c0_F   = 5.d-1               
! Transformation Kinetic law austenite
double precision, parameter::Nu_A      = 1.7d-1
double precision, parameter::XiMax_A   = 3.d-3     ! (s^-1)
double precision, parameter::TDFCrit_A = 227.d-3   ! GPa     !227.d-6   (N/um^2)
! Plastic Kinetic Law austenite
double precision, parameter::mExp_A      = 2.d-2
double precision, parameter::GammaDot0_A = 1.d-3       
double precision, parameter::Phi_A       = 5.13d-9   ! GJ/(kg.K)  !5.13d+6  (N.um/kg.K)		 		        
! Plastic Kinetic Law ferrite
double precision, parameter::mExp_F      = 2.d-2
double precision, parameter::GammaDot0_F = 1.d-3       
double precision, parameter::Phi_F       = 4.27d-9   ! GJ/(kg.K)  !4.27350427d+6   (N.um/kg.K)       
