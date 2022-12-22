import torch

from .constants import consts
from .trip_ferrite_data import SlipSys, ElasStif, ShearMod

I = torch.eye(3)

def sdvini(nstatv):

    statev = torch.zeros(58)
    # slip system, tag 0 = inactive, 1 = active
    tags_gamma = torch.zeros(24, dtype=torch.int)

    assert nstatv == 91 # ?? or 93

    # initial plastic deformation is identity
    statev[ : 9] = torch.eye(3).reshape(-1, 1).squeeze()

#
#   accumulative plastic strain, slip resistance parameter, dislocation 
#   density parameter
#   initial slip resistance = 0.158 for original matrix
#                             0.072 for soft matrix (low yield stress)
#                             0.216 for hard matrix (high yield stress)

    statev[9 : 33] = 0.0   # initial plastic strain                    
    statev[33: 57] = s0_F  # 1.58d-1 !initial slip resistance parameter
   
    statev[57] = 0.0     # initial dislocation density

    return statev, tags_gamma


def nonschmidstressbcc(Schmid,NonSchmid):
    '''
    computes non-Schmid stress according to Bassani's convention
    '''
    NonSchmid = torch.zeros_like(Schmid)
    # compute the non-glide stress (non-Schmid stress)
    NonSchmid[0]  = Schmid[5]      #      NonSchmid(1)  = NGlide*Schmid(6) 
    NonSchmid[1]  = Schmid[2]      #      NonSchmid(2)  = NGlide*Schmid(3) 
    NonSchmid[2]  = Schmid[1]      #      NonSchmid(3)  = NGlide*Schmid(2) 
    NonSchmid[3]  = Schmid[4]      #      NonSchmid(4)  = NGlide*Schmid(5) 
    NonSchmid[4]  = Schmid[3]      #      NonSchmid(5)  = NGlide*Schmid(4) 
    NonSchmid[5]  = Schmid[0]      #      NonSchmid(6)  = NGlide*Schmid(1) 
    #               
    NonSchmid[6]  = Schmid[9]      #      NonSchmid(7)  = NGlide*Schmid(10)
    NonSchmid[7]  = Schmid[10]     #      NonSchmid(8)  = NGlide*Schmid(11)
    NonSchmid[8]  = Schmid[11]     #      NonSchmid(9)  = NGlide*Schmid(12)
    NonSchmid[9] =  Schmid[6]      #      NonSchmid(10) = NGlide*Schmid(7) 
    NonSchmid[10] = Schmid[7]      #      NonSchmid(11) = NGlide*Schmid(8) 
    NonSchmid[11] = Schmid[8]      #      NonSchmid(12) = NGlide*Schmid(9) 
    #               
    NonSchmid[12] = Schmid[15]     #      NonSchmid(13) = NGlide*Schmid(16)
    NonSchmid[13] = Schmid[16]     #      NonSchmid(14) = NGlide*Schmid(17)
    NonSchmid[14] = Schmid[17]     #      NonSchmid(15) = NGlide*Schmid(18)
    NonSchmid[15] = Schmid[12]     #      NonSchmid(16) = NGlide*Schmid(13)
    NonSchmid[16] = Schmid[13]     #      NonSchmid(17) = NGlide*Schmid(14)
    NonSchmid[17] = Schmid[14]     #      NonSchmid(18) = NGlide*Schmid(15)
    #               
    NonSchmid[18] = Schmid[23]     #      NonSchmid(19) = NGlide*Schmid(24)
    NonSchmid[19] = Schmid[20]     #      NonSchmid(20) = NGlide*Schmid(21)
    NonSchmid[20] = Schmid[19]     #      NonSchmid(21) = NGlide*Schmid(20)
    NonSchmid[21] = Schmid[22]     #      NonSchmid(22) = NGlide*Schmid(23)
    NonSchmid[22] = Schmid[21]     #      NonSchmid(23) = NGlide*Schmid(22)
    NonSchmid[23] = Schmid[18]     #      NonSchmid(24) = NGlide*Schmid(19)
    #
    return consts.NGlide * NonSchmid

    
def rotation_matrix(angle1, angle2, angle3):
    '''defines the rotation matrix using 323 euler rotations'''
    # rotation matrix of the first rotation (angle1)
    angle1 = torch.tensor(angle1)
    angle2 = torch.tensor(angle2)
    angle3 = torch.tensor(angle3)

    R1 = torch.zeros([3, 3])
    R1[0, 0] =  torch.cos(angle1)
    R1[0, 1] =  torch.sin(angle1)
    R1[1, 0] = -torch.sin(angle1)
    R1[1, 1] =  torch.cos(angle1)
    R1[2, 2] =  1.0
    #
    #  rotation matrix of the second rotation (angle2)
    R2 = torch.zeros([3, 3])
    R2[0, 0] =  torch.cos(angle2)
    R2[0, 2] = -torch.sin(angle2)
    R2[1, 1] =  1.0
    R2[2, 0] =  torch.sin(angle2)
    R2[2, 2] =  torch.cos(angle2)
    #
    #  rotation matrix of the third rotation (angle3)
    R3 = torch.zeros([3, 3])
    R3[0, 0] =  torch.cos(angle3)
    R3[0, 1] =  torch.sin(angle3)
    R3[1, 0] = -torch.sin(angle3)
    R3[1, 1] =  torch.cos(angle3)
    R3[2, 2] =  1.0

#
#  calculate the overall rotation matrix
    RM = R3 @ R2 @ R1
    return RM


def grain_orientation_bcc(ElasStif, SlipSys, angles):
    '''rotates the stiffness and slip systems with the calculated rotation matrix'''
    rm = rotation_matrix(*angles)
    #
    rotated_slip_system = torch.einsum(
        'kab,ia,jb->kij',
        SlipSys, rm, rm
    )
    #
    rotated_elas_stiffness = torch.einsum(
        'abcd,ia,jb,kc,ld->ijkl',
        ElasStif, rm, rm, rm, rm
    )
    #
    return rotated_slip_system, rotated_elas_stiffness


def material_properties_bcc(ElasStif, SlipSys, angles):
    return grain_orientation_bcc(ElasStif, SlipSys, angles)
    

def get_ks(delta_s, slip_0):

    return consts.k0_F * (
        1 - (slip_0 + delta_s) / consts.sInf_F
    ) ** consts.uExp_F


def get_H_matrix(ks):
    N = len(ks)
    return torch.vstack(N * [ks]) * (
        consts.q0_F * (torch.ones([N, N]) - torch.eye(N)) + torch.eye(N)
    )
    

def get_ws(H):
    N = len(H)
    return (1. / (consts.c0_F * consts.mu_F * N)) * H.sum(axis=0)


def get_beta(dgamma, ws, beta_0):
    return beta_0 + torch.dot(ws, dgamma)


def get_gd(beta, ws):
    return -consts.omega_F * consts.mu_F * beta * ws



def get_gm(F_e1, slip_sys, elas_stiff):
    '''
    F_e1 : F_{e, n+1}
    '''
    C_e1 = F_e1.T @ F_e1 # we use this twice
    S = get_PK2(C_e1, elas_stiff)
    return ((C_e1 @ S).reshape(1, 3, 3) * slip_sys).sum(axis=(1, 2))


def get_r_I(ds, H, dgamma):
    return ds - H @ dgamma


def get_r_II(gs_1, slip_1, dgamma, dt):
    '''
    gs_1 : g^{(i)}_{n+1}
    slip_1 : s^{(i)}_{n+1}
    dgamma : \Delta\gamma_{n+1}
    dt : \Delta t = t_{n+1} - t_n
    '''
    #
    def get_indicator(x, threshold):
        with torch.no_grad():
            return (x > threshold).to(torch.float)
    #
    # first we form the vector we want to output and then
    # zero the entries below the threshold using the get_indicator function
    #
    ret = dgamma - dt * consts.GammaDot0_F * ((gs_1 / slip_1) ** (1 / consts.pExp_F) - 1.0)
    return ret * get_indicator(gs_1, slip_1)

def functional():
    return torch.norm(torch.hstack(
        [get_r_I(), get_r_II()]
    )) + penalty_coeff * (
        torch.max(0, -dgamma) ** 2 + torch.max(0, slip_1 - s_inf) ** 2
    )
    

def plasticdefgradbcc(delta_gamma, id_gamma, slip_sys, Fp0, Fp1):
    ...
    Lp = I - Lp
    Fp1 = torch.linalg.inv(Lp) @ Fp0
    return


def get_PK2(Fe, elas_stiff):
    #
    return torch.einsum(
        'ijkl,kl->ij', elas_stiff, 0.5 * (Fe.T @ Fe - I)
    )


def get_PK1(F, Fp, elas_stiff):
    Fe = F @ torch.linalg.inv(Fp)
    PK2 = get_PK2(F, Fp, elas_stiff)
    
    
def slipsystemcheckbcc(PDF, PDFCrit, DeltaGammaA, idGammaA,
                       nGammaA, TagGamma, UpgradeSlipSys, iterP):
    '''
    Checks the consistency condition of active slip systems
    '''
#
#  declaration of variables
#      double precision PDF(24),PDFCrit(24),DeltaGammaA(nGammaA)
#      integer i,nGammaA,TagGamma(24),idGammaA(nGammaA),iterP
#      logical UpgradeSlipSys,PlotDebug,PlotResult
#
# 
#
#  first check: if there are any active systems become inactive
    for i in range(nGammaA):
        if DeltaGammaA[i] <= 0.0:
            TagGamma[idGammaA[i]] = 0
            UpgradeSlipSys = True
             
    if UpgradeSlipSys: # goto 64
        return UpgradeSlipSys 
#
#  second check: if there are any inactive systems become active
    for i in range(24):
        if TagGamma[i] == 0 :
            if iterP <= 6:
                if PDF[i] > PDFCrit[i]:
                    TagGamma[i] = 1
                    UpgradeSlipSys = True
#     
#  64   continue
    return UpgradeSlipSys


def returnmappingbcc(DeltaGammaA, SlipResA, nGammaA, PDF, PDFCrit,
                     idGammaA, FP0, F1, BetaA, DeltaT, SubStepping,
                     ConTangent, temp, dtemp,
                     ElasStif, SlipSys, ShearMod):
    ...


def umat(stress, statev, tags_gamma, ddsdde, time, dtime,
         temp, nstatv, dfgrd0, dfgrd1, angles):

    I2 = torch.eye(3)
    dtemp = 0.0

    slip_sys, elas_stif = materialpropertiesbcc(ElasStif, SlipSys, angles)

    # initialization of stress-update algorithm
    SubFraction = 0.0                                          
    SubStep     = 1.0                                          
    DeltaF      = dfgrd1 - dfgrd0                               
    StateR      = statev                                        
    DefGrad0    = dfgrd0                                        
    DefGrad1    = dfgrd1

    FP0 = torch.zeros([3, 3])
    FP0 = statev[0 : 9]

    FE0 = dfgrd0 @ torch.linalg.inv(FP0)

    SubStepping = False
    DeltaT  = SubStep * dtime
    DF      = SubStep * DeltaF
    F0 = DefGrad0
    F1 = F0 + DF


    while True:   # substepping beings    60 continue
        SubStepping = False
        DeltaT  = SubStep*dtime
        DF      = SubStep*DeltaF
        F0 = DefGrad0
        F1 = F0 + DF

        FP0 = StateR[0 : 9].reshape(3, 3)
        Gamma0 = StateR[9 : 33]
        SlipRes0 = StateR[33 : 57]
        Beta0    = StateR[57]
        TagGamma = tags_gamma

        # set the initially active slip systems
        iterP = 0
        # 61 continue
        # counting the number of active systems
        nGammaA = Tags_Gamma.sum()

        # allocating according to nGamma and set initial values for iteration
        idGammaA = torch.where(TagGamma == 1)[0]    # indices of nonzero tags
        DeltaGammaA = torch.zeros(nGammaA)
        
        upgradeslipsys = False

        BetaA    = Beta0
        SlipResA = SlipRes0
        
        # compute the stress-update using return-mapping algorithm
        ConTangent = False

        # we should do something about PDF, PDFCrit, ... and those that are not 
        returnmappingbcc(DeltaGammaA,SlipResA,nGammaA,PDF,PDFCrit,
                         idGammaA,FP0,F1,BetaA,DeltaT,SubStepping,
                         ConTangent,temp+dtemp,
                         ElasStif,SlipSys,ShearMod)
        
        if SubStepping:
            ... # goto 68
        SubStepping = False
