import torch
from torch._tensor import Tensor


SlipSys: Tensor = torch.zeros([24, 3, 3])
ElasStif: Tensor = torch.zeros([3, 3, 3, 3])

# plastic slip systems: schmid tensors
SlipSys[0, 0, 0] =  0.0                                
SlipSys[0, 0, 1] =  0.40824829         
SlipSys[0, 0, 2] =  -0.40824829        
SlipSys[0, 1, 0] =  0.0                
SlipSys[0, 1, 1] =  0.40824829         
SlipSys[0, 1, 2] =  -0.40824829        
SlipSys[0, 2, 0] =  0.0                
SlipSys[0, 2, 1] =  0.40824829         
SlipSys[0, 2, 2] =  -0.40824829        
#                                      
SlipSys[1, 0, 0] =  0.0                
SlipSys[1, 0, 1] =  -0.40824829        
SlipSys[1, 0, 2] =  0.40824829         
SlipSys[1, 1, 0] =  0.0                
SlipSys[1, 1, 1] =  -0.40824829        
SlipSys[1, 1, 2] =  0.40824829         
SlipSys[1, 2, 0] =  0.0                
SlipSys[1, 2, 1] =  -0.40824829        
SlipSys[1, 2, 2] =  0.40824829         
#                                      
SlipSys[2, 0, 0] =  -0.40824829        
SlipSys[2, 0, 1] =  0.0                
SlipSys[2, 0, 2] =  0.40824829         
SlipSys[2, 1, 0] =  -0.40824829        
SlipSys[2, 1, 1] =  0.0                
SlipSys[2, 1, 2] =  0.40824829         
SlipSys[2, 2, 0] =  -0.40824829        
SlipSys[2, 2, 1] =  0.0                
SlipSys[2, 2, 2] =  0.40824829         
#                                      
SlipSys[3, 0, 0] =  0.40824829         
SlipSys[3, 0, 1] =  0.0                
SlipSys[3, 0, 2] =  -0.40824829        
SlipSys[3, 1, 0] =  0.40824829         
SlipSys[3, 1, 1] =  0.0                
SlipSys[3, 1, 2] =  -0.40824829        
SlipSys[3, 2, 0] =  0.40824829         
SlipSys[3, 2, 1] =  0.0                
SlipSys[3, 2, 2] =  -0.40824829        
#                                      
SlipSys[4, 0, 0] =  0.40824829         
SlipSys[4, 0, 1] =  -0.40824829        
SlipSys[4, 0, 2] =  0.0                
SlipSys[4, 1, 0] =  0.40824829         
SlipSys[4, 1, 1] =  -0.40824829        
SlipSys[4, 1, 2] =  0.0                
SlipSys[4, 2, 0] =  0.40824829         
SlipSys[4, 2, 1] =  -0.40824829        
SlipSys[4, 2, 2] =  0.0                
#                                      
SlipSys[5, 0, 0] =  -0.40824829        
SlipSys[5, 0, 1] =  0.40824829         
SlipSys[5, 0, 2] =  0.0                
SlipSys[5, 1, 0] =  -0.40824829        
SlipSys[5, 1, 1] =  0.40824829         
SlipSys[5, 1, 2] =  0.0                
SlipSys[5, 2, 0] =  -0.40824829        
SlipSys[5, 2, 1] =  0.40824829         
SlipSys[5, 2, 2] =  0.0                
#                                      
SlipSys[6, 0, 0] =  0.0                
SlipSys[6, 0, 1] =  0.40824829         
SlipSys[6, 0, 2] =  -0.40824829        
SlipSys[6, 1, 0] =  0.0                
SlipSys[6, 1, 1] =  -0.40824829        
SlipSys[6, 1, 2] =  0.40824829         
SlipSys[6, 2, 0] =  0.0                
SlipSys[6, 2, 1] =  -0.40824829        
SlipSys[6, 2, 2] =  0.40824829         
#                                      
SlipSys[7, 0, 0] =  0.0                
SlipSys[7, 0, 1] =  -0.40824829        
SlipSys[7, 0, 2] =  0.40824829         
SlipSys[7, 1, 0] =  0.0                
SlipSys[7, 1, 1] =  0.40824829         
SlipSys[7, 1, 2] =  -0.40824829        
SlipSys[7, 2, 0] =  0.0                
SlipSys[7, 2, 1] =  0.40824829         
SlipSys[7, 2, 2] =  -0.40824829        
#                                      
SlipSys[8, 0, 0] =  0.40824829         
SlipSys[8, 0, 1] =  0.0                
SlipSys[8, 0, 2] =  0.40824829         
SlipSys[8, 1, 0] =  -0.40824829        
SlipSys[8, 1, 1] =  0.0                
SlipSys[8, 1, 2] =  -0.40824829        
SlipSys[8, 2, 0] =  -0.40824829        
SlipSys[8, 2, 1] =  0.0                
SlipSys[8, 2, 2] =  -0.40824829        
#                                      
SlipSys[9, 0, 0] =  -0.40824829        
SlipSys[9, 0, 1] =  0.0                
SlipSys[9, 0, 2] =  -0.40824829        
SlipSys[9, 1, 0] =  0.40824829         
SlipSys[9, 1, 1] =  0.0                
SlipSys[9, 1, 2] =  0.40824829         
SlipSys[9, 2, 0] =  0.40824829         
SlipSys[9, 2, 1] =  0.0                
SlipSys[9, 2, 2] =  0.40824829         
#                                      
SlipSys[10, 0, 0] = -0.40824829        
SlipSys[10, 0, 1] = -0.40824829        
SlipSys[10, 0, 2] = 0.0                
SlipSys[10, 1, 0] = 0.40824829         
SlipSys[10, 1, 1] = 0.40824829         
SlipSys[10, 1, 2] = 0.0                
SlipSys[10, 2, 0] = 0.40824829         
SlipSys[10, 2, 1] = 0.40824829         
SlipSys[10, 2, 2] = 0.0                
#                                      
SlipSys[11, 0, 0] = 0.40824829         
SlipSys[11, 0, 1] = 0.40824829         
SlipSys[11, 0, 2] = 0.0                
SlipSys[11, 1, 0] = -0.40824829        
SlipSys[11, 1, 1] = -0.40824829        
SlipSys[11, 1, 2] = 0.0                
SlipSys[11, 2, 0] = -0.40824829        
SlipSys[11, 2, 1] = -0.40824829        
SlipSys[11, 2, 2] = 0.0                
#                                      
SlipSys[12, 0, 0] = -0.40824829        
SlipSys[12, 0, 1] = 0.0                
SlipSys[12, 0, 2] = -0.40824829        
SlipSys[12, 1, 0] = -0.40824829        
SlipSys[12, 1, 1] = 0.0                
SlipSys[12, 1, 2] = -0.40824829        
SlipSys[12, 2, 0] = 0.40824829         
SlipSys[12, 2, 1] = 0.0                
SlipSys[12, 2, 2] = 0.40824829         
#                                      
SlipSys[13, 0, 0] = 0.40824829         
SlipSys[13, 0, 1] = 0.0                
SlipSys[13, 0, 2] = 0.40824829         
SlipSys[13, 1, 0] = 0.40824829         
SlipSys[13, 1, 1] = 0.0                
SlipSys[13, 1, 2] = 0.40824829         
SlipSys[13, 2, 0] = -0.40824829        
SlipSys[13, 2, 1] = 0.0                
SlipSys[13, 2, 2] = -0.40824829        
#                                      
SlipSys[14, 0, 0] = 0.0                
SlipSys[14, 0, 1] = 0.40824829         
SlipSys[14, 0, 2] = 0.40824829         
SlipSys[14, 1, 0] = 0.0                
SlipSys[14, 1, 1] = 0.40824829         
SlipSys[14, 1, 2] = 0.40824829         
SlipSys[14, 2, 0] = 0.0                
SlipSys[14, 2, 1] = -0.40824829        
SlipSys[14, 2, 2] = -0.40824829        
#                                      
SlipSys[15, 0, 0] = 0.0                
SlipSys[15, 0, 1] = -0.40824829        
SlipSys[15, 0, 2] = -0.40824829        
SlipSys[15, 1, 0] = 0.0                
SlipSys[15, 1, 1] = -0.40824829        
SlipSys[15, 1, 2] = -0.40824829        
SlipSys[15, 2, 0] = 0.0                
SlipSys[15, 2, 1] = 0.40824829         
SlipSys[15, 2, 2] = 0.40824829         
#                                      
SlipSys[16, 0, 0] = 0.40824829         
SlipSys[16, 0, 1] = -0.40824829        
SlipSys[16, 0, 2] = 0.0                
SlipSys[16, 1, 0] = 0.40824829         
SlipSys[16, 1, 1] = -0.40824829        
SlipSys[16, 1, 2] = 0.0                
SlipSys[16, 2, 0] = -0.40824829        
SlipSys[16, 2, 1] = 0.40824829         
SlipSys[16, 2, 2] = 0.0                
#                                      
SlipSys[17, 0, 0] = -0.40824829        
SlipSys[17, 0, 1] = 0.40824829         
SlipSys[17, 0, 2] = 0.0                
SlipSys[17, 1, 0] = -0.40824829        
SlipSys[17, 1, 1] = 0.40824829         
SlipSys[17, 1, 2] = 0.0                
SlipSys[17, 2, 0] = 0.40824829         
SlipSys[17, 2, 1] = -0.40824829        
SlipSys[17, 2, 2] = 0.0                
#                                      
SlipSys[18, 0, 0] = 0.40824829         
SlipSys[18, 0, 1] = 0.0                
SlipSys[18, 0, 2] = -0.40824829        
SlipSys[18, 1, 0] = -0.40824829        
SlipSys[18, 1, 1] = 0.0                
SlipSys[18, 1, 2] = 0.40824829         
SlipSys[18, 2, 0] = 0.40824829         
SlipSys[18, 2, 1] = 0.0                
SlipSys[18, 2, 2] = -0.40824829        
#                                      
SlipSys[19, 0, 0] = -0.40824829        
SlipSys[19, 0, 1] = 0.0                
SlipSys[19, 0, 2] = 0.40824829         
SlipSys[19, 1, 0] = 0.40824829         
SlipSys[19, 1, 1] = 0.0                
SlipSys[19, 1, 2] = -0.40824829        
SlipSys[19, 2, 0] = -0.40824829        
SlipSys[19, 2, 1] = 0.0                
SlipSys[19, 2, 2] = 0.40824829         
#                                      
SlipSys[20, 0, 0] = 0.0                
SlipSys[20, 0, 1] = 0.40824829         
SlipSys[20, 0, 2] = 0.40824829         
SlipSys[20, 1, 0] = 0.0                
SlipSys[20, 1, 1] = -0.40824829        
SlipSys[20, 1, 2] = -0.40824829        
SlipSys[20, 2, 0] = 0.0                
SlipSys[20, 2, 1] = 0.40824829         
SlipSys[20, 2, 2] = 0.40824829         
#                                      
SlipSys[21, 0, 0] = 0.0                
SlipSys[21, 0, 1] = -0.40824829        
SlipSys[21, 0, 2] = -0.40824829        
SlipSys[21, 1, 0] = 0.0                
SlipSys[21, 1, 1] = 0.40824829         
SlipSys[21, 1, 2] = 0.40824829         
SlipSys[21, 2, 0] = 0.0                
SlipSys[21, 2, 1] = -0.40824829        
SlipSys[21, 2, 2] = -0.40824829        
#                                      
SlipSys[22, 0, 0] = -0.40824829        
SlipSys[22, 0, 1] = -0.40824829        
SlipSys[22, 0, 2] = 0.0                
SlipSys[22, 1, 0] = 0.40824829         
SlipSys[22, 1, 1] = 0.40824829         
SlipSys[22, 1, 2] = 0.0                
SlipSys[22, 2, 0] = -0.40824829        
SlipSys[22, 2, 1] = -0.40824829        
SlipSys[22, 2, 2] = 0.0                
#                                      
SlipSys[23, 0, 0] = 0.40824829         
SlipSys[23, 0, 1] = 0.40824829         
SlipSys[23, 0, 2] = 0.0                
SlipSys[23, 1, 0] = -0.40824829        
SlipSys[23, 1, 1] = -0.40824829        
SlipSys[23, 1, 2] = 0.0                
SlipSys[23, 2, 0] = 0.40824829         
SlipSys[23, 2, 1] = 0.40824829         
SlipSys[23, 2, 2] = 0.0                


#  elastic stiffnes ferrite
ElasStif[0, 0, 0, 0] = 233.500000   
ElasStif[0, 0, 0, 1] =   0.000000   
ElasStif[0, 0, 0, 2] =   0.000000   
ElasStif[0, 0, 1, 0] =   0.000000   
ElasStif[0, 0, 1, 1] = 135.500000   
ElasStif[0, 0, 1, 2] =   0.000000   
ElasStif[0, 0, 2, 0] =   0.000000   
ElasStif[0, 0, 2, 1] =   0.000000   
ElasStif[0, 0, 2, 2] = 135.500000   
ElasStif[0, 1, 0, 0] =   0.000000   
ElasStif[0, 1, 0, 1] = 118.000000   
ElasStif[0, 1, 0, 2] =   0.000000   
ElasStif[0, 1, 1, 0] = 118.000000   
ElasStif[0, 1, 1, 1] =   0.000000   
ElasStif[0, 1, 1, 2] =   0.000000   
ElasStif[0, 1, 2, 0] =   0.000000   
ElasStif[0, 1, 2, 1] =   0.000000   
ElasStif[0, 1, 2, 2] =   0.000000   
ElasStif[0, 2, 0, 0] =   0.000000   
ElasStif[0, 2, 0, 1] =   0.000000   
ElasStif[0, 2, 0, 2] = 118.000000   
ElasStif[0, 2, 1, 0] =   0.000000   
ElasStif[0, 2, 1, 1] =   0.000000   
ElasStif[0, 2, 1, 2] =   0.000000   
ElasStif[0, 2, 2, 0] = 118.000000   
ElasStif[0, 2, 2, 1] =   0.000000   
ElasStif[0, 2, 2, 2] =   0.000000   
ElasStif[1, 0, 0, 0] =   0.000000   
ElasStif[1, 0, 0, 1] = 118.000000   
ElasStif[1, 0, 0, 2] =   0.000000   
ElasStif[1, 0, 1, 0] = 118.000000   
ElasStif[1, 0, 1, 1] =   0.000000   
ElasStif[1, 0, 1, 2] =   0.000000   
ElasStif[1, 0, 2, 0] =   0.000000   
ElasStif[1, 0, 2, 1] =   0.000000   
ElasStif[1, 0, 2, 2] =   0.000000   
ElasStif[1, 1, 0, 0] = 135.500000   
ElasStif[1, 1, 0, 1] =   0.000000   
ElasStif[1, 1, 0, 2] =   0.000000   
ElasStif[1, 1, 1, 0] =   0.000000   
ElasStif[1, 1, 1, 1] = 233.500000   
ElasStif[1, 1, 1, 2] =   0.000000   
ElasStif[1, 1, 2, 0] =   0.000000   
ElasStif[1, 1, 2, 1] =   0.000000   
ElasStif[1, 1, 2, 2] = 135.500000   
ElasStif[1, 2, 0, 0] =   0.000000   
ElasStif[1, 2, 0, 1] =   0.000000   
ElasStif[1, 2, 0, 2] =   0.000000   
ElasStif[1, 2, 1, 0] =   0.000000   
ElasStif[1, 2, 1, 1] =   0.000000   
ElasStif[1, 2, 1, 2] = 118.000000   
ElasStif[1, 2, 2, 0] =   0.000000   
ElasStif[1, 2, 2, 1] = 118.000000   
ElasStif[1, 2, 2, 2] =   0.000000   
ElasStif[2, 0, 0, 0] =   0.000000   
ElasStif[2, 0, 0, 1] =   0.000000   
ElasStif[2, 0, 0, 2] = 118.000000   
ElasStif[2, 0, 1, 0] =   0.000000   
ElasStif[2, 0, 1, 1] =   0.000000   
ElasStif[2, 0, 1, 2] =   0.000000   
ElasStif[2, 0, 2, 0] = 118.000000   
ElasStif[2, 0, 2, 1] =   0.000000   
ElasStif[2, 0, 2, 2] =   0.000000   
ElasStif[2, 1, 0, 0] =   0.000000   
ElasStif[2, 1, 0, 1] =   0.000000   
ElasStif[2, 1, 0, 2] =   0.000000   
ElasStif[2, 1, 1, 0] =   0.000000   
ElasStif[2, 1, 1, 1] =   0.000000   
ElasStif[2, 1, 1, 2] = 118.000000   
ElasStif[2, 1, 2, 0] =   0.000000   
ElasStif[2, 1, 2, 1] = 118.000000   
ElasStif[2, 1, 2, 2] =   0.000000   
ElasStif[2, 2, 0, 0] = 135.500000   
ElasStif[2, 2, 0, 1] =   0.000000   
ElasStif[2, 2, 0, 2] =   0.000000   
ElasStif[2, 2, 1, 0] =   0.000000   
ElasStif[2, 2, 1, 1] = 135.500000   
ElasStif[2, 2, 1, 2] =   0.000000   
ElasStif[2, 2, 2, 0] =   0.000000   
ElasStif[2, 2, 2, 1] =   0.000000   
ElasStif[2, 2, 2, 2] = 233.500000   

# equivalent elastic shear moduli ferrite
ShearMod = 55.0000 # GPa
