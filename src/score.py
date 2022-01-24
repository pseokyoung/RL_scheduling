import numpy as np

def R_Squared(prediction, actual):    
    actual_mean = np.mean(actual, axis=0)
    SSR = np.sum( (prediction - actual)**2 , axis=0)    
    TSS = np.sum( (actual - actual_mean )**2, axis=0 )    
    
    R_square = 1 - SSR/TSS
    return R_square

def RMSE(prediction, actual):
    rmse = np.sqrt(np.mean((prediction - actual)**2, axis=0))

    return rmse