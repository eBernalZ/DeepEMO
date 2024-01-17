#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:41:51 2023

@author: Rodolfo Humberto Tamayo
"""

import math


from zcat_tools import *
from zcat_g import *
from zcat_Z import *
from zcat_F import *



ZCAT_F = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20]
ZCAT_G = [g4, g5, g2, g7, g9, g4, g5, g2, g7, g9, g3, g10, g1, g6, g8, g10, g1, g8, g6, g3]

zcat1,zcat2,zcat3,zcat4,zcat5,zcat6,zcat7,zcat8,zcat9,zcat10,zcat11 = 0,1,2,3,4,5,6,7,8,9,10
zcat12,zcat13,zcat14,zcat15,zcat16,zcat17,zcat18,zcat19,zcat20 = 11,12,13,14,15,16,17,18,19



zcat_config = {
    'M': 0,                 # Number of objectives
    'N': 0,                 # Number of decision variables
    'COMPLICATED_PS': 1,    # Complicated PS flag
    'LEVEL': 0,             # Difficulty level
    'BIAS': 0,              # Bias flag
    'IMBALANCE': 0,         # Imbalance flag
    'LB': None,             # Low bound (list or numpy array)
    'UB': None              # Up bound (list or numpy array)
}


ZCAT_TRUE = 1
ZCAT_FALSE = 0



def zcat_set_bounds(LB, UB, n):
    for i in range(1, n + 1):
        LB[i - 1] = -i * 0.5
        UB[i - 1] = i * 0.5
    return LB, UB


def zcat_get_y(x, LB, UB, n):
    y = [0.0] * n
    for i in range(n):
        y[i] = (x[i] - LB[i]) / (UB[i] - LB[i])
        assert 0.0 <= y[i] <= 1.0
    return y


def zcat_get_z(y, m, n, g_function):
    g = g_function(y, m, n)
    z = [0.0] * (n - m)
    for i in range(m, n):
        z[i - m] = y[i] - g[i - m]
    return z


def zcat_get_w(z, m, n):
    w = [0.0] * (n - m)
    for i in range(n - m):
        w[i] = Zbias(z[i]) if zcat_config["BIAS"] == 1 else z[i]
    return w



def zcat_get_J(i, M, w, wsize):
    assert i <= M
    J = []
    size = 0
    for j in range(1, wsize + 1):
        if (j - i) % M == 0:
            size += 1
            J.append(w[j - 1])

    if size == 0:
        size = 1
        J.append(w[0])

    return J, size


def zcat_evaluate_Z(w, ith_objective):
    if zcat_config['IMBALANCE'] == ZCAT_TRUE:
        if ith_objective % 2 == 0:
            z = Z4(w)
        else:
            z = Z1(w)
        return z

    switch_cases = {
        1: Z1,
        2: Z2,
        3: Z3,
        4: Z4,
        5: Z5,
        6: Z6
    }

    z_function = switch_cases.get(zcat_config['LEVEL'], Z1)
    z = z_function(w)
    
    return z


def zcat_get_beta(y, M, m, n, g_function, zcat_config):
    z = zcat_get_z(y, m, n, g_function)
    w = zcat_get_w(z, m, n)

    b = [0.0] * M

    if n == m:  # number of variables equal to the dimension of the PF
        for i in range(1, M + 1):
            b[i - 1] = 0.0
           
    else:
        for i in range(1, M + 1):
            J, Jsize = zcat_get_J(i, M, w, n - m)
            Zvalue = zcat_evaluate_Z(J, i)
            b[i - 1] = (i * i) * Zvalue
            
            #b[i - 1] = 0.0
    return b


def zcat_get_alpha(y, M, f_function):
    a = f_function(y, M)
    for i in range(1, M + 1):
        a[i - 1] = (i * i) * a[i - 1]
    return a


def zcat_mop_definition(alpha, beta, M):
    f = [0.0] * M
    # UNCOMMENT THE NEXT LINE TO BE EVEN MORE PRECISE IN THE GENERATION OF THE PF. SOLUTIONS WILL BE GENERATED BUT MAY BE INCORRECT.
    #beta = [0.0] * M 
    for i in range(1, M + 1):
        f[i - 1] = alpha[i - 1] + beta[i - 1]  # Additive Approach
    return f
        
    
    
# **********************************************************************************************
# Definition of the problems ZCAT1--ZCAT20
# **********************************************************************************************


def ZCAT1(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat1]
    g_function = ZCAT_G[zcat1] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values
    
    return f



def ZCAT2(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat2]
    g_function = ZCAT_G[zcat2] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT3(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat3]
    g_function = ZCAT_G[zcat3] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position  
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT4(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat4]
    g_function = ZCAT_G[zcat4] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position 
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT5(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat5]
    g_function = ZCAT_G[zcat5] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
 
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT6(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat6]
    g_function = ZCAT_G[zcat6] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT7(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat7]
    g_function = ZCAT_G[zcat7] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT8(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat8]
    g_function = ZCAT_G[zcat8] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT9(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat9]
    g_function = ZCAT_G[zcat9] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT10(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat10]
    g_function = ZCAT_G[zcat10] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT11(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat11]
    g_function = ZCAT_G[zcat11] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT12(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat12]
    g_function = ZCAT_G[zcat12] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT13(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat13]
    g_function = ZCAT_G[zcat13] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT14(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = 1

    f_function = ZCAT_F[zcat14]
    g_function = ZCAT_G[zcat14] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f



def ZCAT15(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = 1

    f_function = ZCAT_F[zcat15]
    g_function = ZCAT_G[zcat15] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT16(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = 1

    f_function = ZCAT_F[zcat16]
    g_function = ZCAT_G[zcat16] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT17(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat17]
    g_function = ZCAT_G[zcat17] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT18(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']
    m = zcat_config['M'] - 1

    f_function = ZCAT_F[zcat18]
    g_function = ZCAT_G[zcat18] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT19(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']

    f_function = ZCAT_F[zcat19]
    g_function = ZCAT_G[zcat19] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    m = 1 if zcat_value_in(y[0], 0, 0.2) or zcat_value_in(y[0], 0.4, 0.6) else M - 1
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




def ZCAT20(x, zcat_config):
    M = zcat_config['M']
    n = zcat_config['N']

    f_function = ZCAT_F[zcat20]
    g_function = ZCAT_G[zcat20] if zcat_config['COMPLICATED_PS'] else g0

    y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], n)
    m = 1 if zcat_value_in(y[0], 0.1, 0.4) or zcat_value_in(y[0], 0.6, 0.9) else M - 1
    
    alpha = zcat_get_alpha(y, M, f_function)  # Define Position
    beta = zcat_get_beta(y, M, m, n, g_function, zcat_config)  # Define Distance
    f = zcat_mop_definition(alpha, beta, M)  # Assign fitness values

    return f




# **********************************************************************************************
# **********************************************************************************************
# **********************************************************************************************
# **********************************************************************************************


def zcat_default_settings(nobj):
    # Benchmark Configuration
    zcat_config['M'] = nobj
    zcat_config['N'] = nobj * 10
    zcat_config['LEVEL'] = 1
    zcat_config['BIAS'] = ZCAT_FALSE
    zcat_config['COMPLICATED_PS'] = ZCAT_TRUE
    zcat_config['IMBALANCE'] = ZCAT_FALSE

    zcat_config['LB'] = [-i * 0.5 for i in range(1, zcat_config['N'] + 1)]
    zcat_config['UB'] = [i * 0.5 for i in range(1, zcat_config['N'] + 1)]



def zcat_set(nvar, nobj, Level, Bias, Complicated_PS, Imbalance):
    default_nvars = nobj * 10

    assert 1 <= Level <= 6
    assert Bias == ZCAT_TRUE or Bias == ZCAT_FALSE
    assert Complicated_PS == ZCAT_TRUE or Complicated_PS == ZCAT_FALSE
    assert nobj > 1
    assert nvar >= (nobj - 1) or nvar == -1

    # Benchmark Configuration
    zcat_config['LEVEL'] = Level
    zcat_config['BIAS'] = Bias
    zcat_config['COMPLICATED_PS'] = Complicated_PS
    zcat_config['IMBALANCE'] = Imbalance
    zcat_config['M'] = nobj
    zcat_config['N'] = default_nvars if nvar == -1 else nvar
    zcat_config['LB'] = [-i * 0.5 for i in range(1, zcat_config['N'] + 1)]
    zcat_config['UB'] = [i * 0.5 for i in range(1, zcat_config['N'] + 1)]

    # Return the lower and upper bounds as output
    return zcat_config['LB'], zcat_config['UB'], zcat_config
































