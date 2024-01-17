#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 02:59:56 2023

@author: Rodolfo Humberto Tamayo
"""

import random
import time
import os
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from zcat_tools import *
from zcat_g import *
from zcat_Z import *
from zcat_F import *
from zcat_benckmark import *
from zcat_rnd_opt_sol import *



ZCAT_MOP_STR = [
    "ZCAT1", "ZCAT2", "ZCAT3", "ZCAT4", "ZCAT5", "ZCAT6", "ZCAT7", "ZCAT8",
    "ZCAT9", "ZCAT10", "ZCAT11", "ZCAT12", "ZCAT13", "ZCAT14", "ZCAT15",
    "ZCAT16", "ZCAT17", "ZCAT18", "ZCAT19", "ZCAT20"]


ZCAT_MOP = [ZCAT1, ZCAT2, ZCAT3, ZCAT4, ZCAT5, ZCAT6, ZCAT7, ZCAT8, ZCAT9, ZCAT10,
    ZCAT11, ZCAT12, ZCAT13, ZCAT14, ZCAT15, ZCAT16, ZCAT17, ZCAT18, ZCAT19, ZCAT20]


zcat1,zcat2,zcat3,zcat4,zcat5,zcat6,zcat7,zcat8,zcat9,zcat10,zcat11 = 0,1,2,3,4,5,6,7,8,9,10
zcat12,zcat13,zcat14,zcat15,zcat16,zcat17,zcat18,zcat19,zcat20 = 11,12,13,14,15,16,17,18,19


def print_file_header(file, mop, nobj, nvar):
    for j in range(nobj):
        file.write(f"<obj{j + 1}> ")
    for j in range(nvar):
        file.write(f"<var{j + 1}> ")
    file.write("\n")
    
    
def print_file_solution(file, x, f, nobj, nvar):
    for j in range(nobj):
        file.write(f"{f[j]:e} ")
    for j in range(nvar):
        file.write(f"{x[j]:e} ")
    file.write("\n")
    

def generate_pareto_front(mop, max_solutions, seed, nvar, nobj, Level, Bias_flag, Complicated_PS_flag, Imbalance_flag):
    random.seed(seed)

    # Configuring ZCAT benchmark. LB and UB are the bounds in the ZCAT structures
    LB, UB, zcat_config = zcat_set(nvar, nobj, Level, Bias_flag, Complicated_PS_flag, Imbalance_flag)

    file_directory = "zcat-optimal-solutions"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    fname = f"{file_directory}/PF{seed}-{ZCAT_MOP_STR[mop]}-{nobj}objs.opt"
    with open(fname, "w") as pf:
        print_file_header(pf, ZCAT_MOP_STR[mop], nobj, nvar)
        
        evaluations = []
        
        
        for i in range(max_solutions):
            x, m, g_function = zcat_rnd_opt_sol(mop,nobj,nvar) # Generate optimal solution
            y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], nvar) 
            #print('g_func2: ', g_function)
            #print('y2: ', y)
            #print('m2: ',m)
            #print('n2: ',nvar)
            g = g_function(y,m,nvar)
            #print('g2: ',g)
            beta = zcat_get_beta(y, nobj, m, nvar, g_function, zcat_config)  # Define Distance
            #print('beta 2: ',beta)
            f = ZCAT_MOP[mop](x,zcat_config)  # Evaluate the generated optimal solution
            evaluations.append(f)
            print_file_solution(pf, x, f, nobj, nvar)  # Print solution in the file
            
    return evaluations


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


# MODIFY THIS SECTION OF THE CODE WITH THE DESIRED PROPERTIES OF THE PF, LIKE THE NUMBER OF OBJECTIVES. 
# THIS CODE ALSO PRODUCES THE VARIABLES OF THE PF. 
#SO, IF THE SOLUTION'S CONFIGURATION IS DIFFERENT, THE PF SHOULD VE EXTREMELLY SIMILAR, BUT THE SOLUTIONS WILL BE DIFFERENT.



for mop in [zcat1,zcat2,zcat3,zcat4,zcat5,zcat6,zcat7,zcat8,zcat9,zcat10,zcat11,zcat12,zcat13,zcat14,zcat15,zcat16,zcat17,zcat18,zcat19,zcat20]:
    max_solutions = 5000
    nobj = 3  # number of objectives
    
    seed = 1
    nvar = 10 * nobj  # Standard decision variables
    
    
    #Configure the solutions
    # The standard values are Level 1, Bias_flag and Complicated_PS_flag as False and Imbalance_flag as True.
    
    # Other configurations may work well except Level 5 and 6.
    # They require to be extremely precise so they do not work completely. w = 0.
    # Even small values like w = 10^-20 does not work.
    # DO NOT USE LEVEL 5 or 6 to generate the PFs.
    
    Level = 1  # Level of the problem {1,..,6}
    Bias_flag = 0  # Bias flag (True: 1, False: 0)
    Complicated_PS_flag = 1  # Complicated PS flag (True: 1, False: 0)
    Imbalance_flag = 0  # Imbalance flag (True: 1, False: 0)
    
    
    #Call the function to generate the file with the Optimal Solutions. 
    evaluations = generate_pareto_front(mop, max_solutions, seed, nvar, nobj, Level, Bias_flag, Complicated_PS_flag, Imbalance_flag)












