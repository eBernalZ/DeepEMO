#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:37:12 2023

@author: Rodolfo Humberto Tamayo
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from pandas.plotting import parallel_coordinates

#%matplotlib qt

from zcat_tools import *
from zcat_g import *
from zcat_Z import *
from zcat_F import *
from zcat_benckmark import *
from zcat_rnd_opt_sol import *


ZCAT_MOP = [ZCAT1, ZCAT2, ZCAT3, ZCAT4, ZCAT5, ZCAT6, ZCAT7, ZCAT8, ZCAT9, ZCAT10,
    ZCAT11, ZCAT12, ZCAT13, ZCAT14, ZCAT15, ZCAT16, ZCAT17, ZCAT18, ZCAT19, ZCAT20]

ZCAT_MOP_STR = [
    "ZCAT1", "ZCAT2", "ZCAT3", "ZCAT4", "ZCAT5", "ZCAT6", "ZCAT7", "ZCAT8",
    "ZCAT9", "ZCAT10", "ZCAT11", "ZCAT12", "ZCAT13", "ZCAT14", "ZCAT15",
    "ZCAT16", "ZCAT17", "ZCAT18", "ZCAT19", "ZCAT20"]

zcat1,zcat2,zcat3,zcat4,zcat5,zcat6,zcat7,zcat8,zcat9,zcat10,zcat11 = 0,1,2,3,4,5,6,7,8,9,10
zcat12,zcat13,zcat14,zcat15,zcat16,zcat17,zcat18,zcat19,zcat20 = 11,12,13,14,15,16,17,18,19

# *********************************************************************************************************************
# *********************************************************************************************************************

mop = zcat1 # ZCAT MOP TO PLOT
nobj = 3 # NUMBER OF OBJECTIVES

seed = 1 # SEED USED WHEN GENERATING THE SOLUTIONS.
nvar = 10 * nobj #Standard.

# *********************************************************************************************************************
# *********************************************************************************************************************


def read_opt_file(file_path,nobj,nvar):
    obj_columns = ['obj{}'.format(i) for i in range( 1, nobj + 1 )]
    var_columns = ['var{}'.format(i) for i in range( 1, nvar + 1 )]
    df = pd.read_csv(file_path, sep=' ', names=obj_columns + var_columns, index_col=False)
    return df



file_directory = "zcat-optimal-solutions"
if not os.path.exists(file_directory):
    os.makedirs(file_directory)
fname = f"{file_directory}/PF{seed}-{ZCAT_MOP_STR[mop]}-{nobj}objs.opt"


data_frame = read_opt_file(fname,nobj,nvar)
data_frame = data_frame.drop(0)


# Convert objective columns to numeric data type
for i in range( 1, nobj + 1 ):
    data_frame['obj{}'.format(i)]  = pd.to_numeric(data_frame['obj{}'.format(i)])
    
objective_columns = data_frame[['obj{}'.format(i) for i in range( 1, nobj + 1 )]]

# *********************************************************************************************************************
# *********************************************************************************************************************

# PLOT THE PF AND THE FIRST THREE VARIABLES IF THE NUMBER OF OBJECTIVES IS THREE.


if nobj == 3: 
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the three objective columns as separate variables
    x = objective_columns['obj1']
    y = objective_columns['obj2']
    z = objective_columns['obj3']
    # Plot the points in 3D
    ax.scatter(x, y, z, c='r', marker='o',s=1)
    # Customize the plot
    ax.set_xlabel('Obj 1')
    ax.set_ylabel('Obj 2')
    ax.set_zlabel('Obj 3')
    plt.title('Pareto Front Approximation')
    plt.suptitle(ZCAT_MOP_STR[mop])
    plt.show()
    
    
    # Convert first three variable columns to numeric data type
    for i in range( 1, 3 + 1 ):
        data_frame['var{}'.format(i)]  = pd.to_numeric(data_frame['var{}'.format(i)])
        
    variable_columns = data_frame[['var{}'.format(i) for i in range( 1, 3 + 1 )]]
    
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the three objective columns as separate variables
    x = variable_columns['var1']
    y = variable_columns['var2']
    z = variable_columns['var3']
    # Plot the points in 3D
    ax.scatter(x, y, z, c='b', marker='o',s=1)
    # Customize the plot
    ax.set_xlabel('Var 1')
    ax.set_ylabel('Var 2')
    ax.set_zlabel('Var 3')
    plt.title('3D Scatter Plot of the First Three Variable Values')
    plt.suptitle(ZCAT_MOP_STR[mop])
    plt.show()
    
    
# *********************************************************************************************************************
# *********************************************************************************************************************

# PLOT THE PARALLEL COORDINATES OF THE OBJECTIVES


# Add a new column for coloring (all lines will be black)
objective_columns['Color'] = 1

# Create a parallel coordinates plot
plt.figure(figsize=(10, 6))
parallel_coordinates(objective_columns, 'Color', colormap='viridis', linewidth=0.05)

plt.title('Parallel Coordinates Pareto Front Approximation')
plt.suptitle(ZCAT_MOP_STR[mop])
plt.xlabel('Objective')
plt.ylabel('Evaluation')
plt.show()





    
    

