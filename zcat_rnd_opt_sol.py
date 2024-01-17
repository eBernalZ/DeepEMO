#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 02:50:22 2023

@author: Rodolfo Humberto Tamayo
"""

import random


from zcat_tools import *
from zcat_g import *
from zcat_Z import *
from zcat_F import *
from zcat_benckmark import *


ZCAT_G = [g4, g5, g2, g7, g9, g4, g5, g2, g7, g9, g3, g10, g1, g6, g8, g10, g1, g8, g6, g3]

zcat1,zcat2,zcat3,zcat4,zcat5,zcat6,zcat7,zcat8,zcat9,zcat10,zcat11 = 0,1,2,3,4,5,6,7,8,9,10
zcat12,zcat13,zcat14,zcat15,zcat16,zcat17,zcat18,zcat19,zcat20 = 11,12,13,14,15,16,17,18,19




def zcat_search_space(y, i):
    lb = -i * 0.5
    ub = i * 0.5
    x = y * (ub - lb) + lb
    return x


class Segment:
    def __init__(self):
        self.x1 = 0.0
        self.x2 = 0.0

def zcat_get_segments(k):
    segments = []
    fname = f"seg/K{k}.seg"

    try:
        with open(fname, "r") as fp:
            for i in range(k):
                x1 = float(fp.readline())
                x2 = float(fp.readline())
                segment = Segment()
                segment.x1 = x1
                segment.x2 = x2
                segments.append(segment)
    except FileNotFoundError:
        print(f"ERROR: File {fname} not found.")
        exit(1)
    except ValueError:
        print(f"ERROR: Cannot read {fname} (get_segments).")
        exit(1)

    return segments


def rnd_real(lower, upper):
    return random.uniform(lower, upper)

def rnd_int(lower, upper):
    return random.randint(lower, upper)

def zcat_value_in(value, start, end):
    return start <= value <= end


def zcat_rnd_opt_sol(mop, nobj, nreal):
    x = [0]*nreal
    m = -1
    g_function = ZCAT_G[mop] if zcat_config['COMPLICATED_PS'] else g0

    y0 = random.uniform(0.0, 1.0)
    m = nobj - 1

    if zcat1 <= mop <= zcat13 or zcat17 == mop or mop == zcat18:
        m = nobj - 1
    elif zcat14 <= mop <= zcat16:  # Degenerate PF
        m = 1
    elif mop == zcat19:  # Hybrid
        m = 1 if (0.0 <= y0 <= 0.2) or (0.4 <= y0 <= 0.6) else nobj - 1
    elif mop == zcat20:  # Hybrid
        m = 1 if (0.1 <= y0 <= 0.4) or (0.6 <= y0 <= 0.9) else nobj - 1

    assert m > 0

    # For disconnected problems
    if (zcat11 <= mop <= zcat13) or mop == zcat15 or mop == zcat16:
        # Get the number of disconnections
        k = -1
        if mop == zcat11:
            k = 4
        if mop == zcat12 or mop == zcat13 or mop == zcat15:
            k = 3
        if mop == zcat16:
            k = 5
        assert k > 0

        # Get the range of each segment
        seg = zcat_get_segments(k)

        # Fix optimal solution
        k = random.randint(0, k - 1)  # choose random segment
        y0 = random.uniform(seg[k].x1, seg[k].x2)

    y = [y0]
    for i in range(2, m + 1):
        y.append(random.uniform(0.0, 1.0))

    g = g_function(y,m,nreal)

    # Setting solution
    for i in range(1, m + 1):
        xi = zcat_search_space(y[i - 1], i)
        assert (-i * 0.5) <= xi <= (i * 0.5)
        x[i - 1] = xi

    for j in range(len(g)):
        xi = zcat_search_space(g[j], m + 1 + j)
        i = m + 1 + j
        assert (-i * 0.5) <= xi <= (i * 0.5)
        x[i - 1] = xi
    return x,m,g_function

'''


nobj = 3
nvars = nobj*10
mop = zcat14

zcat_set(nvars, nobj, 1, 0, 1, 0)

print(zcat_config)
        
x,m,g_function = zcat_rnd_opt_sol(mop,nobj,nvars)
print('x: ', x)
y = zcat_get_y(x, zcat_config['LB'], zcat_config['UB'], nvars)   
print('y: ', y)
g = g_function(y,m,nvars)
print('g: ',g)
beta = zcat_get_beta(y, nobj, m, nvars, g_function, zcat_config)  # Define Distance
print('beta: ', beta)


'''




