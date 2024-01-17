#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:48:19 2023

@author: Rodolfo Humberto Tamayo
"""

import sys
from math import fabs
import random


def zcat_fix_to_01(a):
    epsilon = sys.float_info.epsilon
    min_val = 0.0
    max_val = 1.0

    min_epsilon = min_val - epsilon
    max_epsilon = max_val + epsilon

    if a <= min_val and a >= min_epsilon:
        return min_val
    elif a >= max_val and a <= max_epsilon:
        return max_val
    else:
        return a
    

def zcat_lq(y, z):
    epsilon = sys.float_info.epsilon

    if y < z:
        return 1
    if fabs(z - y) < epsilon:
        return 1
    return 0


def zcat_value_in(y, lb, ub):
    return 1 if zcat_lq(lb, y) and zcat_lq(y, ub) else 0


def zcat_forall_value_in(y, m, lb, ub):
    for i in range(m):
        if zcat_value_in(y[i], lb, ub) == 0:
            return 0
    return 1


def rnd_real(lb, ub):
    assert lb < ub
    rnd = random.random()
    rnd = rnd * (ub - lb) + lb
    assert lb <= rnd <= ub
    return rnd


def rnd_perc():
    return rnd_real(0.0, 1.0)


def rnd_int(lb, ub):
    assert lb <= ub
    r = random.randint(lb, ub)
    return r

