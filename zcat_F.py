#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:22:59 2023

@author: Rodolfo Humberto Tamayo
"""

import math
import numpy as np

from zcat_tools import *


def F1(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for i in range(1, M):
        F[0] *= math.sin(y[i - 1] * math.pi / 2)

    for j in range(2, M):
        F[j - 1] = 1.0
        for i in range(1, M - j + 1):
            F[j - 1] *= math.sin(y[i - 1] * math.pi / 2)
        F[j - 1] *= math.cos(y[M - j] * math.pi / 2)
    F[M - 1] = 1.0 - math.sin(y[0] * math.pi / 2)
    return F


def F2(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for i in range(1, M):
        F[0] *= 1.0 - np.cos(y[i - 1] * np.pi / 2)
    assert 0 <= F[0] <= 1.0

    for j in range(2, M):
        F[j - 1] = 1.0
        for i in range(1, M - j + 1):
            F[j - 1] *= 1.0 - np.cos(y[i - 1] * np.pi / 2)
        F[j - 1] *= 1.0 - np.sin(y[M - j] * np.pi / 2)
        assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = 1.0 - np.sin(y[0] * np.pi / 2)
    assert 0 <= F[M - 1] <= 1.0
    return F



def F3(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for j in range(M - 1):
        F[0] = 0.0
        for i in range(1, M):
            F[0] += y[i - 1]
        F[0] /= M - 1
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 0.0
            for i in range(1, M - j + 1):
                F[j - 1] += y[i - 1]
            F[j - 1] += (1 - y[M - j])
            F[j - 1] /= M - j + 1
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = 1 - y[0]
    assert 0 <= F[M - 1] <= 1.0
    return F



def F4(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for j in range(1, M):
        F[j - 1] = y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    sum_val = 0.0
    for i in range(1, M):
        sum_val += y[i - 1]

    F[M - 1] = 1.0 - sum_val / (M - 1)
    assert 0 <= F[M - 1] <= 1.0
    return F



def F5(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for j in range(1, M):
        F[j - 1] = y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    sum_val = 0.0
    for i in range(1, M):
        sum_val += 1.0 - y[i - 1]

    numerator = math.pow(math.exp(sum_val / (M - 1)), 8.0) - 1.0
    denominator = math.pow(math.exp(1), 8.0) - 1.0

    F[M - 1] = numerator / denominator
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F6(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 40.0
    r = 0.05
    
    for j in range(1, M):
        F[j - 1] = y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    mu = 0.0
    for i in range(1, M):
        mu += y[i - 1]
    mu /= M - 1

    numerator = (math.pow(1 + math.exp(2 * k * mu - k), -1.0) - r * mu
                 - math.pow(1 + math.exp(k), -1.0) + r)
    denominator = (math.pow(1 + math.exp(-k), -1.0) - math.pow(1 + math.exp(k), -1.0) + r)

    F[M - 1] = numerator / denominator
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F7(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for j in range(1, M):
        F[j - 1] = y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    sum_val = 0.0
    for i in range(1, M):
        sum_val += math.pow(0.5 - y[i - 1], 5)
    sum_val = sum_val / (2 * (M - 1) * math.pow(0.5, 5))

    F[M - 1] = sum_val + 0.5
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F8(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for i in range(M - 1):
        F[0] = 1.0
        for i in range(1, M):
            F[0] *= 1.0 - math.sin(y[i - 1] * math.pi / 2)
        F[0] = 1.0 - F[0]
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 1.0
            for i in range(1, M - j + 1):
                F[j - 1] *= 1.0 - math.sin(y[i - 1] * math.pi / 2)
            F[j - 1] *= 1.0 - math.cos(y[M - j] * math.pi / 2)
            F[j - 1] = 1.0 - F[j - 1]
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = math.cos(y[0] * math.pi / 2)
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F9(y, M):
    F = [0.0] * M
    F[0] = 1.0
    for i in range(M - 1):
        F[0] = 0.0
        for i in range(1, M):
            F[0] += math.sin(y[i - 1] * math.pi / 2)
        F[0] = F[0] / (M - 1)
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 0.0
            for i in range(1, M - j + 1):
                F[j - 1] += math.sin(y[i - 1] * math.pi / 2)
            F[j - 1] += math.cos(y[M - j] * math.pi / 2)
            F[j - 1] = F[j - 1] / (M - j + 1)
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = math.cos(y[0] * math.pi / 2)
    assert 0 <= F[M - 1] <= 1.0
    return F



def F10(y, M):
    F = [0.0] * M
    F[0] = 1.0
    sum_val = 0.0
    r = 0.02
    for j in range(1, M):
        sum_val += 1 - y[j - 1]
        F[j - 1] = y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    numerator = math.pow(r, -1) - math.pow(sum_val / (M - 1) + r, -1)
    denominator = math.pow(r, -1) - math.pow(1 + r, -1)

    F[M - 1] = numerator / denominator
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F11(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 4.0
    
    for i in range(M - 1):
        F[0] = 0.0
        for i in range(1, M):
            F[0] += y[i - 1]
        F[0] = F[0] / (M - 1)
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 0.0
            for i in range(1, M - j + 1):
                F[j - 1] += y[i - 1]
            F[j - 1] += (1 - y[M - j])
            F[j - 1] = F[j - 1] / (M - j + 1)
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = (math.cos((2 * k - 1) * y[0] * math.pi) + 2 * y[0] + 4 * k * (1 - y[0]) - 1) / (4 * k)
    assert 0 <= F[M - 1] <= 1.0
    return F



def F12(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 3.0
    
    for i in range(M - 1):
        F[0] = 1.0
        for i in range(1, M):
            F[0] *= (1.0 - y[i - 1])
        F[0] = 1.0 - F[0]
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 1.0
            for i in range(1, M - j + 1):
                F[j - 1] *= (1.0 - y[i - 1])
            F[j - 1] *= y[M - j]
            F[j - 1] = 1.0 - F[j - 1]
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = (math.cos((2 * k - 1) * y[0] * math.pi) + 2 * y[0] + 4 * k * (1 - y[0]) - 1) / (4 * k)
    assert 0 <= F[M - 1] <= 1.0
    return F


def F13(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 3.0
    
    for i in range(M - 1):
        F[0] = 0.0
        for i in range(1, M):
            F[0] += math.sin(y[i - 1] * math.pi / 2)
        F[0] = 1.0 - F[0] / (M - 1.0)
        assert 0 <= F[0] <= 1.0

        for j in range(2, M):
            F[j - 1] = 0.0
            for i in range(1, M - j + 1):
                F[j - 1] += math.sin(y[i - 1] * math.pi / 2)
            F[j - 1] += math.cos(y[M - j] * math.pi / 2)
            F[j - 1] = 1.0 - F[j - 1] / (M - j + 1)
            assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = 1.0 - (math.cos((2 * k - 1) * y[0] * math.pi) + 2 * y[0] + 4 * k * (1 - y[0]) - 1) / (4.0 * k)
    assert 0 <= F[M - 1] <= 1.0
    return F



def F14(y, M):
    F = [0.0] * M
    F[0] = 1.0
    F[0] = math.pow(math.sin(y[0] * math.pi / 2), 2.0)
    assert 0 <= F[0] <= 1.0

    for j in range(2, M - 1):
        F[j - 1] = math.pow(math.sin(y[0] * math.pi / 2), 2.0 + (j - 1.0) / (M - 2.0))
        assert 0 <= F[j - 1] <= 1.0

    if M > 2:
        F[M - 2] = 0.5 * (1 + math.sin(6 * y[0] * math.pi / 2 - math.pi / 2))
        assert 0 <= F[M - 2] <= 1.0

    F[M - 1] = math.cos(y[0] * math.pi / 2)
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F15(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 3.0
    
    for j in range(1, M):
        F[j - 1] = math.pow(y[0], 1.0 + (j - 1.0) / (4.0 * M))
        assert 0 <= F[j - 1] <= 1.0

    F[M - 1] = (math.cos((2 * k - 1) * y[0] * math.pi) + 2 * y[0] + 4 * k * (1 - y[0]) - 1) / (4 * k)
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F16(y, M):
    F = [0.0] * M
    F[0] = 1.0
    k = 5
    
    F[0] = math.sin(y[0] * math.pi / 2)
    assert 0 <= F[0] <= 1.0

    for j in range(2, M - 1):
        F[j - 1] = math.pow(math.sin(y[0] * math.pi / 2), 1.0 + (j - 1.0) / (M - 2.0))
        assert 0 <= F[j - 1] <= 1.0

    if M > 2:
        F[M - 2] = 0.5 * (1 + math.sin(10 * y[0] * math.pi / 2 - math.pi / 2))
        assert 0 <= F[M - 2] <= 1.0

    F[M - 1] = (math.cos((2 * k - 1) * y[0] * math.pi) + 2 * y[0] + 4 * k * (1 - y[0]) - 1) / (4 * k)
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    
    

def F17(y, M):
    F = [0.0] * M
    F[0] = 1.0
    wedge_flag = zcat_forall_value_in(y, M - 1, 0.0, 0.5)
    sum_val = 0.0

    for j in range(1, M):
        if wedge_flag:
            F[j - 1] = y[0]
        else:
            F[j - 1] = y[j - 1]
            sum_val += 1 - y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    if wedge_flag:
        F[M - 1] = (math.pow(math.exp(1.0 - y[0]), 8.0) - 1.0) / (math.pow(math.exp(1.0), 8.0) - 1.0)
    else:
        F[M - 1] = (math.pow(math.exp(sum_val / (M - 1)), 8.0) - 1.0) / (math.pow(math.exp(1.0), 8.0) - 1.0)
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F18(y, M):
    F = [0.0] * M
    F[0] = 1.0
    f1 = zcat_forall_value_in(y, M - 1, 0.0, 0.4)
    f2 = zcat_forall_value_in(y, M - 1, 0.6, 1.0)
    wedge_flag = 1 if f1 == 1 or f2 == 1 else 0

    sum_val = 0.0
    for j in range(1, M):
        if wedge_flag:
            F[j - 1] = y[0]
        else:
            F[j - 1] = y[j - 1]
            sum_val += math.pow(0.5 - y[j - 1], 5.0)
        assert 0 <= F[j - 1] <= 1.0

    if wedge_flag:
        F[M - 1] = (math.pow(0.5 - y[0], 5.0) + math.pow(0.5, 5.0)) / (2.0 * math.pow(0.5, 5.0))
    else:
        F[M - 1] = sum_val / (2.0 * (M - 1.0) * math.pow(0.5, 5.0)) + 0.5
    assert 0 <= F[M - 1] <= 1.0
    return F
    
    

def F19(y, M):
    F = [0.0] * M
    F[0] = 1.0
    A = 5.0
    mu = 0.0
    
    flag_deg = zcat_value_in(y[0], 0.0, 0.2) or zcat_value_in(y[0], 0.4, 0.6)

    for j in range(1, M):
        if flag_deg:
            F[j - 1] = y[0]
        else:
            F[j - 1] = y[j - 1]
        F[j - 1] = zcat_fix_to_01(F[j - 1])
        assert 0 <= F[j - 1] <= 1.0
        mu += y[j - 1]

    if flag_deg:
        mu = y[0]
    else:
        mu /= (M - 1)

    F[M - 1] = 1.0 - mu - math.cos(2.0 * A * math.pi * mu + math.pi / 2) / (2.0 * A * math.pi)
    F[M - 1] = zcat_fix_to_01(F[M - 1])
    assert 0 <= F[M - 1] <= 1.0
    return F




def F20(y, M):
    F = [0.0] * M
    F[0] = 1.0
    deg_flag = zcat_value_in(y[0], 0.1, 0.4) or zcat_value_in(y[0], 0.6, 0.9)

    sum_val = 0.0
    for j in range(1, M):
        sum_val += math.pow(0.5 - y[j - 1], 5.0)
        F[j - 1] = y[0] if deg_flag else y[j - 1]
        assert 0 <= F[j - 1] <= 1.0

    if deg_flag:
        F[M - 1] = (math.pow(0.5 - y[0], 5.0) + math.pow(0.5, 5.0)) / (2.0 * math.pow(0.5, 5.0))
    else:
        F[M - 1] = sum_val / (2.0 * (M - 1.0) * math.pow(0.5, 5.0)) + 0.5
    assert 0 <= F[M - 1] <= 1.0
    return F



