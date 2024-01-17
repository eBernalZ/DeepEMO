#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:42:49 2023

@author: macbook
"""
import math


def Z1(J):
    assert len(J) > 0
    Z = 0.0
    for value in J:
        Z += value * value
    Z = (10.0 / len(J)) * Z
    assert 0 <= Z <= 10.0
    return Z


def Z2(J):
    assert len(J) > 0
    Z = -float("inf")
    for value in J:
        Z = max(Z, abs(value))
    Z = 10.0 * Z
    assert 0 <= Z <= 10.0
    return Z


def Z3(J):
    assert len(J) > 0
    k = 5.0
    Z = 0.0
    for value in J:
        Z += (pow(value, 2.0) - math.cos((2.0 * k - 1) * math.pi * value) + 1.0) / 3.0
    Z = (10.0 / len(J)) * Z
    assert 0 <= Z <= 10.0
    return Z


def Z4(J):
    assert len(J) > 0
    k = 5.0
    Z = 0.0
    pow1 = -float("inf")
    pow2 = 0.0
    for value in J:
        pow1 = max(pow1, abs(value))
        pow2 += 0.5 * (math.cos((2.0 * k - 1) * math.pi * value) + 1.0)
    Z = (10.0 / (2.0 * math.exp(1) - 2.0)) * (math.exp(pow(pow1, 0.5)) - math.exp(pow2 / len(J)) - 1.0 + math.exp(1))
    assert 0 <= Z <= 10.0
    return Z


def Z5(J):
    assert len(J) > 0
    Z = 0.0
    for value in J:
        Z += pow(abs(value), 0.002)
    Z = -0.7 * Z3(J) + (10.0 / len(J)) * Z
    assert 0 <= Z <= 10.0
    return Z


def Z6(J):
    assert len(J) > 0
    Z = 0.0
    for value in J:
        Z += abs(value)
    Z = -0.7 * Z4(J) + 10.0 * pow(Z / len(J), 0.002)
    assert 0 <= Z <= 10.0
    return Z


def Zbias(z):
    gamma = 0.05
    w = pow(abs(z), gamma)
    assert 0.0 <= w <= 1.0
    return w




