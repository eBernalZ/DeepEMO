#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:31:01 2023

@author: Rodolfo Humberto Tamayo
"""

import math


def Thetaj(j, m, n):
    assert 1 <= j <= n - m
    Tj = 2.0 * math.pi * ( j + m - ( m + 1 ) ) / n
    return Tj


def g0(y, m, n):
    assert 0 <= m <= n
    g = [0.2210 for _ in range(n - m)]
    return g


def g1(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = 0.0
        for i in range(1, m + 1):
            sum_val += math.sin(1.5 * math.pi * y[i - 1] + Thetaj(j, m, n))
        g_j = sum_val / (2.0 * m) + 0.5
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g2(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = 0.0
        for i in range(1, m + 1):
            sum_val += pow(y[i - 1], 2.0) * math.sin(4.5 * math.pi * y[i - 1] + Thetaj(j, m, n))
        g_j = sum_val / (2.0 * m) + 0.5
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g3(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = 0.0
        for i in range(1, m + 1):
            sum_val += pow(math.cos(math.pi * y[i - 1] + Thetaj(j, m, n)), 2.0)
        g_j = sum_val / m
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g4(y, m, n):
    g = []

    for j in range(1, n - m + 1):
        mu = sum(y[:m]) / m
        theta_j = Thetaj(j, m, n)
        g_j = (mu / 2.0) * math.cos(4.0 * math.pi * mu + theta_j) + 0.5
        assert 0.0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g5(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = 0.0
        for i in range(1, m + 1):
            sum_val += pow(math.sin(2.0 * math.pi * y[i - 1] - 1 + Thetaj(j, m, n)), 3.0)
        g_j = sum_val / (2.0 * m) + 0.5
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g6(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        s1 = sum(pow(y[i - 1], 2.0) for i in range(1, m + 1)) / m
        s2 = sum(pow(math.cos(11.0 * math.pi * y[i - 1] + Thetaj(j, m, n)), 3.0) for i in range(1, m + 1)) / m
        numerator = -10.0 * math.exp((-2.0 / 5.0) * math.sqrt(s1)) - math.exp(s2) + 10.0 + math.exp(1.0)
        denominator = -10.0 * math.exp(-2.0 / 5.0) - pow(math.exp(1.0), -1) + 10.0 + math.exp(1.0)
        g_j = numerator / denominator
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g7(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        mu = 0
        for i in range(m):
            mu = mu + y[i]
        mu = mu / m
        numerator = mu + math.exp(math.sin(7.0 * math.pi * mu - math.pi / 2.0 + Thetaj(j, m, n))) - math.exp(-1.0)
        denominator = 1.0 + math.exp(1) - math.exp(-1)
        g_j = numerator / denominator
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g8(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = sum(abs(math.sin(2.5 * math.pi * (y[i - 1] - 0.5) + Thetaj(j, m, n))) for i in range(1, m + 1))
        g_j = sum_val / m
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g9(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = sum(abs(math.sin(2.5 * math.pi * y[i - 1] - math.pi / 2.0 + Thetaj(j, m, n))) for i in range(1, m + 1))
        mu = 0
        for i in range(m):
            mu = mu + y[i]
        mu = mu / m
        g_j = mu / 2.0 - sum_val / (2.0 * m) + 0.5
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g


def g10(y, m, n):
    assert 0 <= m <= n
    g = []
    for j in range(1, n - m + 1):
        sum_val = sum(math.sin((4.0 * y[i - 1] - 2.0) * math.pi + Thetaj(j, m, n)) for i in range(1, m + 1))
        g_j = pow(sum_val, 3.0) / (2.0 * pow(m, 3.0)) + 0.5
        assert 0 <= g_j <= 1.0
        g.append(g_j)
    return g

