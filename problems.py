from global_vars import *
import math
from zcat_benckmark import *

## PROBLEM FUNCTIONS ##
def vie1_f1(candidate):
	val = candidate[0]**2 + ((candidate[1]-1)**2)
	return val

def vie1_f2(candidate):
	val = (candidate[0]**2) + ((candidate[1]+1)**2) + 1
	return val

def vie1_f3(candidate):
	val = ((candidate[0]-1)**2) + (candidate[1]**2) + 2
	return val
	
def vie2_f1(candidate):
	val = (((candidate[0] - 2) ** 2) / 2) + (((candidate[1] + 1) ** 2) / 13) + 3
	return val

def vie2_f2(candidate):
	val = (((candidate[0] + candidate[1] - 3) ** 2) / 36) + (((-candidate[0] + candidate[1] + 2) ** 2) / 8) - 17
	return val

def vie2_f3(candidate):
	val = (((candidate[0] + (2 * candidate[1]) - 1) ** 2) / 175) + ((((2 * candidate[1]) - candidate[0]) ** 2) / 17) - 13
	return val

def vie3_f1(candidate):
	val = 0.5 * ((candidate[0] ** 2) + (candidate[1] ** 2)) + math.sin((candidate[0] ** 2) + (candidate[1] ** 2))
	return val

def vie3_f2(candidate):
	val = ((((3 * candidate[0]) - (2 * candidate[1]) + 4) ** 2) / 8) + ((((candidate[0]) - (candidate[1]) + 1) ** 2) / 27) + 15
	return val

def vie3_f3(candidate):
	val = (1 / ((candidate[0] ** 2) + (candidate[1] ** 2) + 1)) + (1.1 * math.exp(-((candidate[0] ** 2) + (candidate[1] ** 2))))
	return val

def dtlz1_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = 0.5 * (1 + dtlz1_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= candidate[i]
	val *= product
	return val

def dtlz1_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	index = USER_M_OBJ - 1 - j
	val = 0.5 * (1 + dtlz1_g(y)) * (1 - candidate[index])
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= candidate[i]
	val *= product
	return val

def dtlz1_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = 0.5 * (1 + dtlz1_g(y)) * (1 - candidate[0])
	return val

def dtlz1_g(y):
	val = 0
	ksum = 0
	for i in range(DTLZ1_K):
		ksum += ((y[i] - 0.5) ** 2) - math.cos(20 * math.pi * (y[i] - 0.5))
	val = 100 * (DTLZ1_K + ksum)
	return val

def dtlz2_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz2_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.cos((math.pi / 2) * candidate[i])
	val *= product
	return val

def dtlz2_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	index = USER_M_OBJ - 1 - j
	val = (1 + dtlz2_g(y)) * (math.sin((math.pi / 2) * candidate[index]))
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.cos((math.pi / 2) * candidate[i])
	val *= product
	return val

def dtlz2_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz2_g(y)) * (math.sin((math.pi / 2) * candidate[0]))
	return val

def dtlz2_g(y):
	val = 0
	for i in range(DTLZ2T6_K):
		val += ((y[i] - 0.5) ** 2)
	return val

def dtlz3_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz3_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.cos((math.pi / 2) * candidate[i])
	val *= product
	return val

def dtlz3_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	index = USER_M_OBJ - 1 - j
	val = (1 + dtlz3_g(y)) * (math.sin((math.pi / 2) * candidate[index]))
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.cos((math.pi / 2) * candidate[i])
	val *= product
	return val

def dtlz3_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz3_g(y)) * (math.sin((math.pi / 2) * candidate[0]))
	return val

def dtlz3_g(y):
	val = 0
	ksum = 0
	for i in range(DTLZ2T6_K):
		ksum += ((y[i] - 0.5) ** 2) - math.cos(20 * math.pi * (y[i] - 0.5))
	val = 100 * (DTLZ2T6_K + ksum)
	return val

def dtlz4_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz4_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.cos((math.pi / 2) * (candidate[i] ** DTLZ4_ALPHA))
	val *= product
	return val

def dtlz4_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	index = USER_M_OBJ - 1 - j
	val = (1 + dtlz4_g(y)) * (math.sin((math.pi / 2) * (candidate[index] ** DTLZ4_ALPHA)))
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.cos((math.pi / 2) * (candidate[i] ** DTLZ4_ALPHA))
	val *= product
	return val

def dtlz4_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz4_g(y)) * (math.sin((math.pi / 2) * (candidate[0] ** DTLZ4_ALPHA)))
	return val

def dtlz4_g(y):
	val = 0
	for i in range(DTLZ2T6_K):
		val += ((y[i] - 0.5) ** 2)
	return val

def dtlz5_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	theta = build_theta(candidate)
	val = (1 + dtlz5_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.cos((math.pi / 2) * theta[i])
	val *= product
	return val

def dtlz5_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	theta = build_theta(candidate)
	index = USER_M_OBJ - 1 - j
	val = (1 + dtlz5_g(y)) * (math.sin((math.pi / 2) * theta[index]))
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.cos((math.pi / 2) * (theta[i]))
	val *= product
	return val

def dtlz5_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz5_g(y)) * (math.sin((math.pi / 2) * candidate[0]))
	return val

def dtlz5_g(y):
	val = 0
	for i in range(DTLZ2T6_K):
		val += ((y[i] - 0.5) ** 2)
	return val

def build_theta(candidate):
	y = candidate[USER_M_OBJ - 1:]
	theta = []
	theta.append(candidate[0])
	for i in range(1, USER_M_OBJ - 1):
		theta.append(((1 + (2 * dtlz5_g(y))) / (2 * (1 + dtlz5_g(y)))) * candidate[i])
	return theta

def dtlz6_f1(candidate):
	y = candidate[USER_M_OBJ - 1:]
	theta = build_theta(candidate)
	val = (1 + dtlz6_g(y))
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.cos((math.pi / 2) * theta[i])
	val *= product
	return val

def dtlz6_fi(candidate, j):
	y = candidate[USER_M_OBJ - 1:]
	theta = build_theta(candidate)
	index = USER_M_OBJ - 1 - j
	val = (1 + dtlz6_g(y)) * (math.sin((math.pi / 2) * theta[index]))
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.cos((math.pi / 2) * (theta[i]))
	val *= product
	return val

def dtlz6_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz6_g(y)) * (math.sin((math.pi / 2) * candidate[0]))
	return val

def dtlz6_g(y):
	val = 0
	for i in range(DTLZ2T6_K):
		val += (y[i] ** 0.1)
	return val

def dtlz7_fi(candidate, j):
	val = candidate[j]
	return val

def dtlz7_fm(candidate):
	y = candidate[USER_M_OBJ - 1:]
	val = (1 + dtlz7_g(y))
	isum = 0
	for i in range(USER_M_OBJ - 1):
		isum += (candidate[i] / (1 + dtlz7_g(y))) * (1 + math.sin(3 * math.pi * candidate[i]))
	val *= (USER_M_OBJ - isum)
	return val

def dtlz7_g(y):
	val = 0
	ksum = 0
	for i in range(DTLZ7_K):
		ksum += y[i]
	val = 1 + (9 / DTLZ7_K) * ksum
	return val

def zcat(candidate, zcat_settings, problem):
	if problem == "ZCAT1":
		val = ZCAT1(candidate, zcat_settings)
	elif problem == "ZCAT2":
		val = ZCAT2(candidate, zcat_settings)
	elif problem == "ZCAT3":
		val = ZCAT3(candidate, zcat_settings)
	elif problem == "ZCAT4":
		val = ZCAT4(candidate, zcat_settings)
	elif problem == "ZCAT5":
		val = ZCAT5(candidate, zcat_settings)
	elif problem == "ZCAT6":
		val = ZCAT6(candidate, zcat_settings)
	elif problem == "ZCAT7":
		val = ZCAT7(candidate, zcat_settings)
	elif problem == "ZCAT8":
		val = ZCAT8(candidate, zcat_settings)
	elif problem == "ZCAT9":
		val = ZCAT9(candidate, zcat_settings)
	elif problem == "ZCAT10":
		val = ZCAT10(candidate, zcat_settings)
	elif problem == "ZCAT11":
		val = ZCAT11(candidate, zcat_settings)
	elif problem == "ZCAT12":
		val = ZCAT12(candidate, zcat_settings)
	elif problem == "ZCAT13":
		val = ZCAT13(candidate, zcat_settings)
	elif problem == "ZCAT14":
		val = ZCAT14(candidate, zcat_settings)
	elif problem == "ZCAT15":
		val = ZCAT15(candidate, zcat_settings)
	elif problem == "ZCAT16":
		val = ZCAT16(candidate, zcat_settings)
	elif problem == "ZCAT17":
		val = ZCAT17(candidate, zcat_settings)
	elif problem == "ZCAT18":
		val = ZCAT18(candidate, zcat_settings)
	elif problem == "ZCAT19":
		val = ZCAT19(candidate, zcat_settings)
	elif problem == "ZCAT20":
		val = ZCAT20(candidate, zcat_settings)
	return val

def wfg1_f1(candidate):
	x = wfg1_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= 1 - math.cos((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg1_fi(candidate, j):
	x = wfg1_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= 1 - math.cos((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * (1 - math.sin((math.pi / 2) * x[USER_M_OBJ - 1 - j]))
	return val

def wfg1_fm(candidate):
	x = wfg1_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * (1 - x[0] - ((math.cos(10 * math.pi * x[0] + math.pi / 2)) / (10 * math.pi)))
	return val

def wfg1_build_x(candidate):
	y = wfg1_build_y(candidate)
	yk_xm = y[WFG_K:]
	kn_xm = [2 * ((i + 1) + WFG_K) for i in range(WFG_L)]
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		weight_min = 2 * minIndex
		weight_max = 2 * maxIndex
		kn_xi = [j for j in range(weight_min, weight_max + 1)]
		val = r_sum(y[minIndex-1:maxIndex], kn_xi)
		x.append(val)
	x.append(r_sum(yk_xm, kn_xm))
	return x

def wfg1_build_y(candidate):
	y = []
	yp = wfg1_build_yp(candidate)
	for i in range(WFG_K + WFG_L):
		val = b_poly(yp[i], 0.02)
		y.append(val)
	return y

def wfg1_build_yp(candidate):
	yp = []
	y2p = wfg1_build_y2p(candidate)
	for i in range(WFG_K):
		yp.append(y2p[i])
	for i in range(WFG_K, WFG_K + WFG_L):
		yp.append(b_flat(y2p[i], 0.8, 0.75, 0.85))
	return yp

def wfg1_build_y2p(candidate):
	y2p = []
	for i in range(WFG_K):
		y2p.append(candidate[i] / (2 * (i + 1)))
	
	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(candidate[i] / (2 * (i + 1)), 0.35)
		y2p.append(val)
	return y2p

def wfg2_f1(candidate):
	x = wfg2_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= 1 - math.cos((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg2_fi(candidate, j):
	x = wfg2_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= 1 - math.cos((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * (1 - math.sin((math.pi / 2) * x[USER_M_OBJ - 1 - j]))
	return val

def wfg2_fm(candidate):
	x = wfg2_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * (1 - x[0] * (math.cos(5 * x[0] * math.pi) ** 2))
	return val

def wfg2_build_x(candidate):
	y = wfg2_build_y(candidate)
	yk_xm = y[WFG_K:(WFG_K + WFG_L)//2]
	kn_xm = [1 for _ in range(WFG_K, ((WFG_K + WFG_L)//2) + 1)]
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		kn_xi = [1 for _ in range(minIndex, maxIndex + 1)]
		val = r_sum(y[minIndex-1:maxIndex], kn_xi)
		x.append(val)
	x.append(r_sum(yk_xm, kn_xm))
	return x

def wfg2_build_y(candidate):
	yp = wfg2_build_yp(candidate)
	y = yp[:WFG_K]
	for i in range(WFG_K+1, ((WFG_K+WFG_L) // 2) + 1):
		val = r_nonsep([yp[WFG_K + 2 * (i - WFG_K) - 1], yp[WFG_K + 2 * (i - WFG_K)]], 2)
		y.append(val)
	return y

def wfg2_build_yp(candidate):
	yp = []
	for i in range(WFG_K):
		yp.append(candidate[i] / (2 * (i + 1)))
	
	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(candidate[i] / (2 * (i + 1)), 0.35)
		yp.append(val)
	return yp

def wfg3_f1(candidate):
	x = wfg3_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= x[i]
	val += 2 * product
	return val

def wfg3_fi(candidate, j):
	x = wfg3_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= x[i]
	val += 2 * (j+1) * product * (1 - x[USER_M_OBJ - 1 - j])
	return val

def wfg3_fm(candidate):
	x = wfg3_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * (1 - x[0])
	return val

def wfg3_build_x(candidate):
	y = wfg3_build_y(candidate)
	yk_xm = y[WFG_K:(WFG_K + WFG_L)//2]
	kn_xm = [1 for _ in range(WFG_K, ((WFG_K + WFG_L)//2) + 1)]
	xm = r_sum(yk_xm, kn_xm)
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		kn_xi = [1 for _ in range(minIndex, maxIndex + 1)]
		ui = r_sum(y[minIndex-1:maxIndex], kn_xi)
		if (i > 0):
			xi = xm * (ui - 0.5) + 0.5
		else:
			xi = ui
		x.append(xi)
	x.append(xm)
	return x

def wfg3_build_y(candidate):
	yp = wfg3_build_yp(candidate)
	y = yp[:WFG_K]
	for i in range(WFG_K+1, ((WFG_K+WFG_L) // 2) + 1):
		val = r_nonsep([yp[WFG_K + 2 * (i - WFG_K) - 1], yp[WFG_K + 2 * (i - WFG_K)]], 2)
		y.append(val)
	return y

def wfg3_build_yp(candidate):
	yp = []
	for i in range(WFG_K):
		yp.append(candidate[i] / (2 * (i + 1)))
	
	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(candidate[i] / (2 * (i + 1)), 0.35)
		yp.append(val)
	return yp

def wfg4_f1(candidate):
	x = wfg4_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg4_fi(candidate, j):
	x = wfg4_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg4_fm(candidate):
	x = wfg4_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg4_build_x(candidate):
	y = wfg4_build_y(candidate)
	yk_xm = y[WFG_K:]
	kn_xm = [1 for _ in range(WFG_K, WFG_K + WFG_L)]
	xm = r_sum(yk_xm, kn_xm)
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		kn_xi = [1 for _ in range(minIndex, maxIndex + 1)]
		xi = r_sum(y[minIndex-1:maxIndex], kn_xi)
		x.append(xi)
	x.append(xm)
	return x

def wfg4_build_y(candidate):
	y = []
	for i in range(WFG_K + WFG_L):
		val = s_multi((candidate[i] / (2 * (i+1))), 30, 10, 0.35)
		y.append(val)
	return y

def wfg5_f1(candidate):
	x = wfg5_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg5_fi(candidate, j):
	x = wfg5_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg5_fm(candidate):
	x = wfg5_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg5_build_x(candidate):
	y = wfg5_build_y(candidate)
	yk_xm = y[WFG_K:]
	kn_xm = [1 for _ in range(WFG_K, WFG_K + WFG_L)]
	xm = r_sum(yk_xm, kn_xm)
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		kn_xi = [1 for _ in range(minIndex, maxIndex + 1)]
		xi = r_sum(y[minIndex-1:maxIndex], kn_xi)
		x.append(xi)
	x.append(xm)
	return x

def wfg5_build_y(candidate):
	y = []
	for i in range(WFG_K + WFG_L):
		val = s_decept((candidate[i] / (2 * (i+1))), 0.35, 0.001, 0.05)
		y.append(val)
	return y

def wfg6_f1(candidate):
	x = wfg6_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg6_fi(candidate, j):
	x = wfg6_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg6_fm(candidate):
	x = wfg6_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg6_build_x(candidate):
	y = wfg6_build_y(candidate)
	yk_xm = y[WFG_K:]	
	xm = r_nonsep(yk_xm, WFG_L)
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		gap = (WFG_K // (USER_M_OBJ - 1))
		# print(y[minIndex-1:maxIndex])
		# print(y[minIndex:maxIndex])
		xi = r_nonsep(y[minIndex-1:maxIndex], gap)
		x.append(xi)
	x.append(xm)
	return x

def wfg6_build_y(candidate):
	y = []
	for i in range(WFG_K):
		val = candidate[i] / (2 * (i + 1))
		y.append(val)

	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(candidate[i] / (2 * (i + 1)), 0.35)
		y.append(val)
	return y

def wfg7_f1(candidate):
	x = wfg7_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg7_fi(candidate, j):
	x = wfg7_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg7_fm(candidate):
	x = wfg7_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg7_build_x(candidate):
	y = wfg7_build_y(candidate)
	yk_xm = y[WFG_K:]	
	xm = r_sum(yk_xm, [1 for _ in range(len(yk_xm))])
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		ones = [1 for _ in range(minIndex, maxIndex + 1)]
		xi = r_sum(y[minIndex-1:maxIndex], ones)
		x.append(xi)
	x.append(xm)
	return x

def wfg7_build_y(candidate):
	yp = wfg7_build_yp(candidate)
	y = yp[:WFG_K]

	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(yp[i], 0.35)
		y.append(val)
	return y

def wfg7_build_yp(candidate):
	yp = []
	for i in range(WFG_K):
		z_arr = []
		for j in range(i+1, WFG_K + WFG_L):
			z_arr.append(candidate[j] / (2 * (j + 1)))
		ones = [1 for _ in range(len(z_arr))]
		rsum = r_sum(z_arr, ones)
		val = b_param((candidate[i] / (2 * (i + 1))), rsum, 0.98/49.98, 0.02, 50)
		yp.append(val)

	for i in range(WFG_K, WFG_K + WFG_L):
		val = candidate[i] / (2 * (i + 1))
		yp.append(val)
	return yp

def wfg8_f1(candidate):
	x = wfg8_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg8_fi(candidate, j):
	x = wfg8_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg8_fm(candidate):
	x = wfg8_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg8_build_x(candidate):
	y = wfg8_build_y(candidate)
	yk_xm = y[WFG_K:]	
	xm = r_sum(yk_xm, [1 for _ in range(len(yk_xm))])
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		ones = [1 for _ in range(minIndex, maxIndex + 1)]
		xi = r_sum(y[minIndex-1:maxIndex], ones)
		x.append(xi)
	x.append(xm)
	return x

def wfg8_build_y(candidate):
	yp = wfg8_build_yp(candidate)
	y = yp[:WFG_K]

	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_linear(yp[i], 0.35)
		y.append(val)
	return y

def wfg8_build_yp(candidate):
	yp = []
	for i in range(WFG_K):
		val = candidate[i] / (2 * (i + 1))
		yp.append(val)

	for i in range(WFG_K, WFG_K + WFG_L):
		z_arr = []
		for j in range(i-1):
			z_arr.append(candidate[j] / (2 * (j+1)))
		ones = [1 for _ in range(len(z_arr))]
		rsum = r_sum(z_arr, ones)
		val = b_param((candidate[i] / (2 * (i + 1))), rsum, 0.98/49.98, 0.02, 50)
		yp.append(val)
	return yp

def wfg9_f1(candidate):
	x = wfg9_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * product
	return val

def wfg9_fi(candidate, j):
	x = wfg9_build_x(candidate)
	val = x[-1]
	product = 1
	for i in range(USER_M_OBJ - 1 - j):
		product *= math.sin((math.pi / 2) * x[i])
	val += 2 * (j+1) * product * math.cos((math.pi / 2) * x[USER_M_OBJ - 1 - j])
	return val

def wfg9_fm(candidate):
	x = wfg9_build_x(candidate)
	val = x[-1] + (2 * USER_M_OBJ) * math.cos((math.pi / 2) * x[0])
	return val

def wfg9_build_x(candidate):
	y = wfg9_build_y(candidate)
	yk_xm = y[WFG_K:]	
	xm = r_nonsep(yk_xm, WFG_L)
	x = []
	for i in range(USER_M_OBJ - 1):
		minIndex = int(((i+1) - 1) * WFG_K / (USER_M_OBJ - 1) + 1)
		maxIndex = int((i+1) * WFG_K / (USER_M_OBJ - 1))
		gap = (WFG_K // (USER_M_OBJ - 1))
		xi = r_nonsep(y[minIndex-1:maxIndex], gap)
		x.append(xi)
	x.append(xm)
	return x

def wfg9_build_y(candidate):
	yp = wfg9_build_yp(candidate)
	y = []
	for i in range(WFG_K):
		val = s_decept(yp[i], 0.35, 0.001, 0.05)
		y.append(val)

	for i in range(WFG_K, WFG_K + WFG_L):
		val = s_multi(yp[i], 30, 95, 0.35)
		y.append(val)
	return y

def wfg9_build_yp(candidate):
	yp = []
	for i in range((WFG_K + WFG_L) - 1):
		z_arr = []
		for j in range(i+1, WFG_K + WFG_L):
			z_arr.append(candidate[j] / (2 * (j+1)))
		ones = [1 for _ in range(len(z_arr))]
		rsum = r_sum(z_arr, ones)
		val = b_param((candidate[i] / (2 * (i + 1))), rsum, 0.98/49.98, 0.02, 50)
		yp.append(val)
		
	yp.append(candidate[-1] / (2 * (WFG_K + WFG_L)))
	return yp

def imop_y1(candidate):
	val = 0
	for i in range(IMOP_K):
		val += candidate[i]
	val *= 1 / IMOP_K
	return val ** IMOP_A1

def imop_y2(candidate):
	val = 0
	for i in range(math.ceil(IMOP_K / 2)):
		val += candidate[i]
	val *= 1 / (math.ceil(IMOP_K / 2))
	return val ** IMOP_A2

def imop_y3(candidate):
	val = 0
	for i in range(math.ceil(IMOP_K / 2), IMOP_K):
		val += candidate[i]
	val *= 1 / (math.floor(IMOP_K / 2))
	return val ** IMOP_A3

def imop_g(candidate):
	val = 0
	for i in range(IMOP_K, IMOP_K + IMOP_L):
		val += (candidate[i] - 0.5) ** 2
	return val

def imop1_f1(candidate):
	y1 = imop_y1(candidate)
	g = imop_g(candidate)
	return g + (math.cos((math.pi / 2) * y1) ** 8)

def imop1_f2(candidate):
	y1 = imop_y1(candidate)
	g = imop_g(candidate)
	return g + (math.sin((math.pi / 2) * y1) ** 8)

def imop2_f1(candidate):
	y1 = imop_y1(candidate)
	g = imop_g(candidate)
	return g + (math.cos((math.pi / 2) * y1) ** 0.5)

def imop2_f2(candidate):
	y1 = imop_y1(candidate)
	g = imop_g(candidate)
	return g + (math.sin((math.pi / 2) * y1) ** 0.5)

def imop3_f1(candidate):
	g = imop_g(candidate)
	y1 = imop_y1(candidate)
	return g + 1 + ((1 / 5) * math.cos(10 * math.pi * y1)) - y1

def imop3_f2(candidate):
	g = imop_g(candidate)
	y1 = imop_y1(candidate)
	return g + y1

def imop4_f1(candidate):
	g = imop_g(candidate)
	y1 = imop_y1(candidate)
	return (1 + g) * y1

def imop4_f2(candidate):
	g = imop_g(candidate)
	y1 = imop_y1(candidate)
	return (1 + g) * (y1 + ((1 / 10) * math.sin(10 * math.pi * y1)))

def imop4_f3(candidate):
	g = imop_g(candidate)
	y1 = imop_y1(candidate)
	return (1 + g) * (1 - y1)

def imop5_h1(candidate):
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	val = (0.4 * math.cos((math.pi / 4) * math.ceil(8 * y2))) + (0.1 * y3 * math.cos(16 * math.pi * y2))
	return val

def imop5_h2(candidate):
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	val = (0.4 * math.sin((math.pi / 4) * math.ceil(8 * y2))) + (0.1 * y3 * math.sin(16 * math.pi * y2))
	return val

def imop5_f1(candidate):
	g = imop_g(candidate)
	h1 = imop5_h1(candidate)
	return g + h1

def imop5_f2(candidate):
	g = imop_g(candidate)
	h2 = imop5_h2(candidate)
	return g + h2

def imop5_f3(candidate):
	g = imop_g(candidate)
	h1 = imop5_h1(candidate)
	h2 = imop5_h2(candidate)
	return g + 0.5 - h1 - h2

def imop6_r(candidate):
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	val1 = (math.sin(3 * math.pi * y2) ** 2)
	val2 = (math.sin(3 * math.pi * y3) ** 2)
	return max(0, min(val1, val2) - 0.5)

def imop6_f1(candidate):
	g = imop_g(candidate)
	y2 = imop_y2(candidate)
	r = imop6_r(candidate)
	return (1 + g) * y2 + math.ceil(r)

def imop6_f2(candidate):
	g = imop_g(candidate)
	y3 = imop_y3(candidate)
	r = imop6_r(candidate)
	return (1 + g) * y3 + math.ceil(r)

def imop6_f3(candidate):
	g = imop_g(candidate)
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	r = imop6_r(candidate)
	return (0.5 + g) * (2 - y2 - y3) + math.ceil(r)

def imop7_h1(candidate):
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	g = imop_g(candidate)
	val = (1 + g) * math.cos((math.pi / 2) * y2) * math.cos((math.pi / 2) * y3)
	return val

def imop7_h2(candidate):
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	g = imop_g(candidate)
	val = (1 + g) * math.cos((math.pi / 2) * y2) * math.sin((math.pi / 2) * y3)
	return val

def imop7_h3(candidate):
	y2 = imop_y2(candidate)
	g = imop_g(candidate)
	val = (1 + g) * math.sin((math.pi / 2) * y2)
	return val

def imop7_r(candidate):
	h1 = imop7_h1(candidate)
	h2 = imop7_h2(candidate)
	h3 = imop7_h3(candidate)
	val_min1 = abs(h1 - h2)
	val_min2 = abs(h2 - h3)
	val_min3 = abs(h3 - h1)
	return min(min(val_min1, val_min2), val_min3)

def imop7_f1(candidate):
	r = imop7_r(candidate)
	h1 = imop7_h1(candidate)
	return h1 + 10 * max(0, r - 0.1)

def imop7_f2(candidate):
	r = imop7_r(candidate)
	h2 = imop7_h2(candidate)
	return h2 + 10 * max(0, r - 0.1)

def imop7_f3(candidate):
	r = imop7_r(candidate)
	h3 = imop7_h3(candidate)
	return h3 + 10 * max(0, r - 0.1)

def imop8_f1(candidate):
	y2 = imop_y2(candidate)
	return y2

def imop8_f2(candidate):
	y3 = imop_y3(candidate)
	return y3

def imop8_f3(candidate):
	g = imop_g(candidate)
	y2 = imop_y2(candidate)
	y3 = imop_y3(candidate)
	ys = [y2, y3]
	summation = 0
	for y in ys:
		summation += ((y * (1 + math.sin(19 * math.pi * y))) / (1 + g))
	return (1 + g) * (3 - summation)

## TRANSFORMATION FUNCTIONS ##
def s_linear(y, A):
	return abs(y - A) / abs(math.floor(A - y) + A)

def b_poly(y, a):
	return y ** a

def b_flat(y, A, B, C):
	return A + min(0, math.floor(y - B)) * ((A * (B - y)) / B) - min(0, math.floor(C - y)) * (((1 - A) * (y - C)) / (1 - C))

def b_param(y, u, A, B, C):
	return y ** (B + (C - B) * (A - (1 - 2 * u) * (abs(math.floor(0.5 - u) + A))))

def r_sum(y, w):
	numerator = 0
	denominator = 0
	for i in range(len(y)):
		numerator += w[i] * y[i]
		denominator += w[i]
	return numerator / denominator
	
def r_nonsep(y, A):
	val = 0
	jsum = 0
	denominator = ((len(y) / A) * math.ceil(A / 2) * (1 + 2 * A - 2 * math.ceil(A / 2)))
	for j in range(len(y)):
		ksum = 0
		for k in range(A - 2):
			ksum += abs(y[j] - y[(j + k + 1) % len(y)])
		jsum += y[j] + ksum
	val = jsum / denominator
	return val

def s_multi(y, A, B, C):
	numerator = (1 + math.cos((4 * A + 2) * math.pi * (0.5 - ((abs(y-C)) / (2 * (math.floor(C - y) + C))))) + (4 * B * (((abs(y - C)) / (2 * (math.floor(C - y) + C))) ** 2)))
	denominator = B + 2
	return numerator / denominator

def s_decept(y, A, B, C):
	numerator1 = math.floor(y - A + B) * (1 - C + ((A - B) / (B)))
	frac1 = numerator1 / (A - B)
	numerator2 = math.floor(A + B - y) * (1 - C + ((1 - A - B) / (B)))
	frac2 = numerator2 / (1 - A - B)
	return 1 + (abs(y - A) - B) * (frac1 + frac2 + (1 / B)) 