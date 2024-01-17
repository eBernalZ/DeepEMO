from hv import HyperVolume
import math
import numpy as np
from global_vars import *
import scipy.linalg as la

P_VAL = 2
S_VAL = USER_M_OBJ + 1
THETA = 10
# Solow Polasky se maximiza y no debe ser mayor a N
# VERIFIED
def hv_de(A, ref_point, normalized = True):
	hv = HyperVolume(ref_point)
	total = hv.compute(A, normalized)
	C = [0 for _ in range(len(A))]
	a = A.copy()

	for i in range(len(A)):
		a.pop(i)
		C[i] = abs(total - hv.compute(a, normalized))
		a.insert(i, A[i])
	return C, total

# VERIFIED
def gd(A, Z, p = False, normalized = True):
	memoization = [[[math.inf, math.inf] for _ in range(2)] for _ in range(len(A))]
	pre = 1 / len(A)
	outer_sum = 0
	for i, a in enumerate(A):
		minz = math.inf
		for z in Z:
			inner_sum = 0
			for j in range(len(a.normalized_fitness)):
				if normalized:
					inner_sum += ((a.normalized_fitness[j] - z.normalized_fitness[j]) ** 2)
				else:
					inner_sum += ((a.fitness[j] - z.fitness[j]) ** 2)
			inner_sum = math.sqrt(inner_sum)
			if inner_sum < minz:
				memoization[i][1] = memoization[i][0].copy()
				memoization[i][0][0] = inner_sum
				memoization[i][0][1] = z
				minz = inner_sum
			elif inner_sum < memoization[i][1][0]:
				memoization[i][1][0] = inner_sum
				memoization[i][1][1] = z
		outer_sum += (minz ** P_VAL)
		if p:
			total = (pre * outer_sum) ** (1 / P_VAL)
		else:
			total = pre * (outer_sum ** (1 / P_VAL))

	return total, memoization

# VERIFIED
def gd_de(A, Z, p = False, normalized = True):
	C = [0 for _ in range(len(A))]
	total, memo = gd(A, Z, p=p, normalized=normalized)
	# print("Total GD: ", total)
	for i in range(len(A)):
		v = 0
		for j in range(len(A)):
			if (i != j):
				v += (memo[j][0][0] ** P_VAL)
		v = v / len(Z)
		v = v ** (1 / P_VAL)
		C[i] = abs(total - v)
	return C, total

# VERIFIED
def igd(A, Z, p = False, normalized = True):
	total, memoization = gd(Z, A, p=p, normalized=normalized)
	return total, memoization

# VERIFIED
def igd_de(A, Z, normalized = True):
	C = [0 for _ in range(len(A))]
	total, memo = igd(A, Z, normalized=normalized)
	# print("Total IGD: ", total)
	for i in range(len(A)):
		v = 0
		for j in range(len(Z)):
			if A[i] == memo[j][0][1]:
				v += (memo[j][1][0] ** P_VAL)
			else:
				v += (memo[j][0][0] ** P_VAL)
		v = v / len(Z)
		v = v ** (1 / P_VAL)
		C[i] = abs(total - v)
	return C, total

# VERIFIED
def igdplus(A, Z, normalized = True):
	memoization = [[[math.inf, math.inf] for _ in range(2)] for _ in range(len(Z))]
	pre = 1 / len(Z)
	outer_sum = 0
	for i, z in enumerate(Z):
		mina = math.inf
		for a in A:
			inner_sum = 0
			for j in range(len(a.normalized_fitness)):
				if normalized:
					inner_sum += max(0, ((a.normalized_fitness[j] - z.normalized_fitness[j]) ** 2))
				else:
					inner_sum += max(0, ((a.fitness[j] - z.fitness[j]) ** 2))
			inner_sum = math.sqrt(inner_sum)
			if inner_sum < mina:
				memoization[i][1] = memoization[i][0].copy()
				memoization[i][0][0] = inner_sum
				memoization[i][0][1] = a
				mina = inner_sum
			elif inner_sum < memoization[i][1][0]:
				memoization[i][1][0] = inner_sum
				memoization[i][1][1] = a
		outer_sum += mina
	return (pre * outer_sum), memoization

# VERIFIED
def igdplus_de(A, Z, normalized = True):
	C = [0 for _ in range(len(A))]
	total, memo = igdplus(A, Z, normalized=normalized)
	for i in range(len(A)):
		v = 0
		for j in range(len(Z)):
			if A[i] == memo[j][0][1]:
				v += memo[j][1][0]
			else:
				v += memo[j][0][0]
		v = v / len(Z)
		C[i] = abs(total - v)
	return C, total

# VERIFIED
def epsip(A, Z, normalized = True):
	memoization = [[[math.inf, math.inf] for _ in range(2)] for _ in range(len(Z))]
	maxz = -math.inf
	for i, z in enumerate(Z):
		mina = math.inf
		for a in A:
			maxm = -math.inf
			for j in range(len(a.normalized_fitness)):
				if normalized:
					maxm = max(maxm, (z.normalized_fitness[j] - a.normalized_fitness[j]))
				else:
					maxm = max(maxm, (z.fitness[j] - a.fitness[j]))
			if maxm < mina:
				memoization[i][1] = memoization[i][0].copy()
				memoization[i][0][0] = maxm
				memoization[i][0][1] = a
				mina = maxm
			elif maxm < memoization[i][1][0]:
				memoization[i][1][0] = maxm
				memoization[i][1][1] = a
		maxz = max(maxz, mina)
	return maxz, memoization

# VERIFIED
def epsip_de(A, Z, normalized = True):
	C = [0 for _ in range(len(A))]
	total, memo = epsip(A, Z, normalized=normalized)
	# print("Total Epsilon+: ", total)
	for i in range(len(A)):
		v = 0
		for j in range(len(Z)):
			if A[i] == memo[j][0][1]:
				v += memo[j][1][0]
			else:
				v += memo[j][0][0]
		v = v / len(Z)
		C[i] = abs(total - v)
	return C, total

# VERIFIED
def deltap_de(A, Z, normalized = True):
	total_gdp, memo_gdp = gd(A, Z, p=True, normalized=normalized)
	total_igdp, memo_igdp = igd(A, Z, p=True, normalized=normalized)
	C = [0 for _ in range(len(A))]
	if total_gdp > total_igdp:
		# print("GDP")
		total = total_gdp
		for i in range(len(A)):
			v = 0
			for j in range(len(A)):
				if (i != j):
					v += (memo_gdp[j][0][0] ** P_VAL)
			v = v / len(Z)
			v = v ** (1 / P_VAL)
			C[i] = abs(total - v)
	else:
		# print("IGDP")
		total = total_igdp
		for i in range(len(A)):
			v = 0
			for j in range(len(Z)):
				if A[i] == memo_igdp[j][0][1]:
					v += (memo_igdp[j][1][0] ** P_VAL)
				else:
					v += (memo_igdp[j][0][0] ** P_VAL)
			v = v / len(Z)
			v = v ** (1 / P_VAL)
			C[i] = abs(total - v)
	# print("Total Deltap: ", total)
	return C, total

# VERIFIED
def u(a, w, z, normalized = True, ignoreZ = False):
	if normalized:
		f_prime = a.normalized_fitness.copy()
	else:
		f_prime = a.fitness.copy()

	maxval = -math.inf
	for i in range(len(a.normalized_fitness)):
		if not ignoreZ:
			f_prime[i] -= z[i]	
		val = f_prime[i] / w[i]
		maxval = max(maxval, val)
	return maxval

# VERIFIED
def r2(A, W, z, normalized = True, ignoreZ = False):
	memoization = [[[math.inf, math.inf] for _ in range(2)] for _ in range(len(W))]
	pre = 1.0 / len(W)
	outer_sum = 0
	for i, w in enumerate(W):
		mina = math.inf
		for a in A:
			val = u(a, w, z, normalized, ignoreZ)
			if val < mina:
				memoization[i][1] = memoization[i][0].copy()
				memoization[i][0][0] = val
				memoization[i][0][1] = a
				mina = val
			elif val < memoization[i][1][0]:
				memoization[i][1][0] = val
				memoization[i][1][1] = a
		outer_sum += mina
	return pre * outer_sum, memoization

# VERIFIED
def r2_de(A, W, z, normalized = True, ignoreZ = False):
	C = [0 for _ in range(len(A))]
	total, memo = r2(A, W, z, normalized, ignoreZ)
	# print("Total R2: ", total)
	for i in range(len(A)):
		v = 0
		for j in range(len(W)):
			if A[i] == memo[j][0][1]:
				v += memo[j][1][0]
			else:
				v += memo[j][0][0]
		C[i] = abs(total - (v / len(W)))
	return C, total

# VERIFIED
def s_energy(A, normalized = True):
	memoization = [[0 for _ in range(len(A)+1)] for _ in range(len(A))]
	total = 0
	for i in range(len(A)):
		rowsum = 0
		for j in range(len(A)):
			if i != j:
				euclidean_distance = 0	
				for k in range(len(A[i].fitness)):
					if normalized:
						euclidean_distance += ((A[i].normalized_fitness[k] - A[j].normalized_fitness[k]) ** 2)
					else:
						euclidean_distance += ((A[i].fitness[k] - A[j].fitness[k]) ** 2)
				if euclidean_distance == 0:
					euclidean_distance = 1e-6
				euclidean_distance = math.sqrt(euclidean_distance)
				euclidean_distance = euclidean_distance ** -S_VAL
				# Store in aiaj
				memoization[i][j] = euclidean_distance
				# Store row sum in aip
				rowsum += euclidean_distance
		memoization[i][len(A)] = rowsum
		total += rowsum
	# for row in memoization:
	# 	print(row)
	return total, memoization

# VERIFIED
def s_energy_de(A, normalized = True):
	total, memo = s_energy(A, normalized)
	C = [0 for _ in range(len(A))]
	for i in range(len(A)):
		aux = 0
		for j in range(len(A)):
			if i != j:
				aux += memo[i][len(A)] - memo[i][j]
		C[i] = (total - aux) / 2.0
	return C, total

def ns_energy(A, normalized = True):
	# memoization = [[0 for _ in range(len(A))] for _ in range(len(A))]
	r = [0 for _ in range(len(A))]
	total = 0
	for i in range(len(A)):
		rowsum = 0
		for j in range(len(A)):
			if i != j:
				euclidean_distance = 0	
				for k in range(len(A[i].fitness)):
					if normalized:
						euclidean_distance += ((A[i].normalized_fitness[k] - A[j].normalized_fitness[k]) ** 2)
					else:
						euclidean_distance += ((A[i].fitness[k] - A[j].fitness[k]) ** 2)
				if euclidean_distance == 0:
					euclidean_distance = 1e-6
				euclidean_distance = math.sqrt(euclidean_distance)
				euclidean_distance = euclidean_distance ** -S_VAL
				# memoization[i][j] = euclidean_distance
				rowsum += euclidean_distance
		r[i] = rowsum
		total += rowsum
	# return memoization, r, total
	return r, total

def ns_energy_de(A, normalized = True):
	r, total = ns_energy(A, normalized)
	# K, r, total = ns_energy(A, normalized)
	# S = set(range(len(A)))
	# limit = len(S) - 1
	# while len(S) > MAX_POP:
	# while len(S) > limit:
	worst = r.index(max(r))
		# for i in S:
		# for i in range(len(A)):
			# r[i] -= K[i][worst]
		# r.pop(worst)
		# S.remove(worst)
	return worst, total

def solow_polansky(A, normalized = True):
	M = [[0.0 for _ in range(len(A))] for _ in range(len(A))]
	for i in range(len(A)):
		for j in range(len(A)):
			dist = 0
			for k in range(len(A[i].fitness)):
				if normalized:
					dist += ((A[i].normalized_fitness[k] - A[j].normalized_fitness[k]) ** 2)
				else:
					dist += ((A[i].fitness[k] - A[j].fitness[k]) ** 2)
			dist = math.sqrt(dist)
			val = math.exp(-THETA * dist)
			M[i][j] = val
	# if la.det(M) == 0:
	# 	total = 0
	# else:
	# 	p,l,u = la.lu(M, permute_l=False)
	# 	l = np.dot(p, u)
	# 	l_inv = np.linalg.inv(l)
	# 	u_inv = np.linalg.inv(u)
	# 	C = np.dot(u_inv, l_inv)
	# 	total = 0

	try:
		C = np.linalg.inv(M)
		total = 0
		for i in range(len(A)):
			for j in range(len(A)):
				total += C[i][j]
		if total > len(A) or total < 0:
			total = 0
	except np.linalg.LinAlgError:
		total = 0
	return total