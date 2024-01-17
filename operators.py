import numpy as np
from global_vars import *

def recombination(parent1, parent2, settings):
	child1 = parent1.copy()
	child2 = parent2.copy()
	if np.random.uniform() <= CROSSOVER_RATE:
		for i in range(settings["dim"]):
			if np.fabs(parent1[i] - parent2[i]) > 1.2e-7:
				if parent1[i] < parent2[i]:
					y1 = parent1[i]
					y2 = parent2[i]
				else:
					y1 = parent2[i]
					y2 = parent1[i]
				yl = settings["lower_bound"]
				if "WFG" in settings["problem"]:
					yu = settings["upper_bound"] * (i + 1)
				else:
					yu = settings["upper_bound"]

				rnd = 0
				while rnd == 0:
					rnd = np.random.uniform()

				betaq = np.power(2*rnd,1.0/(NC + 1)) if rnd <= 0.5 else np.power(2 - 2*rnd,-1/(NC + 1))
				rnd = 0
				while rnd == 0:
					rnd = np.random.uniform()
				betaq = 1 if rnd <= 0.5 else -1 * betaq

				rnd = 0
				while rnd == 0:
					rnd = np.random.uniform()
				betaq = 1 if rnd <= 0.5 else betaq

				rnd = 0
				while rnd == 0:
					rnd = np.random.uniform()
				betaq = 1 if rnd > CROSSOVER_RATE else betaq

				c1 = 0.5*((y1 + y2) - betaq*(parent1[i] - parent2[i]))
				c2 = 0.5*((y1 + y2) + betaq*(parent1[i] - parent2[i]))

				c1 = yl if c1 < yl else c1
				c2 = yl if c2 < yl else c2
				c1 = yu if c1 > yu else c1
				c2 = yu if c2 > yu else c2

				if np.random.uniform() >= 0.5:
					child1[i] = c2
					child2[i] = c1
				else:
					child1[i] = c1
					child2[i] = c2
			else:
				child1[i] = parent1[i]
				child2[i] = parent2[i]
	return [child1, child2]

def mutation(candidate, settings, force = False):
	PM = 1 / settings["dim"]
	for j in range(settings["dim"]):
		if np.random.rand() <= PM or force:
			y = candidate[j]
			yl = settings["lower_bound"]
			if "WFG" in settings["problem"]:
				yu = settings["upper_bound"] * (j + 1)
			else:
				yu = settings["upper_bound"]
			delta1 = (y - yl)/(yu - yl)
			delta2 = (yu - y)/(yu - yl)
			rnd = 0
			while rnd == 0:
				rnd = np.random.uniform()
			mut_pow = 1.0 / (NC + 1.0)
			if rnd <= 0.5:
				xy = 1 - delta1
				val = 2*rnd + (1 - 2*rnd) * np.power(xy, (NM + 1))
				deltaq = np.power(val, mut_pow) - 1
			else:
				xy = 1 - delta2
				val = 2*(1 - rnd) + 2*(rnd - 0.5)*np.power(xy, NM + 1)
				deltaq = 1 - pow(val, mut_pow)
			y = y + deltaq*(yu - yl)
			y = yl if y < yl else y
			y = yu if y > yu else y
			candidate[j] = y