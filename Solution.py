from global_vars import *

class Solution:
	def __init__(self, candidate, settings):
		self.candidate = candidate
		self.fitness = evaluate(self.candidate, settings)
		self.rank = 0
		self.rank2 = 0
		self.distance = 0
		self.s_list = []
		self.n = 0
		self.normalized_fitness = [0 for _ in range(len(self.fitness))]
		self.ds = []
	
	def __str__(self):
		return str(self.candidate) + " " + str(self.fitness) + " " + str(self.normalized_fitness)

def evaluate(candidate, settings):
	fitness = []
	for j, function in enumerate(settings["functions"]):
		if ("DTLZ" in settings["problem"] and (j > 0 and j < USER_M_OBJ-1)) or ("DTLZ7" in settings["problem"] and j < USER_M_OBJ-1) or ("WFG" in settings["problem"] and j > 0 and j < USER_M_OBJ-1):
			if "Minus" in settings["problem"]:
				fitness.append(-function(candidate, j))
			else:
				fitness.append(function(candidate, j))
		elif "ZCAT" in settings["problem"]:
			fitness = function(candidate, settings["zcat_settings"], settings["problem"])
		else:
			if "Minus" in settings["problem"]:
				fitness.append(-function(candidate))
			else:
				fitness.append(function(candidate))
	return fitness