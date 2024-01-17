import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from global_vars import *
from indicators import *
from problems import *
from SLD import * 
from model import DGCNN
import torch.nn as nn
# from EvalSolution import EvalSolution

def plot_front(fronts, run, problem, directory):
	if len(fronts[0][0].fitness) == 2:
		fig, ax = plt.subplots(figsize=(6,6))
	else:
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6,6))

	for front in fronts:
		for solution in front:
			if len(solution.fitness) == 2:
				ax.scatter(solution.fitness[0], solution.fitness[1], color='red')
			else:
				ax.scatter(solution.fitness[0], solution.fitness[1], solution.fitness[2], color='red')
	ax.tick_params(axis='both', which='major', labelsize=18)
	if len(fronts[0][0].fitness) == 3:
		ax.view_init(elev=30, azim=45)

	# ax.set_xlabel("F1")
	# ax.set_ylabel("F2")
	# if USER_M_OBJ == 3:
		# ax.set_zlabel("F3")

	# if "DTLZ" in problem:
	#     ax.set_xlim([0, 1.2])
	#     ax.set_ylim([0, 1.2])
	#     if USER_M_OBJ == 3:
	#         ax.set_zlim([0, 1.2])
	# ax.text2D(0.05, 0.95, "Prediction: \"" + geometry + "\" Percentage: " + str(np.round(certainty,2)), transform=ax.transAxes, color='green')
	plt.savefig(directory + "/graph_" + problem + "_" + str(run) + ".svg")
	plt.close()


def classify_geometry(front, model):
	data = np.array([])
	for solution in front:
		if len(solution.normalized_fitness) == 3:
			data = np.append(data, solution.normalized_fitness)
		else:
			# print("ELSE")
			arr = np.array(solution.normalized_fitness)
			arr = np.append(arr, 0)
			data = np.append(data, arr)
	
	data = data.reshape(-1, 3)
	
	data = torch.from_numpy(data).float()
	data = data.unsqueeze(0)
	data = data.transpose(2, 1)
	data = data.to('cpu')
	
	logits = model(data)
	geometry = logits.max(dim=1)[1].detach().cpu().numpy()[0]
	certainty = torch.softmax(logits, dim=-1)
	certainty = certainty.max(dim=1)[0].detach().cpu().numpy()[0]
	return geometry, certainty

def buildZ(A):
	z = []
	for i in range(len(A[0].fitness)):
		minval = math.inf
		for a in A:
			minval = min(minval, a.fitness[i])
		z.append(minval)
	return z

def save_pop(pop, run, problem, directory, initial = False):
	if initial:
		filename = "/pop_" + problem + "_" + str(run) + "_initial.pof"
	else:
		filename = "/pop_" + problem + "_" + str(run) + ".pof"
	with open(directory + filename, "w") as f:
		f.write("# " + str(len(pop)) + " " + str(len(pop[0].fitness)) + "\n")
		for solution in pop:
			for i in range(len(solution.fitness)-1):
				f.write(str(solution.fitness[i]) + " ")
			f.write(str(solution.fitness[-1]))
			f.write("\n")

def calculate_metrics(pop, problem):
	if problem == "DTLZ1":
		hv_ref = [1.0 for _ in range(USER_M_OBJ)]
	elif problem == "MinusDTLZ1":
		hv_ref = [0.1 for _ in range(USER_M_OBJ)]
	elif problem == "DTLZ2":
		hv_ref = [1.2 for _ in range(USER_M_OBJ)]
	elif problem == "MinusDTLZ2":
		hv_ref = [0.1 for _ in range(USER_M_OBJ)]
	elif "WFG" in problem:
		if "Minus" in problem:
			hv_ref = [0.1, 0.1, 0.1]
		else:
			hv_ref = [3, 5, 7]
	elif problem == "IMOP1" or problem == "IMOP2" or problem == "IMOP3" or problem == "IMOP4" or problem == "IMOP6" or problem == "IMOP7":
		hv_ref = [2.0 for _ in range(len(pop[0].fitness))]
	elif "IMOP5" in problem:
		hv_ref = [1.5, 1.5, 2.0]
	elif "IMOP8" in problem:
		hv_ref = [2.0, 2.0, 4.0]
	elif "VNT1" in problem:
		hv_ref = [5.0, 6.0, 5.0]
	elif "VNT2" in problem:
		hv_ref = [6.0, 6.0, 6.0]
	elif "VNT3" in problem:
		hv_ref = [19.0, 19.0, 19.0]
	elif problem == "MinusDTLZ7":
		hv_ref = [0.1 for _ in range(USER_M_OBJ)]
	elif problem == "DTLZ7":
		hv_ref = [1.0 for _ in range(USER_M_OBJ-1)]
		hv_ref.append(21.0)
	W = SLD(MAX_POP,len(pop[0].fitness))
	_, hv = hv_de(pop, hv_ref, False)
	_, r2 = r2_de(pop, W, [], False, True)
	_, s_energy = ns_energy_de(pop, False)
	spd = solow_polansky(pop, False)
	metrics = [hv, r2, s_energy, spd]
	return metrics

def normalize_population(pop):
	for i in range(len(pop[0].fitness)):
		minval = math.inf
		maxval = -math.inf
		for solution in pop:
			minval = min(minval, solution.fitness[i])
			maxval = max(maxval, solution.fitness[i])
		for solution in pop:
			if maxval != minval:
				solution.normalized_fitness[i] = (solution.fitness[i] - minval) / (maxval - minval)
			else:
				solution.normalized_fitness[i] = 0

def find_in_pop(solution, pop):
	for i in range(len(pop)):
		if pop[i].fitness == solution.fitness:
			return i
	return -1

def setProblemSettings(problem):
	config = {}
	config["problem"] = problem
	if problem == "VNT1":
		config["functions"] = [vie1_f1, vie1_f2, vie1_f3]
		config["dim"] = 2
		config["lower_bound"] = -2.0
		config["upper_bound"] = 2.0
	elif problem == "VNT2":
		config["functions"] = [vie2_f1, vie2_f2, vie2_f3]
		config["dim"] = 2
		config["lower_bound"] = -4.0
		config["upper_bound"] = 4.0
	elif problem == "VNT3":
		config["functions"] = [vie3_f1, vie3_f2, vie3_f3]
		config["dim"] = 2
		config["lower_bound"] = -3.0
		config["upper_bound"] = 3.0
	elif problem == "DTLZ1" or problem == "MinusDTLZ1":
		config["functions"] = [dtlz1_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz1_fi)
		config["functions"].append(dtlz1_fm)
		config["dim"] = USER_M_OBJ + DTLZ1_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "DTLZ2" or problem == "MinusDTLZ2":
		config["functions"] = [dtlz2_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz2_fi)
		config["functions"].append(dtlz2_fm)
		config["dim"] = USER_M_OBJ + DTLZ2T6_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "DTLZ3" or problem == "MinusDTLZ3":
		config["functions"] = [dtlz3_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz3_fi)
		config["functions"].append(dtlz3_fm)
		config["dim"] = USER_M_OBJ + DTLZ2T6_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0

	elif problem == "DTLZ4" or problem == "MinusDTLZ4":
		config["functions"] = [dtlz4_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz4_fi)
		config["functions"].append(dtlz4_fm)
		config["dim"] = USER_M_OBJ + DTLZ2T6_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "DTLZ5" or problem == "MinusDTLZ5":
		config["functions"] = [dtlz5_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz5_fi)
		config["functions"].append(dtlz5_fm)
		config["dim"] = USER_M_OBJ + DTLZ2T6_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0

	elif problem == "DTLZ6" or problem == "MinusDTLZ6":
		config["functions"] = [dtlz6_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(dtlz6_fi)
		config["functions"].append(dtlz6_fm)
		config["dim"] = USER_M_OBJ + DTLZ2T6_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0

	elif problem == "DTLZ7" or problem == "MinusDTLZ7":
		config["functions"] = []
		for _ in range(USER_M_OBJ-1):
			config["functions"].append(dtlz7_fi)
		config["functions"].append(dtlz7_fm)
		config["dim"] = USER_M_OBJ + DTLZ7_K - 1
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0

	elif "ZCAT" in problem:
		config["dim"] = 10
		lb, ub, zcat_settings = zcat_set(config["dim"], USER_M_OBJ, ZCAT_LEVEL, ZCAT_BIAS, ZCAT_COMPLICATED_PS, ZCAT_IMBALANCE)
		config["functions"] = [zcat]
		config["lower_bound"] = lb[0]
		config["upper_bound"] = ub[0]
		config["zcat_settings"] = zcat_settings

	elif problem == "WFG1" or problem == "MinusWFG1":
		config["functions"] = [wfg1_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg1_fi)
		config["functions"].append(wfg1_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0

	elif problem == "WFG2" or problem == "MinusWFG2":
		config["functions"] = [wfg2_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg2_fi)
		config["functions"].append(wfg2_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0

	elif problem == "WFG3" or problem == "MinusWFG3":
		config["functions"] = [wfg3_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg3_fi)
		config["functions"].append(wfg3_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "WFG4" or problem == "MinusWFG4":
		config["functions"] = [wfg4_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg4_fi)
		config["functions"].append(wfg4_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "WFG5" or problem == "MinusWFG5":
		config["functions"] = [wfg5_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg5_fi)
		config["functions"].append(wfg5_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "WFG6" or problem == "MinusWFG6":
		config["functions"] = [wfg6_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg6_fi)
		config["functions"].append(wfg6_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0

	elif problem == "WFG7" or problem == "MinusWFG7":
		config["functions"] = [wfg7_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg7_fi)
		config["functions"].append(wfg7_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "WFG8" or problem == "MinusWFG8":
		config["functions"] = [wfg8_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg8_fi)
		config["functions"].append(wfg8_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "WFG9" or problem == "MinusWFG9":
		config["functions"] = [wfg9_f1]
		for i in range(1, USER_M_OBJ-1):
			config["functions"].append(wfg9_fi)
		config["functions"].append(wfg9_fm)
		config["dim"] = WFG_K + WFG_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 2.0
	
	elif problem == "IMOP1":
		config["functions"] = [imop1_f1, imop1_f2]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP2":
		config["functions"] = [imop2_f1, imop2_f2]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP3":
		config["functions"] = [imop3_f1, imop3_f2]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP4":
		config["functions"] = [imop4_f1, imop4_f2, imop4_f3]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP5":
		config["functions"] = [imop5_f1, imop5_f2, imop5_f3]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP6":
		config["functions"] = [imop6_f1, imop6_f2, imop6_f3]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP7":
		config["functions"] = [imop7_f1, imop7_f2, imop7_f3]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	
	elif problem == "IMOP8":
		config["functions"] = [imop8_f1, imop8_f2, imop8_f3]
		config["dim"] = IMOP_K + IMOP_L
		config["lower_bound"] = 0.0
		config["upper_bound"] = 1.0
	return config

def load_model():
	args = type('', (), {})()
	args.emb_dims = 1024
	args.k = 5 * (MAX_POP // 50)
	args.dropout = 0.5
	args.num_points = MAX_POP
	# These settings are for your GPU, if the computer freezes, lower them
	args.batch_size = 32
	args.test_batch_size = 16
	model = DGCNN(args)
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load('models/model_50_norm.pth'))
	if torch.cuda.is_available():
		model = model.cuda()
	else:
		model.to('cpu')
	model = model.eval()
	torch.no_grad()
	return model

""" ParetoDominance 
 * A solution x strictly dominates a solution y, if
 * x[i] <= y[i] for all i in {1,..,n} and x[j] < y[j] for at least one j in {1,..,n}.
 * Return value: 
 *  1 - if p dominates q
 *  0 - if p and q are mutually non dominated  (INCOMPARABLE)
 * -1 - if q dominates p
 *  2 - if p and q are equal
"""
def dominates(p,q):
	fx = 0
	fy = 0
	eq = 0
	for i in range(len(p.fitness)):
		if p.fitness[i] < q.fitness[i]:
			fx += 1
		elif p.fitness[i] > q.fitness[i]:
			fy += 1
		else:
			eq += 1

	if fx > 0 and fy == 0:
		return 1
	if fy > 0 and fx == 0:
		return -1
	if eq == len(p.fitness):
		return 2
	return 0