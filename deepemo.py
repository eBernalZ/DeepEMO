import numpy as np
from numpy.random import randint
from numpy.random import rand
import os
from indicators import *
from SLD import SLD
from util import *
from Solution import Solution
from global_vars import *
from zcat_benckmark import *
from M3 import M3_NDS
from operators import *

def main():
	seeds = open("seed.dat", "r")
	seeds = seeds.read().split("\n")
	seeds = [int(seed) for seed in seeds]
	for problem in PROBLEMS:
		global settings
		settings = setProblemSettings(problem)
		print("PROBLEM: ", problem)
		for version in VERSIONS:
			print("VERSION: ", version)
			if not os.path.exists("runs/" + version + "/" + problem + "/" + str(len(settings["functions"])) + "D"):
				os.makedirs("runs/" + version + "/" + problem + "/" + str(len(settings["functions"])) + "D")
			directory = "runs/" + version + "/" + problem + "/" + str(len(settings["functions"])) + "D"
			if version == "DEEPEMO":
					model = load_model()
					shape_names = [line.rstrip() for line in open('shape_names.txt')]
			if version == "R2" or version == "DEEPEMO" or version == "RANDOM":
				if "ZCAT" in problem:
					W = SLD(MAX_POP, USER_M_OBJ)
				else:
					W = SLD(MAX_POP, len(settings["functions"]))
			for run in range(RUNS):
				metrics = open(directory + "/metrics_" + str(run) + ".csv", "w")
				metrics.write("HV,R2,S_ENERGY\n")
				np.random.seed(seeds[run])
				print("RUN: ", str(run + 1))
				if version == "DEEPEMO" or version == "RANDOM":
					classifications = open(directory + "/classifications_" + problem + "_" + str(run) + ".csv", "w")
					classifications.write("Gen,Shape,Certainty,Indicator\n")
				if "WFG" in problem:
					pop = [Solution(np.array([np.random.uniform(settings["lower_bound"], settings["upper_bound"] * (i + 1)) for i in range(settings["dim"])]), settings) for _ in range(MAX_POP)]
				else:
					pop = [Solution(np.random.uniform(settings["lower_bound"], settings["upper_bound"], settings["dim"]), settings) for _ in range(MAX_POP)]
				for gen in range(MAX_ITER):
					if gen % 10000 == 0:
						print("Gen: ", str(gen + 1))
					# ## STEADY STATE
					children = recombination(pop[randint(0, len(pop))].candidate, pop[randint(0, len(pop))].candidate, settings)
					q = children[0]
					mutation(q, settings)
					pop.append(Solution(q, settings))
		
					## NON DOMINATED SORT
					fronts = M3_NDS(pop)

					## INDICATORS CALCULATION
					if (len(fronts[-1]) > 1):
						A = fronts[-1]
						Z = fronts[0]
						
						if version == "DEEPEMO":
							# NORMALIZE POPULATION
							normalize_population(pop)
							geometry, certainty = classify_geometry(pop, model)
							classifications.write(str(gen) + "," + str(shape_names[geometry]) + "," + str(certainty))
							
							if shape_names[geometry] == "concave" and certainty >= BETA:
								classifications.write(",R2\n")
								z = buildZ(A)
								C_r2, _ = r2_de(A, W, z)
								i = C_r2.index(min(C_r2))
							elif shape_names[geometry] == "convex" and certainty >= BETA:
								classifications.write(",HV\n")
								if len(A[0].fitness) == 2:
									hvref = [2, 2]
								else:
									hvref = [2, 2, 2]
								C_hv, _ = hv_de(A, hvref)
								i = C_hv.index(min(C_hv))
							else:
								classifications.write(",S_energy\n")
								i, _ = ns_energy_de(A)

						elif version == "RANDOM":
							ind = randint(0, 3)
							if ind == 0:
								classifications.write(str(gen) + ",,,R2\n")
								z = buildZ(A)
								C_r2, _ = r2_de(A, W, z)
								i = C_r2.index(min(C_r2))
							elif ind == 1:
								classifications.write(str(gen) + ",,,HV\n")
								if len(A[0].fitness) == 2:
									hvref = [2, 2]
								else:
									hvref = [2, 2, 2]
								C_hv, _ = hv_de(A, hvref)
								i = C_hv.index(min(C_hv))
							elif ind == 2:
								classifications.write(str(gen) + ",,,S_energy\n")
								i, _ = ns_energy_de(A)

						elif version == "HV":
							if len(A[0].fitness) == 2:
								hvref = [2, 2]
							else:
								hvref = [2, 2, 2]
							C_hv, _ = hv_de(A, hvref)
							i = C_hv.index(min(C_hv))

						elif version == "IGD+":
							print("IGD+")
							C_igdplus, _ = igdplus_de(A, Z)
							i = C_igdplus.index(min(C_igdplus))

						elif version == "GD":
							print("GD")
							C_gd, _ = gd_de(A, Z)
							i = C_gd.index(min(C_gd))
						
						elif version == "IGD":
							print("IGD")
							C_igd, _ = igd_de(A, Z)
							i = C_igd.index(min(C_igd))

						elif version == "E+":
							# print("E+")
							C_epsip, _ = epsip_de(A, Z)
							i = C_epsip.index(min(C_epsip))
						
						elif version == "DELTAP":
							print("DELTAP")
							C_deltap, _ = deltap_de(A, Z)
							i = C_deltap.index(min(C_deltap))
						
						elif version == "R2":
							z = buildZ(A)
							C_r2, _ = r2_de(A, W, z)
							i = C_r2.index(min(C_r2))
							
						elif version == "S_ENERGY":
							i, _ = ns_energy_de(A)
						pop.remove(A[i])
						fronts[-1].remove(A[i])
					else:
						if version == "DEEPEMO" or version == "RANDOM":
							classifications.write(str(gen) + ",,,P\n")
						pop.remove(fronts[-1][0])
						fronts[-1].remove(fronts[-1][0])
					## PLOT FRONTS
					if gen == MAX_ITER - 1:
						save_pop(pop, run, problem, directory)
						plot_front(fronts, run, problem, directory)
						run_metrics = calculate_metrics(pop, problem)
						for i in range(len(run_metrics)-1):
							metrics.write(str(run_metrics[i]) + ",")
						metrics.write(str(run_metrics[-1]) + "\n")
				if version == "DEEPEMO":
					classifications.close()
				metrics.close()
				print("-------------------------------------")
			# times.close()
		print("=====================================")

if __name__ == "__main__":
	main()