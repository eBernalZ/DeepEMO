from util import dominates
from indicators import ns_energy_de
def M3(pop):
    topMax = 0
    front = [pop[0]]
    for p in pop[1:]:
        j = 0
        while j <= topMax:
            domination = dominates(front[j], p)
            if domination == 1:
                front.insert(0, front.pop(j))
                j = topMax + 2
            elif domination == -1:  
                front[j:topMax] = front[j+1:topMax+1]
                front.pop(-1)
                topMax -= 1
            elif domination == 2:
                j = topMax + 2
            else:
                j += 1
        if j == topMax + 1:
            topMax += 1
            front.append(p)
    return front

def M3_NDS(pop):
    work = pop.copy()
    fronts = []
    while len(work) > 0:
        front = M3(work)
        fronts.append(front)
        for p in front:
            work.remove(p)
    return fronts