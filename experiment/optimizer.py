import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from experiment.utils import *


def de(fun_opt, metrics, bounds, mut=0.5, crossp=0.5, popsize=20, itrs=10):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds)[:,0], np.asarray(bounds)[:,1]
    diff = np.fabs(min_b - max_b)
    # pop_denorm = min_b + pop * diff
    pop_denorm = np.rint(min_b + pop * diff).astype(np.int)
    fitness = np.asarray([fun_opt(*ind) for ind in pop_denorm])
    if metrics == 0:
        best_idx = np.argmin(fitness)
    if metrics == 1:
        best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]
    for i in range(itrs):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            # trial_denorm = min_b + trial * diff
            trial_denorm = np.rint(min_b + trial * diff).astype(np.int)
            f = fun_opt(*trial_denorm)
            if metrics == 0:
                if f < fitness[j]:  ####### MRE
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:  ####### MRE
                        best_idx = j
                        best = trial_denorm
            if metrics == 1:
                if f > fitness[j]:  ####### SA
                    fitness[j] = f
                    pop[j] = trial
                    if f > fitness[best_idx]:  ####### SA
                        best_idx = j
                        best = trial_denorm
        return fitness[best_idx], best


def rd(fun_opt, metrics, bounds, popsize=20, itrs=10):
    dimensions = len(bounds)
    pop = np.random.rand(1, dimensions)
    min_b, max_b = np.asarray(bounds)[:,0], np.asarray(bounds)[:,1]
    diff = np.fabs(min_b - max_b)
    pop_denorm = np.rint(min_b + pop * diff).astype(np.int)
    fitness = np.asarray([fun_opt(*ind) for ind in pop_denorm])
    if metrics == 0:
        best_idx = np.argmin(fitness)
    if metrics == 1:
        best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]
    for i in range(itrs * popsize):
        sample = np.random.rand(1, dimensions)
        min_bb, max_bb = np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1]
        difff = np.fabs(min_bb - max_bb)
        sample_denorm = np.rint(min_b + sample * difff).astype(np.int)[0]
        f = fun_opt(*sample_denorm)
        if metrics == 0:
            if f < fitness[0]:  ####### MRE
                fitness[0] = f
                best = sample_denorm
        if metrics == 1:
            if f > fitness[0]:  ####### SA
                fitness[0] = f
                best = sample_denorm
    return fitness[best_idx], best
