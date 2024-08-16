#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
import argparse

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

os.environ['PYTHONUNBUFFERED'] = '1'

'''
Fixar: Taxa de aceite

2 objetivos:
- Acurácia total do sistema
- Tempo médio de inferência

4 parâmetros -> 
Limiar de normal / ataque na primeira
Limiar de normal / ataque na segunda
'''

def get_objectives(df, normal_threshold_exit1, attack_threshold_exit1, normal_threshold_exit2, attack_threshold_exit2):
    f_n_exit1 = 'y_exit_1 == 0 and cnf_exit_1 >= @normal_threshold_exit1'
    f_a_exit1 = 'y_exit_1 == 1 and cnf_exit_1 >= @attack_threshold_exit1'

    f_n_exit2 = 'y_exit_2 == 0 and cnf_exit_2 >= @normal_threshold_exit2'
    f_a_exit2 = 'y_exit_2 == 1 and cnf_exit_2 >= @attack_threshold_exit2'

    exit1_normal = df.query(f_n_exit1)
    exit1_attack = df.query(f_a_exit1)

    exit2 = df.query(f'not ({f_n_exit1}) and not ({f_a_exit1})')

    exit2_normal = exit2.query(f_n_exit2)
    exit2_attack = exit2.query(f_a_exit2)

    not_accepted = exit2.query(f'not ({f_n_exit2}) and not ({f_a_exit2})')

    total = df['y'].count()

    exit1_normal_cnt = exit1_normal['y'].count()
    exit1_attack_cnt = exit1_attack['y'].count()
    exit2_normal_cnt = exit2_normal['y'].count()
    exit2_attack_cnt = exit2_attack['y'].count()

    accepted = exit1_normal_cnt + exit1_attack_cnt + exit2_normal_cnt + exit2_attack_cnt

    acceptance_rate = accepted / total

    correct = exit1_normal.query('y == y_exit_1')['y'].count() + \
              exit1_attack.query('y == y_exit_1')['y'].count() + \
              exit2_normal.query('y == y_exit_2')['y'].count() + \
              exit2_attack.query('y == y_exit_2')['y'].count()

    accuracy = correct / accepted

    exit1_total_time = exit1_normal['bb_time_exit_1'].sum() + exit1_normal['exit_time_exit_1'].sum() + \
                       exit1_attack['bb_time_exit_1'].sum() + exit1_attack['exit_time_exit_1'].sum()

    exit2_total_time = exit2_normal['bb_time_exit_1'].sum() + exit2_normal['bb_time_exit_2'].sum() + exit2_normal['exit_time_exit_2'].sum() + \
                       exit2_attack['bb_time_exit_1'].sum() + exit2_attack['bb_time_exit_2'].sum() + exit2_attack['exit_time_exit_2'].sum()

    not_accepted_total_time = not_accepted['bb_time_exit_1'].sum() + not_accepted['bb_time_exit_2'].sum() + not_accepted['exit_time_exit_2'].sum()

    total_time = exit1_total_time + exit2_total_time + not_accepted_total_time

    # print(f"Total: {total}")
    # print(f"exit1_normal_cnt: {exit1_normal_cnt}, exit1_attack_cnt: {exit1_attack_cnt}")
    # print(f"exit2_normal_cnt: {exit2_normal_cnt}, exit2_attack_cnt: {exit2_attack_cnt}")
    # print(f"Accepted: {accepted}, Accepted: {total - not_accepted['y'].count()}")
    # print(f"exit1_total_time: {exit1_total_time:.4f}, exit2_total_time: {exit2_total_time:.4f}, not_accepted_total_time: {not_accepted_total_time:.4f}")
    # print(f"exit1_rate: {100 * ( exit1_normal_cnt + exit1_attack_cnt ) / total:.2f}, exit2_rate: {100 * ( exit2_normal_cnt + exit2_attack_cnt ) / total:.2f}")
    # print(f"Accuracy: {100 * accuracy:.2f}, Acceptance: {100 * acceptance_rate:.2f}, Average Time: {1e6 * total_time / total:.2f}")

    return [ accuracy, acceptance_rate, 1e6 * total_time / total ]

class MyProblem(ElementwiseProblem):
    def __init__(self, df, min_acceptance=0.7):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([ 0.5, 0.5, 0.5, 0.5 ]),
                         xu=np.array([ 1, 1, 1, 1 ]))
            
        self.df = df
        self.accuracy_e1, self.acceptance_e1, self.min_time = get_objectives(self.df, 0, 0, 1, 1)
        self.accuracy_e2, self.acceptance_e2, self.max_time = get_objectives(self.df, 2, 2, 0, 0)
        self.min_acceptance = min_acceptance      

    def _evaluate(self, x, out, *args, **kwargs):
        accuracy, acceptance, time = get_objectives(self.df, *x)
        out["F"] = [ 1 - accuracy, (time - self.min_time) / (self.max_time - self.min_time) ]
        out["G"] = [ self.min_acceptance - acceptance ]

def process(eval_file, min_acceptance=0.7, population_size=100, n_offsprings=80, n_gen=1000):
    df = pd.read_csv(eval_file)

    problem = MyProblem(df, min_acceptance=min_acceptance)

    algorithm = NSGA2(
        pop_size=population_size, # 100
        n_offsprings=n_offsprings, # 80
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen) # 1000

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    print(f'Exit1: Accuracy e1: {problem.accuracy_e1*100:.2f}% - Acceptance e1: {problem.acceptance_e1*100:.2f}% - Min time: {problem.min_time:.2f}us')
    print(f'Exit2: Accuracy e2: {problem.accuracy_e2*100:.2f}% - Acceptance e2: {problem.acceptance_e2*100:.2f}% - Max time: {problem.max_time:.2f}us')
    print()
    
    print("Parameters: normal_threshold_exit1, attack_threshold_exit1, normal_threshold_exit2, attack_threshold_exit2\n")
    for i in range(len(F)):
        f = F[i]
        x = X[i]
        print(f'{i:02d}: Accuracy: {100 * (1 - f[0]):.2f}% : Avg time: {problem.min_time + (f[1] * (problem.max_time - problem.min_time)):.2f}us', end='')
        print(f'\tParameters: {x[0]:.4f} : {x[1]:.4f} : {x[2]:.4f} : {x[3]:.4f}')

    return X, F, problem.min_time, problem.max_time, problem.accuracy_e1, problem.acceptance_e1, problem.accuracy_e2, problem.acceptance_e2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-acceptance", type=float, default=0.7, help="Minimum acceptance rate (default: 0.7)")
    parser.add_argument("--eval-file", help="Evaluation file pattern")
    parser.add_argument("--savefile", help="Save file name")
    parser.add_argument("--offspring", type=int, default=80, help="Number of offsprings (default: 80)")
    parser.add_argument("--gen", type=int, default=1000, help="Number of generations (default: 1000)")
    parser.add_argument("--population", type=int, default=100, help="Population size (default: 100)")
    args = parser.parse_args()

    min_acceptance = args.min_acceptance
    eval_file = args.eval_file
    savefile = args.savefile
    offspring = args.offspring
    gen = args.gen
    population = args.population

    print(f"Processing {eval_file} with min_acceptance = {min_acceptance}")

    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = process(eval_file, min_acceptance, population, offspring, gen)
    print(f"{min_time}, {max_time}, {accuracy_e1}, {acceptance_e1}, {accuracy_e2}, {acceptance_e2}")
    with open(savefile, 'wb') as f:
        pickle.dump([X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2], f)
