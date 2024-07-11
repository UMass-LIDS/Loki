# This optimization takes the number of devices and batch size across
# different stages for each pipeline variant from the offline optimization
# and allocates resources to it (assuming a homogeneous cluster)

import itertools
import numpy as np
import pandas as pd
import gurobipy as gp


def parse_accuracy_file(filepath):
    df = pd.read_csv(filepath)
    combined_models = df.iloc[:, 0].values
    model_1 = []
    model_2 = []
    for combined_model in combined_models:
        separated = combined_model.split('_')
        model_1.append(separated[-1])
        model_2.append(separated[0])
    df['model_1'] = model_1
    df['model_2'] = model_2
    print(df)
    return df


models = {
        #   'yolov5': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
          'yolov5': ['yolov5n', 'yolov5m', 'yolov5l', 'yolov5x'],
          'efficientnet': ['eb0', 'eb1', 'eb2', 'eb3', 'eb4', 'eb5', 'eb6']
         }

variants = list(itertools.product(models['yolov5'], models['efficientnet']))

# TODO: remove this when it is profiled
variants.remove(('yolov5l', 'eb4'))

# TODO: profile eb0-6 with yolov5m,s,n

print(f'variants: {variants}')

# # TODO: use actual pipeline variants, perhaps calculated from
# #       the no. of variants of each stage
# variants = list(range(0, 105))

acc = {}
thput = {}
mult_factor = {}
servers = {}

runtime_df = pd.read_csv('../../profiling/profiled/runtimes_all.csv')
accuracy_df = parse_accuracy_file('../../profiling/profiled/accuracy_all.csv')
offline_opt_df = pd.read_csv('results_offline_opt.csv')

for variant in variants:
    print(f'variant: {variant}')
    # TODO: use actual accuracy and throughput values
    # print(f"{accuracy_df[(accuracy_df['model_1'] == variant[0]) & (accuracy_df['model_2'] == variant[1])]['e2e_acc'].values}")
    # acc[variant] = 0.8 + np.random.random() * 0.2       # randomized from 0.8 - 1.0
    acc[variant] = accuracy_df[(accuracy_df['model_1'] == variant[0]) &
                               (accuracy_df['model_2'] == variant[1])]['e2e_acc'].values[0]

    # # TODO: use actual values with the batch size that we got from the offline optimization
    # # TODO: should lie on a convex function w.r.t accuracy
    # # thput[variant] = 50 + np.random.random() * 50       # should lie on a convex function
    # #                                                     # w.r.t accuracy
    # thput[variant] = 1000 / (acc[variant] * 10)          # should lie on a convex function
    #                                                     # w.r.t accuracy
    # runtime = runtime_df
    thput[variant] = offline_opt_df[(offline_opt_df['model_1'] == variant[0]) &
                                    (offline_opt_df['model_2'] == f'{variant[1]}_checkpoint_150epochs')]['objective'].values[0]
    # print(f'thput[{variant}]: {thput[variant]}')
    
    # exit()

    # TODO: use the servers that we got from the offline optimization
    # if acc[variant] > 0.9:
    #     servers[variant] = 10
    # else:
    #     servers[variant] = 5
    # servers[variant] = round(acc[variant] * 5)
    servers[variant] = np.random.randint(1, 10)
    # servers[variant] = offline_opt_df[(offline_opt_df['model_1'] == variant[0]) &
    #                                   (offline_opt_df['model_2'] == f'{variant[1]}_checkpoint_150epochs')][]

    # # TODO: use actual multiplicative factor calculated from the factors
    # #       in each variant
    # mult_factor[variant] = 1 + np.random.random() * 10  # what should this be?
    mult_factor[variant] = offline_opt_df[(offline_opt_df['model_1'] == variant[0]) &
                                          (offline_opt_df['model_2'] == f'{variant[1]}_checkpoint_150epochs')]['pipeline_mult_factor'].values[0]


# the total number of servers
S = 10

# the total incoming demand into the system
D = 200


gp_model = gp.Model()

x = gp_model.addVars(variants, vtype=gp.GRB.INTEGER)
z = gp_model.addVar(vtype=gp.GRB.CONTINUOUS)


gp_model.addConstr(sum(x[v] * servers[v] for v in variants) <= S)

gp_model.addConstr(sum(x[v] * thput[v] for v in variants) >= D)

gp_model.addConstr(sum(x[v] * thput[v] * mult_factor[v] for v in variants) * z == 1.0)

gp_model.setObjective(sum(x[v] * thput[v] * mult_factor[v] * acc[v]
                          for v in variants) * z, gp.GRB.MAXIMIZE)

gp_model.setParam('NonConvex', 2)
gp_model.optimize()

for v in variants:
    if x[v].x > 0:
        print(f'x[{v}]: {x[v].x}, acc: {acc[v]}, thput: {thput[v]}, servers needed: '
            f'{servers[v]}')
