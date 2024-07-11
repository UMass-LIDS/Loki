# This optimization is used offline to select the maximum batch size and
# device ratio for each model in every pipeline variant

import itertools
import numpy as np
import pandas as pd
import gurobipy as gp


# TODO: need multiplicative factor for every model q(i)
#       and branching probability for every edge (r(i,i'))
#       and throughput for every model variant for every batch size

# Latency SLO for the entire pipeline (seconds)
T = 10

runtime_df = pd.read_csv('../../profiling/profiled/runtimes_all.csv')
branching_df = pd.read_csv('../../profiling/profiled/branching_all.csv')

# Set of models
# models = ['yolov5x', 'efficientnet-b7', 'vgg16']
all_models = {
        #   'yolov5': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
          'yolov5': ['yolov5n', 'yolov5m', 'yolov5l', 'yolov5x'],
          'efficientnet': ['eb0_checkpoint_150epochs', 'eb1_checkpoint_150epochs',
                           'eb2_checkpoint_150epochs', 'eb3_checkpoint_150epochs',
                           'eb4_checkpoint_150epochs', 'eb5_checkpoint_150epochs',
                           'eb6_checkpoint_150epochs']
         }

pipeline_variants = list(itertools.product(all_models['yolov5'], all_models['efficientnet']))

results_columns = ['model_1', 'model_1_batch_size', 'model_1_device_share', 'model_1_mult_factor',
                   'model_2', 'model_2_batch_size', 'model_2_device_share',
                   'pipeline_mult_factor', 'k', 'objective']
results_df = pd.DataFrame(columns=results_columns)

# pv2 = zip(pipeline_variants, range(pipeline_variants))
pv2 = list(enumerate(pipeline_variants))
# pv2 = list(map(lambda x: (x[1], x[0]), pv2))
print(list(pv2))


# Initialize the edges
e = {}

# Set of possible batch sizes
B = [1, 2, 4, 8, 16, 32, 64]
# B = [64]

# Peak profiled throughput (QPS) for each model with each batch size
P = {}
q = {}

total_servers = 10
total_demand = 800
# total_demand = 1000

runtimes = {}

# Initialize branching factor for each edge
r = {}

for pipeline_variant in pipeline_variants:
    print(pipeline_variant)
    # continue

    models = list(pipeline_variant)

    all_pairs = [(model_1, model_2) for model_1 in models for model_2 in models]
    for _tuple in all_pairs:
        # e[('yolov5x', 'efficientnet-b7')] = 1
        # e[('yolov5x', 'vgg16')] = 1
        (model_1, model_2) = _tuple
        if 'yolov5' in model_1 and 'eb' in model_2:
            e[_tuple] = 1
            # TODO: use this when both branches are used
            # r[_tuple] = branching_df[branching_df['model'] == model_1]['avg_car_branch_prob'].values[0]
            r[_tuple] = 1
        else:
            # We are not making self-edges, that is intentional
            e[_tuple] = 0
            r[_tuple] = 0
    print(f'e: {e}')

    print(sum(e.values()))
    print(len(pipeline_variants))

    for model in pipeline_variant:
        print(f'\nmodel: {model}\n')
        # TODO: replace by actual profiled multiplicative factor of model

        branching_row = branching_df[branching_df['model'] == model]
        if branching_row.empty:
            q[model] = 1
        else:
            q[model] = branching_row['avg_mult_factor'].values[0]

        print(f'q[{model}]: {q[model]}\n')

        for batch_size in B:
            print(f'model: {model}, batch size: {batch_size}')
            runtime = runtime_df[(runtime_df['model'] == model) &
                                 (runtime_df['batch_size'] == batch_size)]['90th_pct_runtime'].values[0]
            print(f'runtime: {runtime}')
            runtimes[(model, batch_size)] = runtime
            P[(model, batch_size)] = batch_size / runtime

print(f'\nP: {P}')
print(f'\nruntimes: {runtimes}')
print(f'\nq: {q}\n')

print(f'\npipeline variants: {pipeline_variants}')

print(f'\ne: {e}')


# While model is not solved, keep trying by increasing k
k = 10
k_increment = 10
opt_solved = False

while not(opt_solved):

    # Initializing the Gurobi model
    gp_model = gp.Model('Offline batch size and device ratio selection')
    
    b_indices = []
    # for model in models:
    #     for j in range(len(B)):
    #         b_indices.append((model, j))
    b_counter = 0
    for pipeline_variant in pipeline_variants:
        for model_variant in pipeline_variant:
            for j in range(len(B)):
                b_indices.append((model_variant, pipeline_variant, j))
                b_counter += 1

    print(f'\nb_indices: {b_indices}')
    print(f'\nb_counter: {b_counter}')

    iv_indices = []
    iv_counter = 0
    for pipeline_variant in pipeline_variants:
        for model_variant in pipeline_variant:
            iv_indices.append((model_variant, pipeline_variant))
            iv_counter += 1

    # Initializing decision variables
    b = gp_model.addVars(b_indices, vtype=gp.GRB.BINARY)
    z = gp_model.addVars(iv_indices, vtype=gp.GRB.CONTINUOUS)
    x = gp_model.addVars(iv_indices, vtype=gp.GRB.INTEGER)
    s = gp_model.addVars(iv_indices, vtype=gp.GRB.CONTINUOUS)
    # s_inv = gp_model.addVars(models, vtype=gp.GRB.CONTINUOUS)

    print(f'\nx: {x}')
    print(f'\niv_counter: {iv_counter}')
    print(f'\nnum of pipeline variants: {len(pipeline_variants)}')

    # print(f'\n\nr: {r}\n\n')
    # print(f'\n\nq: {q}\n\n')
    # exit()

    gp_model.setParam("LogToConsole", 0)
    gp_model.setParam('Threads', 8)
    gp_model.setParam('NonConvex', 2)

    # # Optimization objective (sum of all throughputs)
    # gp_model.setObjective(sum(d[i]*sum(P[i, B[j]]*b[i, j] for j in range(len(B)))
    #                           for i in models), gp.GRB.MAXIMIZE)

    first_model = None
    second_model = None
    for i in models:
        if 'yolov5' in i:
            first_model = i
        if 'eb' in i:
            second_model = i

    # gp_model.addConstr(b[first_model, 5] == 1)
    # gp_model.addConstr(b[second_model, 6] == 1)

    # # Optimization objective (first stage throughput only)
    # objective = d[first_model]*sum(P[first_model, B[j]]*b[first_model, j]
    #                                 for j in range(len(B)))
    # gp_model.setObjective(objective, gp.GRB.MAXIMIZE)


    # Constraints
    for v in pipeline_variants:
        for i in v:
            gp_model.addConstr(sum(b[i, v, j] for j in range(len(B))) == 1)

    for v in pipeline_variants:
        for i in v:
            gp_model.addConstr(sum(P[i, B[j]]*b[i, v, j] for j in range(len(B)))
                               * z[i, v] == 1)

    for v in pipeline_variants:
        gp_model.addConstr(sum(z[i, v] * sum(B[j]*b[i, v, j] for j in range(len(B)))
                               for i in v) <= T)
    
    gp_model.addConstr(sum(sum(x[i, v] for i in v) for v in pipeline_variants)
                       <= total_servers)
    
    gp_model.addConstr(sum(x[v[0], v]*sum(P[v[0], B[j]]*b[v[0], v, j] for j in range(len(B)))
                           for v in pipeline_variants) >= total_demand)
    
    for v in pipeline_variants:
        for i in v:
            gp_model.addConstr(s[i, v] == sum(x[i, v]*P[(i, B[j])]*b[i, v, j]
                                              for j in range(len(B))))
    
    for v in pipeline_variants:
        for i in v:
            for i_p in v:
                # We want the absolute value | s.q.r - s' | . e <= k
                # So we express it as two constraints:
                gp_model.addConstr((s[i, v]*q[i]*r[i, i_p] - s[i_p, v]) * e[i, i_p] <= k)

                gp_model.addConstr((s[i_p, v] - s[i, v]*q[i]*r[i, i_p]) * e[i, i_p] <= k)

    # Optimize the model
    gp_model.optimize()

    if gp_model.Status == 2:
        # Gurobi OPTIMAL status code
        opt_solved = True

    elif gp_model.Status == 3:
        # Gurobi INFEASIBLE status code
        k += k_increment

    _sum = 0
    throughput = 0
    for v in pipeline_variants:
        for i in v:
            if x[i, v].x <= 0:
                continue

            print(f'\nx[{i}, {v}]: {x[i, v].x}')
            _sum += x[i, v].x
            print(f'batch size: {sum(B[j]*b[i, v, j].x for j in range(len(B)))}')
            print(f'1 replica throughput: {sum(P[i, B[j]]*b[i,v,j].x for j in range(len(B)))}')
            replicas_times_throughput = x[i, v].x * sum(P[i, B[j]]*b[i,v,j].x for j in range(len(B)))
            print(f'{x[i, v].x} replicas throughput: {replicas_times_throughput}')
            print(f'{x[i, v].x} replicas throughput * q: {replicas_times_throughput * q[i]}')
            print(f'throughputs w/ different batch sizes: {[P[i, B[j]] for j in range(len(B))]}')
        throughput += x[v[0], v].x * sum(P[v[0], B[j]]*b[v[0], v, j].x
                                         for j in range(len(B)))
    print(f'\n_sum: {_sum}')
    print(f'throughput: {throughput}')

    print(f'solved with k: {k}')

# print(f'\ne: {e}')

# print('b values..')
# for i in models:
#     print(f'b[{i}]: {sum(b[i, j].x*B[j] for j in range(len(B)))}')
#     for j in range(len(B)):
#         print(f'b[{i, j}]: {b[i, j].x}, B[{j}]: {B[j]}')
# print()

# print('d values..')
# for i in models:
#     print(f'd[{i}]: {d[i].x}')
# print()

# print('s values..')
# for i in models:
#     print(f's[{i}]: {s[i].x}')
#     print(f'sum(s[{i}]*q[i]*r): {sum(s[i].x*q[i]*r[i, i_p] for i_p in models)}')
# print()

# print('z values..')
# for i in models:
#     print(f'z[{i}]*b[{i}]: {sum(b[i,j].x*B[j]*z[i].x for j in range(len(B)))}')
# print()

# print(f'sum(z*b): {sum(sum(B[j]*b[i,j].x*z[i].x for j in range(len(B))) for i in models)}, '
#     f'should be close to T')

# print(f'\nSolved model with k: {k}')

# # print(f'first_model: {first_model}')

# model_names = []
# model_batch_sizes = []
# model_device_shares = []

# for i in models:
#     model_names.append(i)
#     model_batch_sizes.append(sum(B[j]*b[i,j].x for j in range(len(B))))
#     model_device_shares.append(d[i].x)
#     if 'yolo' in i:
#         yolo_model_name = i

# # print(f'objective: {objective}')
# # objective = sum(sum(P[i, B[j]]*b[i, j].x for j in range(len(B))) for i in models)
# objective = sum(d[first_model].x*P[first_model, B[j]]*b[first_model, j].x
#                 for j in range(len(B)))

# new_row = {'model_1': model_names[0], 'model_1_batch_size': model_batch_sizes[0],
#             'model_1_device_share': model_device_shares[0],
#             'model_1_mult_factor': q[yolo_model_name], 'model_2': model_names[1],
#             'model_2_batch_size': model_batch_sizes[1],
#             'model_2_device_share': model_device_shares[1],
#             # TODO: pipeline_mult_factor will change if downstream models also have
#             #       their own multiplicative factors
#             'pipeline_mult_factor': q[yolo_model_name], 'k': k, 'objective': objective}
# results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# print(f'results_df: {results_df}')
# results_df.to_csv('results_offline_opt.csv')
