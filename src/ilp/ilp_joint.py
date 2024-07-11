import time
import pprint
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class IlpPipelineJointFormulation:
    def __init__(self):
        self.m = gp.Model('Pipeline joint formulation')
        return
    

    def create(self, devices, graph, deadline, allowed_batch_sizes):
        print(f'devices: {devices}')
        print(f'graph: {graph}')
        print(f'deadline: {deadline} milliseconds')
        print(f'allowed_batch_sizes: {allowed_batch_sizes}')

        v = graph['V']
        # tasks = v.shape[0]

        print(f'v: {v}')
        # print(f'v shape: {v.shape}')
        # print(f'tasks: {tasks}')

        gp_model = self.m

        gp_model.setParam('NonConvex', 2)

        # Profiled peak throughput of model variant on device for a batch size
        P = {}
        L = {}
        x_idx = {}
        z_idx = {}
        q = {}
        for model_family in v:
            model_variants = v[model_family]
            for model_variant in model_variants:
                # TODO: use profiled or estimated values (or atleast placeholder values)
                # q[model_family, model_variant] = 0
                q[model_family, model_variant] = 2
                for device in devices:
                    x_idx[model_family, model_variant, device] = 0
                    for batch_size in allowed_batch_sizes:
                        # TODO: use profiled values (or atleast placeholder values)
                        # P[model_family, model_variant, device, batch_size] = 0
                        # P[model_family, model_variant, device, batch_size] = 20 + 50 * \
                        #                                                     np.random.rand()
                        P[model_family, model_variant, device, batch_size] = 100
                        L[model_family, model_variant, device, batch_size] = 1000 / P[model_family,
                                                                                      model_variant,
                                                                                      device,
                                                                                      batch_size]

                        z_idx[batch_size, device] = 0

        # print(f'P: {P}')

        # TODO: hard-coded value, needs to be input into the function
        l = 50

        # print(f'multidict: {gp.multidict(P)}')

        # P_indices, _ = gp.multidict(P)
        x_indices, _ = gp.multidict(x_idx)
        z_indices, _ = gp.multidict(z_idx)
        model_pairs, _ = gp.multidict(q)

        fin_aux_idx = {}
        s_idx = {}
        for (i, m) in model_pairs:
            for device in devices:
                for (i_p, m_p) in model_pairs:
                    for device_p in devices:
                        fin_aux_idx[i, m, device, i_p, m_p, device_p] = 1
                        s_idx[device, device_p] = 1

        fin_aux_indices, _ = gp.multidict(fin_aux_idx)
        s_indices, _ = gp.multidict(s_idx)

        # print(f'x_indices: {x_indices}')
        x = gp_model.addVars(x_indices, vtype=GRB.BINARY)
        _in = gp_model.addVars(x_indices)
        proc = gp_model.addVars(x_indices)
        out = gp_model.addVars(x_indices)
        fin = gp_model.addVars(x_indices)
        # aux = gp_model.addVars
        b = gp_model.addVars(devices)
        z = gp_model.addVars(z_indices, vtype=GRB.BINARY)
        makespan = gp_model.addVar(vtype=GRB.CONTINUOUS, name='makespan')
        fin_aux = gp_model.addVars(fin_aux_indices)
        s = gp_model.addVars(s_indices)

        edge_indicator = {}
        for variant1 in model_pairs:
            for variant2 in model_pairs:
                if (variant1, variant2) in graph['E']:
                    edge_indicator[variant1, variant2] = 1
                else:
                    edge_indicator[variant1, variant2] = 0
        # print(f'\nedge_indicator: {edge_indicator}\n')

        print(f'\nx variable: {x}\n')
        print(f'\nb variable: {b}\n')
        print(f'\nz variable: {z}\n')
        print(f'\nmodel_pairs: {model_pairs}\n')

        for (i, m) in model_pairs:
            print(f'i: {i}, m: {m}')

        # -------------------------------
        # Constraints on pipeline latency
        # -------------------------------

        for device in devices:
            # print(f'(batch_size, device): {[(batch_size, device) for batch_size in allowed_batch_sizes]}')
            # print(f'(i, m, device): {[(i, m, device) for (i, m) in model_pairs]}')
            # print(f'batch_size: {[allowed_batch_sizes[batch_size] for batch_size in range(len(allowed_batch_sizes))]}')
            gp_model.addConstr(b[device] == sum(sum(z[allowed_batch_sizes[j], device] *
                                                    allowed_batch_sizes[j] *
                                                    x[i, m, device]
                                                    for (i, m) in model_pairs)
                                                for j in range(len(allowed_batch_sizes))),
                                            f'c_batch_{device}')
            
            gp_model.addConstr((sum(z[allowed_batch_sizes[j], device]
                                    for j in range(len(allowed_batch_sizes))) == 1),
                               f'c_z_{device}')


        # -------------------------------
        # Constraints on pipeline latency
        # -------------------------------

        for device in devices:
            for (i, m) in model_pairs:
                gp_model.addConstr(fin[i, m, device] ==
                                   gp.max_([fin_aux[i, m, device, i_p, m_p, device_p]
                                            for (i_p, m_p) in model_pairs
                                            for device_p in devices]))

                for (i_p, m_p) in model_pairs:
                    for device_p in devices:
                        gp_model.addConstr(fin_aux[i, m, device, i_p, m_p, device_p] ==
                                           sum(L[i, m, device, allowed_batch_sizes[j]] *
                                               z[allowed_batch_sizes[j], device]
                                               for j in range(len(allowed_batch_sizes))) +
                                           edge_indicator[(i_p, m_p), (i, m)] *
                                           fin[i_p, m_p, device_p])

            print(f'\ndata: {gp.max_(fin)}\n')

        gp_model.addConstr(makespan == gp.max_(fin))
        gp_model.addConstr(makespan <= deadline)
        print(f'\nmakespan: {makespan}\n')
        
        # ------------------------------
        # Constraints on model placement
        # ------------------------------

        for device in devices:
            gp_model.addConstr(sum(x[i, m, device] for (i, m) in model_pairs) <= 1)

        # --------------------------
        # Constraints on each vertex
        # --------------------------

        for device in devices:
            for (i, m) in model_pairs:
                gp_model.addConstr(out[i, m, device] == proc[i, m, device] * q[i, m])

        # ----------------------------------
        # Constraints on workload assignment
        # ----------------------------------
        
        for device in devices:
            for (i, m) in model_pairs:
                gp_model.addConstr(proc[i, m, device] <= _in[i, m, device])
                gp_model.addConstr(proc[i, m, device] <= x[i, m, device] *
                                                        sum(P[i, m, device, allowed_batch_sizes[j]] *
                                                            z[allowed_batch_sizes[j], device]
                                                            for j in range(len(allowed_batch_sizes))))
        
        # Fixing the sink vertex to the sink device
        gp_model.addConstr(x['sink', 'sink', 'sink_device'] == 1)
                
        # Only have non-zero values for in when x is 1 (for a given i, m, d)
        gp_model.addConstr(sum(sum(_in[i, m, device]
                                   for (i, m) in model_pairs)
                               for device in devices) ==
                           sum(sum(x[i, m, device] * _in[i, m, device]
                                   for (i, m) in model_pairs)
                               for device in devices))
                
        # --------------------
        # Constraints on edges
        # --------------------

        for (i, m) in model_pairs:
            # gp_model.addConstr(sum(out[i, m, device] for device in devices) ==
            #                        sum(sum(x[i_p, m_p, device_p] *
            #                                _in[i_p, m_p, device_p] *
            #                                edge_indicator[(i, m), (i_p, m_p)]
            #                                for (i_p, m_p) in model_pairs
            #                                )
            #                            for device_p in devices
            #                            )
            #                    )
            for device in devices:
                for (i_p, m_p) in model_pairs:
                    for device_p in devices:
                        gp_model.addConstr(_in[i_p, m_p, device_p] ==
                                           out[i, m, device] *
                                           s[device, device_p] *
                                           x[i, m, device] *
                                           x[i_p, m_p, device_p] *
                                           edge_indicator[(i, m), (i_p, m_p)])
                        
        for device in devices:
            gp_model.addConstr(sum(s[device, device_p] for device_p in devices) == 1)
            
        # -------------------------------------------
        # Constraint on incoming load into the system
        # -------------------------------------------
        print(f'\nmodel_pairs: {model_pairs}\n')
        first_task_pairs = []

        # TODO: hard-coded value, needs to be input or read from the graph
        first_task = 'yolo'

        for (i, m) in model_pairs:
            if first_task == i:
                first_task_pairs.append((i, m))
        print(f'\nfirst_task_pairs: {first_task_pairs}\n')

        gp_model.addConstr(l == sum(sum(_in[i, m, device] for (i, m) in first_task_pairs)
                                    for device in devices))

        # ----------------------
        # Optimization objective
        # ----------------------

        gp_model.setObjective(gp.quicksum(out), GRB.MAXIMIZE)

        # ------------
        # Book-keeping
        # ------------

        self.x = x
        self.z = z
        self.b = b
        self._in = _in
        self.proc = proc
        self.out = out
        self.fin = fin
        self.fin_aux = fin_aux

        self.m = gp_model
        return
    

    def solve(self):
        start_time = time.time()
        print(f'Solving ILP..')

        # Solve ILP
        self.m.optimize()

        end_time = time.time()
        print(f'Time to solve ILP: {end_time-start_time} seconds')
        print(f'Solution: \n{self.m}')
        # print(f'\nx: {self.x}')
        print('\nx:')
        pprint.pprint(self.x)
        print('\nz:')
        pprint.pprint(self.z)
        # print(f'\nb: {self.b}')
        print('\nb:')
        pprint.pprint(self.b)
        # print(f'\nin: {self._in}')
        print('\nin:')
        pprint.pprint(self._in)
        # print(f'\nproc: {self.proc}')
        print('\nproc:')
        pprint.pprint(self.proc)
        # print(f'\nout: {self.out}')
        print('\nout:')
        pprint.pprint(self.out)
        print('\nfin:')
        pprint.pprint(self.fin)
        # print('\nfin_aux:')
        # pprint.pprint(self.fin_aux)

        return


if __name__ == "__main__":

    devices = ['CPU1', 'GPU1', 'GPU2', 'sink_device']
    graph = {
            'V': {'yolo': ['yolov5m'],
                  'resnet': ['resnet50', 'resnet101'],
                  'sink': ['sink']
                 },
            'E': {(('yolo', 'yolov5m'), ('resnet', 'resnet50')),
                  (('yolo', 'yolov5m'), ('resnet', 'resnet101')),
                  (('resnet', 'resnet50'), ('sink', 'sink')),
                  (('resnet', 'resnet101'), ('sink', 'sink'))
                 }
            }
    # graph = {
    #         'V': {'yolo': ['yolov5m'],
    #               'resnet': ['resnet50'],
    #               'gan': ['upscalegan'],
    #               'sink': ['sink']
    #              },
    #         'E': {(('yolo', 'yolov5m'), ('resnet', 'resnet50')),
    #               (('resnet', 'resnet50'), ('gan', 'upscalegan')),
    #               (('gan', 'upscalegan'), ('sink', 'sink'))
    #              }
    #         }
    deadline = 200
    allowed_batch_sizes = [1, 2, 4, 8, 16, 32]

    ilp = IlpPipelineJointFormulation()
    ilp.create(devices, graph, deadline, allowed_batch_sizes)
    ilp.solve()