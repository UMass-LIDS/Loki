import pprint
import itertools
import logging
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from typing import List
from algorithms.base import SchedulingAlgorithm
from common.app import App
from core.simulator import Simulator


MSecsInSec = 1000
USecsInSec = 1000 * 1000


class PipelineIlp(SchedulingAlgorithm):
    def __init__(self, simulator: Simulator=None, loggingLevel=logging.INFO,
                 accuracies: pd.DataFrame=None, runtimes: dict=None,
                 beta: float=None):
        SchedulingAlgorithm.__init__(self, 'Pipeline ILP')

        self.log = logging.getLogger(__name__)
        # self.log.addHandler(logging.FileHandler('logs/pipeline_ilp/output.log',
        #                                         mode='w'))
        self.log.setLevel(loggingLevel)

        self.simulator = simulator

        self.accuracies = accuracies
        self.runtimes = runtimes

        self.beta = beta

        expColumns = ['demand', 'accuracy', 'x_variable', 'c_variable',
                      'y_variable']
        self.expDf = pd.DataFrame(columns=expColumns)

        return
    

    def run(self, observation: np.ndarray, num_acc_types: int, num_max_acc: int):
        ''' Runs hardware scaling if possible, otherwise runs accuracy scaling
        '''
        self.log.info(f'Solving ILP. Clock: {self.simulator.clock}')

        num_isi = observation.shape[0] - 1
        current_alloc = observation[0:num_isi, 0:num_acc_types]

        appName = 'traffic_analysis'
        # appName = 'social_media'
        
        # TODO: allow multiple apps instead of just 1
        # TODO: do not use hard-coded app name
        app = self.simulator.apps[appName]

        # Getting all the app tasks
        tasks = app.getAllTasks()

        # We don't need to include sink in the ILP, we will just allocate
        # atleast one server for sink at all times
        if 'sink' in tasks:
            tasks.remove('sink')
        if 'facial_recognition' in tasks:
            tasks.remove('facial_recognition')

        # Loading the variants for each task from the config file
        variants = []
        for task in tasks:
            appNode = app.findNodeByTask(task=task)
            taskVariants = appNode.modelVariants
            for variant in taskVariants:
                variants.append((task, variant))
        if ('object_detection', 'yolov5s') in variants:
            variants.remove(('object_detection', 'yolov5s'))
        if ('car_classification', 'efficientnet-b7') in variants:
            variants.remove(('car_classification', 'efficientnet-b7'))
        
        # TODO: Construct taskPaths from the app graph
        # TODO: Currently only considers chain. Make it more generic by using
        #       branches so we can construct any tree (but not DAG)
        if appName == 'traffic_analysis':
            taskPaths = [
                        ('object_detection', 'car_classification'),
                        # ('object_detection', 'facial_recognition')
                        ]
        elif appName == 'social_media':
            taskPaths = [
                        ('source', 'object_detection'),
                        ('source', 'image_captioning')
                        ]
        else:
            raise Exception(f'Unexpected app: {appName}')

        # This is needed to generate all paths
        taskToVariants = {}
        for task in tasks:
            taskVariants = list(filter(lambda x: task in x, variants))
            taskVariants = list(map(lambda x: x[1], taskVariants))
            taskToVariants[task] = taskVariants

        print('\ntaskToVariants:')
        pprint.pprint(taskToVariants)

        pathsForTaskPaths = {}

        # All possible paths in the system
        paths = []
        for taskPath in taskPaths:
            # We want to add all variants for every taskPath
            currentProduct = None
            for task in taskPath:
                if currentProduct is None:
                    currentProduct = taskToVariants[task]
                else:
                    currentProduct = list(itertools.product(currentProduct,
                                                            taskToVariants[task]))
            pathsForTaskPath = list(currentProduct)
            paths.extend(pathsForTaskPath)

            pathsForTaskPaths[taskPath] = pathsForTaskPath
        # print(f'\n\npaths: (len {len(paths)})')
        # pprint.pprint(paths)
        # print(f'taskPaths:')
        # pprint.pprint(taskPaths)

        # Multiplicative factor for every task, task', variant combination
        multFactor = {}
        # TODO: For now, we use hard-coded values to test optimization
        if appName == 'traffic_analysis':
            multFactor[('object_detection', 'car_classification', 'yolov5n')] = 2.584 * 0.768 * self.beta
            multFactor[('object_detection', 'car_classification', 'yolov5s')] = 5.464 * 0.932 * self.beta
            multFactor[('object_detection', 'car_classification', 'yolov5m')] = 14.373 * 0.958 * self.beta
            multFactor[('object_detection', 'car_classification', 'yolov5l')] = 18.207 * 0.959 * self.beta
            multFactor[('object_detection', 'car_classification', 'yolov5x')] = 18.894 * 0.943 * self.beta

            multFactor[('object_detection', 'facial_recognition', 'yolov5n')] = 2.584 * 0.232 * self.beta
            multFactor[('object_detection', 'facial_recognition', 'yolov5s')] = 5.464 * 0.068 * self.beta
            multFactor[('object_detection', 'facial_recognition', 'yolov5m')] = 14.373 * 0.042 * self.beta
            multFactor[('object_detection', 'facial_recognition', 'yolov5l')] = 18.207 * 0.041 * self.beta
            multFactor[('object_detection', 'facial_recognition', 'yolov5x')] = 18.894 * 0.057 * self.beta
        elif appName == 'social_media':
            # TODO: For now, we use hard-coded values to test optimization
            # Multiplying the bottleneck with the over-provisioning factor
            multFactor[('source', 'object_detection', 'source')] = 1 * self.beta
            multFactor[('source', 'image_captioning', 'source')] = 1 * self.beta
        else:
            raise Exception(f'Unexpected app: {appName}')

        self.log.warning('Do not use hard-coded multiplicative factors in the ILP')

        # # Updating multFactor by reading real-time values from predictor
        # # If no predictor is present for a model variant, system will default
        # # to average profiled values
        # clock = self.simulator.clock
        # yoloVariants = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        # for yoloVariant in yoloVariants:
        #     foundPredictor = None
        #     for predictor in self.simulator.get_all_predictors():
        #         if predictor.variant_name == yoloVariant:
        #             foundPredictor = predictor
        #             break
        #     if foundPredictor is None:
        #         continue
        #     else:
        #         request_trace_max_time = self.simulator.request_trace_max_times['object_detection']
        #         scaled_clock_time = clock / request_trace_max_time
        #         # We now scale it to be in [0,branching_trace_max_time]
        #         scaled_clock_time = int(scaled_clock_time *
        #                                 foundPredictor.branchingTraceMaxTime)
        #         df_row = foundPredictor.branchingTrace.loc[scaled_clock_time]
        #         factor = df_row['car_classification']
        #     multFactor[('object_detection', 'car_classification', yoloVariant)] = factor

        # print('\nmultFactor:')
        # pprint.pprint(multFactor)
        
        # Throughput for every task, variant combination
        # TODO: Currently only using CPU runtimes. If cluster is heterogeneous,
        #       remove this hard-coding
        throughputs = {}
        # runtimes = self.simulator.model_variant_runtimes[1]
        runtimes = self.runtimes[1]
        for key in runtimes:
            (task, variant, batchSize) = key
            runtime = runtimes[key]
            throughput = batchSize / (runtime / MSecsInSec)
            throughputs[key] = throughput

        # print(f'simulator.model_variant_runtimes: '
        #       f'{self.simulator.model_variant_runtimes}')
        # print(f'throughputs: {throughputs}')

        # Accuracy for every path
        accuracy = {}

        # # TODO: Hard-coded for two stages currently. Generalize it
        # scaleStartPoint = 70
        # scaleEndPoint = 100
        # scaleWindow = (scaleEndPoint - scaleStartPoint) / scaleEndPoint
        # for _, row in self.accuracies.iterrows():
        #     elements = row['path'].split('_')
        #     firstVariant = elements[-1]
        #     secondVariant = elements[0]
        #     secondVariant = secondVariant.replace('eb', 'efficientnet-b')
        #     path = (firstVariant, secondVariant)
        #     pathAccuracy = float(row['e2e_acc']) * 100
        #     scaledAccuracy = pathAccuracy * scaleWindow + scaleStartPoint
        #     accuracy[path] = scaledAccuracy

        # TODO: Assuming end-to-end accuracy to be multiplicative
        singleModelAccuracies = self.simulator.model_variant_accuracies
        for path in paths:
            pathAccuracy = 100
            for model in path:
                modelTask = app.findTaskFromModelVariant(model)
                modelAccuracy = singleModelAccuracies[(modelTask, model)] / 100
                pathAccuracy *= modelAccuracy
            accuracy[path] = pathAccuracy

        # We get the total number of servers from the simulator
        # TODO: ILP does not work with heterogeneous cluster
        totalServers = sum(self.simulator.predictors_max) - 2

        # Reading the first task's demand measured by the simulator as a
        # moving average
        topSortedTasks = app.getAllTasksTopSorted()
        firstTask = topSortedTasks[0]
        demand_since_last = self.simulator.ewma_demand.ravel()
        demand = demand_since_last / (self.simulator.allocation_window / 1000)
        firstTaskIdx = self.simulator.isi_to_idx[firstTask]
        firstTaskEWMADemand = demand[firstTaskIdx]
        
        # TODO: Don't use hard-coded value, but the idea here is to use a
        #       sufficiently large enough demand so that cluster can be
        #       set up and warmed up
        # TODO: However, we will not see the effects of hardware scaling here
        if firstTaskEWMADemand < 50:
            firstTaskEWMADemand = 50

        totalDemand = firstTaskEWMADemand

        # Allowed batch sizes
        allowedBatchSizes = [1, 2, 4, 8, 16, 32, 64]
        # allowedBatchSizes = self.simulator.allowed_batch_sizes
        # allowedBatchSizes = [1]

        # Latency SLO in seconds
        slo = app.getLatencySLO() / USecsInSec

        x = None
        c = None

        # Modifying certain data structures for hardware scaling to only include
        # most accurate model variants and paths
        highestAccuracyVariants = []
        for (task, variant) in variants:
            if task == 'object_detection' and variant != 'yolov5x':
                continue
            if task == 'car_classification' and variant != 'efficientnet-b6':
                continue
            if task == 'facial_recognition' and variant != 'genderNet_19':
                continue
            if task == 'image_captioning' and variant != 'clip-vit-large-patch14-336':
                continue
            highestAccuracyVariants.append((task, variant))
        hwScalingVariants = highestAccuracyVariants

        newPaths = []
        for path in paths:
            allPathVariantsFound = True
            for pathVariant in path:
                pathVariantFound = False
                for (task, variant) in hwScalingVariants:
                    if variant == pathVariant:
                        pathVariantFound = True
                allPathVariantsFound = allPathVariantsFound and pathVariantFound
            if allPathVariantsFound:
                newPaths.append(path)
        hwScalingPaths = newPaths

        newTaskToVariants = {}
        for task in taskToVariants:
            if task == 'source':
                newTaskToVariants[task] = ['source']
            if task == 'object_detection':
                newTaskToVariants[task] = ['yolov5x']
            if task == 'car_classification':
                newTaskToVariants[task] = ['efficientnet-b6']
            if task == 'facial_recognition':
                newTaskToVariants[task] = ['genderNet_19']
            if task == 'image_captioning':
                newTaskToVariants[task] = ['clip-vit-large-patch14-336']
        hwScalingTaskToVariants = newTaskToVariants

        newPathsForTaskPaths = {}
        for taskPath in pathsForTaskPaths:
            pathsForTaskPath = pathsForTaskPaths[taskPath]
            newPathsForTaskPath = []
            for pp in pathsForTaskPath:
                if pp not in paths:
                    continue
                newPathsForTaskPath.append(pp)
            newPathsForTaskPaths[taskPath] = newPathsForTaskPath
        hwScalingPathsForTaskPaths = newPathsForTaskPaths

        startTime = time.time()
        status, x, c, y = self.solveForHardwareScaling(accuracy=accuracy,
                                                       pathsForTaskPaths=hwScalingPathsForTaskPaths,
                                                        paths=hwScalingPaths, tasks=tasks,
                                                        throughputs=throughputs,
                                                        taskToVariants=hwScalingTaskToVariants,
                                                        multFactor=multFactor,
                                                        variants=hwScalingVariants,
                                                        app=app,
                                                        totalDemand=totalDemand,
                                                        totalServers=totalServers,
                                                        allowedBatchSizes=allowedBatchSizes,
                                                        slo=slo,
                                                        beta=self.beta)
        
        if status == gp.GRB.Status.OPTIMAL:
            variants = hwScalingVariants
            paths = hwScalingPaths
            taskToVariants = hwScalingTaskToVariants
            pathsForTaskPaths = hwScalingPathsForTaskPaths
            print(f'Hardware scaling successful')

        else:
            status, x, c, y = self.solveForAccuracyScaling(accuracy=accuracy,
                                                           pathsForTaskPaths=pathsForTaskPaths,
                                                            paths=paths, tasks=tasks,
                                                            throughputs=throughputs,
                                                            taskToVariants=taskToVariants,
                                                            multFactor=multFactor,
                                                            variants=variants,
                                                            app=app,
                                                            totalDemand=totalDemand,
                                                            totalServers=totalServers,
                                                            allowedBatchSizes=allowedBatchSizes,
                                                            slo=slo,
                                                            beta=self.beta)
            
            if status != gp.GRB.Status.OPTIMAL:
                raise Exception(f'Neither hardware scaling nor accuracy scaling '
                                f'is possible')

        solveTime = time.time() - startTime
        print(f'ILP_SOLVE_TIME, clock: {self.simulator.clock}, solveTime: '
              f'{solveTime}')

        pendingRemovalPredictors = len(self.simulator.get_pending_removal_predictors())
        print(f'PENDING_REMOVAL_PREDICTORS, clock: {self.simulator.clock}, '
              f'pendingRemoval: {pendingRemovalPredictors}')

        # Allocating atleast one server for other branch
        if appName == 'traffic_analysis':
            x[('facial_recognition', 'genderNet_16')] = 1
            y[('facial_recognition', 'genderNet_16', allowedBatchSizes[0])] = 1
        elif appName == 'social_media':
            x[('image_captioning', 'clip-vit-large-patch14-336')] = 1
            y[('image_captioning', 'clip-vit-large-patch14-336', allowedBatchSizes[0])] = 1
        else:
            raise Exception(f'Unexpected app: {appName}')

        # Allocating atleast one server for sink
        x[('sink', 'sink')] = 1
        y[('sink', 'sink', allowedBatchSizes[-1])] = 1

        # print(f'\nthroughputs')
        # pprint.pprint(throughputs)

        # print(f'\nmultFactors')
        # pprint.pprint(multFactor)

        print('\nx')
        xString = ''
        for variant in variants:
            if x[variant] > 0:
                print(f'x[{variant}]: {x[variant]}')

                if xString == '':
                    xString = f'{variant[1]}:{x[variant]}'
                else:
                    xString += f';{variant[1]}:{x[variant]}'

        # print(f'xString: {xString}')
        # time.sleep(1)

        print('\nc')
        cString = ''
        for path in paths:
            if c[path] > 0:
                print(f'c[{path}]: {c[path]}')

                pathString = ''
                for variant in path:
                    if pathString == '':
                        pathString = variant
                    else:
                        pathString += '|' + variant

                if cString == '':
                    cString = f'{pathString}:{c[path]}'
                else:
                    cString += f';{pathString}:{c[path]}'

        yString = ''
        for task in tasks:
            for variant in taskToVariants[task]:
                for batchSize in allowedBatchSizes:
                    if y.get((task, variant, batchSize), 0) > 0:
                        print(f'y[{task}, {variant}, {batchSize}]: '
                                f'{y[task, variant, batchSize]}')
                        
                        yKeyString = (f'{task}|{variant}|{batchSize}:'
                                        f'{y[task, variant, batchSize]}')
                        
                        if yString == '':
                            yString = yKeyString
                        else:
                            yString += f';{yKeyString}'

        objective = sum(c[path]*accuracy[path] for path in paths)
        print(f'\nobjective: {objective}, demand: {totalDemand}, observed '
              f'demand:{demand[firstTaskIdx]}')

        # print(f'\naccuracy:')
        # pprint.pprint(accuracy)

        # pathThroughputs = {}
        # for path in paths:
        #     (yoloVariant, ebVariant) = path

        #     yoloThroughput = throughputs[('object_detection', yoloVariant, 1)]
        #     ebThroughput = throughputs[('car_classification', ebVariant, 1)]

        #     yoloRepFactor = 1
        #     ebRepFactor = yoloThroughput * yoloRepFactor * multFactor[('object_detection', 'car_classification', yoloVariant)] / ebThroughput

        #     totalRepFactor = yoloRepFactor + ebRepFactor

        #     yoloRepFactor /= totalRepFactor
        #     ebRepFactor /= totalRepFactor

        #     # We could also get it through ebThroughput * ebRepFactor, since
        #     # they are perfectly balanced
        #     pathThroughput = yoloThroughput * yoloRepFactor * multFactor[('object_detection', 'car_classification', yoloVariant)]

        #     pathThroughputs[path] = pathThroughput

        # pathAccToThput = {}
        # for path in paths:
        #     pathAccToThput[path] = accuracy[path] / pathThroughputs[path]

        # print(f'path throughputs:')
        # pprint.pprint(pathThroughputs)

        # print(f'path accuracy to throughput value:')
        # pprint.pprint(pathAccToThput)

        # Allocate atleast one server for all tasks that do not have any
        # servers allocated
        for task in tasks:
            taskHasServers = False
            for allocationEntry in x:
                if task in allocationEntry:
                    taskHasServers = True
                    break
            if not(taskHasServers):
                # Allocate one server for any of its model variants and set its
                # batch size to 1
                appNode = app.findNodeByTask(task)
                variants = appNode.modelVariants
                selectedVariant = variants[-1]
                x[(task, selectedVariant)] = 1
                y[(task, selectedVariant, 1)] = 1

        newRow = {'demand': totalDemand, 'accuracy': objective,
                  'x_variable': xString, 'c_variable': cString,
                  'y_variable': yString}
        self.expDf = self.expDf.append(newRow, ignore_index=True)

        # Apply solution to simulator
        self.simulator.apply_pipeline_ilp_homogeneous_solution(x=x, y=y, c=c)

        # print(f'expDf: {expDf}')
        expLogFilename = f'logs/pipeline_ilp/experiment_servers_{totalServers}.csv'
        self.expDf.to_csv(expLogFilename)

        return
    

    def solveForHardwareScaling(self, accuracy: dict, pathsForTaskPaths: dict,
                                paths: dict, tasks: List[str], throughputs: dict,
                                taskToVariants: dict, multFactor: dict, 
                                variants: List[tuple], app: App, totalDemand: float,
                                totalServers: int, allowedBatchSizes: List[int],
                                slo: float, beta: float) -> tuple:
        ''' We first try to solve the optimization for hardware scaling with
            the most accurate model variants
        '''
        # Creating the Gurobi optimization model
        gpModel = gp.Model('Hardware scaling ILP')

        # Make Gurobi non-verbose
        gpModel.setParam('LogToConsole', 0)

        # Optimization variables
        x = gpModel.addVars(variants, vtype=gp.GRB.INTEGER, name='x')
        c = gpModel.addVars(paths, name='c')
        y = gpModel.addVars(throughputs, vtype=gp.GRB.BINARY, name='y')

        # Intermediate variables
        ind = gpModel.addVars(paths, vtype=gp.GRB.BINARY, name='ind')

        # Derived variables
        # Latency of a model variant with its set batch size
        l = gpModel.addVars(variants, name='l')

        # These constants are for setting the value of the indicator variables
        epsilon = 0.001
        M = 100000

        # The objective is the number of servers
        gpModel.setObjective(sum(x[variant] for variant in variants),
                             gp.GRB.MINIMIZE)

        # We disallow batch sizes that are not feasible or those that are not
        # profiled. This is different for each (task, variant) pair
        taskVariantAllowedBatchSizes = {}
        for task in tasks:
            for variant in taskToVariants[task]:
                batchSizes = []
                for batchSize in allowedBatchSizes:
                    if throughputs[(task, variant, batchSize)] > 0:
                        batchSizes.append(batchSize)
                taskVariantAllowedBatchSizes[(task, variant)] = batchSizes

        # For every task in the graph, we need to construct constraints based
        # on preceding tasks (because we need to account for their mult factors)
        for task in tasks:
            for variant in taskToVariants[task]:
                # sum of all flows going through this path must be supported by
                # replicas of this variant
                # We need to derive the multiplicative factor for each path upfront
                # Not only each path, but also every task of each path
                # Basically, the derived q variable

                # We need all paths that contain this variant
                pathsWithVariant = list(filter(lambda x: variant in x, paths))
                
                multFactorsWithVariant = {}
                for path in pathsWithVariant:
                    # Basically, we want to go edge by edge (i.e., pairs of
                    # vertices), and we break when we run out of edges or we
                    # reach the edge where the first vertex is the current
                    # variant's task
                    multFactorsWithVariant[path] = 1
                    for i in range(len(path)-1):
                        vertex1Variant = path[i]
                        vertex2Variant = path[i+1]

                        vertex1Task = app.findTaskFromModelVariant(vertex1Variant)
                        vertex2Task = app.findTaskFromModelVariant(vertex2Variant)

                        if vertex1Variant == variant:
                            break
                        
                        key = (vertex1Task, vertex2Task, vertex1Variant)
                        if key not in multFactor:
                            # If there is no entry in multFactor, raise a warning
                            # and assume factor of 1
                            self.log.warning(f'Key {key} not found in multFactor')
                            continue
                        else:
                            multFactorsWithVariant[path] *= multFactor[key]

                # print(f'multFactorsWithVariant: {multFactorsWithVariant}')
                # exit(0)

                gpModel.addConstr(sum(c[path] * totalDemand * multFactorsWithVariant[path]
                                      for path in pathsWithVariant)
                                  <= sum(x[task, variant] *
                                         throughputs[task, variant, batchSize] *
                                         y[task, variant, batchSize] for batchSize
                                         in taskVariantAllowedBatchSizes[(task, variant)])
                                 )

        for task in tasks:
            for variant in taskToVariants[task]:
                # y is a binary variable to set the batch size of each model
                # variant
                # The replicas of a model variant must all use only one of the
                # allowed batch sizes
                gpModel.addConstr(sum(y[task, variant, batchSize] for batchSize
                                      in taskVariantAllowedBatchSizes[(task, variant)]) <= 1)

                # We derive the latency of each model variant based on the
                # configured batch size and its profiled throughput w/ that
                # batch size
                gpModel.addConstr(l[task, variant] == sum(y[task, variant, batchSize] * 
                                                          batchSize / 
                                                          throughputs[task, variant,
                                                                      batchSize]
                                                          for batchSize in 
                                                          taskVariantAllowedBatchSizes[(task, variant)]))

        # Setting the indicator variables for every path
        # If there is no traffic through a path, ind will be 0
        # If there is non-zero (atleast epsilon) traffic through a path, ind
        # will be 1
        for path in paths:
            gpModel.addConstr(c[path] >= epsilon + M * (ind[path] - 1))
            gpModel.addConstr(c[path] <= epsilon + M * ind[path])

        # For every path that has any traffic going through it, the sum of its
        # latencies should be at most SLO / 2
        for path in paths:
            pathLatency = None
            for variant in path:
                task = app.findTaskFromModelVariant(variant)
                if pathLatency is None:
                    pathLatency = l[task, variant]
                else:
                    pathLatency += l[task, variant]

            gpModel.addConstr(pathLatency * ind[path] <= slo / 2)
        
        # We want effective accuracy to be the highest possible, i.e.,
        # Fraction of requests going through a path multiplied by accuracy of
        # the path, summed up over all paths
        maxAccuracy = 0
        for path in paths:
            maxAccuracy = max(maxAccuracy, accuracy[path])

        # gpModel.addConstr(sum(c[path] * accuracy[path] for path in paths)
        #                   >= maxAccuracy)

        # (Branch agnostic) Sum of fraction of requests through all paths
        # must equal 1
        gpModel.addConstr(sum(c[path] for path in paths) == 1)

        # # (Branch aware)
        # for taskPath in pathsForTaskPaths:
        #     pathsForTaskPath = pathsForTaskPaths[taskPath]

        #     # # Sum of fraction of requests through all paths for a taskPath
        #     # # must equal 1 (A task path is a path through the application graph
        #     # # of tasks, not model variants). This is to ensure that traffic
        #     # # flows through each branch of the application
        #     # gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) == 1)

        #     # # We want effective accuracy to be the highest possible, i.e.,
        #     # # Fraction of requests going through a path multiplied by accuracy of
        #     # # the path, summed up over all paths
        #     # maxAccuracyForTaskPath = 0
        #     # for path in pathsForTaskPath:
        #     #     maxAccuracyForTaskPath = max(maxAccuracyForTaskPath,
        #     #                                  accuracy[path])

        #     # gpModel.addConstr(sum(c[path] * accuracy[path] for path in pathsForTaskPath)
        #     #                   >= maxAccuracyForTaskPath)

        #     # print(f'taskPath: {taskPath}')
        #     # time.sleep(1)
        #     if 'car_classification' in taskPath:
        #         print(f'pathsForTaskPath: {pathsForTaskPath}')
        #         # time.sleep(1)
        #         gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) >= 16.0/18.0)
        #     if 'image_captioning' in taskPath:
        #         gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) >= 0.3)
        #         gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) <= 0.7)

        # Must not allocate more replicas than servers in the system
        gpModel.addConstr(sum(x[variant] for variant in variants) <= totalServers)

        # Solve the optimization
        gpModel.optimize()
        
        # Constructing x values to return
        xValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for variant in variants:
                xValues[variant] = round(x[variant].X)

        # Constructing c values to return
        cValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for path in paths:
                cValues[path] = c[path].X

        # Constructing y values to return
        yValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for task in tasks:
                for variant in taskToVariants[task]:
                    for batchSize in taskVariantAllowedBatchSizes[(task, variant)]:
                        yValues[task, variant, batchSize] = y[task, variant,
                                                              batchSize].X

        return gpModel.status, xValues, cValues, yValues
    

    def solveForAccuracyScaling(self, accuracy: dict, pathsForTaskPaths: dict,
                                paths: dict, tasks: List[str], throughputs: dict,
                                taskToVariants: dict, multFactor: dict, 
                                variants: List[tuple], app: App, totalDemand: float,
                                totalServers: int, allowedBatchSizes: List[int],
                                slo: float, beta: float) -> tuple:
        ''' If hardware scaling is infeasible, we solve the optimization for
            accuracy scaling
        '''
        # Creating the Gurobi optimization model
        gpModel = gp.Model('Accuracy scaling ILP')

        # Make Gurobi non-verbose
        gpModel.setParam('LogToConsole', 0)

        # Optimization variables
        x = gpModel.addVars(variants, vtype=gp.GRB.INTEGER, name='x')
        c = gpModel.addVars(paths, name='c')
        y = gpModel.addVars(throughputs, vtype=gp.GRB.BINARY, name='y')

        # Intermediate variables
        ind = gpModel.addVars(paths, vtype=gp.GRB.BINARY, name='ind')

        # Derived variables
        # Latency of a model variant with its set batch size
        l = gpModel.addVars(variants, name='l')

        # These constants are for setting the value of the indicator variables
        epsilon = 0.001
        M = 100000

        # TODO: ILP should form obj/constraints based on edges of the graph,
        #       which it should get from the app

        # The objective is effective accuracy, i.e.,
        # Fraction of requests going through a path multiplied by accuracy of
        # the path, summed up over all paths
        gpModel.setObjective(sum(c[path]*accuracy[path] for path in paths),
                             gp.GRB.MAXIMIZE)

        # We disallow batch sizes that are not feasible or those that are not
        # profiled. This is different for each (task, variant) pair
        taskVariantAllowedBatchSizes = {}
        for task in tasks:
            for variant in taskToVariants[task]:
                batchSizes = []
                for batchSize in allowedBatchSizes:
                    if throughputs[(task, variant, batchSize)] > 0:
                        batchSizes.append(batchSize)
                taskVariantAllowedBatchSizes[(task, variant)] = batchSizes

        # For every task in the graph, we need to construct constraints based
        # on preceding tasks (because we need to account for their mult factors)
        for task in tasks:
            for variant in taskToVariants[task]:
                # sum of all flows going through this path must be supported by
                # replicas of this variant
                # We need to derive the multiplicative factor for each path upfront
                # Not only each path, but also every task of each path
                # Basically, the derived q variable

                # We need all paths that contain this variant
                pathsWithVariant = list(filter(lambda x: variant in x, paths))
                
                multFactorsWithVariant = {}
                # TODO: We need to debug this to see if this is correct
                for path in pathsWithVariant:
                    # Basically, we want to go edge by edge (i.e., pairs of
                    # vertices), and we break when we run out of edges or we
                    # reach the edge where the first vertex is the current
                    # variant's task
                    multFactorsWithVariant[path] = 1
                    for i in range(len(path)-1):
                        vertex1Variant = path[i]
                        vertex2Variant = path[i+1]

                        vertex1Task = app.findTaskFromModelVariant(vertex1Variant)
                        vertex2Task = app.findTaskFromModelVariant(vertex2Variant)

                        if vertex1Variant == variant:
                            break
                        
                        # print(f'\npath: {path}')
                        # print(f'vertex1Variant: {vertex1Variant}, vertex1Task: '
                        #       f'{vertex1Task}, vertex2Variant:  {vertex2Variant}, '
                        #       f'vertex2Task: {vertex2Task}, key: {key}')
                        key = (vertex1Task, vertex2Task, vertex1Variant)
                        if key not in multFactor:
                            # If there is no entry in multFactor, raise a warning
                            # and assume factor of 1
                            self.log.warning(f'Key {key} not found in multFactor')
                            continue
                        else:
                            multFactorsWithVariant[path] *= multFactor[key]

                # TODO: Remove this when the one above has been debugged
                #       This has changed from vertex-based to edge-based
                # for path in pathsWithVariant:
                #     # For each path, we derive the appropriate multiplicative
                #     # factor
                #     multFactorsWithVariant[path] = 1
                #     for pathVariant in path:
                #         if pathVariant == variant:
                #             break
                #         multFactorsWithVariant[path] *= multFactor[pathVariant]

                # print(f'\nvariant: {variant}, pathsWithVariant: {pathsWithVariant}')
                # print(f'variant: {variant}, multFactorsWithVariant: ')
                # pprint.pprint(multFactorsWithVariant)
                gpModel.addConstr(sum(c[path] * totalDemand * multFactorsWithVariant[path]
                                      for path in pathsWithVariant)
                                  <= sum(x[task, variant] *
                                         throughputs[task, variant, batchSize] *
                                         y[task, variant, batchSize] for batchSize
                                         in taskVariantAllowedBatchSizes[(task, variant)])
                                 )

        for task in tasks:
            for variant in taskToVariants[task]:
                # y is a binary variable to set the batch size of each model
                # variant
                # The replicas of a model variant must all use only one of the
                # allowed batch sizes
                gpModel.addConstr(sum(y[task, variant, batchSize] for batchSize
                                      in taskVariantAllowedBatchSizes[(task, variant)]) <= 1)

                # We derive the latency of each model variant based on the
                # configured batch size and its profiled throughput w/ that
                # batch size
                gpModel.addConstr(l[task, variant] == sum(y[task, variant, batchSize] * 
                                                          batchSize / 
                                                          throughputs[task, variant,
                                                                      batchSize]
                                                          for batchSize in 
                                                          taskVariantAllowedBatchSizes[(task, variant)]))

        # Setting the indicator variables for every path
        # If there is no traffic through a path, ind will be 0
        # If there is non-zero (atleast epsilon) traffic through a path, ind
        # will be 1
        for path in paths:
            gpModel.addConstr(c[path] >= epsilon + M * (ind[path] - 1))
            gpModel.addConstr(c[path] <= epsilon + M * ind[path])

        # For every path that has any traffic going through it, the sum of its
        # latencies should be at most SLO / 2
        for path in paths:
            pathLatency = None
            for variant in path:
                task = app.findTaskFromModelVariant(variant)
                if pathLatency is None:
                    pathLatency = l[task, variant]
                else:
                    pathLatency += l[task, variant]

            gpModel.addConstr(pathLatency * ind[path] <= slo / 2)

        # (Branch agnostic)
        # Sum of fraction of requests through all paths must equal 1
        gpModel.addConstr(sum(c[path] for path in paths) == 1)

        # (Branch aware)
        for taskPath in pathsForTaskPaths:
            pathsForTaskPath = pathsForTaskPaths[taskPath]

            # # Sum of fraction of requests through all paths for a taskPath
            # # must equal 1 (A task path is a path through the application graph
            # # of tasks, not model variants). This is to ensure that traffic
            # # flows through each branch of the application
            # gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) == 1)

            print(f'taskPath: {taskPath}')
            # time.sleep(1)
            if 'car_classification' in taskPath:
                print(f'pathsForTaskPath: {pathsForTaskPath}')
                # time.sleep(1)
                gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) >= 16.0/18.0)
            if 'image_captioning' in taskPath:
                gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) >= 0.3)
                gpModel.addConstr(sum(c[path] for path in pathsForTaskPath) <= 0.7)

        # Must not allocate more replicas than servers in the system
        gpModel.addConstr(sum(x[variant] for variant in variants) <= totalServers)

        # # TODO: How does this constraint play out with above constraint on sum
        # #       of c == 1?
        # gpModel.addConstr(sum(c[path] for path in paths) * totalDemand
        #                   >= totalDemand)

        # Solve the optimization
        gpModel.optimize()

        # Constructing x values to return
        xValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for variant in variants:
                xValues[variant] = round(x[variant].X)

        # Constructing c values to return
        cValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for path in paths:
                cValues[path] = c[path].X

        # Constructing y values to return
        yValues = {}
        if gpModel.status == gp.GRB.Status.OPTIMAL:
            for task in tasks:
                for variant in taskToVariants[task]:
                    for batchSize in taskVariantAllowedBatchSizes[(task, variant)]:
                        yValues[task, variant, batchSize] = y[task, variant,
                                                              batchSize].X

        return gpModel.status, xValues, cValues, yValues
        

if __name__=='__main__':
    pipelineIlp = PipelineIlp()
    pipelineIlp.run()
