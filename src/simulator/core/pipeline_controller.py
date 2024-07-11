import logging
import pickle
import sys
from enum import Enum
import numpy as np
from algorithms.inferline import Inferline
from algorithms.pipeline_ilp import PipelineIlp
from core.executor import Executor
from core.simulator import Simulator
from core.predictor import Predictor


class AllocationPolicy(Enum):
    ROUND_ROBIN = 1
    TAIL_HEAVY = 2
    ILP_COST = 3
    INFERLINE = 4


class WorkerEntry:
    def __init__(self, hostID: str, model: str=None, task: str=None,
                 appID: str=None, predictor: Predictor=None,
                 executor: Executor=None):
        # TODO: for every task we need an executor as well
        self.hostID = hostID
        self.model = model
        self.task = task
        self.appID = appID
        self.currentLoad = 0
        self.queueSize = 0
        self.branchingSinceHeartbeat = None
        return
    

    def setModel(self, model: str, task: str, appID: str):
        self.model = model
        self.task = task
        self.appID = appID
        return


class PipelineController:
    ''' Class to simulate the Controller from the system implementation
        of pipelines
    '''
    def __init__(self, simulator: Simulator, allocationPolicy: AllocationPolicy=None,
                 beta: float=None):
        self.simulator = simulator

        # TODO: Someone (probably Simulator) needs to populate the dict of
        #       workers
        self.workers = {}

        # # TODO: use policy from parameter instead of hard-coded policy
        if allocationPolicy is not None:
            self.allocationPolicy = allocationPolicy
        else:
            self.allocationPolicy = AllocationPolicy.ILP_COST

        if self.allocationPolicy == AllocationPolicy.ILP_COST:
            self.pipelineIlp = PipelineIlp(simulator=self.simulator,
                                           accuracies=self.simulator.pipeline_accuracies,
                                           runtimes=self.simulator.model_variant_runtimes,
                                           beta=beta)
        elif self.allocationPolicy == AllocationPolicy.INFERLINE:
            self.inferline =  Inferline(simulator=self.simulator,
                                        accuracies=self.simulator.pipeline_accuracies,
                                        runtimes=self.simulator.model_variant_runtimes)
        else:
            raise Exception(f'Unexpected allocation policy: {self.allocationPolicy}')

        # TODO: allow all models, perhaps use model families as well?
        self.allocatedModels = {'yolov5m': 0, 'efficientnet-b6': 0, 'sink': 0}

        return
    
    # TODO: We want to mimic the controller.py as much as possible here
    #       so that any further changes we make to this file can be
    #       easily reflected there

    def eventLoop(self, observation: np.ndarray, num_acc_types: int,
                  num_max_acc: int):
        ''' The event loop is called every second by the Simulator
        '''
        # TODO: In the real system, we keep the same workers but change the
        #       model loaded on them
        #       In the simulator, we can't change model variants on Predictors,
        #       we remove the Predictor and add a new one with the desired
        #       mode variant
        #       How to workaround this?
        #       One way to do this is to keep the same hostID but change the
        #       underlying Predictor (we can keep Predictor pointer and ID
        #       in the Worker class)

        # TODO: Another difference is that in the Simulator, we will have to
        #       add Predictors through the Executor, not directly. However,
        #       the Executor does give the predictor.id that we can keep

        # TODO: We can choose to remove Predictors by id through the respective
        #       Executor for fine-grained control

        # TODO: The other thing this does not consider is the model loading and
        #       warmup time. Can we simulate that somehow?
        #       But if we simulate that, only the PipelineController and
        #       InferLine would be simulating it and not Proteus/INFaaS

        for hostID in self.workers:
            worker = self.workers[hostID]
            self.checkWorkerHeartbeat(hostID=hostID, worker=worker)

        # This doesn't necessarily have to run every time Controller checks
        # heartbeats
        self.allocateResources(observation=observation,
                               num_acc_types=num_acc_types,
                               num_max_acc=num_max_acc)

        # Controller sleeps here for specified time period. We do not need to
        # do that here since the eventLoop will be called from outside after
        # the simulated time period elapses

        return
    

    def checkWorkerHeartbeat(self, hostID: str, worker: WorkerEntry):
        ''' See if worker still exists and update information related to the
            worker
        '''
        raise Exception('Not verified in Simulator')
        try:
            response = worker.Heartbeat()
            worker.currentLoad = response.queriesSinceHeartbeat
            worker.queueSize = response.queueSize
            worker.branchingSinceHeartbeat = pickle.loads(response.branchingSinceHeartbeat)
            logging.info(f'Heartbeat from worker {hostID} received, model variant: '
                         f'{worker.model}, currentLoad: {worker.currentLoad}, total '
                         f'queries received: {response.totalQueries}, queue size: '
                         f'{worker.queueSize}, branching since heartbeat: '
                         f'{worker.branchingSinceHeartbeat}')
        except Exception as e:
            # TODO: remove worker after certain number of missed heartbeats?
            logging.warning(f'No heartbeat from worker: {hostID}')
        return
    
    
    def loadModelOnWorker(self, worker: WorkerEntry, model: str):
        ''' Loads the given model on a worker
        '''
        # TODO: We have to figure out the Executor from model name
        #       Use app.findTaskFromModelVariant as below?
        #       Then we add a Predictor with model variant name

        connection = worker.connection
        stub = worker_pb2_grpc.WorkerDaemonStub(connection)
        # TODO: hard-coded application index
        app = self.apps[0]
        appID = app.appID
        task = app.findTaskFromModelVariant(model)
        childrenTasks = pickle.dumps(app.getChildrenTasks(task))
        labelToChildrenTasks = pickle.dumps(app.getLabelToChildrenTasksDict(task))

        request = worker_pb2.LoadModelRequest(modelName=model,
                                                applicationID=appID,
                                                task=task,
                                                childrenTasks=childrenTasks,
                                                labelToChildrenTasks=labelToChildrenTasks)
        
        # Update book-keeping on Controller to reflect that model is loaded
        self.allocatedModels[model] += 1
        worker.setModel(model=model, task=task, appID=appID)

        raise Exception('Not verified in Simulator')
        return
    

    def allocateResources(self, observation: np.ndarray, num_acc_types: int,
                          num_max_acc: int):
        ''' Run the resource allocation algorithm with the appropriate policy
        '''
        if self.allocationPolicy == AllocationPolicy.ROUND_ROBIN:
            self.allocateByRoundRobin()
        elif self.allocationPolicy == AllocationPolicy.TAIL_HEAVY:
            self.allocateByTailHeavy()
        elif self.allocationPolicy == AllocationPolicy.ILP_COST:
            self.allocateByCostILP(observation=observation,
                                   num_acc_types=num_acc_types,
                                   num_max_acc=num_max_acc)
        elif self.allocationPolicy == AllocationPolicy.INFERLINE:
            self.allocateByInferLine(observation=observation,
                                     num_acc_types=num_acc_types,
                                     num_max_acc=num_max_acc)
        else:
            raise Exception(f'Unknown allocation policy: {self.allocationPolicy}')
        return
    
    
    def allocateByRoundRobin(self):
        ''' Perform round-robin resource allocation of models to workers
        '''
        for hostID in self.workers:
            worker = self.workers[hostID]
            
            # TODO: the assumption is that workers will be all started in
            #       Simulator even if Predictor is not up
            if worker.predictor is None:
                logging.info(f'No model loaded at worker {hostID}')
                modelToLoad = self.getModelByRoundRobin()

                logging.info(f'Loading model {modelToLoad} on worker {hostID}')
                self.loadModelOnWorker(worker=worker, model=modelToLoad)
                    
        raise Exception('Not verified in Simulator')
        return
    

    def getModelByRoundRobin(self):
        ''' Returns a model such that self.allocatedModels would have equal
            number of allocated workers for all tasks in the pipeline
            (except the sink which only needs one worker)
        '''
        # TODO: This function depends on the current data structure
        #       self.allocatedModels. If that changes, this will too
        minValue = sys.maxsize
        minModel = None
        for model in self.allocatedModels:
            if self.allocatedModels[model] < minValue:
                # We only want one copy of the sink for each application
                if model == 'sink' and self.allocatedModels[model] == 1:
                    continue
                
                minValue = self.allocatedModels[model]
                minModel = model
        return minModel
    

    def allocateByTailHeavy(self):
        raise Exception(f'Tail-heavy allocation policy not yet implemented, '
                        f'controller crashing')
        return
    

    def allocateByCostILP(self, observation: np.ndarray, num_acc_types: int,
                          num_max_acc: int):
        self.pipelineIlp.run(observation=observation, num_acc_types=num_acc_types,
                             num_max_acc=num_max_acc)
        return
    

    def allocateByInferLine(self, observation: np.ndarray, num_acc_types: int,
                            num_max_acc: int):
        self.inferline.run(observation=observation, num_acc_types=num_acc_types,
                           num_max_acc=num_max_acc)
        return

    
    def allocateByInferLineBackup(self):
        raise Exception('Not verified in Simulator')
        if 'inferline_initiated' not in self.allocationMetadata:
            self.allocateByInferLineInitial()
        else:
            self.allocateByInferLinePeriodic()
        raise Exception('InferLine planner solve interval unknown')
        raise Exception('InferLine scale down not yet implemented')
        raise Exception('InferLine traffic envelopes/tuner not yet implemented')
        raise Exception('InferLine check SLO with simulator not yet implemented')
        return
    

    def allocateByInferLineInitial(self):
        raise Exception('Not verified in Simulator')
        # TODO: Assuming one app for now
        app = self.apps[0]
        allocationPlan = {}

        tasks = app.getAllTasks()
        for task in tasks:
            modelVariants = app.getModelVariantsFromTaskName(task)
            # TODO: Get the most accurate model variant
            # TODO: Assuming they are sorted by accuracy, but this is a weak
            #       assumption
            logging.warning(f'allocateByInferLineInitial assumes models are '
                            f'sorted by accuracy')
            modelVariant = modelVariants[-1]

            # An allocation plan is a dict with the following definiton:
            # key:    (model variant, batch size, hardware)
            # value:  replicas

            batchSize = 1

            # TODO: remove hard-coded value of hardware, find best hardware instead
            #       hardware = bestHardware(modelVariant)
            hardware = '1080ti'

            allocationPlan[(modelVariant, batchSize, hardware)] = 1

        serviceTime = app.getServiceTimeForAllocation(allocationPlan=allocationPlan,
                                                      executionProfiles=self.executionProfiles)
        logging.info(f'service time: {serviceTime} micro-seconds')

        if serviceTime >= app.getLatencySLO():
            raise Exception(f'allocateByInferLineInitial(): No allocation possible '
                            f'as serviceTime ({serviceTime}) is more than application '
                            f'latency SLO ({app.getLatencySLO()})')
        else:
            totalWorkers = len(self.workers)
            # totalWorkers = 20
            assignedWorkers = sum(allocationPlan.values())

            logging.info(f'assignedWorkers: {assignedWorkers}, totalWorkers: {totalWorkers}')

            while assignedWorkers < totalWorkers:
                task = app.findMinThroughput(allocationPlan=allocationPlan,
                                             executionProfiles=self.executionProfiles,
                                             branchingProfiles=self.branchingProfiles)
                modelVariants = app.getModelVariantsFromTaskName(task)
                modelVariant = modelVariants[-1]
                batchSize = 1
                # TODO: change hard-coded hardware, use same hardware as before
                hardware = '1080ti'
                key = (modelVariant, batchSize, hardware)

                if key not in allocationPlan:
                    raise Exception(f'Key {key} should already be in allocation plan')
                
                allocationPlan[key] += 1
                assignedWorkers += 1
                logging.info(f'Incremented replica for {key} by 1')
        
        self.allocationMetadata['inferline_initiated'] = True
        self.allocationMetadata['plan'] = allocationPlan

        logging.error(f'allocateByInferLineInitial did not materialize allocationPlan')
        return
    

    def allocateByInferLinePeriodic(self):
        raise Exception('Not verified in Simulator')
        # TODO: Again assuming only 1 app
        app = self.apps[0]
        tasks = app.getAllTasks()

        actions = ['IncreaseBatch', 'RemoveReplica', 'DowngradeHW']

        bestPlan = self.allocationMetadata['plan']
        for task in tasks:
            for action in actions:
                newPlan = applyActionToPlan(action, task, bestPlan)

                if checkPlanFeasibility(newPlan):
                    if cost(newPlan) < cost(bestPlan):
                        bestPlan = newPlan
        
        if bestPlan != self.allocationMetadata['plan']:
            applyPlanToSystem(bestPlan)
            self.allocationMetadata['plan'] = bestPlan

        raise Exception(f'allocateByInferLinePeriodic not yet implemented, '
                        f'controller crashing')
    

    def getModelByRoundRobin(self):
        ''' Returns a model such that self.allocatedModels would have equal
            number of allocated workers for all tasks in the pipeline
            (except the sink which only needs one worker)
        '''
        raise Exception('Not verified in Simulator')
        minValue = sys.maxsize
        minModel = None
        for model in self.allocatedModels:
            if self.allocatedModels[model] < minValue:
                # We only want one copy of the sink for each application
                if model == 'sink' and self.allocatedModels[model] == 1:
                    continue
                
                minValue = self.allocatedModels[model]
                minModel = model
        return minModel
    
