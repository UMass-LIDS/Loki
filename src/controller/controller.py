# This is the Controller microservice
import argparse
import sys
import grpc
import logging
import pandas as pd
import pickle
import threading
import time
from concurrent import futures
from enum import Enum
from common.app import App, AppNode, registerApplication
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc


class AllocationPolicy(Enum):
    ROUND_ROBIN = 1
    TAIL_HEAVY = 2
    ILP_COST = 3
    INFERLINE = 4


class WorkerEntry:
    def __init__(self, IP: str, port: str, hostID: str, connection: grpc.insecure_channel,
                 model: str=None, task: str=None, appID: str=None):
        self.IP = IP
        self.port = port
        self.hostID = hostID
        self.connection = connection
        self.model = model
        self.task = task
        self.appID = appID
        self.currentLoad = 0
        self.queueSize = 0

    def setModel(self, model, task, appID):
        self.model = model
        self.task = task
        self.appID = appID


class Controller(controller_pb2_grpc.ControllerServicer):
    def __init__(self, allocationPolicy: AllocationPolicy):
        self.lbIP = None
        self.lbPort = None
        self.lbConnection = None
        # key: hostID, value: WorkerEntry
        self.workers = {}
        # Time in seconds after which controller is invoked
        self.period = 1

        self.allocationPolicy = allocationPolicy
        self.allocationMetadata = {}

        self.apps = [registerApplication('../../traces/apps/traffic_analysis.json')]

        # TODO: update execution and branching profiles based on real-time data
        self.executionProfiles = pd.read_csv('../../profiling/profiled/runtimes_all_vgghardcoded.csv')
        self.branchingProfiles = pd.read_csv('../../profiling/profiled/branching.csv')

        eventLoopThread = threading.Thread(target=self.eventLoop)
        eventLoopThread.start()

        # TODO: allow all models, perhaps use model families as well?
        self.allocatedModels = {'yolov5m': 0, 'eb6': 0, 'sink': 0}

    
    def WorkerSetup(self, request, context):
        try:
            logging.info(f'Trying to establish GRPC connection with worker..')
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                workerIP = splitContext[1]
            else:
                workerIP = 'localhost'
            connection = grpc.insecure_channel(f'{workerIP}:{request.hostPort}')
            workerEntry = WorkerEntry(IP=workerIP, port=request.hostPort,
                                      hostID=request.hostID, connection=connection,
                                      model=None)
            self.workers[request.hostID] = workerEntry
            logging.info(f'Established GRPC connection with worker (hostID: '
                         f'{request.hostID}, IP: {workerIP}, port: '
                         f'{request.hostPort})')
            
            return controller_pb2.RegisterWorkerResponse(lbIP=self.lbIP,
                                                         lbPort=self.lbPort)
        except Exception as e:
            message = f'Exception while setting up worker: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterWorkerResponse(lbIP=None, lbPort=None,
                                                         message=message)
    

    def LBSetup(self, request, context):
        try:
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                lbIP = splitContext[1]
            else:
                lbIP = 'localhost'
            logging.info(f'Trying to establish GRPC connection with load balancer..')
            connection = grpc.insecure_channel(f'{lbIP}:{request.lbPort}')
            self.lbConnection = connection
            self.lbIP = lbIP
            self.lbPort = request.lbPort
            logging.info(f'Established GRPC connection with load balancer '
                         f'(IP: {lbIP}, port: {request.lbPort})')
            
            return controller_pb2.RegisterLBResponse(message='Done!')
        
        except Exception as e:
            message = f'Exception while setting up load balancer: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterLBResponse(message=message)
    

    def eventLoop(self):
        while True:
            self.checkLBHeartbeat()

            for hostID in self.workers:
                worker = self.workers[hostID]
                self.checkWorkerHeartbeat(hostID, worker)
            
            # This doesn't necessary have to run every time Controller checks
            # heartbeats
            self.allocateResources()

            time.sleep(self.period)
            logging.debug('Woke up from sleep, running eventLoop again..')

    
    def allocateResources(self):
        ''' Run the resource allocation algorithm with the appropriate policy
        '''
        if self.allocationPolicy == AllocationPolicy.ROUND_ROBIN:
            self.allocateByRoundRobin()
        elif self.allocationPolicy == AllocationPolicy.TAIL_HEAVY:
            self.allocateByTailHeavy()
        elif self.allocationPolicy == AllocationPolicy.ILP_COST:
            self.allocateByCostILP()
        elif self.allocationPolicy == AllocationPolicy.INFERLINE:
            self.allocateByInferLine()
        else:
            raise Exception(f'Unknown allocation policy: {self.allocationPolicy}')
        return
    

    def allocateByRoundRobin(self):
        ''' Perform round-robin resource allocation of models to workers
        '''
        for hostID in self.workers:
            worker = self.workers[hostID]
            
            if worker.model is None:
                logging.info(f'No model loaded at worker {hostID}')
                modelToLoad = self.getModelByRoundRobin()
                try:
                    logging.info(f'Trying to load model {modelToLoad} on worker {hostID}')
                    self.loadModelOnWorker(worker, modelToLoad)
                except Exception as e:
                    logging.exception(f'Error while loading model {modelToLoad} on '
                                        f'worker {hostID}: {e}')
                    
        return
    

    def allocateByTailHeavy(self):
        raise Exception(f'Tail-heavy allocation policy not yet implemented, '
                        f'controller crashing')
        return
    

    def allocateByCostILP(self):
        raise Exception(f'Cost ILP allocation policy not yet implemented, '
                        f'controller crashing')
        return
    

    def allocateByInferLine(self):
        if 'inferline_initiated' not in self.allocationMetadata:
            self.allocateByInferLineInitial()
        else:
            self.allocateByInferLinePeriodic()
        return
    

    def allocateByInferLineInitial(self):
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
            totalWorkers = 20
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
                    raise Exception(f'Error! Key {key} not already in allocation plan')
                
                allocationPlan[key] += 1
                assignedWorkers += 1
                logging.info(f'Incremented replica for {key} by 1')
        
        self.allocationMetadata['inferline_initiated'] = True
        return
    

    def allocateByInferLinePeriodic(self):
        raise Exception(f'allocateByInferLinePeriodic not yet implemented, '
                        f'controller crashing')

        
    def checkLBHeartbeat(self):
        try:
            connection = self.lbConnection
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            message = 'Still alive?'
            request = load_balancer_pb2.LBHeartbeat(message=message)
            response = stub.LBAlive(request)
            logging.info(f'Heartbeat from load balancer received')
        except Exception as e:
            logging.warning('No heartbeat from load balancer')

    
    def checkWorkerHeartbeat(self, hostID: str, worker: WorkerEntry):
        try:
            connection = worker.connection
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            message = 'Still alive?'
            request = worker_pb2.HeartbeatRequest(message=message)
            response = stub.Heartbeat(request)
            worker.currentLoad = response.queriesSinceHeartbeat
            worker.queueSize = response.queueSize
            branchingSinceHeartbeat = pickle.loads(response.branchingSinceHeartbeat)
            logging.info(f'Heartbeat from worker {hostID} received, model variant: '
                         f'{worker.model}, currentLoad: {worker.currentLoad}, total '
                         f'queries received: {response.totalQueries}, queue size: '
                         f'{worker.queueSize}, branching since heartbeat: '
                         f'{branchingSinceHeartbeat}')
            
        except Exception as e:
            # TODO: remove worker after certain number of missed heartbeats?
            logging.warning(f'No heartbeat from worker: {hostID}')
            # logging.exception(f'Exception while checking heartbeat for worker {hostID}: {e}')


    def loadModelOnWorker(self, worker: WorkerEntry, model: str):
        ''' Loads the given model on a worker
        '''
        try:
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
            response = stub.LoadModel(request)
            logging.info(f'LOAD_MODEL_RESPONSE from host {worker.hostID}: {response.response}, '
                         f'{response.message}')
            
            # Model loaded without any errors
            if response.response == 0:
                self.allocatedModels[model] += 1
            # If there is an error while loading model, raise Exception
            else:
                raise Exception(f'Error occurred while loading model {model} on worker '
                                f'{worker.hostID}: {response.message}')
            
            worker.setModel(model, task, appID)
        except Exception as e:
            raise e


    def getModelByRoundRobin(self):
        ''' Returns a model such that self.allocatedModels would have equal
            number of allocated workers for all tasks in the pipeline
            (except the sink which only needs one worker)
        '''
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
    

def getargs():
    parser = argparse.ArgumentParser(description='Controller micro-service')
    parser.add_argument('--port', '-p', required=False, dest='port', default='50050',
                        help='Port to start the controller on')
    parser.add_argument('--allocation_policy', '-ap', required=True,
                        dest='allocationPolicy', choices=['1', '2', '3', '4'],
                        help=(f'Allocation policy for the controller. 1: Round Robin, '
                              f'2: Tail-Heavy, 3: Cost-based ILP, 4: InferLine'))

    return parser.parse_args()


def serve(args):
    port = args.port
    allocationPolicy = AllocationPolicy(int(args.allocationPolicy))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controller = Controller(allocationPolicy=allocationPolicy)
    controller_pb2_grpc.add_ControllerServicer_to_server(controller, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f'Controller started, listening on port {port}...')
    logging.info(f'Using resource allocation policy {allocationPolicy}')
    server.wait_for_termination()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    serve(getargs())
