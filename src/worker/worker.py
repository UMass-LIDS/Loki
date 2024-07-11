import os
import sys
import argparse
import grpc
import logging
import pickle
import random
import threading
import time
import uuid
from array import array
from concurrent import futures
from multiprocessing import Pipe
from torch import tensor
from model import LoadedModel
from common.host import getRoutingTableStr
from protos import worker_pb2, worker_pb2_grpc
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from utils.vision import Dataset


class WorkerDaemon(worker_pb2_grpc.WorkerDaemonServicer):
    def __init__(self, workerIP: str, workerPort: str, controllerIP: str,
                 controllerPort: str):
        self.hostID = str(uuid.uuid4())

        self.appID = ''
        self.task = None
        self.childrenTasks = []

        self.dataset = Dataset('../../data/preprocessed')

        # Model executor and preprocessing function
        self.preprocess = None
        childReadPipe, self.writePipe = Pipe()
        self.readPipe, childWritePipe = Pipe()
        self.loadedModel = LoadedModel((childReadPipe, self.writePipe), 
                                       (self.readPipe, childWritePipe),
                                       self.dataset)
        
        self.IP = workerIP
        self.port = workerPort

        self.controllerIP = controllerIP
        self.controllerPort = controllerPort

        self.lbIP = None
        self.lbPort = None

        self.controllerConnection = None
        self.lbConnection = None

        self.routingTable = []
        self.connections = {}
        self.queryMetaStore = {}

        self.stats = {'queries_received': 0, 'queries_since_heartbeat': 0,
                      'queue_size': 0, 'branching': {}, 'branching_since_heartbeat': {}}

        # Setting up controller
        self.setupController(self.controllerIP, self.controllerPort)
        
        # Starting worker event loop
        eventLoopThread = threading.Thread(target=self.eventLoop)
        eventLoopThread.start()
        return
    

    def Heartbeat(self, request, context):
        message = f'Host {self.hostID} is still alive!'

        queriesSinceHeartbeat = self.stats['queries_since_heartbeat']
        totalQueries = self.stats['queries_received']
        queueSize = self.stats['queue_size']
        branchingSinceHeartbeat = pickle.dumps(self.stats['branching_since_heartbeat'])

        # Reset some stats at every heartbeat
        self.stats['queries_since_heartbeat'] = 0
        self.stats['branching_since_heartbeat'] = {}

        return worker_pb2.HeartbeatResponse(message=message,
                                            queriesSinceHeartbeat=queriesSinceHeartbeat,
                                            totalQueries=totalQueries,
                                            queueSize=queueSize,
                                            branchingSinceHeartbeat=branchingSinceHeartbeat)
    

    def setupController(self, IP, port):
        logging.info('Setting up controller..')
        try:
            connection = grpc.insecure_channel(f'{IP}:{port}')
            stub = controller_pb2_grpc.ControllerStub(connection)
            request = controller_pb2.RegisterWorker(hostID=self.hostID, hostIP=self.IP,
                                                   hostPort=self.port)
            response = stub.WorkerSetup(request)

            self.controllerConnection = connection

            # If load balancer resides on same host as controller, use the same
            # IP for it
            if response.lbIP == 'localhost':
                self.lbIP = self.controllerIP
            else:
                self.lbIP = response.lbIP
                
            self.lbPort = response.lbPort
            logging.info(f'Response from controller: {response}')
        except Exception as e:
            logging.exception(f'Could not connect to controller, exception: {e}')
        
        return
    
    def setupLoadBalancer(self):
        ''' This is done every time a new model is loaded
        '''
        logging.info('Setting up load balancer..')
        try:
            connection = grpc.insecure_channel(f'{self.lbIP}:{self.lbPort}')
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            request = load_balancer_pb2.RegisterWorkerAtLB(hostID=self.hostID,
                                                           hostIP=self.IP,
                                                           port=self.port,
                                                           appID=self.appID,
                                                           task=self.task,
                                                           loadedModel=self.modelName)
            response = stub.WorkerSetup(request)
            
            self.lbConnection = connection
            logging.info(f'Response from load balancer: {response}')

        except Exception as e:
            logging.exception(f'Could not connect to load balancer, exception: {e}')
        
        return
    
    
    def LoadModel(self, request, context):
        modelName = request.modelName
        appID = request.applicationID
        task = request.task
        childrenTasks = pickle.loads(request.childrenTasks)
        labelToChildrenTasks = pickle.loads(request.labelToChildrenTasks)

        try:
            self.writePipe.send('LOAD_MODEL')
            self.writePipe.send(f'{modelName},{appID},{task}')
            self.writePipe.send(childrenTasks)
            self.writePipe.send(labelToChildrenTasks)
            
            self.appID = appID
            self.task = task
            self.childrenTasks = childrenTasks

            response = worker_pb2.LoadModelEnum.LOAD_INITIATED
            return worker_pb2.LoadModelResponse(response=response)
            # else:
            #     raise Exception(f'Unknown message received: {message}')
        
        except Exception as e:
            # If model loading failed, respond with fail code and error message
            logging.error(f'Model loading failed with exception: {e}')
            response = worker_pb2.LoadModelEnum.LOAD_FAILED
            return worker_pb2.LoadModelResponse(response=response,
                                                loadingTimeInUSec=0,
                                                message=str(e))


    def InitiateRequest(self, request, context):
        logging.info(f'Received initial request: {request}')

        # We need a requestID whether we can serve it or not
        requestID = str(uuid.uuid4())

        return self.serveQuery(request, context, requestID)
        
    
    def IntermediateRequest(self, request, context):
        logging.info(f'Received intermediate request: {request}')

        requestID = request.requestID

        return self.serveQuery(request, context, requestID)
    

    def serveQuery(self, request, context, requestID):
        message = None

        # TODO: we are assuming self.appID is already set, but it needs to be set by
        #       someone (either controller or load balancer)
        if self.appID == request.applicationID:
            try:
                # We are assuming this is where the query's latency timer starts
                # and do not take into account the network delay for query reaching
                # this point
                timestamp = time.time()

                request.requestID = requestID
                request.timestamp = timestamp

                event = {'event': 'WORKER_RECEIVED_REQUEST',
                         'requestID': request.requestID, 'queryID': None,
                         'userID': request.userID, 'appID': request.applicationID,
                         'sequenceNum': request.sequenceNum, 'timestamp': timestamp}
                logging.info(f'EVENT,{str(event)}')
                
                logging.info(f'Putting request in worker IPC pipe, time: {time.time()}')
                self.writePipe.send('REQUEST')
                self.writePipe.send(request)
                logging.info(f'Done putting request in worker IPC pipe, time: {time.time()}')

                status = worker_pb2.RequestStatus.ACCEPTED

            except Exception as e:
                status = worker_pb2.RequestStatus.REQUEST_FAILED
                message = str(e)
                print(f'Request failed with exception: {e}')
        else:
            logging.warning(f'Request received for invalid application ID: '
                            f'{request.applicationID}, worker runs application ID: '
                            f'{self.appID}')
            status = worker_pb2.RequestStatus.INVALID_APPLICATION
        
        # Return ACK
        if message is None:
            return worker_pb2.InferenceRequestAck(requestID=requestID,
                                                  status=status)
        else:
            return worker_pb2.InferenceRequestAck(requestID=requestID,
                                                  status=status,
                                                  message=message)
    

    def eventLoop(self):
        logging.info(f'Worker daemon event loop waiting')
        while True:
            message = self.readPipe.recv()
            logging.info(f'Message received by worker daemon eventLoop: {message}')

            if message == 'LOAD_MODEL_RESPONSE':
                message = self.readPipe.recv()
                modelName, loadedFrom, loadingTime = message.split(',')
                self.modelName = modelName
                # Notify load balancer of change in model
                self.setupLoadBalancer()
            
            elif 'QUEUED_QUERY' in message:
                queueSize = int(message.split(',')[1])
                self.stats['queue_size'] = queueSize

                query = self.readPipe.recv()
                self.registerQuery(query)

            elif 'QUEUE_SIZE' in message:
                queueSize = int(message.split(',')[1])
                self.stats['queue_size'] = queueSize

            elif message == 'COMPLETED_INFERENCE':
                while True:
                    message = self.readPipe.recv()
                    if message == 'DONE_SENDING':
                        logging.info('Message received by worker daemon eventLoop: DONE_SENDING')
                        break
                    
                    logging.info(f'inference results received from worker at time {time.time()}')
                    logging.info(f'serialized message length: {len(message)}')

                    # Forward intermediate query using routing table to find where to
                    # direct each query

                    # Perhaps we can have task ID to distinguish?
                    # This ID could be configured by controller
                    # Daemon configures executor with task IDs
                    # Executor sends results separated by task IDs to daemon

                    queryID, results = message.split(',', 1)
                    logging.info(f'results: {results}')
                    results = eval(results)

                    # TODO: this will execute sequentially. is there any benefit
                    #       to parallelizing?
                    for task in results:
                        tensor_data = results[task]
                        # data = bytes(tensor)
                        byte_data = pickle.dumps(tensor_data)
                        print(f'length of pickled tensor: {len(byte_data)}')
                        self.forwardIntermediateQuery(queryID, task, byte_data)

                    self.logBranching(results)

    
    def logBranching(self, results: dict):
        branching = {}
        for task in results:
            tensor_data = results[task]
            branching[task] = tensor_data.shape[0]
        resultsShape = list(map(lambda x: x.shape[0], list(results.values())))

        for task in self.childrenTasks:
            if task not in self.stats['branching']:
                self.stats['branching'][task] = []

            if task not in self.stats['branching_since_heartbeat']:
                self.stats['branching_since_heartbeat'][task] = []
            
            if task in branching:
                self.stats['branching'][task].append(branching[task])
                self.stats['branching_since_heartbeat'][task].append(branching[task])
            else:
                self.stats['branching'][task].append(0)
                self.stats['branching_since_heartbeat'][task].append(0)

        print(f'\n\nresults shape: {resultsShape}')
        print(f'branching: {branching}')
        print(f'self.stats[branching]: {self.stats["branching"]}')
        print(f'self.stats[branching_since_heartbeat]: {self.stats["branching_since_heartbeat"]}\n\n')
        return

    
    def registerQuery(self, query):
        self.queryMetaStore[str(query.queryID)] = query
        self.stats['queries_received'] += 1
        self.stats['queries_since_heartbeat'] += 1
        return

    
    def forwardIntermediateQuery(self, queryID, task, data):
        # TODO: a periodic cleanup thread can delete metadata for old queries
        #       that have completed execution, or they could be cleaned up when
        #       they get forwarded through this function

        queryMetadata = self.queryMetaStore[queryID]

        appID = queryMetadata.applicationID
        hostID = self.getHostID(appID, task)
        
        if hostID is None:
            logging.warning(f'forwardIntermediateQuery(): getHostID returned None, '
                            f'cannot forward query')
            return

        logging.debug(f'forwardIntermediateQuery -- appID: {appID}, hostID: {hostID}, '
                     f'queryMetadata: {queryMetadata}, data: {data}')

        # Send query to intermediate worker
        try:
            connection = self.connections[hostID]
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            inference_request = worker_pb2.InferenceRequest(requestID=queryMetadata.requestID,
                                                            queryID=queryID,
                                                            userID=queryMetadata.userID,
                                                            applicationID=queryMetadata.applicationID,
                                                            data=data,
                                                            sequenceNum=queryMetadata.sequenceNum)
            response = stub.IntermediateRequest(inference_request)

            event = {'event': 'WORKER_FORWARDED_QUERY',
                    'requestID': queryMetadata.requestID, 'queryID': queryID,
                    'userID': queryMetadata.userID, 'appID': queryMetadata.applicationID,
                    'task': self.task, 'sequenceNum': queryMetadata.sequenceNum,
                    'timestamp': time.time()}
            
            logging.info(f'EVENT,{str(event)}')
            
            logging.info(f'Received response from worker for intermediate query: {response}')
        
        except Exception as e:
            logging.error(f'Error sending intermediate query (queryID: {queryID}, '
                          f'requestID: {queryMetadata.requestID}) to worker '
                          f'(hostID: {hostID}): {e}')
        return
    

    def SetRoutingTable(self, request, context):
        routingTable = pickle.loads(request.routingTable)
        logging.info(f'Setting routing table at worker: {getRoutingTableStr(routingTable)}')
        self.routingTable = routingTable
        response = worker_pb2.RoutingTableResponse(message='Routing table set successfully!')

        for routingEntry in routingTable:
            if routingEntry.hostID not in self.connections:
                try:
                    connection = grpc.insecure_channel(f'{routingEntry.ip}:{routingEntry.port}')
                    self.connections[routingEntry.hostID] = connection
                    logging.info(f'SetRoutingTable(): Established connection with new '
                                f'worker (hostID: {routingEntry.hostID}, IP: {routingEntry.ip}, '
                                f'port: {routingEntry.port})')
                except Exception as e:
                    logging.error(f'Could not establish connection with worker (hostID: '
                                  f'{routingEntry.hostID}, IP: {routingEntry.ip}, port: '
                                  f'{routingEntry.port}): {e}')
                
        logging.warning(f'SetRoutingTable(): Should we remove old connections no longer '
                        f'in routing table? Or perhaps we may get them again. It depends '
                        f'on the overhead of keeping old connections open')
        return response
    

    def getHostID(self, appID, task):
        ''' This function implements probability-based routing from a routing
            table lookup.
        '''
        filteredWorkers = list(filter(lambda x: x.task == task, self.routingTable))

        if len(filteredWorkers) == 0:
            logging.error(f'getHostID(): No hosts found for task: {task}')
            return None

        weights = list(map(lambda x: x.percentage, filteredWorkers))
        selected = random.choices(filteredWorkers, weights, k=1)

        hostID = selected[0].hostID

        logging.debug(f'\n\nfilteredWorkers: {filteredWorkers}')
        logging.debug(f'weights: {weights}')
        logging.debug(f'selected: {selected}, hostID: {hostID}')

        return hostID
    

def serve(args):
    workerIP = args.workerIP
    workerPort = args.workerPort
    controllerIP = args.controllerIP
    controllerPort = args.controllerPort
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    workerDaemon = WorkerDaemon(workerIP=workerIP, workerPort=workerPort,
                                controllerIP=controllerIP, controllerPort=controllerPort)
    worker_pb2_grpc.add_WorkerDaemonServicer_to_server(workerDaemon, server)
    server.add_insecure_port(f'[::]:{workerPort}')
    server.start()
    logging.info(f'Worker daemon started, listening on port {workerPort}...')
    server.wait_for_termination()


def getargs():
    parser = argparse.ArgumentParser(description='Worker daemon')
    parser.add_argument('--ip_address', '-ip', required=False, dest='workerIP',
                        default='localhost', help='IP address to start worker on')
    parser.add_argument('--port', '-p', required=False, dest='workerPort', default='50051',
                        help='Port to start worker on')
    parser.add_argument('--controller_ip', '-cip', required=False, dest='controllerIP',
                        default='localhost', help='IP address of the controller')
    parser.add_argument('--controller_port', '-cport', required=False,
                        dest='controllerPort', default='50050',
                        help='Port of the controller')

    return parser.parse_args()


if __name__=='__main__':
    logfile_name = f'../../logs/worker_{time.time()}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    # asyncio.run(serve())
    serve(getargs())
