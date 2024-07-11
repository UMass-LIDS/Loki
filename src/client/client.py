import argparse
import grpc
import pickle
import logging
import random
import threading
import time
import uuid
from concurrent import futures
from common.host import getRoutingTableStr
from protos import client_pb2, client_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc


class ClientDaemon(client_pb2_grpc.ClientServicer):
    def __init__(self, ip: str, port: str, lbIP: str, lbPort: str):
        self.hostID = str(uuid.uuid4())
        logging.info(f'Client started with ID: {self.hostID}')

        # TODO: replace hard-coded appID
        self.appID = '123'

        self.routingTable = []
        self.connections = {}

        self.ip = ip
        self.port = port

        self.lbIP = lbIP
        self.lbPort = lbPort
        self.lbConnection = None

        self.setupLB()

        sendRequestsThread = threading.Thread(target=self.sendRequests)
        sendRequestsThread.start()

    
    def setupLB(self):
        logging.info('Setting up load balancer at client..')
        try:
            connection = grpc.insecure_channel(f'{self.lbIP}:{self.lbPort}')
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            request = load_balancer_pb2.RegisterClient(hostID=self.hostID,
                                                       hostIP=self.ip,
                                                       port=self.port,
                                                       appID=self.appID)
            response = stub.ClientSetup(request)
            
            self.lbConnection = connection
            logging.info(f'Response from load balancer: {response}')

        except Exception as e:
            logging.exception(f'Could not connect to load balancer, exception: {e}')


    def SetRoutingTable(self, request, context):
        routingTable = pickle.loads(request.routingTable)
        logging.info(f'Setting routing table at client: {getRoutingTableStr(routingTable)}')
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
                
        # TODO: fix this warning
        logging.warning(f'SetRoutingTable(): Should we remove old connections no longer '
                        f'in routing table? Or perhaps we may get them again. It depends '
                        f'on the overhead of keeping old connections open')
        
        # TODO: fix this warning
        # logging.warning(f'SetRoutingTable() at client has the same code as on worker, '
        #                 f'so a change in one may not reflect in both places. Perhaps '
        #                 f'they should both inherit from a Host class')
        return response
    

    def getHostID(self):
        # TODO: create a host daemon class and inherit this from it
        ''' This function will implement probability-based routing from a routing
            table lookup.
        '''
        task = self.routingTable[0].task

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
    

    def sendRequests(self):
        logging.info('Sleeping for 5 seconds before sending requests..')
        time.sleep(5)

        request_start_time = time.time()
        for data_idx in range(7120):
            logging.info('Sleeping for 500 milliseconds before sending another request..')
            time_difference = time.time() - request_start_time
            time.sleep(0.5 - time_difference)

            logging.info(f'Time difference was: {time_difference}')

            request_start_time = time.time()
            inference_request = worker_pb2.InferenceRequest(userID=self.hostID,
                                                            applicationID='123',
                                                            sequenceNum=data_idx)
            hostID = self.getHostID()

            try:
                connection = self.connections[hostID]
                stub = worker_pb2_grpc.WorkerDaemonStub(connection)

                logging.info(f'Sending request (sequenceNum: {data_idx}) to '
                            f'worker (hostID: {hostID})..')
                
                response = stub.InitiateRequest(inference_request)
                logging.info(f'Message from worker for inference request number '
                            f'{data_idx}.. requestID: {response.requestID}, request '
                            f'status: {response.status}, message: {response.message}')
                
            except Exception as e:
                logging.warning(f'Could not send request (sequenceNum: {data_idx}) '
                                f'to worker (hostID: {hostID}): {e}')


def serve(args):
    ip = args.ip
    port = args.port
    lbIP = args.lbIP
    lbPort = args.lbPort
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    client = ClientDaemon(ip=ip, port=port, lbIP=lbIP, lbPort=lbPort)
    client_pb2_grpc.add_ClientServicer_to_server(client, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f'Client started, listening on port {port}...')
    server.wait_for_termination()


def getargs():
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--ip_address', '-ip', required=False, dest='ip',
                        default='localhost', help='IP address to start client on')
    parser.add_argument('--port', '-p', required=False, dest='port', default='51050',
                        help='Port to start client on')
    parser.add_argument('--lb_ip', '-lbip', required=False, dest='lbIP',
                        default='localhost', help='IP address of the load balancer')
    parser.add_argument('--lb_port', '-lbport', required=False, dest='lbPort',
                        default='50049', help='Port of the controller')

    return parser.parse_args()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    # client = Client()
    # client.run()
    # print('Done')

    serve(getargs())
