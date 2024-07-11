import bisect
import logging
import pprint
import random
import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
from enum import Enum
from typing import List
from common.host import RoutingPolicy
from core.common import Event, RoutingEntry, TaskAssignment
from core.exceptions import PredictorException


class AccType(Enum):
    CPU = 1
    GPU = 2
    VPU = 3
    FPGA = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
           return self.value < other.value
        else:
            return NotImplemented


class Predictor:
    def __init__(self, logging_level, acc_type=AccType.CPU, qos_level=0,
                 profiled_accuracy=100.0, profiled_latencies={}, variant_name=None,
                 executor=None, simulator=None, configured_max_batch_size=None,
                 application=None):
        # attributes related to predictor hardware
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)
        self.logging_level = logging_level

        self.id = uuid.uuid4().hex
        self.acc_type = acc_type
        self.variant_name = variant_name
        self.qos_level = qos_level
        self.profiled_accuracy = profiled_accuracy
        self.profiled_latencies = profiled_latencies

        # attributes related to current status
        self.busy = False
        self.busy_till = None
        self.request_dict = {}

        # Batching-related variables (by default we have a batch size of 1)
        self.request_queue = []
        self.event_counter = 0
        self.slo_expiring_dict = {}
        self.expiring_waiting = False

        self.load = None

        self.executor = executor
        self.simulator = simulator
        self.application = application

        self.batch_sizes_allowed = self.simulator.allowed_batch_sizes
        self.configured_max_batch_size = configured_max_batch_size
        self.max_batch_size = self.get_largest_batch_size()

        self.served_requests_per_step = 0
        self.incoming_requests_per_step = 0
        if self.max_batch_size == 0:
            self.peak_throughput = 0
        else:
            self.peak_throughput = self.max_batch_size * 1000 / profiled_latencies[(self.executor.isi,
                                                                                    variant_name,
                                                                                    self.max_batch_size)] 

        self.batching_algo = self.simulator.batching_algo

        # Only needed if model assignment and job scheduling policies are INFaaS v2
        self.infaas_batch_size = self.max_batch_size
        self.infaas_cost = np.inf
        self.set_infaas_cost()
        self.aimd_batch_size = 1

        self.batch_expiring_set = False

        self.task_assignment = self.executor.task_assignment

        # For request forwarding in multiple-task applications
        self.routingTable = []
        self.routingPolicy = None

        # This is set to True when predictor is no longer accepting new requests
        # and only processing outstanding requests. When its queue becomes empty,
        # it will signal the system to remove it from pending-removal lists
        self.pendingRemoval = False

        # # For simulation of muliplicative factor and branching
        # # This dictionary maps task/isi to probability (total should be exactly 1)
        self.branching_probabilities = {}

        # Default values for mult factor (1) and branching probabilities (equal
        # for each child)
        node = self.application.findNodeByTask(self.executor.isi)
        numChildren = len(node.children)
        for child in node.children:
            self.branching_probabilities[child.getTaskName()] = 1.0 / numChildren
        self.multiplicative_factor = 1

        # Load the branching trace for the model variant if given
        self.loadBranchingTrace(branchingTracePath=self.simulator.branching_trace_path)

        # Used for Most-Accurate-First request routing
        self.incoming_requests_per_path = {}
        self.remaining_capacity = self.peak_throughput
        
        # If the maximum batch size is 0, that means that predictor cannot even
        # serve a batch size of 1 without violating latency SLO
        if self.max_batch_size == 0:
            self.busy = True

        # Enable dynamic request rerouting if simulator has flag set
        self.dynamic_rerouting = self.simulator.dynamic_rerouting

        # Disable early dropping if simulator has flag set
        self.no_early_dropping = self.simulator.no_early_dropping
        # Enable last stage early dropping
        self.last_stage_early_drop = self.simulator.last_stage_early_drop
        
        predictor_log = logging.FileHandler(f'logs/per_predictor/{self.id}.txt')
        predictor_log.setLevel(logging.INFO)
        self.predictor_log = logging.getLogger(self.id)
        self.predictor_log.addHandler(predictor_log)
        self.predictor_log.debug(f'{self.variant_name},{self.acc_type},{self.max_batch_size},'
                                f'{self.executor.isi}')
        self.predictor_log.debug(self.profiled_latencies)

        self.predictor_log.info(f'Added predictor {self.id} for executor {executor.isi} '
                                f'and application {application.getName()}')

        # Default batch size is 1 for pipeline ILP unless otherwise set
        self.pipeline_batch_size = 1

        return
    

    def loadBranchingTrace(self, branchingTracePath):
        self.branchingTrace = None

        # If experiment does not specify any branching trace, use default values
        if not(os.path.exists(branchingTracePath)):
            return
        
        # If branching trace does not exist for this model variant, use default values
        branchingTraceFile = os.path.join(branchingTracePath, self.variant_name+'.csv')
        print(f'branchingTraceFile: {branchingTraceFile}, exists: '
              f'{os.path.exists(branchingTraceFile)}')
        if not(os.path.exists(branchingTraceFile)):
            # time.sleep(1)
            return
        
        # Set the branching trace as a Pandas DataFrame
        self.branchingTrace = pd.read_csv(branchingTraceFile)
        self.branchingTraceMaxTime = self.branchingTrace['frame'].values[-1]

        return
    

    def clearRoutingTable(self):
        oldTable = self.routingTable
        self.routingTable = []
        return oldTable
    
    
    def setRoutingTable(self, routingTable: List[RoutingEntry]):
        self.routingTable = routingTable
        return
    
    
    def addRoutingEntry(self, routingEntry: RoutingEntry):
        self.routingTable.append(routingEntry)
        return

    
    def set_load(self, load):
        self.load = load
        return
    

    def set_pending_removal(self):
        self.pendingRemoval = True
        return


    def set_pipeline_batch_size(self, batch_size: int):
        self.pipeline_batch_size = batch_size
        self.max_batch_size = batch_size
        return

    
    def increase_aimd_batch_size(self):
        ''' Since we only allow multiple of 4 batch sizes after 8, AIMD can only
         go one step up to the next available batch size
        '''
        self.simulator.aimd_stats['increased'] += 1
        self.log.debug(f'increase_aimd_batch_size: current batch size: {self.aimd_batch_size}')
        current_idx = self.batch_sizes_allowed.index(self.aimd_batch_size)
        if current_idx < len(self.batch_sizes_allowed) - 1:
            new_idx = current_idx + 1
            self.aimd_batch_size = self.batch_sizes_allowed[new_idx]
            current_idx = new_idx
        self.log.debug(f'increase_aimd_batch_size: batch size set: {self.aimd_batch_size}')
        return

    
    def decrease_aimd_batch_size(self):
        ''' Since we only allow multiple of 4 batch sizes after 8, AIMD can only
         go one step down to the previous available batch size
        '''
        self.simulator.aimd_stats['decreased'] += 1
        self.log.debug(f'decrease_aimd_batch_size: current batch size: {self.aimd_batch_size}')
        current_idx = self.batch_sizes_allowed.index(self.aimd_batch_size)
        if current_idx > 0:
            new_idx = current_idx - 1
            self.aimd_batch_size = self.batch_sizes_allowed[new_idx]
            current_idx = new_idx
        self.log.debug(f'decrease_aimd_batch_size: batch size set: {self.aimd_batch_size}')
        return

    
    def get_infaas_cost(self):
        return self.infaas_cost

    
    def set_infaas_cost(self):
        ''' The cost of this model variant is the drop in accuracy when compared
        to the most accurate model in the model family
        '''
        all_variant_accuracies = self.executor.variant_accuracies
        model_name = self.executor.isi
        model_variant_accuracies = dict(filter(lambda x: x[0][0]==model_name,
                                               all_variant_accuracies.items()))

        highest_accuracy = max(model_variant_accuracies.values())
        accuracy_drop = highest_accuracy - self.profiled_accuracy
        self.infaas_cost = accuracy_drop
        return

    
    def get_infaas_batch_size(self):
        return self.infaas_batch_size

    
    def set_infaas_batch_size(self, batch_size):
        self.infaas_batch_size = batch_size
        return

    
    def get_largest_batch_size(self):
        largest_batch_sizes = self.simulator.get_largest_batch_sizes()

        acc_type = self.acc_type
        if acc_type == 1:
            acc_type = 'CPU'
        elif acc_type == 2:
            acc_type = 'GPU_AMPERE'
        elif acc_type == 3:
            acc_type = 'VPU'
        elif acc_type == 4:
            acc_type = 'GPU_PASCAL'
        
        largest_batch_size = largest_batch_sizes[(acc_type, self.variant_name)]
        
        # If self.configured_max_batch_size is None, there is no maximum batch size
        # specified, so we can use the largest batch size for the given accelerator
        # Otherwise, we need to cap it by self.configured_max_batch_size which is a
        # configuration parameter
        if self.configured_max_batch_size is not None:
            largest_batch_size = min(largest_batch_size, self.configured_max_batch_size)
        return largest_batch_size
    

    def assign_request(self, event, clock):
        # If request can be finished within deadline, return end_time,
        # else return None (failed request)
        
        if self.busy:
            end_time = self.busy_till + event.runtime
        else:
            end_time = clock + event.runtime
        if end_time <= event.start_time + event.deadline:
            self.busy = True
            self.busy_till = end_time
            self.request_dict[event.id] = 1
            return end_time
        else:
            return None

    
    def finish_request(self, event, clock):
        if event.id not in self.request_dict:
            return False
        
        self.served_requests_per_step += 1
        del self.request_dict[event.id]
        if len(self.request_dict) == 0:
            self.busy_till = None

        # For multi-task applications:
        # Generate multiple sub-requests or queries based on predictor's
        # multiplicative factor and then forward them
        if self.branchingTrace is None:
            subrequestsPerChild = self.generateSubrequestsWithoutTrace()
        else:
            subrequestsPerChild = self.generateSubrequestsWithTrace(event)
        
        # Send the profiled branching info to Controller (simulator) so that
        # it can keep a history of branching info
        self.simulator.add_profiled_branching_info(predictor=self,
                                                   branchingInfo=subrequestsPerChild)

        # Checking incoming path and setting outgoing path
        if event.path == '':
            raise Exception('Event path should be None')
        elif event.path is None:
            event.path = ''
            outgoingPath = self.variant_name
        else:
            outgoingPath = event.path + ',' + self.variant_name
        
        # Generate subrequests
        for isi in subrequestsPerChild:
            subrequestsGenerated = subrequestsPerChild[isi]
            for i in range(subrequestsGenerated):

                target_predictor_id = None
                if event.dynamic_rerouted:
                    print(f'REROUTING_PLAN_REACHED, clock: {clock}, event: {event.getStr()}')
                    if isi in event.rerouted_target_predictors:
                        target_predictor_id = event.rerouted_target_predictors[isi].id
                        print(f'REROUTING_DONE, target_predictor_id: {target_predictor_id} '
                              f'clock: {clock}, event: {event.getStr()}')
                    else:
                        print(f'REROUTING_FAILED, clock: {clock}, could not find task: {isi}, '
                              f'rerouted_target_predictors: {event.rerouted_target_predictors}'
                              f', event: {event.getStr()}')

                if target_predictor_id is None:
                    # Find target predictor through routing table
                    target_predictor_id = self.find_predictor_through_routing_table(isi=isi,
                                                                                    event=event)

                self.log.debug(f'Sending request to predictor: {target_predictor_id}')

                if target_predictor_id is None:
                    self.log.error(f'isi_name: {isi}')

                # Generate intermediate request for it
                self.simulator.add_intermediate_request_from_predictor(isi_name=isi,
                                                                       parent_request_id=event.parent_request_id,
                                                                       start_time=clock,
                                                                       target_predictor_id=target_predictor_id,
                                                                       sequence_num=event.sequence_num,
                                                                       path=outgoingPath)

        return True
    

    def resetRemainingCapacity(self):
        ''' Resets the remaining capacity of predictor to maximum capacity
        '''
        self.remaining_capacity = self.peak_throughput
        return
    

    def generateSubrequestsWithTrace(self, event):
        ''' If branching trace is given, generate the exact number of subrequests
            based on the sequence number of the request
        '''
        subrequestsPerChild = {}
        # --- Deprecated ---
        # sequenceNum = event.sequence_num
        # df_row = self.branchingTrace.loc[sequenceNum]
        # ------------------

        # Instead of using sequence_num, we use the arrival time of the parent
        # request and scale it to get the index
        # --- New method: Time-alignment of traces ---
        df_row = self.getTimescaledProfiledBranching(event)
        # --------------------------------------------
        for column, value in df_row.iteritems():
            # If column name corresponds to a child task
            if column in self.branching_probabilities:
                subrequestsPerChild[column] = value

        return subrequestsPerChild


    def getTimescaledProfiledBranching(self, event):
        ''' Instead of using sequence_num of a parent request to get its profiled
            branching factor, we use the arrival time of the parent request and
            scale it to get the index in the branching trace
        '''
        parent_arrival_time = self.simulator.parent_request_arrival_times[event.parent_request_id]
        # We scale it to be in [0,1]
        isi = event.desc
        if isi not in self.simulator.request_trace_max_times:
            return None

        request_trace_max_time = self.simulator.request_trace_max_times[isi]
        scaled_parent_arrival_time = parent_arrival_time / request_trace_max_time
        # We now scale it to be in [0,branching_trace_max_time]
        scaled_parent_arrival_time = int(scaled_parent_arrival_time *
                                         self.branchingTraceMaxTime)
        print(f'Generating subrequest, parent_arrival_time: {parent_arrival_time}'
              f', scaled_parent_arrival_time: {scaled_parent_arrival_time}')
        df_row = self.branchingTrace.loc[scaled_parent_arrival_time]
        return df_row
    

    def generateSubrequestsWithoutTrace(self):
        ''' If no branching trace is provided, generate subrequests based on
            default multiplicative factor (1) and branching probabilities (equal
            for all children)
        '''
        subrequestsPerChild = {}
        sortedProbs = sorted(self.branching_probabilities.items())
        branchingProbIsiArray = list(map(lambda x: x[0], sortedProbs))
        branchingProbValArray = list(map(lambda x: x[1], sortedProbs))

        if len(branchingProbIsiArray) > 0:
            for i in range(self.multiplicative_factor):
                # generate a random number and assign it to ISI based on prefix sum
                subrequestIsi = random.choices(branchingProbIsiArray,
                                               branchingProbValArray,
                                               k=1)[0]
                subrequestsPerChild[subrequestIsi] = subrequestsPerChild.setdefault(subrequestIsi, 0) + 1

        return subrequestsPerChild
    

    def set_routing_policy(self, routingPolicy: RoutingPolicy):
        self.routingPolicy = routingPolicy
        return
    

    def find_predictor_through_routing_table(self, isi: str, event: Event) -> str:
        ''' Finds a predictor for a given ISI through routing table using
            probability-based routing and returns its ID
        '''
        if self.routingPolicy is None:
            self.simulator.loadBalancer.balanceLoad()
            if self.routingPolicy is None:
                raise Exception('No routing policy set at predictor')

        # First, filter predictors by task
        taskFilteredPredictors = list(filter(lambda x: x.task == isi, self.routingTable))
        filteredPredictors = taskFilteredPredictors
        
        # If we are using MostAccurateFirst routing, filter events by path as well
        if self.routingPolicy == RoutingPolicy.MOST_ACCURATE_FIRST:
            pathAndTaskFilteredPredictors = list(filter(lambda x: x.path == event.path,
                                                        taskFilteredPredictors))
            if len(taskFilteredPredictors) > 0 and len(pathAndTaskFilteredPredictors) == 0:
                # TODO: fix this particular error
                # self.log.error('FIX THIS PARTICULAR ERROR IN PREDICTOR')
                # raise Exception(f'Predictors found for outgoing task {isi} but '
                #                 f'not for incoming path: {event.path}')
                pass
            else:
                filteredPredictors = pathAndTaskFilteredPredictors
        
        # If there are no predictors to serve request, return None
        if len(filteredPredictors) == 0:
            self.log.error(f'find_predictor_through_routing_table(): No predictor '
                           f'found for task: {isi}')
            self.log.error(f'Routing table: {self.routingTable}')
            return None
        
        # Choose a predictor based on routing probability
        weights = list(map(lambda x: x.percentage, filteredPredictors))
        selected = random.choices(filteredPredictors, weights, k=1)

        predictor_id = selected[0].predictor.id

        filteredPredictorsIds = list(map(lambda x: x.predictor.id,
                                         filteredPredictors))
        routingTableStr = [entry.predictor.id for entry in self.routingTable]
        self.log.debug(f'predictor: {self.id}, isi: {self.executor.isi}, '
                      f'filteredPredictorsIds: {filteredPredictorsIds}, '
                      f'self.routingTable: {routingTableStr}, chosen: '
                      f'{predictor_id}')

        return predictor_id
    
    
    def enqueue_request(self, event, clock):
        ''' Add the request to the request queue of this predictor
        '''
        if self.pendingRemoval:
            raise Exception(f'We should not be enqueuing new requests since this '
                            f'predictor is pending removal')
        
        self.predictor_log.debug(f'enqueued,{clock}')
        self.request_dict[event.id] = 1
        bisect.insort(self.request_queue, event, key=lambda x: x.start_time +
                      x.deadline)
        # print(f'request queue sorted: {list(map(lambda x: x.start_time+x.deadline,
        #                                         self.request_queue))}')
        # time.sleep(1)
        # self.request_queue.append(event)
        self.simulator.generate_predictor_enqueued_event(event, clock)
        self.event_counter += 1
        self.incoming_requests_per_step += 1

        # If predictor is busy, we have to wait until we get a FINISH_BATCH event
        # before we further process this request
        if self.busy:
            if self.batching_algo in ['aimd', 'nexus', 'infaas', 'fixed_size']:
                if self.batch_expiring_set == False:
                    if self.max_batch_size == 0:
                        self.simulator.bump_failed_request_stats(event, clock)
                        return
                    if self.batching_algo == 'nexus':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.max_batch_size, event)
                    elif self.batching_algo == 'aimd':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.aimd_batch_size, event)
                    elif self.batching_algo == 'infaas':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.infaas_batch_size, event)
                    elif self.batching_algo == 'fixed_size':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.pipeline_batch_size, event)
                    self.generate_batch_expiring(event, batch_expiring_set)
                    return
            else:
                self.generate_head_slo_expiring(clock)
            return

        if self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'aimd':
            batch_size = self.aimd_batch_size
            if len(self.request_queue) >= self.aimd_batch_size:
                self.log.debug(f'aimd calling process_batch from enqueue_request')
                self.process_batch(clock, self.aimd_batch_size)
            return
        elif self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'nexus':
            if len(self.request_queue) >= self.max_batch_size:
                self.process_batch(clock, self.max_batch_size)
            return
        elif self.task_assignment == TaskAssignment.INFAAS:
            batch_size = self.infaas_batch_size
            if len(self.request_queue) >= batch_size:
                self.process_batch(clock, batch_size)
            return
        elif self.task_assignment == TaskAssignment.MOST_ACCURATE_FIRST and self.batching_algo == 'fixed_size':
            print(f'predictor id: {self.id}')
            batch_size = min(len(self.request_queue), self.pipeline_batch_size)
            if batch_size > 0:
                self.process_batch(clock, batch_size)
            else:
                self.log.info(f'Request queue is empty, not executing batch')
            return
        elif (self.task_assignment == TaskAssignment.CANARY or self.task_assignment == TaskAssignment.MOST_ACCURATE_FIRST) and self.batching_algo == 'accscale':
            # If we are past t_w(q), execute queue with q-1 requests
            # If q-1 is 0 at this point, the request deadlines are not set properly
            # and request cannot possibly be executed within deadline
            # print(f'pre-popping, request queue size: {len(self.request_queue)}')
            if clock > self.request_queue[0].start_time:
                self.pop_while_first_expires(clock)
            # print(f'post-popping, request queue size: {len(self.request_queue)}')
            if len(self.request_queue) == 0:
                raise Exception('Batch cannot be executed even with size 1 with given deadline')
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'enqueue_request: Trying to find appropriate batch size. Number '
                           f'of requests in queue: {len(self.request_queue)}, batch size '
                           f'returned: {batch_size}')
        else:
            raise PredictorException(f'Unexpected combination, task assignment: '
                                     f'{self.task_assignment},  batching algorithm: '
                                     f'{self.batching_algo}')

        if batch_size == -1:
            # requests in queue exceed maximum batch size
            self.process_batch(clock, self.max_batch_size)
        else:
            first_request = self.request_queue[0]
            first_request_expiration = first_request.start_time + first_request.deadline

            if clock > first_request_expiration:
                raise PredictorException('Expired request has not been removed from the queue')

            if self.batch_processing_latency(1, first_request) > first_request.deadline:
                if 'infaas' in self.simulator.model_assignment:
                    self.simulator.bump_failed_request_stats(first_request, clock)
                else:
                    raise PredictorException(f'Request cannot be processed even with batch size '
                                            f'of 1. deadline: {first_request.deadline}, processing '
                                            f'latency: {self.batch_processing_latency(1, first_request)}')

            max_waiting_time = first_request_expiration - self.batch_processing_latency(batch_size, first_request)

            if clock < max_waiting_time:
                # we can still wait with new request in queue
                # print(f'\n\ncan still wait, batch size: {batch_size}, task: '
                #       f'{self.executor.isi}, clock: {clock}, max_waiting_time: '
                #       f'{max_waiting_time}, predictor id: {self.id}\n\n')
                # time.sleep(0.1)

                self.generate_slo_expiring(first_request, max_waiting_time)
                self.log.debug(f'Generated SLO_EXPIRING event for request {event.desc} '
                               f'to expire at {max_waiting_time}')
            else:
                # print(f'\n\nhere instead, predictor id: {self.id}\n\n')
                # time.sleep(0.1)

                # if we execute a batch with the latest request, we will miss SLO
                # of first request in the queue. therefore, execute batch size q-1 requests
                if self.batching_algo == 'accscale':
                    batch_size = self.find_batch_size(requests=len(self.request_queue)-1)
                elif self.task_assignment == TaskAssignment.INFAAS:
                    batch_size = self.infaas_batch_size
                else:
                    raise PredictorException(f'Unexpected combination, task assignment: '
                                             f'{self.task_assignment}, batching algorithm: '
                                             f'{self.batching_algo}')
                self.log.debug(f'Calling process batch from enqueue_request')
                self.process_batch(clock, batch_size)
                return

        return

    
    def process_batch(self, clock, batch_size):
        ''' Dequeue the first `batch_size` requests from the queue and process
        them in a batch.
        '''
        if self.busy:
            raise PredictorException('process_batch called when predictor is busy')
        
        model_asn = self.simulator.model_assignment
        if 'ilp' in model_asn or 'infaas' in model_asn or 'clipper' in model_asn:
            self.drop_expired_requests(clock)
        
        self.simulator.batch_size_counters[batch_size] += 1

        self.log.debug(f'process_batch called with batch size of {batch_size}')

        self.log.debug(f'Requests in queue before popping: {len(self.request_queue)}')

        if batch_size == -1:
            self.log.error(f'process_batch received batch size of -1')
            time.sleep(10)


        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        temp_queue = []
        dequeued_requests = 0

        while dequeued_requests < batch_size and len(self.request_queue) > 0:
            first_request = self.request_queue.pop(0)
            temp_queue.append(first_request)
            self.simulator.generate_predictor_dequeued_event(first_request, clock)
            dequeued_requests += 1

        if dequeued_requests == 0:
            return

        # Since requests have been popped from the queue, we need to generate an
        # SLO_EXPIRING event for the new request at the head of the queue
        if len(self.request_queue) > 0:
            self.generate_head_slo_expiring(clock)

        self.log.debug(f'Batch size given: {batch_size}, requests in queue after popping: '
                       f'{len(self.request_queue)}, dequeued_requests: {dequeued_requests}')

        batch_processing_time = self.batch_processing_latency(batch_size, temp_queue[0])
        finish_time = clock + batch_processing_time
        accuracy_seen = self.profiled_accuracy

        qos_met = True

        batch_finished_late = False
        aimd_negative_feedback = False
        for request in temp_queue:
            if finish_time > request.start_time + request.deadline:
                if self.task_assignment == TaskAssignment.CANARY:
                    if self.batching_algo == 'accscale':
                        batch_finished_late = True
                        pass
                    # AIMD performs lazy-dropping
                    elif self.batching_algo == 'aimd':
                        aimd_negative_feedback = True
                        batch_finished_late = True
                        pass
                    # Since Nexus already performed early-drop, no need to drop here
                    elif self.batching_algo == 'nexus':
                        batch_finished_late = True
                        pass
                    elif self.batching_algo == 'fixed_size':
                        batch_finished_late = True
                        print(f'batch finishing late')
                        time.sleep(1)
                        pass
                    else:
                        raise PredictorException(f'unexpected batching algorithm: {self.batching_algo}')
                elif self.task_assignment == TaskAssignment.INFAAS:
                    batch_finished_late = True
                    pass
            self.simulator.generate_end_request_event(request, finish_time,
                                                      accuracy_seen, qos_met,
                                                      sequence_num=request.sequence_num)
            
        self.simulator.generate_finish_batch_event(finish_time=finish_time,
                                                   predictor=self,
                                                   executor=self.executor)
        
        if aimd_negative_feedback:
            self.decrease_aimd_batch_size()
        else:
            self.increase_aimd_batch_size()

        self.busy = True
        self.busy_till = finish_time

        self.predictor_log.debug(f'process_batch,{clock},{batch_size},{batch_finished_late}')

        return

    
    def finish_batch_callback(self, clock):
        ''' Callback to handle a FINISH_BATCH event
        '''
        self.predictor_log.debug(f'finish_batch_callback,{clock}')

        self.busy = False
        self.busy_till = None

        if len(self.request_queue) == 0:
            return

        if self.task_assignment == TaskAssignment.CANARY or self.task_assignment == TaskAssignment.MOST_ACCURATE_FIRST:
            if self.batching_algo == 'accscale':
                self.pop_while_first_expires(clock)
                if self.expiring_waiting:
                    self.expiring_waiting = False
                    batch_size = self.find_batch_size(len(self.request_queue))
                    if batch_size == -1:
                        batch_size = self.max_batch_size
                    elif batch_size is None:
                        if len(self.request_queue) > 0:
                            raise Exception(f'find_batch_size returned None while '
                                            f'there are requests in the queue')
                        return
                    self.process_batch(clock, batch_size)
                else:
                    if len(self.request_queue) > self.max_batch_size:
                        self.process_batch(clock, self.max_batch_size)
                return
            elif self.batching_algo == 'aimd':
                if len(self.request_queue) >= self.aimd_batch_size:
                    self.log.debug(f'Calling process_batch from finish_batch_callback')
                    self.process_batch(clock, self.aimd_batch_size)
                    return
                elif len(self.request_queue) > 0 and len(self.request_queue) < self.aimd_batch_size:
                    first_request = self.request_queue[0]
                    first_request_deadline = first_request.deadline
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.aimd_batch_size, first_request)
                    self.generate_batch_expiring(first_request, batch_expiring_time)
                    return
            elif self.batching_algo == 'nexus':
                self.batch_expiring_set = False
                if len(self.request_queue) >= self.max_batch_size:
                    self.process_batch(clock, self.max_batch_size)
                elif len(self.request_queue) > 0 and len(self.request_queue) < self.max_batch_size:
                    first_request = self.request_queue[0]
                    first_request_deadline = first_request.deadline
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.max_batch_size, first_request)
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.max_batch_size, first_request)
                    self.generate_batch_expiring(first_request, batch_expiring_time)
                return
            elif self.batching_algo == 'fixed_size':
                self.batch_expiring_set = False
                if not(self.no_early_dropping):
                    # TODO: Do not hard-code last stage tasks. Read them from
                    #       the app configuration
                    if 'traffic_analysis' in self.simulator.apps:
                        last_stage_tasks = ['car_classification', 'facial_recognition']
                    elif 'social_media' in self.simulator.apps:
                        last_stage_tasks = ['object_detection', 'image_captioning']
                    if self.last_stage_early_drop:
                        if self.executor.isi in last_stage_tasks:
                            self.pop_while_first_expires(clock)
                        else:
                            pass
                    else:
                        self.pop_while_first_expires(clock)

                if len(self.request_queue) > 0:
                    batch_size = min(len(self.request_queue), self.pipeline_batch_size)
                    self.process_batch(clock, batch_size)
                return
            else:
                self.log.error(f'finish_batch_callback: Unexpected batching algo: {self.batching_algo}')
        elif self.task_assignment == TaskAssignment.INFAAS:
            self.log.debug(f'infaas batch size: {self.infaas_batch_size}')
            if len(self.request_queue) >= self.infaas_batch_size:
                self.log.debug(f'Calling process_batch from finish_batch_callback for INFaaS')
                self.process_batch(clock, self.infaas_batch_size)
        else:
            self.log.error(f'finish_batch_callback encountered unexpected task '
                          f'assignment algorithm: {self.task_assignment}')

        return

    
    def pop_while_first_expires(self, clock):
        ''' This function removes the first request in the queue if it would expire
        by executing all the requests in the queue in a batch or with the maximum
        batch size allowed, iteratively until we do not have any request in the queue
        that would expire
        '''
        if self.batching_algo in ['aimd']:
            raise PredictorException(f'{self.batching_algo} should not be using '
                                     f'pop_while_first_expires() since it drops request using '
                                     f'an assumption of SLO_EXPIRING events which are not '
                                     f'used for this batching algorithm')
        
        # Assume first request is expiring
        ptr = 0
        ptr_request_expiring = True
        popped = False
        queued_requests = len(self.request_queue)

        while ptr_request_expiring and ptr < queued_requests:
            ptr_request = self.request_queue[ptr]
            ptr_request_expiration = ptr_request.start_time + ptr_request.deadline

            batch_size = self.find_batch_size(queued_requests)
            if batch_size == -1:
                batch_size = self.max_batch_size
            batch_processing_time = self.batch_processing_latency(batch_size, ptr_request)

            self.log.debug(f'FINISH_BATCH callback: current time: {clock}, ptr request '
                           f'starts at {ptr_request.start_time} and expires at {ptr_request_expiration}, '
                           f'time to process it will be {batch_processing_time} with batch '
                           f'size of {batch_size}, requests in queue: {queued_requests}')

            rerouted = False
            if self.dynamic_rerouting:
                # TODO: What to do about multiple branches? Do we reroute for
                # (1) all of them? Or (2) only the critical branch?
                # Answer: 1 is correct, 2 may not be

                # For (1), what do we do if we can only reroute one branch
                # and not the other?
                # Depends on the application. For traffic analysis, if there
                # are 20 child subrequests, if we can save some of them and
                # and not others, that is fine
                # one problem is this: we may not drop 1 request at
                # early stage but it leads to 20 subrequests. Now if we drop
                # 2 out of 20 subrequests, is our SLO violation ratio higher
                # or lower? It should be lower
                # For (2), how do we find the critical branch? No, we need
                # all branches

                # If there are available predictors, check them
                # Get the assigned_next_stage_budget for this path
                # First, let's get the executor for the downstream task
                # Check all its predictors. Filter those that have capacity
                # If there are no shortlisted predictors at this point, get out
                # of IF condition

                currentTask = self.executor.isi
                # TODO: Do not use hard-coded app name
                if 'traffic_analysis' in self.simulator.apps:
                    app = self.simulator.apps['traffic_analysis']
                if 'social_media' in self.simulator.apps:
                    app = self.simulator.apps['social_media']
                currentTaskNode = app.findNodeByTask(task=currentTask)
                childNodes = currentTaskNode.children
                for childNode in childNodes:
                    childTask = childNode.task
                    print(f'childTask: {childTask}, executors: {self.simulator.executors}')
                    executor = self.simulator.executors[childTask]
                    for pred_id in executor.predictors:
                        predictor = executor.predictors[pred_id]
                        if predictor.pendingRemoval:
                            continue
                        variant = predictor.variant_name
                        if variant not in self.simulator.variant_slo_dict:
                            continue
                            # raise Exception(f'Active predictor not in simulator '
                            #                 f'variant_slo_dict (variant: {variant})')
                        potential_next_stage_budget = self.simulator.variant_slo_dict[variant]

                        assigned_next_stage_predictor_id = self.find_predictor_through_routing_table(isi=childTask,
                                                                                                     event=ptr_request)
                        if assigned_next_stage_predictor_id is None or assigned_next_stage_predictor_id not in executor.predictors:
                            # reroute in this case
                            assigned_next_stage_budget = sys.maxsize
                        else:
                            assigned_next_stage_predictor = executor.predictors[assigned_next_stage_predictor_id]
                            assigned_variant = assigned_next_stage_predictor.variant_name
                            if assigned_variant in self.simulator.variant_slo_dict:
                                assigned_next_stage_budget = self.simulator.variant_slo_dict[assigned_variant]
                            else:
                                # reroute in this case
                                assigned_next_stage_budget = sys.maxsize

                        if potential_next_stage_budget >= assigned_next_stage_budget:
                            print('potential budget is >= assigned budget')
                            # time.sleep(1)
                            continue

                        if predictor.remaining_capacity <= 0:
                            print('potential predictor has no remaining capacity')
                            # time.sleep(1)
                            continue

                        if clock + batch_processing_time <= ptr_request_expiration - potential_next_stage_budget:
                            # Update the predictor's load
                            # TODO: This only works if requests are in QPS,
                            #       i.e., remaining_capacity should be reset
                            #       every second
                            self.log.warning(f'If ILP is not solved every second, '
                                             f'fix this TODO')
                            df_row = self.getTimescaledProfiledBranching(ptr_request)
                            if df_row is None:
                                # It is for a task which has no branching, skip for now
                                print(f'task: {self.executor.isi} has no branching')
                                continue

                            if childTask in df_row:
                                downstream_requests = df_row[childTask]
                            else:
                                print(f'childTask is not in df_row')
                                # time.sleep(1)
                                continue
                            predictor.remaining_capacity -= downstream_requests

                            ptr_request.dynamic_rerouted = True
                            ptr_request.rerouted_target_predictors[childTask] = predictor
                            rerouted = True

                            print(f'REROUTING_PLANNED, clock: {clock}, event: '
                                  f'{ptr_request.getStr()}')
                            # time.sleep(1)
                            break

                # otherwise, use default method
                pass

            # If we reroute a query, we are just ignoring a potentially
            # expiring request. Later requests in the queue could also be
            # expiring. So we keep going until the request at the pointer
            # does not expire

            if not(rerouted):
                if clock + batch_processing_time > ptr_request_expiration:
                    failed_request = self.request_queue.pop(ptr)
                    self.simulator.pop_expired_request_in_queue(failed_request, clock)
                    self.simulator.bump_failed_request_stats(failed_request, clock)
                    queued_requests = len(self.request_queue)
                    popped = True
                else:
                    ptr_request_expiring = False

            ptr += 1

        if popped:
            # Since requests have been popped from the queue, we need to
            # generate an SLO_EXPIRING event for the new request at the head
            # of the queue
            self.generate_head_slo_expiring(clock)

        return

    
    def generate_slo_expiring(self, event, time):
        ''' Call the simulator's handler for generating an SLO_EXPIRING
        event
        '''
        if event.id in self.slo_expiring_dict and self.slo_expiring_dict[event.id] == time:
            # We don't need to generate a new SLO_EXPIRING event since we already have
            # an event with the same expiration time (because batch size hasn't changed)
            return
        self.log.debug(f'Generating SLO_EXPIRING event for request {event.id} '
                      f'to expire at {time}')
        self.simulator.generate_slo_expiring_event(time, event,
                                                   predictor=self,
                                                   executor=self.executor,
                                                   event_counter=event.id)
        self.slo_expiring_dict[event.id] = time
        self.event_counter += 1
        return
    

    def generate_batch_expiring(self, event, time):
        ''' Call the simulator's handler for generating a BATCH_EXPIRING
        event
        '''
        self.simulator.generate_batch_expiring_event(time, event,
                                                     predictor=self,
                                                     executor=self.executor,
                                                     event_counter=event.id)
        self.batch_expiring_set = True
        return

    
    def slo_expiring_callback(self, event, clock):
        ''' Callback to handle an SLO_EXPIRING event
        '''
        if self.busy is True:
            self.expiring_waiting = True
            return
        
        if self.batching_algo == 'nexus' or self.batching_algo == 'aimd' or self.batching_algo == 'infaas' or self.batching_algo == 'fixed_size':
            self.log.debug(f'SLO expiring event encountered, ignoring for task assignment: '
                          f'{self.task_assignment}, batching algo: {self.batching_algo}')
            return

        self.log.debug(f'SLO expiring callback, Current clock: {clock}, event counter: '
            f'{event.event_counter}, slo_expiring_dict entry: {self.slo_expiring_dict[event.event_counter]}')
        if clock != self.slo_expiring_dict[event.event_counter]:
            # this means we have encountered an older SLO_EXPIRING callback
            # we do not need to handle it
            self.log.debug(f'slo_expiring_callback: Encountered an older SLO_EXPIRING '
                f'event for request {event.event_counter}. Latest value: '
                f'{self.slo_expiring_dict[event.event_counter]}, Value in '
                f'current event: {clock}')
            return

        if len(self.request_queue) > 0 and self.request_queue[0].id != event.event_counter:
            self.log.debug(f'Request at head of queue: {self.request_queue[0].id}, SLO_EXPIRING event '
                           f'received for request: {event.event_counter}. Previous request: '
                           f'{(event.event_counter in self.slo_expiring_dict)}')
            return

        if len(self.request_queue) == 0:
            # We might have already processed the request before reaching its
            # SLO EXPIRING
            return

        if (self.task_assignment == TaskAssignment.CANARY or self.task_assignment == TaskAssignment.MOST_ACCURATE_FIRST) and self.batching_algo == 'accscale':
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'slo_expiring_callback: Trying to find appropriate batch size. '
                           f'Number of requests in queue: {len(self.request_queue)}, '
                           f'batch size returned: {batch_size}')
        elif self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'aimd':
            batch_size = self.aimd_batch_size
        elif self.task_assignment == TaskAssignment.INFAAS:
            batch_size = self.infaas_batch_size
            if batch_size == 0:
                batch_size = 1
        else:
            raise PredictorException(f'Unexpected combination, task assignment: '
                                     f'{self.task_assignment},  batching algorithm: '
                                     f'{self.batching_algo}')
        
        if batch_size == -1:
            # this can happen in two cases:
            # 1. if we added new requests while the predictor was busy processing another batch
            # 2. if the requests in batch were already more than max batch size when head
            #   SLO was generated
            self.log.warn('slo_expiring_callback: find_batch_size returned -1')
            batch_size = self.max_batch_size
            return
        
        # if self.batching_algo is not 'aimd':
        self.log.debug(f'Calling process_batch from slo_expiring_callback')
        self.process_batch(clock, batch_size)
        return
    

    def batch_expiring_callback(self, event, clock):
        if self.batching_algo == 'nexus':
            self.nexus_expiring_callback(event, clock)
        elif self.batching_algo == 'aimd':
            self.aimd_expiring_callback(event, clock)
        elif self.batching_algo == 'infaas':
            self.infaas_expiring_callback(event, clock)
        elif self.batching_algo == 'fixed_size':
            self.fixed_size_expiring_callback(event, clock)
        else:
            raise PredictorException(f'Unexpected batching algo: {self.batching_algo}')
    
    
    def nexus_expiring_callback(self, event, clock):
        ''' Callback to handle a Nexus BATCH_EXPIRING event
        '''
        if self.busy is True:
            return

        self.pop_while_first_expires(clock)

        if len(self.request_queue) == 0:
            self.log.debug(f'no requests in queue, returning from nexus expiring callback')
            return
        
        if len(self.request_queue) >= self.max_batch_size:
            batch_size = self.max_batch_size
        else:
            batch_size = self.find_batch_size(len(self.request_queue))
        self.process_batch(clock, batch_size)
        return
    

    def aimd_expiring_callback(self, event, clock):
        ''' Callback to handle an AIMD BATCH_EXPIRING event
        '''
        # AIMD does not respond to BATCH_EXPIRING. If this functionality is changed,
        # this function will be implemented similar to nexus_expiring_callback()
        pass


    def infaas_expiring_callback(self, event, clock):
        ''' Callback to handle an INFaaS BATCH_EXPIRING event
        '''
        if self.busy is True:
            return
        
        self.pop_while_first_expires(clock)
        
        if len(self.request_queue) == 0:
            return
        
        batch_size = min(self.infaas_batch_size, self.find_batch_size(len(self.request_queue)))
        if batch_size == -1:
            batch_size = self.infaas_batch_size

        self.process_batch(clock, batch_size)
        return
    

    def fixed_size_expiring_callback(self, event, clock):
        ''' Callback to handle a BATCH_EXPIRING event with 'fixed_size' batching
        '''
        print(f'reached fixed_size_expiring_callback, queue_size: {len(self.request_queue)}, self.busy: {self.busy}')
        # time.sleep(1)
        if self.busy is True:
            return
        
        if len(self.request_queue) == 0:
            return
        
        batch_size = min(len(self.request_queue), self.max_batch_size)
        self.process_batch(clock, batch_size)
        return
    

    def drop_expired_requests(self, clock):
        ''' Drop all expired requests from the queue
        '''
        drop_indices = []
        for i in range(len(self.request_queue)):
            request = self.request_queue[i]
            if clock > request.start_time + request.deadline * 2:
                drop_indices.append(i)

        dropped = 0
        for i in range(len(drop_indices)):
            drop_idx = drop_indices[i]
            request = self.request_queue.pop(drop_idx-dropped)
            self.simulator.bump_failed_request_stats(request, clock)
            dropped += 1
        return

    
    def generate_head_slo_expiring(self, clock):
        ''' If any request is popped from the head of the queue and the queue has
        a new head, we need to generate an SLO_EXPIRING event for it to keep track
        of its expiry
        '''
        if len(self.request_queue) == 0:
            return

        first_request = self.request_queue[0]
        first_request_expiration = first_request.start_time + first_request.deadline

        while self.batch_processing_latency(batch_size=1, request=first_request) > first_request.deadline:
            if 'accscale' in self.simulator.model_assignment or 'accscale' in self.simulator.batching_algo:
                self.simulator.bump_failed_request_stats(first_request, clock)
                self.request_queue.pop(0)
                if len(self.request_queue) == 0:
                    return
                first_request = self.request_queue[0]
                first_request_expiration = first_request.start_time + first_request.deadline
            else:
                raise PredictorException(f'Request cannot be processed even with batch size of 1')
        
        batch_size = self.find_batch_size(requests=len(self.request_queue))

        if batch_size == -1:
            batch_size = self.max_batch_size

        max_waiting_time = first_request_expiration - self.batch_processing_latency(batch_size, first_request)

        self.generate_slo_expiring(first_request, max_waiting_time)
        self.log.debug(f'head: Generated SLO_EXPIRING event for request {first_request.desc} '
                      f'to expire at {max_waiting_time}')
        return

    
    def batch_processing_latency(self, batch_size, request):
        ''' Return the latency to process a batch of a given size
        '''
        self.log.debug('batch_processing_latency()')
        self.log.debug(f'Profiled latencies: {self.profiled_latencies}')
        self.log.debug(f'Request desc: {request.desc}, qos_level: {request.qos_level}, '
                      f'profiled latency: {self.profiled_latencies[(request.desc, self.variant_name, batch_size)]}')
        processing_latency = self.profiled_latencies[(request.desc, self.variant_name, batch_size)]
        return processing_latency

    
    def find_batch_size(self, requests):
        ''' Find the appropriate batch size for a given number of requests by
        rounding up to the nearest bigger batch size
        '''
        if requests == 0:
            return None

        batch_size_index = self.find_maximum_that_fills(requests)
        if batch_size_index >= len(self.batch_sizes_allowed):
            return -1
        else:
            batch_size = self.batch_sizes_allowed[batch_size_index]
            return batch_size

    
    def find_maximum_that_fills(self, requests):
        ''' Alternate way of finding an appropriate batch size. We select a
        batch size that gets completely filled up by the current queue,
        instead of rounding up to the nearest bigger batch size
        '''
        if requests == 0:
            return -1

        idx = 0
        batch_idx = 0
        while idx < len(self.batch_sizes_allowed):
            # We cannot exceed the maximum batch size in our search
            if self.batch_sizes_allowed[idx] > self.max_batch_size:
                return batch_idx

            if requests >= self.batch_sizes_allowed[idx]:
                batch_idx = idx
            else:
                return batch_idx
            idx += 1
        return batch_idx

    
    def binary_search_index(self, arr, number):
        ''' Finds the appropriate batch size for a given number of requests
        by rounding up to the nearest bigger batch size available.
        For example, if we support batch sizes of 2 and 4, a queue of 3
        requests will be given a batch size of 4 according to this.
        Note: This might result in lower throughput as we are increasing
        latency but finishing a smaller number of requests
        '''
        # Lower and upper bounds
        start = 0
        end = len(arr) - 1

        # Traverse the search space
        while start <= end:
            mid = (start + end) // 2
            if arr[mid] == number:
                return mid
            elif arr[mid] < number:
                start = mid + 1
            else:
                end = mid - 1
        # Return the insert position
        return end + 1
