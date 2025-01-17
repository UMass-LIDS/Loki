import copy
import logging
import random
import requests
import os
import sys
import time
import pprint
import numpy as np
import pandas as pd
from core.common import Event, EventType
from core.executor import Executor, AccType
from core.exceptions import SimulatorException
from core.load_balancer import SimLoadBalancer
from core.predictor import Predictor
from common.app import registerApplication


USecInMSec = 1000
# Setting default parameter values in case none are specified

MAX_CLOCK_VALUE = 0
# If the simulator clock is within this threshold of the starting time of
# the last request in queue for a given executor, it will refill the event
# queue
QUEUE_REFILL_THRESHOLD = 2000
UTILIZATION_THRESHOLD = 0.65
CHANGE_PROPORTION_THRESHOLD = 1 - UTILIZATION_THRESHOLD
EWMA_WINDOW = 20
DEFAULT_EWMA_DECAY = 2.1
DEFAULT_INFAAS_SLACK = 0.95

# Pipeline-specific
PROFILED_BRANCHING_MAX_HISTORY = 50


class Simulator:
    def __init__(self, job_sched_algo, logging_level, app_definitions_path=None,
                 trace_path=None, mode='training', branching_trace_path=None,
                 max_acc_per_type=0, predictors_max=[10, 10, 10, 10], n_qos_levels=1,
                 random_runtimes=False, fixed_seed=0, batching=False,
                 model_assignment=None, batching_algo=None, profiling_data=None,
                 allowed_variants_path=None, max_batch_size=None, ewma_decay=None,
                 infaas_slack=None, infaas_downscale_slack=None,
                 allocation_window=None, pipeline_accuracy_file=None,
                 dynamic_rerouting=False, no_early_dropping=None,
                 last_stage_early_drop=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)
        self.logging_level=logging_level

        self.clock = 0
        self.event_queue = []
        self.apps = {}
        self.executors = {}
        self.idx_to_executor = {}
        self.isi_to_idx = {}
        self.failed_requests = 0
        self.total_failed = 0
        self.total_successful = 0
        self.total_successful_per_model = {}
        self.failed_requests_arr = []
        self.successful_requests = 0
        self.successful_requests_arr = []
        self.total_requests_arr = []
        self.mode = mode
        self.completed_requests = 0

        self.max_acc_per_type = max_acc_per_type

        if fixed_seed != 0:
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)

        self.job_sched_algo = job_sched_algo
        self.n_qos_levels = n_qos_levels

        self.store_file_pointers = True
        self.requests_added = 0
        self.parent_requests_added = 0

        self.branching_trace_path = branching_trace_path

        self.allocation_window = allocation_window

        self.total_accuracy = 0
        self.total_accuracy_per_model = {}

        self.accelerator_types = ['CPU', 'GPU_AMPERE', 'VPU', 'GPU_PASCAL']

        self.cpu_runtimes = {}
        self.gpu_runtimes = {}
        self.vpu_runtimes = {}
        self.fpga_runtimes = {}

        self.cpu_loadtimes = {}
        self.gpu_loadtimes = {}
        self.vpu_loadtimes = {}
        self.fpga_loadtimes = {}

        self.model_variants = {}
        self.model_variant_runtimes = {}
        self.model_variant_loadtimes = {}
        self.model_variant_accuracies = {}

        self.cpu_variant_runtimes = {}
        self.gpu_variant_runtimes = {}
        self.vpu_variant_runtimes = {}
        self.fpga_variant_runtimes = {}

        self.requests_per_model = {}
        self.failed_per_model = {}
        self.successful_per_model = {}

        self.accuracy_per_model = {}

        self.trace_files = {}
        self.trace_file_finished = set()
        self.last_request_starting = {}

        # Used to profile the most recent branching information for a given
        # predictor
        self.profiled_branching = {}

        if pipeline_accuracy_file is None:
            self.pipeline_accuracies = None
        else:
            self.pipeline_accuracies = pd.read_csv(pipeline_accuracy_file)

        self.accuracy_logfile = open('logs/log_ilp_accuracy.txt', mode='w')
        self.latency_logfilename = os.path.join('logs', 'latency',
                                                f'{time.time()}_{model_assignment}.csv')
        self.latency_logfile = open(self.latency_logfilename, mode='w')

        self.predictors_max = predictors_max
        self.available_predictors = copy.deepcopy(predictors_max)
        self.added_predictors = copy.deepcopy(predictors_max)
        self.log.info(f'available predictors at the start: {self.available_predictors}')

        self.batching_enabled = batching
        self.batching_algo = batching_algo
        self.largest_batch_sizes = {}
        self.max_batch_size = max_batch_size
        # The difference between slo_dict and variant_slo_dict is that slo_dict
        # contains a value for every executor, whereas variant_slo_dict contains
        # a value for every model variant as set by the pipeline ILP
        self.slo_dict = {}
        self.variant_slo_dict = {}
        self.root_isi_to_app_name = {}
        # self.allowed_batch_sizes = range(64+1)
        self.allowed_batch_sizes = range(1, 65)
        self.profiled_filename = profiling_data

        self.parent_request_arrival_times = {}
        self.request_trace_max_times = {}

        self.model_assignment = model_assignment
        self.use_proportional = None

        self.waiting_queries = {}
        self.executing_queries = {}
        self.request_times = {}

        # Flag to indicate whether we are allowing dynamically rerouting
        # requests for early dropping in pipelines
        self.dynamic_rerouting = dynamic_rerouting

        # Flag to indicate disabling of early dropping
        self.no_early_dropping = no_early_dropping
        # Flag to indicate whether to do early dropping at last stage
        self.last_stage_early_drop = last_stage_early_drop

        if self.no_early_dropping and self.last_stage_early_drop:
            raise Exception(f'Early dropping is disabled but last stage early '
                            f'dropping is enabled. This system configuration '
                            f'is incompatible')

        if ewma_decay is None:
            self.ewma_decay = DEFAULT_EWMA_DECAY
        else:
            self.ewma_decay = ewma_decay

        if infaas_slack is None:
            self.infaas_slack = 1 - DEFAULT_INFAAS_SLACK
        else:
            self.infaas_slack = 1 - infaas_slack

        if infaas_downscale_slack is None:
            self.infaas_downscale_slack = 1 - DEFAULT_INFAAS_SLACK
        else:
            self.infaas_downscale_slack = 1 - infaas_downscale_slack

        idx = 0

        if app_definitions_path is None:
            raise SimulatorException(f'No application definitions provided')
        if trace_path is None:
            raise SimulatorException(f'No trace file provided. trace_path: {trace_path}')
        
        self.log.info(f'Reading applications from: {app_definitions_path}')
        appFiles = sorted(os.listdir(app_definitions_path))
        for appFile in appFiles:
            filename = appFile.split('/')[-1]
            if len(filename.split('.')) == 1:
                # it is a directory
                continue
            appName, extension = filename.split('.')
            if not extension == 'json':
                continue
            self.log.info('App filename: ' + appFile)

            app = registerApplication(filename=os.path.join(app_definitions_path,
                                                            filename))
            # Store app with key appName
            self.apps[appName] = app

            numStages = app.getDepth()
            appLatencySLOInMS = app.getLatencySLO() / USecInMSec
            perStageSLO = appLatencySLOInMS / numStages

            # Add executors for corresponding tasks
            tasks = app.getAllTasks()
            self.log.info(f'tasks: {tasks}')

            for task in tasks:
                # Add executor for task
                isi_name = task

                # By default, the SLO is equally divided across all stages
                # unless otherwise modified
                if task == 'source' or task == 'sink':
                    self.slo_dict[task] = 5.0
                else:
                    self.slo_dict[isi_name] = perStageSLO

                print(f'slo_dict: {self.slo_dict}')
                time.sleep(1)

                variant_list_filename = os.path.join(allowed_variants_path, isi_name)
                self.log.info('Variant list filename:' + variant_list_filename)

                # TODO: read_variants_from_file also sets the single-model
                #       accuracy for variants
                model_variant_list = self.read_variants_from_file(variant_list_filename)
                self.set_model_variants(isi_name, model_variant_list)

                self.initialize_model_variant_loadtimes(isi_name)

                self.add_executor(isi_name, self.job_sched_algo, runtimes={},
                                  model_variant_runtimes={}, model_variant_loadtimes={},
                                  max_acc_per_type=self.max_acc_per_type, simulator=self,
                                  application=app)
                self.idx_to_executor[idx] = isi_name
                self.isi_to_idx[isi_name] = idx
                self.failed_requests_arr.append(0)
                self.successful_requests_arr.append(0)
                self.total_requests_arr.append(0)
                idx += 1

            # Read the trace for the application
            traceFile = os.path.join(trace_path, appName+'.txt')

            # For every application, there should be a corresponding trace file
            # in trace_path
            if not(os.path.exists(traceFile)):
                raise Exception(f'No trace file exists for app {appName} at '
                                f'path {traceFile}')
            
            root_isi_name = app.getRootNode().getTaskName()
            self.root_isi_to_app_name[root_isi_name] = appName

            self.request_trace_max_times[root_isi_name] = self.read_request_max_time(readfileName=traceFile)

            readfile = open(traceFile, mode='r')
            self.trace_files[root_isi_name] = readfile
            self.add_requests_from_trace_pointer(
                root_isi_name, readfile, read_until=10000)
        
        self.initialize_model_variant_runtimes()

        self.runtimes = {1: self.cpu_runtimes, 2: self.gpu_runtimes,
                         3: self.vpu_runtimes, 4: self.fpga_runtimes}

        self.loadtimes = {1: self.cpu_loadtimes, 2: self.gpu_loadtimes,
                          3: self.vpu_loadtimes, 4: self.fpga_loadtimes}

        self.set_executor_runtimes()
        self.set_executor_loadtimes()
        self.log.info(f'Model variant accuracies: {self.model_variant_accuracies}')

        self.set_executor_model_variants()

        self.set_executor_variant_accuracies()
        self.set_executor_variant_runtimes()
        self.set_executor_variant_loadtimes()

        # TODO: Currently, this is only set on initialization, but will need to
        #       be set everytime the SLO of different stages changes
        self.set_largest_batch_sizes()

        self.qos_stats = np.zeros((len(self.executors), n_qos_levels * 2))
        self.slo_timeouts = {'total': 0, 'succeeded': 0, 'timeouts': 0, 'late': 0}
        self.slo_timeouts_per_executor = {'total': {}, 'succeeded': {}, 'timeouts': {},
                                          'late': {}}
        self.requests_per_executor = {}
        self.aimd_stats = {'increased': 0, 'decreased': 0}
        self.batch_size_counters = self.initialize_batch_size_counters()
        self.ilp_stats = {'estimated_throughput': 0, 'estimated_effective_accuracy': 0,
                          'demand': 0, 'ilp_utilization': 0}
        self.demand_moving_window = np.zeros((len(self.executors), 1))
        self.ewma_demand = np.zeros((len(self.executors), 1))

        self.loadBalancer = SimLoadBalancer(simulator=self)

        if not (mode == 'training' or mode == 'debugging'):
            self.insert_event(0, EventType.SCHEDULING,
                              'invoke scheduling agent')

        # how frequently the scheduling agent is invoked (in milliseconds)
        self.sched_interval = 200

        self.sched_agent_uri = 'http://127.0.0.1:8000'

        self.log.warning(f'Models which are not profiled have max batch size '
                         f'set to 0 without the simulator crashing')
        time.sleep(1)

    
    def set_executor_runtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_runtimes(self.runtimes)

    
    def set_executor_model_variants(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_model_variants(self.model_variants)


    def set_executor_variant_accuracies(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_accuracies(self.model_variant_accuracies)


    def set_executor_variant_loadtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_loadtimes(self.model_variant_loadtimes)

    
    def set_executor_variant_runtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_runtimes(self.model_variant_runtimes)


    def set_executor_loadtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_loadtimes(self.loadtimes)

    
    def read_variants_from_file(self, filename):
        model_variants = []
        isi_name = filename.split('/')[-1]
        print(f'filename: {filename}')
        if os.path.exists(filename):
            with open(filename, mode='r') as rf:
                for line in rf.readlines():
                    model_variant = line.split(',')[0]
                    accuracy = float(line.rstrip('\n').split(',')[1])

                    model_variants.append(model_variant)
                    self.model_variant_accuracies[(isi_name, model_variant)] = accuracy
        else:
            raise SimulatorException(f'path not found: {filename}')
        return model_variants


    def reset(self):
        print('Resetting simulator')
        return
    

    def add_moving_demand(self, demand):
        ''' Adds latest demand to moving window and updates EWMA demand
        '''
        self.demand_moving_window = np.hstack((self.demand_moving_window, demand))

        while self.demand_moving_window.shape[1] > EWMA_WINDOW:
            self.demand_moving_window = self.demand_moving_window[:, 1:]
        
        self.log.debug(f'demand_moving_window: {self.demand_moving_window}')
        for i in range(len(self.executors)):
            df = pd.DataFrame({'demand': self.demand_moving_window[i, :]})
            ewma = df.ewm(com=self.ewma_decay).mean()
            ewma_value = ewma['demand'].to_list()[-1]
            self.ewma_demand[i] = ewma_value
        return

    def read_request_max_time(self, readfileName):
        ''' Reads the end time of the request trace
        '''
        max_time = 0
        with open(readfileName, mode='r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                request_description = line.rstrip('\n').split(',')
                if len(request_description) > 0:
                    start_time = int(float(request_description[0]))
                    max_time = max(max_time, start_time)

        return max_time

    def add_requests_from_trace_pointer(self, isi_name, readfile, read_until=500):
        ''' Partially read trace file into memory and store pointer.
        '''
        while True:
            line = readfile.readline()
            if not line:
                return True
            if line.strip()[0] == '#':
                # Skip comment lines in file
                continue
            request_description = line.rstrip('\n').split(',')
            start_time = int(float(request_description[0]))
            # if qos_level is defined, use that. otherwise use qos_level 0 by default
            if len(request_description) >= 2:
                try:
                    qos_level = int(request_description[1])
                except ValueError:
                    accuracy = float(request_description[1])
                    qos_level = 0
            else:
                qos_level = 0
            # if n_qos_levels == 1, ignore qos info from trace
            if self.n_qos_levels == 1:
                qos_level = 0

            # TODO: The third column in trace files is no longer used
            #       Instead, the SLO is read from application's JSON file

            # if len(request_description) >= 3:
            #     deadline = float(request_description[2])
            #     # raise Exception(f'This is a hack. We are setting SLO from last '
            #     #                 f'request seen in the trace. It should be read '
            #     #                 f'from the application file. Moreover, we need '
            #     #                 f'to figure out how to set SLO of each task in '
            #     #                 f'a multi-task pipeline. Divide equally? Where '
            #     #                 f'to set it?')
            #     self.slo_dict[isi_name] = deadline
            # else:
            #     deadline = 1000

            deadline = self.slo_dict[isi_name]

            if len(request_description) >= 4:
                accuracy = float(request_description[3])
            else:
                accuracy = 100.0

            app_name = self.root_isi_to_app_name[isi_name]

            # The path is assigned when the Executor assigns the request
            # to a Predictor
            _, event = self.insert_event(start_time, EventType.START_REQUEST, isi_name,
                                      runtime=None, deadline=deadline,
                                      qos_level=qos_level, accuracy=accuracy,
                                      sequence_num=self.parent_requests_added,
                                      path=None, app_name=app_name,
                                      app_parent_task=isi_name)
            self.requests_added += 1
            self.parent_requests_added += 1

            # We want to keep track of the arrival times of all parent requests
            parent_request_id = event.parent_request_id
            if parent_request_id == '' or parent_request_id is None:
                raise Exception(f'Event was not assigned a parent_request_id')
            self.parent_request_arrival_times[parent_request_id] = start_time

            if start_time >= read_until:
                break

            # We want to keep track of where we stopped adding requests for
            # each file, so that when the clock gets closer to that time,
            # we can refill the event queue
            self.last_request_starting[isi_name] = start_time
        return False
    
    def add_profiled_branching_info(self, predictor: Predictor,
                                    branchingInfo: dict):
        ''' Updates the profiled info
        '''
        if predictor not in self.profiled_branching:
            self.profiled_branching[predictor] = {}

        for isi in branchingInfo:
            if isi in self.profiled_branching[predictor]:
                self.profiled_branching[predictor][isi].append(branchingInfo[isi])

                while len(self.profiled_branching[predictor][isi]) > PROFILED_BRANCHING_MAX_HISTORY:
                    self.profiled_branching[predictor][isi].pop(0)
            else:
                self.profiled_branching[predictor][isi] = [branchingInfo[isi]]
        return

    def add_intermediate_request_from_predictor(self, isi_name: str,
                                                parent_request_id: str,
                                                start_time: int,
                                                target_predictor_id: str,
                                                sequence_num: int,
                                                path: str):
        # If target predictor is present and its deadline is set in
        # self.variant_slo_dict, use the deadline value
        # Otherwise, default to equal deadline per stage
        deadline = None
        executor = self.executors[isi_name]
        if target_predictor_id in executor.predictors:
            target_predictor = executor.predictors[target_predictor_id]
            target_model_variant = target_predictor.variant_name
            if target_model_variant in self.variant_slo_dict:
                deadline = self.variant_slo_dict[target_model_variant]

        if deadline is None:
            deadline = self.slo_dict[isi_name]
        # deadline = self.slo_dict[isi_name]

        self.insert_event(start_time=start_time, type=EventType.START_REQUEST,
                          desc=isi_name, runtime=None, deadline=deadline,
                          qos_level=0, accuracy=0,
                          parent_request_id=parent_request_id,
                          target_predictor_id=target_predictor_id,
                          sequence_num=sequence_num,
                          path=path)
        self.requests_added += 1
        print(f'Adding intermediate request for isi: {isi_name}')
        return

    def set_model_variants(self, isi_name, model_variant_list):
        self.model_variants[isi_name] = model_variant_list

        self.requests_per_model[isi_name] = 0
        self.failed_per_model[isi_name] = 0
        self.successful_per_model[isi_name] = 0

        self.accuracy_per_model[isi_name] = []

    def initialize_dummy_runtimes(self, isi_name, random_runtimes=False):
        for qos_level in range(self.n_qos_levels):
            for batch_size in self.allowed_batch_sizes:
                if random_runtimes:
                    self.cpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.gpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.vpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.fpga_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                else:
                    self.cpu_runtimes[isi_name, qos_level, batch_size] = 50
                    self.gpu_runtimes[isi_name, qos_level, batch_size] = 25
                    self.vpu_runtimes[isi_name, qos_level, batch_size] = 35
                    self.fpga_runtimes[isi_name, qos_level, batch_size] = 30
        
        # Initialize the runtimes per model variant as well
        self.initialize_dummy_model_variant_runtimes(isi_name, random_runtimes=random_runtimes)
        return

    def get_thput_accuracy_per_model(self):
        requests = []
        failed = []
        accuracy = []
        successful = []
        for model in self.requests_per_model:
            requests.append(self.requests_per_model[model])
            self.requests_per_model[model] = 0
            
            failed.append(self.failed_per_model[model])
            self.failed_per_model[model] = 0

            total_accuracy = sum(self.accuracy_per_model[model])
            accuracy.append(total_accuracy)
            self.accuracy_per_model[model] = []

            successful.append(self.successful_per_model[model])
            self.successful_per_model[model] = 0

        return requests, failed, accuracy, successful

    def initialize_model_variant_loadtimes(self, isi_name):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            self.model_variant_loadtimes[(isi_name, model_variant)] = 0

    def initialize_batch_size_counters(self):
        batch_size_counters = {}
        for batch_size in self.allowed_batch_sizes:
            batch_size_counters[batch_size] = 0
        return batch_size_counters

    def initialize_model_variant_runtimes(self):
        profiled = self.import_profiled_data()
        model_variants = profiled['model_variant']
        acc_types = profiled['acc_type']
        batch_sizes = profiled['batch_size']
        latencies = profiled['latency']
        isi_list = []
        for idx in range(profiled.shape[0]):
            model_variant = model_variants[idx]
            if model_variant not in sum(self.model_variants.values(), []):
                # the model variant is profiled but not in allowed_variants from config
                continue
            isi_name = self.get_isi_from_variant_name(model_variant)
            if isi_name not in isi_list:
                isi_list.append(isi_name)
            acc_type = acc_types[idx]
            batch_size = batch_sizes[idx]
            latency = latencies[idx]

            if batch_size not in self.allowed_batch_sizes:
                continue

            # Since the profiled data does not have VPUs, we make the following
            # assumption to reconcile this data with the simulator experiment
            # setting: Assume that VPUs behave the same as CPUs. So effectively
            # we are left with twice the number of CPUs and 0 VPUs
            tuple = (isi_name, model_variant, batch_size)
            if acc_type == 'CPU':
                self.cpu_variant_runtimes[tuple] = latency
                self.vpu_variant_runtimes[tuple] = latency
            elif acc_type == 'GPU_AMPERE':
                self.gpu_variant_runtimes[tuple] = latency
            elif acc_type == 'GPU_PASCAL':
                self.fpga_variant_runtimes[tuple] = latency

            self.log.debug(f'({isi_name}, {model_variant}, {batch_size}): {latency}, {acc_type}')
        
        # If there are unfilled entries that have not been profiled, set their
        # latencies to be effectively infinite so we don't use them
        for isi_name in isi_list:
            model_variant_list = self.model_variants[isi_name]
            for model_variant in model_variant_list:
                for batch_size in self.allowed_batch_sizes:
                    tuple = (isi_name, model_variant, batch_size)

                    if tuple not in self.cpu_variant_runtimes:
                        self.cpu_variant_runtimes[tuple] = np.inf
                    if tuple not in self.gpu_variant_runtimes:
                        self.gpu_variant_runtimes[tuple] = np.inf
                    if tuple not in self.vpu_variant_runtimes:
                        self.vpu_variant_runtimes[tuple] = np.inf
                    if tuple not in self.fpga_variant_runtimes:
                        self.fpga_variant_runtimes[tuple] = np.inf

        self.log.debug(f'Total profiled entries: {profiled.shape[0]}')

        self.model_variant_runtimes = {1: self.cpu_variant_runtimes, 2: self.gpu_variant_runtimes,
                                       3: self.vpu_variant_runtimes, 4: self.fpga_variant_runtimes}
        return

    def set_largest_batch_sizes(self):
        ''' Accesses the profiled data to establish the maximum batch size that
        does not violate latency SLO for each (variant, accelerator) pair
        '''
        max_batch_size_dict = {}
        
        for isi_name in self.model_variants:
            model_variants = self.model_variants[isi_name]
            for model_variant in model_variants:
                # isi_name = self.simulator.get_isi_from_variant_name(model_variant)
                for acc_type in self.accelerator_types:
                    acc_latencies = {}
                    if acc_type == 'CPU':
                        acc_latencies = self.model_variant_runtimes[1]
                    elif acc_type == 'GPU_AMPERE':
                        acc_latencies = self.model_variant_runtimes[2]
                    elif acc_type == 'VPU':
                        acc_latencies = self.model_variant_runtimes[3]
                    elif acc_type == 'GPU_PASCAL':
                        acc_latencies = self.model_variant_runtimes[4]

                    max_batch_size = 0
                    for batch_size in self.allowed_batch_sizes:
                        if (isi_name, model_variant, batch_size) not in acc_latencies:
                            acc_latencies[(isi_name, model_variant, batch_size)] = np.inf
                        latency = acc_latencies[(isi_name, model_variant, batch_size)]

                        if batch_size > max_batch_size and latency < self.slo_dict[isi_name] / 2:
                            max_batch_size = batch_size

                    max_batch_size_dict[(acc_type, model_variant)] = max_batch_size
                    self.log.debug(f'({acc_type}, {model_variant}): {max_batch_size}')

        self.log.debug(f'len(largest_batch_sizes): {len(max_batch_size_dict)}')
        self.log.info(f'largest_batch_sizes:')
        pprint.pprint(max_batch_size_dict)
        self.largest_batch_sizes = max_batch_size_dict
        return
    
    def get_largest_batch_sizes(self):
        return self.largest_batch_sizes
    
    def initialize_dummy_model_variant_runtimes(self, isi_name, random_runtimes=False):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            for batch_size in self.allowed_batch_sizes:
                if random_runtimes:
                    self.cpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.gpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.vpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.fpga_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                else:
                    self.cpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 50
                    self.gpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 25
                    self.vpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 35
                    self.fpga_variant_runtimes[(isi_name, model_variant, batch_size)] = 30
        
        self.model_variant_runtimes = {1: self.cpu_variant_runtimes, 2: self.gpu_variant_runtimes,
                                        3: self.vpu_variant_runtimes, 4: self.fpga_variant_runtimes}
        return

    def replace_profiled_strings(self, element):
        if type(element) is not str:
            return element
        elif element == 'onnxruntime_gpu_ampere':
            return 'GPU_AMPERE'
        elif element == 'onnxruntime_gpu_pascal':
            return 'GPU_PASCAL'
        elif element == 'onnxruntime_cpu':
            return 'CPU'
        elif '.onnx' in element:
            return element.rstrip('.onnx').split('_')[0]
        else:
            return element

    def get_isi_from_variant_name(self, model_variant):
        ''' Gets ISI name from the variant name based on filename
        ''' 
        for element in self.model_variant_accuracies:
            (isi_name, variant) = element
            if model_variant == variant:
                return isi_name

        raise SimulatorException(f'isi_name not found. model variant: {model_variant}')

    def import_profiled_data(self):
        ''' Reads and parses the profiled latency data
        '''
        profiled = pd.read_csv(self.profiled_filename)
        profiled = profiled.rename({'Model': 'model_variant', 'Accel': 'acc_type',
                                    'batchsize': 'batch_size', '50th_pct': 'latency'},
                                    axis='columns')
        profiled = profiled.applymap(self.replace_profiled_strings)
        
        pd.set_option('display.max_columns', None)
        self.log.debug(f'profiled data: {profiled}')
        
        return profiled

    def initialize_loadtimes(self, isi_name):
        self.initialize_model_variant_loadtimes(isi_name)
        for qos_level in range(self.n_qos_levels):
            self.cpu_loadtimes[isi_name, qos_level] = 0
            self.gpu_loadtimes[isi_name, qos_level] = 0
            self.vpu_loadtimes[isi_name, qos_level] = 0
            self.fpga_loadtimes[isi_name, qos_level] = 0
        return

    def trigger_infaas_upscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_upscaling()

    def trigger_infaas_v2_upscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v2_upscaling()

    def trigger_infaas_v3_upscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v3_upscaling()

    def trigger_infaas_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_downscaling()

    def trigger_infaas_v2_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v2_downscaling()

    def trigger_infaas_v3_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v3_downscaling()

    def get_runtimes(self, isi_index):
        runtimes = []
        for qos_level in range(self.n_qos_levels):
            isi_name = self.idx_to_executor[isi_index]
            runtimes.append(self.cpu_runtimes[isi_name, qos_level])
            runtimes.append(self.gpu_runtimes[isi_name, qos_level])
            runtimes.append(self.vpu_runtimes[isi_name, qos_level])
            runtimes.append(self.fpga_runtimes[isi_name, qos_level])
        return runtimes

    def simulate_until(self, until=-1):
        '''
        Run the simulator until the specified clock time.
        If -1 is specified, run till the end.
        '''
        while len(self.event_queue) > 0 and (self.event_queue[0].start_time <= until or until == -1):
            current_event = self.event_queue.pop(0)
            self.clock = current_event.start_time
            self.process(current_event, self.clock)

            for isi in self.last_request_starting:
                last_request_starting = self.last_request_starting[isi]
                if self.clock + QUEUE_REFILL_THRESHOLD >= last_request_starting:
                    # in case we are loading file pointers instead of entire file
                    if self.store_file_pointers:
                        finished = self.refill_event_queue(isi)
        return

    def refill_event_queue(self, isi=None):
        if len(self.trace_files) == 0:
            # we are in a deepcopied simulator instance
            return False
        
        finished = True

        if isi is None:
            trace_files = self.trace_files
        else:
            trace_files = {isi: self.trace_files[isi]}

        for isi_name in trace_files:
            if isi_name in self.trace_file_finished:
                continue
            readfile = self.trace_files[isi_name]
            file_finished = self.add_requests_from_trace_pointer(isi_name, readfile,
                                                                read_until=self.clock+10000)
            if file_finished:
                self.trace_file_finished.add(isi_name)
                self.log.info('Trace file {} finished'.format(isi_name))
                # time.sleep(1)
            finished = finished and file_finished
        return finished

    def is_done(self):
        '''
        Returns true if the request trace has finished, false otherwise.
        '''
        if len(self.event_queue) == 0 or (MAX_CLOCK_VALUE > 0 and self.clock > MAX_CLOCK_VALUE):
            return True
        else:
            return False

    def simulate_requests(self, requests=10):
        '''
        Run the simulator until a given number of requests (instead of a clock time).
        '''
        while len(self.event_queue) > 0 and requests > 0:
            current_event = self.event_queue.pop(0)
            # we only want to process requests and skip scheduling events
            if current_event.type == EventType.SCHEDULING:
                self.reset_request_count()
                continue
            self.clock = current_event.start_time
            self.process(current_event, self.clock)
            requests -= 1

            if self.store_file_pointers:
                finished = self.refill_event_queue()
                if finished:
                    print('oops')
                    break
        return

    def reset_request_count(self):
        for i in range(len(self.total_requests_arr)):
            self.total_requests_arr[i] = 0
            self.failed_requests_arr[i] = 0
            self.failed_requests = 0
            self.successful_requests = 0
            self.successful_requests_arr[i] = 0
        self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels * 2))

    def get_total_requests_arr(self):
        return self.total_requests_arr

    def get_qos_stats(self):
        return self.qos_stats
    
    def get_num_executors(self) -> int:
        return len(self.executors)

    def get_failed_requests_arr(self):
        return self.failed_requests_arr

    def get_failed_requests(self):
        return self.failed_requests

    def get_successful_requests_arr(self):
        return self.successful_requests_arr

    def get_successful_requests(self):
        return self.successful_requests

    def print_assignment(self):
        self.log.debug('Printing assignment in simulator...')
        for isi in self.executors:
            assignment = self.executors[isi].num_predictor_types
            self.log.debug(
                'Executor {} has assignment: {}'.format(isi, assignment))
        return
    
    def apply_pipeline_ilp_homogeneous_solution(self, x: dict, y: dict,
                                                c: dict):
        """ Takes in three dictionaries and applies them to the current
            simulator allocation
            1. x: maps (task, variant) -> no. of replicas
            2. y: maps (task, variant, batchSize) -> boolean indicating
                  batch size to use for each variant
            3. c: maps path -> [0, 1] indicating ratio of traffic to send
                  through path
            Assumes homogeneous cluster, so accelerator type is not specified
            in the ILP solution
        """
        self.apply_homogeneous_predictor_dict(required_predictors=x)
        self.apply_pipeline_batch_size_dict(batch_sizes=y)
        self.update_slo_by_latency_budgets(batch_sizes=y)
        # TODO: Also apply batch size dict (y), and c?
        self.log.warning(f'apply_pipeline_ilp_homogeneous_solution '
                         f'implementation incomplete')
        # time.sleep(1)
        return

    def apply_ilp_solution(self, required_predictors, canary_dict, ilp_x):
        """ Takes in two dictionaries, (i) required_predictors to indicate
        model assignment, and (ii) canary_dict to define the canary routing
        table for the given model assignment
        """
        self.log.debug(f'len: {len(required_predictors)}, required_predictors: {required_predictors}')
        self.log.debug('')
        self.log.debug(f'len: {len(canary_dict)}, canary_dict: {canary_dict}')
        self.log.debug('')
        self.log.debug(f'len: {len(ilp_x)}, ilp_x: {ilp_x}')
        required_predictors, ilp_x = self.postprocess_predictor_dict(required_predictors,
                                                                     canary_dict,
                                                                     ilp_x)
        self.apply_predictor_dict(required_predictors)
        self.apply_canary_dict(canary_dict, ilp_x)
        return
    
    def postprocess_predictor_dict(self, required_predictors, canary_dict, ilp_x):
        """ Upgrade predictors that are not present in canary dict to highest
        accuracy.
        """
        if len(required_predictors) == 0 or len(canary_dict) == 0 or len(ilp_x) == 0:
            return required_predictors
        
        new_ilp_x = {}
        
        accelerators_in_canary = set(map(lambda x: x[0], canary_dict))
        self.log.debug(f'accelerators_in_canary: {accelerators_in_canary}')

        changed = 0
        total = 0
        for tuple in ilp_x:
            total += 1
            (variant, accelerator) = tuple
            # We don't want to replace those that are present in canary_dict
            if accelerator in accelerators_in_canary:
                new_ilp_x[tuple] = ilp_x[tuple]
                continue

            # But if it is not present, replace it with highest accuracy variant
            # This is to ensure the cluster does not go under utilized since ILP
            # will only try to meet throughput, but will not use beyond it
            # This is used in conjunction with proportional routing
            upgraded_variant = self.find_most_accurate_feasible_variant(variant,
                                                                        accelerator.split('-')[0])
            new_tuple = (upgraded_variant, accelerator)
            new_ilp_x[new_tuple] = ilp_x[tuple]
            
            changed += 1

        new_required_predictors = self.get_required_predictors_from_ilpx(new_ilp_x)
        ilp_utilization = 1 - changed / total

        self.use_proportional = True
        self.log.debug(f'Use proportional: {self.use_proportional}, changed '
                      f'proportion: {changed/total}, ilp utilization: {ilp_utilization}')
        self.ilp_stats['ilp_utilization'] = ilp_utilization

        return new_required_predictors, new_ilp_x
    
    def get_required_predictors_from_ilpx(self, ilp_x):
        """ Takes in an ilp_x dictionary and forms a required_predictors dictionary
        """
        required_predictors = {}
        for tuple in ilp_x:
            (variant, accelerator) = tuple
            acc_type = accelerator.split('-')[0]

            if (variant, acc_type) in required_predictors:
                required_predictors[(variant, acc_type)] += 1
            else:
                required_predictors[(variant, acc_type)] = 1

        return required_predictors
    
    def find_most_accurate_feasible_variant(self, variant, accelerator):
        """ Given a model variant, finds the most accurate variant in its
        model family whose maximum batch size is not 0 on given accelerator
        """
        isi = self.get_isi_from_variant_name(variant)
        variants = list(filter(lambda x: x[0] == isi, self.model_variant_accuracies))

        max_accuracy = 0
        max_accuracy_variant = None

        largest_batch_sizes = self.get_largest_batch_sizes()

        for candidate in variants:
            candidate_accuracy = self.model_variant_accuracies[candidate]
            candidate_largest_batch_size = largest_batch_sizes[(accelerator, candidate[1])]

            if candidate_accuracy > max_accuracy and candidate_largest_batch_size > 0:
                max_accuracy = candidate_accuracy
                max_accuracy_variant = candidate

        if max_accuracy_variant is None:
            return variant
        else:
            return max_accuracy_variant[1]
        
    def apply_homogeneous_predictor_dict(self, required_predictors: dict):
        """ Takes in a dictionary 'required_predictors' with
            key: (task/executor, model_variant)
            value: number of predictors
            And applies the assignment to current system
        """
        if len(required_predictors) == 0:
            raise SimulatorException('No required predictors passed')
        
        # The basic idea from apply_predictor_dict() is that there are two
        # steps
        # For each task/executor:
        #   For each model variant of task:
        #       1. If there are any excess predictors (more than required),
        #       remove them
        #       2. If we need more predictors, add them

        # First we need to remove all the predictors that need to be removed
        # to make space
        # self.log.warning(f'apply_homogeneous_predictor_dict, removal phase')
        for executor_key in self.executors:
            executor = self.executors[executor_key]
            for variant in executor.variants_for_this_executor:
                key = (executor.isi, variant)
                required = required_predictors.get(key, 0)

                existingForAllVariants = executor.predictors
                existingForThisVariant = list(filter(lambda x: existingForAllVariants[x].variant_name == variant,
                                                     existingForAllVariants))
                existing = len(existingForThisVariant)

                # self.log.warning(f'\npre-removal, executor: {executor.isi}, variant: '
                #       f'{variant}, required: {required}, existing: {existing}')

                # If there are any excess predictors (more than required),
                # remove them
                while existing > required:
                    # Remove a predictor of any type
                    # Since we have a free hand to choose here, we can remove
                    # the one with the smallest queue
                    to_remove_predictor = None
                    min_queue_length = sys.maxsize
                    # self.log.error(f'existingForThisVariant: {existingForThisVariant}')
                    for predictor_id in existingForThisVariant:
                        predictor = executor.predictors[predictor_id]
                        if predictor.pendingRemoval:
                            continue
                        queue_length = len(predictor.request_queue)
                        if queue_length < min_queue_length:
                            min_queue_length = queue_length
                            to_remove_predictor = predictor
                    acc_type = to_remove_predictor.acc_type
                    print(f'predictor: {predictor.id}, acc_type: {predictor.acc_type}, '
                          f'pendingRemoval: {predictor.pendingRemoval}')
                    removed = executor.remove_predictor_by_id(id=to_remove_predictor.id)

                    if removed:
                        # Update simulator data structures
                        # TODO: We should have a simulator function here so that
                        #       we don't miss anything and don't have to worry
                        #       about the same details everytime
                        # We subtract 1 from acc_type because it is derived from
                        # AccType enum which starts from 1, whereas list indices
                        # start from 0
                        existingForThisVariant.remove(to_remove_predictor.id)
                        self.available_predictors[acc_type-1] += 1

                        max_acc_per_type = self.max_acc_per_type
                        if any(num > max_acc_per_type for num in self.available_predictors):
                            raise Exception(f'Unexpected simulator state. Predictors: '
                                            f'{self.available_predictors}')

                        self.log.info(f'Removed predictor (id: {to_remove_predictor.id}, '
                                      f'variant: {to_remove_predictor.variant_name},'
                                      f'acc_type: {to_remove_predictor.acc_type}), self.'
                                      f'available_predictors: {self.available_predictors}')
                        # time.sleep(1)
                    else:
                        raise Exception(f'No predictor found with id: '
                                        f'{to_remove_predictor.id}')

                    existing -= 1

                # self.log.warning(f'\npost-removal, executor: {executor.isi}, variant: '
                #       f'{variant}, required: {required}, existing: {existing}')

        # self.log.warning(f'apply_homogeneous_predictor_dict, addition phase')
        # Now we add predictors if needed
        for executor_key in self.executors:
            executor = self.executors[executor_key]
            for variant in executor.variants_for_this_executor:
                key = (executor.isi, variant)
                required = required_predictors.get(key, 0)

                existingForAllVariants = executor.predictors
                existingForThisVariant = list(filter(lambda x: existingForAllVariants[x].variant_name == variant,
                                                     existingForAllVariants))
                existing = len(existingForThisVariant)

                # self.log.warning(f'\npre-addition, executor: {executor.isi}, variant: '
                #       f'{variant}, required: {required}, existing: {existing}')
                
                # If we need more predictors, add them
                while required > existing:
                    # Add a predictor for the given model variant of any
                    # accelerator type

                    # We have to figure out which accelerator type has
                    # available hardware
                    found_acc_type = False
                    for acc_type in range(len(self.available_predictors)):
                        if self.available_predictors[acc_type] > 0:
                            found_acc_type = True
                            break
                    
                    if not(found_acc_type):
                        self.log.error(f'task: {executor_key}, variant: {variant}, '
                                       f'required: {required}, existing: {existing}, '
                                       f'available_predictors: {self.available_predictors}, '
                                       f'required_predictors: {required_predictors}')
                        self.log.error(f'Current allocation: '
                                       f'{self.get_variant_allocation_by_acc_type()}')
                        for executor_dummy_isi in self.executors:
                            exec = self.executors[executor_dummy_isi]
                            pred_list = exec.predictors
                            preds = list(map(lambda x: pred_list[x].variant_name, pred_list))
                            self.log.error(f'executor: {executor_dummy_isi}, '
                                            f'predictors: {preds}')
                        raise Exception('No predictors are available')
                    
                    # We add 1 to acc_type since it is an index that starts from
                    # 0, while the AccType enum starts from 1
                    added = executor.add_predictor(acc_type=AccType(acc_type+1),
                                                   variant_name=variant)
                    added_predictor = executor.predictors[added]
                    
                    if added is not None:
                        # Update simulator data structures
                        # TODO: We should have a simulator function here so that
                        #       we don't miss anything and don't have to worry
                        #       about the same details everytime
                        # We don't need to subtract or add anything to acc_type
                        # because both are list indices
                        self.available_predictors[acc_type] -= 1

                        if any(num < 0 for num in self.available_predictors):
                            raise Exception(f'Unexpected simulator state. Predictors: '
                                            f'{self.available_predictors}')

                        self.log.info(f'Added predictor (variant: {added_predictor.variant_name})')
                        # time.sleep(1)
                    else:
                        raise Exception('Predictor could not be added')

                    existing += 1

                # self.log.warning(f'\npost-addition, executor: {executor.isi}, variant: '
                #       f'{variant}, required: {required}, existing: {existing}')

                # Only for debugging purposes
                # existingForAllVariants = executor.predictors
                # existingForThisVariant = list(map(lambda x: existingForAllVariants[x].variant_name == variant,
                #                                   existingForAllVariants))
                # existing = len(existingForThisVariant)
                # print(f'post-modification, executor: {executor.isi}, variant: '
                #       f'{variant}, required: {required}, existing: {existing}\n')

        return
    
    def apply_pipeline_batch_size_dict(self, batch_sizes: dict):
        """ Apply the batch sizes dict to every predictor
        """
        for key in batch_sizes:
            (isi, variant, batch_size) = key

            if batch_sizes[key] < 1:
                continue

            executor = self.executors[isi]
            for predictor_id in executor.predictors:
                predictor = executor.predictors[predictor_id]
                if predictor.variant_name == variant:
                    # predictor.pipeline_batch_size = batch_size
                    predictor.set_pipeline_batch_size(batch_size)
                    print(f'Setting batch size for {predictor.id}, '
                          f'{predictor.variant_name}: {batch_size}')

        for isi in self.executors:
            executor = self.executors[isi]
            for predictor_id in executor.predictors:
                predictor = executor.predictors[predictor_id]
                if predictor.pipeline_batch_size is None:
                    raise Exception(f'Predictor batch size not set (variant: '
                                    f'{predictor.variant_name})')
        # raise Exception('not yet implemented')
        return
    
    def update_slo_by_latency_budgets(self, batch_sizes: dict):
        """ Updates the variant_slo_dict by setting an SLO (or latency budget)
            for each model variant based on the batch size set by the pipeline
            ILP
        """
        variant_slo_dict = {}
        for key in batch_sizes:

            (task, variant, batch_size) = key
            batch_size_set = batch_sizes[key]

            if batch_size_set < 1:
                continue

            if int(batch_size_set) == 1:
                # TODO: Since we are considering homogeneous cluster,
                #       we can use processing latency of any acc type
                self.log.warning(f'If we change ILP to be heterogeneous, '
                                 f'change this')
                processing_latency = self.model_variant_runtimes[1][key]
                variant_slo_dict[variant] = 2 * processing_latency

        variant_slo_dict['source'] = 5.0
        variant_slo_dict['sink'] = 5.0
        # print(f'variant_slo_dict: {variant_slo_dict}')
        self.variant_slo_dict = variant_slo_dict
        # time.sleep(1)

        # raise Exception('not yet implemented')
        # raise Exception(f'Everywhere we have used slo_dict, use variant_slo_dict '
        #                 f'if algorithm is ILP and if key is available in it '
        #                 f'Otherwise, raise an exception. Otherwise, revert back '
        #                 f'to slo_dict')
        # raise Exception('Also potentially update largest batch sizes')
        # raise Exception('Also set gendernet because it is not currently set by ILP')
        return

    def apply_predictor_dict(self, required_predictors={}):
        """ Takes in a dictionary 'required_predictors' with
        key: (model_variant, accelerator_type)
        value: number of predictors
        And applies the assignment to current system
        """
        if len(required_predictors) == 0:
            raise SimulatorException('No required predictors passed')

        self.log.debug(f'apply_predictor_dict: required_predictors: {required_predictors}')

        ilp_to_acc_type = {}
        ilp_to_acc_type['CPU'] = 1
        ilp_to_acc_type['GPU_AMPERE'] = 2
        ilp_to_acc_type['VPU'] = 3
        ilp_to_acc_type['GPU_PASCAL'] = 4

        variant_to_executor = {}

        # Build a dictionary 'existing_predictors'
        existing_predictors = {}
        for ex_key in self.executors:
            executor = self.executors[ex_key]
            for pred_list_key in executor.predictors:
                predictor = executor.predictors[pred_list_key]
                model_variant = predictor.variant_name
                acc_type = predictor.acc_type
                
                tuple_key = (model_variant, acc_type)

                if tuple_key in existing_predictors:
                    existing_predictors[tuple_key].append(predictor)
                else:
                    existing_predictors[tuple_key] = [predictor]

            # Build a reverse dictionary to get executor from model variant name
            model_variants = executor.variants_for_this_executor
            for variant in model_variants:
                variant_to_executor[variant] = executor

        self.log.debug(f'Existing predictors: {existing_predictors}')

        existing_count = {}
        for key in existing_predictors:
            (model_variant, acc_type) = key
            existing_count[(model_variant, AccType(acc_type))] = len(existing_predictors[key])

        self.log.debug(f'Existing count: {existing_count}')

        all_tuple_combinations = self.generate_all_model_acc_keys()

        # Go through each key:
        for tuple_key in all_tuple_combinations:
            # Remove predictors if needed
            (model_variant, accelerator_type) = tuple_key
            acc_type = ilp_to_acc_type[accelerator_type]
            
            existing_key = (model_variant, acc_type)
            existing = len(existing_predictors.get(existing_key, {}))
            required = required_predictors.get(tuple_key, 0)
            self.log.debug(f'tuple_key: {tuple_key}, existing: {existing}, required: {required}')

            while existing > required:
                predictor = existing_predictors[(model_variant, acc_type)].pop()
                id = predictor.id
                executor = predictor.executor
                if (executor.remove_predictor_by_id(id)):
                    self.log.debug(f'removed predictor {id} of type {predictor.acc_type} for '
                          f'model variant {model_variant}')
                    self.available_predictors[acc_type-1] += 1

                    max_acc_per_type = self.max_acc_per_type
                    if any(num > max_acc_per_type for num in self.available_predictors):
                        raise Exception(f'Unexpected simulator state. Predictors: '
                                        f'{self.simulator.available_predictors}')

                else:
                    self.log.debug('no predictor found with id:', id)
                    time.sleep(2)

                existing -= 1

        # Go through each key:
        added = 0
        for tuple_key in required_predictors:
            (model_variant, accelerator_type) = tuple_key
            acc_type = ilp_to_acc_type[accelerator_type]

            required = required_predictors[tuple_key]
            if not((model_variant, acc_type) in existing_predictors):
                existing = 0
            else:
                existing = len(existing_predictors[(model_variant, acc_type)])

            # Add predictors if needed
            while required > existing:
                if (self.available_predictors[acc_type-1]) > 0:
                    executor = variant_to_executor[model_variant]
                    executor.add_predictor(acc_type=AccType(acc_type), variant_name=model_variant)
                    self.available_predictors[acc_type-1] -= 1
                    self.log.debug(f'added predictor of type {acc_type} for model variant '
                                   f'{model_variant} at isi: {executor.isi}, id: {executor.id}')
                    self.log.debug(f'isi {executor.isi} now has following predictors: {executor.predictors}')
                    added += 1
                if (self.available_predictors[acc_type-1]) < 0:
                    raise SimulatorException(f'Available predictors {self.available_predictors[acc_type-1]} for type {acc_type}')
                required -= 1
        self.log.debug(f'added predictors: {added}')

        total_predictors = 0
        total_queued = 0
        total_capacity = 0
        for key in self.executors:
            predictors = self.executors[key].predictors
            total_predictors += len(predictors)
            total_queued += sum(list(map(lambda x: len(x.request_queue), predictors.values())))
            total_capacity += sum(list(map(lambda x: x.peak_throughput if 'prajjwal' not in x.variant_name else 0,
                                           predictors.values())))
            predictors = list(map(lambda x: (x.variant_name, f'acc_type: {x.acc_type}',
                                             f'queue length: {len(x.request_queue)}',
                                             f'peak throughput: {x.peak_throughput}',),
                                  predictors.values()))
        self.log.debug(f'Total queued requests: {total_queued}, system serving capacity: '
                       f'{total_capacity:.2f} reqs/sec')
        self.log.debug(f'Total predictors: {total_predictors}')
        
        return

    def generate_all_model_acc_keys(self):
        ''' Generates a list of tuples of the format (model variant, acc type)
        that enumerates all possible combinations
        '''
        all_model_variants = list(map(lambda x: x[1], self.model_variants.items()))
        # merge list of lists into one list
        all_model_variants = sum(all_model_variants, [])

        all_acc_types = ['CPU', 'GPU_PASCAL', 'VPU', 'GPU_AMPERE']

        all_tuples = []
        for variant in all_model_variants:
            for acc_type in all_acc_types:
                tuple = (variant, acc_type)
                all_tuples.append(tuple)
        return all_tuples

    def apply_canary_dict(self, canary_dict, ilp_x):
        ''' Takes in d dictionary 'canary_dict' with
        key: (model_variant, isi number)
        value: canary routing percentage
        And applies it to the canary routing table for each ISI (executor)
        '''
        self.log.debug(f'len: {len(canary_dict)}, canary_dict: {canary_dict}')
        if len(canary_dict) == 0:
            self.log.error('apply_canary_dict: No canary dictionary passed')
            time.sleep(5)
            return
        
        routing_table_ijk = {}
        for key1 in canary_dict:
            for key2 in ilp_x:
                (accelerator_1, rtype_1) = key1
                (variant, accelerator_2) = key2
                canary_value = canary_dict[key1]
                if accelerator_1 == accelerator_2:
                    routing_table_ijk[(accelerator_1, variant, rtype_1)] = canary_value
        self.previous_ilp_x = ilp_x
        self.previous_canary_dict = canary_dict
        self.log.debug(f'len: {len(canary_dict)}, canary_dict: {canary_dict}')
        self.log.debug('')
        self.log.debug(f'len: {len(routing_table_ijk)}, routing_table_ijk: {routing_table_ijk}')
        self.log.debug('')
        self.log.debug(f'len: {len(ilp_x)}, ilp_x: {ilp_x}')

        for idx in self.idx_to_executor:
            isi = self.idx_to_executor[idx]
            executor = self.executors[isi]

            executor_routing_table = dict(filter(lambda x: x[0][2] == idx, routing_table_ijk.items()))
            # just cleaning up from format {('CPU-0', variant, 0): canary_pct}
            # to {('CPU', variant): canary_pct}
            executor_routing_table = dict(map(lambda x: ((x[0][0].split('-')[0], x[0][1]), x[1]),
                                              executor_routing_table.items()))
            executor.apply_routing_table(executor_routing_table)
        return

    def null_action(self, action, idx):
        action[0] = idx
        action[1:] = np.zeros(len(action)-1)
        executor_idx = self.idx_to_executor[idx]
        executor = self.executors[executor_idx]
        self.log.debug('isi:', executor.isi)
        predictor_list = executor.predictors
        for predictor_key in predictor_list:
            predictor = predictor_list[predictor_key]
            action[predictor.acc_type] += 1
        return action

    def apply_assignment(self, assignment):
        self.log.debug(f'Applying assignment: {assignment}')
        assignment = np.round(np.nan_to_num(
            assignment / np.sum(assignment, axis=0)) * self.predictors_max)
        new_assignment = np.zeros(assignment.shape)

        for idx in range(len(assignment)):
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[idx]
            # look up executor idx and modify it
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]
            while cpu_pred > executor.num_predictor_types[AccType.CPU.value - 1]:
                executor.add_predictor(acc_type=AccType.CPU)
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.CPU)
            new_assignment[idx][0] = executor.num_predictor_types[AccType.CPU.value - 1]

            while gpu_pred > executor.num_predictor_types[AccType.GPU.value - 1]:
                executor.add_predictor(acc_type=AccType.GPU)
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.GPU)
            new_assignment[idx][1] = executor.num_predictor_types[AccType.GPU.value - 1]

            while vpu_pred > executor.num_predictor_types[AccType.VPU.value - 1]:
                executor.add_predictor(acc_type=AccType.VPU)
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.VPU)
            new_assignment[idx][2] = executor.num_predictor_types[AccType.VPU.value - 1]

            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value - 1]:
                executor.add_predictor(acc_type=AccType.FPGA)
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.FPGA)
            new_assignment[idx][3] = executor.num_predictor_types[AccType.FPGA.value - 1]

        self.log.debug(f'New applied assignment: {new_assignment}')

        if np.array_equal(new_assignment, assignment):
            self.log.debug('Was able to match given assignment exactly')
        else:
            self.log.debug(
                'Could not match given assignment exactly, had to scale')
        return

    def apply_assignment_vector(self, assignment):
        ''' Vector version of apply_assignment.
        '''
        self.log.debug(f'Assignment: {assignment}')
        idx = assignment[0]
        new_assignment = np.zeros(assignment.shape[0] - 1)

        for qos_level in range(self.n_qos_levels):
            self.log.debug('Assignment for QoS {}: {}'.format(
                qos_level, assignment[qos_level * 4 + 1:qos_level * 4 + 5]))
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[qos_level *
                                                                 4 + 1:qos_level * 4 + 5]
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]

            while cpu_pred > executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[0] > 0:
                executor.add_predictor(
                    acc_type=AccType.CPU, qos_level=qos_level)
                self.available_predictors[0] -= 1
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4]:
                executor.remove_predictor_by_type(
                    acc_type=AccType.CPU.value, qos_level=qos_level)
                self.available_predictors[0] += 1
            new_assignment[0 + qos_level *
                           4] = executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4]

            while gpu_pred > executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[1] > 0:
                executor.add_predictor(
                    acc_type=AccType.GPU, qos_level=qos_level)
                self.available_predictors[1] -= 1
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4]:
                executor.remove_predictor_by_type(
                    acc_type=AccType.GPU.value, qos_level=qos_level)
                self.available_predictors[1] += 1
            new_assignment[1 + qos_level *
                           4] = executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4]

            while vpu_pred > executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[2] > 0:
                executor.add_predictor(
                    acc_type=AccType.VPU, qos_level=qos_level)
                self.available_predictors[2] -= 1
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4]:
                executor.remove_predictor_by_type(
                    acc_type=AccType.VPU.value, qos_level=qos_level)
                self.available_predictors[2] += 1
            new_assignment[2 + qos_level *
                           4] = executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4]

            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4] \
                    and self.available_predictors[3] > 0:
                executor.add_predictor(
                    acc_type=AccType.FPGA, qos_level=qos_level)
                self.available_predictors[3] -= 1
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4]:
                executor.remove_predictor_by_type(
                    acc_type=AccType.FPGA.value, qos_level=qos_level)
                self.available_predictors[3] += 1
            new_assignment[3 + qos_level *
                           4] = executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4]

        self.log.debug('Old assignment for executor {}: {}'.format(
            assignment[0], assignment[1:]))
        self.log.debug('New applied assignment: {}'.format(new_assignment))
        self.log.debug('Remaining available predictors: {}'.format(
            self.available_predictors))

        return new_assignment

    def get_available_predictors(self):
        return self.available_predictors

    def get_variant_allocation_by_acc_type(self):
        """ Returns the current model variants allocated for each accelerator
            type
        """
        allocatedPerAccType = {}

        predictors = self.get_all_predictors()
        for predictor in predictors:
            acc_type = AccType(predictor.acc_type)
            variant = predictor.variant_name
            pending_removal = predictor.pendingRemoval

            if acc_type not in allocatedPerAccType:
                allocatedPerAccType[acc_type] = [(variant, pending_removal)]
            else:
                allocatedPerAccType[acc_type].append((variant, pending_removal))

        return allocatedPerAccType

    def evaluate_reward(self, K):
        '''
        RL-Cache version of reward: play the trace out K steps into the future,
        find the # of missed requests, and then roll back
        '''
        self.log.debug(
            '--- temp_simulator: Playing trace into the future to evaluate reward ---')

        if len(self.event_queue) == 0:
            self.log.warn('WARN: Ran out of requests while evaluating reward.')

        # We are not using the RL agent, so return a reward of 0
        return 0

    def process(self, event, clock):
        '''
        Process the given event according to its EventType.
        '''
        if event.type == EventType.START_REQUEST:
            self.process_start_event(event, clock)
        elif event.type == EventType.END_REQUEST:
            self.process_end_event(event, clock)
        elif event.type == EventType.SCHEDULING:
            self.process_scheduling_event(event, clock)
        elif event.type == EventType.FINISH_BATCH:
            self.process_finish_batch_event(event, clock)
        elif event.type == EventType.SLO_EXPIRING:
            self.process_slo_expiring_event(event, clock)
        elif event.type == EventType.BATCH_EXPIRING:
            self.process_batch_expiring_event(event, clock)
        elif event.type == EventType.PREDICTOR_ENQUEUED_QUERY:
            self.process_predictor_enqueued_query_event(event, clock)
        elif event.type == EventType.PREDICTOR_DEQUEUED_QUERY:
            self.process_predictor_dequeued_query_event(event, clock)
        else:
            raise Exception(f'Unknown simulator event type: {event.type}')

        return None

    def process_start_event(self, event, clock):
        ''' Process EventType.START_REQUEST
        '''
        self.log.debug('Starting event {}. (Time: {})'.format(event.desc, clock))
        isi = event.desc
        if isi not in self.executors:
            self.add_executor(isi, self.job_sched_algo, self.runtimes, self.model_variant_runtimes,
                                self.model_variant_loadtimes, max_acc_per_type=self.max_acc_per_type,
                                simulator=self)

        # call executor.process_request() on relevant executor
        executor = self.executors[isi]

        if not self.batching_enabled:
            self.process_start_event_without_batch(event, clock, executor)
            raise SimulatorException('This code flow should be deprecated')
        else:
            self.process_start_event_with_batch(event, clock, executor,
                                                event.target_predictor_id)
        
        # Bump overall request statistics
        self.bump_overall_request_stats(event)
        return

    def process_start_event_without_batch(self, event, clock, executor):
        ''' If we do not have batching enabled, we will immediately know
        whether an incoming request can be processed or not. This function
        handles the simulator's behavior for the no-batching scenario.
        '''
        end_time, qos_met, accuracy_seen = executor.process_request(
            event, clock, self.runtimes)

        # We report QoS failure whenever:
        # (i) request fails, (ii) request succeeds but QoS is not met

        if end_time is None:
            # Cannot process request, no end event generated, bump failure stats
            self.bump_failed_request_stats(event, clock)
            self.log.debug('WARN: Request id {} for {} failed. (Time: {})'.format(
                event.id, event.desc, clock))
        else:
            # Request can be processed, generate an end event for it
            self.generate_end_request_event(event, end_time, accuracy_seen, qos_met)
            
        return

    def process_start_event_with_batch(self, event, clock, executor,
                                       target_predictor_id):
        ''' If batching is enabled, the request goes through a different pipeline
        and we will not immediately know if request gets processed. So we simply
        add the request to an appropriate predictor.
        '''
        executor.enqueue_request(event, clock, target_predictor_id)
        return
    
    def generate_predictor_enqueued_event(self, event, clock):
        print(f'PREDICTOR_ENQUEUED_QUERY, clock: {clock}, event: {event.getStr()}')
        self.waiting_queries[event.id] = (clock, event)

        self.insert_event(clock, EventType.PREDICTOR_ENQUEUED_QUERY,
                          event.desc, id=event.id, qos_level=event.qos_level,
                          event_counter=event.start_time,
                          parent_request_id=event.parent_request_id)
        
    def generate_predictor_dequeued_event(self, event, clock):
        self.executing_queries[event.id] = (clock, event)

        self.insert_event(clock, EventType.PREDICTOR_DEQUEUED_QUERY,
                          event.desc, id=event.id, qos_level=event.qos_level,
                          event_counter=event.start_time,
                          parent_request_id=event.parent_request_id)

    def generate_end_request_event(self, event, end_time, accuracy_seen, qos_met,
                                   sequence_num=None):
        ''' Generate an END_REQUEST event and insert it into the simulator's
        event queue.
        '''
        # print(f'generate_end_request_event from: event: {event.getStr()}')
        # time.sleep(1)
        isi = event.desc
        self.insert_event(end_time, EventType.END_REQUEST,
                          event.desc, id=event.id, qos_level=event.qos_level,
                          event_counter=event.start_time, accuracy_seen=accuracy_seen,
                          sequence_num=sequence_num, path=event.path,
                          parent_request_id=event.parent_request_id,
                          dynamic_rerouted=event.dynamic_rerouted,
                          rerouted_target_predictors=event.rerouted_target_predictors)
        self.accuracy_logfile.write(str(accuracy_seen) + '\n')
        self.accuracy_per_model[isi].append(accuracy_seen)
        
        if not qos_met:
            self.bump_qos_unmet_stats(event)

        return

    def generate_finish_batch_event(self, finish_time, predictor, executor):
        ''' Generate a FINISH_BATCH event and insert it into the simulator's
        event queue
        '''
        self.insert_event(start_time=finish_time, type=EventType.FINISH_BATCH,
                        desc=executor.isi, predictor=predictor,
                        executor=executor)
        return
    
    def process_finish_batch_event(self, event, clock):
        ''' When a FINISH_BATCH event is encountered, trigger the appropriate
        callbacks
        '''
        executor = event.executor
        predictor = event.predictor

        predictor.finish_batch_callback(clock)
        return

    def generate_slo_expiring_event(self, time, request, predictor, executor, event_counter):
        ''' Generate an SLO_EXPIRING event and insert it into the simulator's
        event queue
        '''
        self.insert_event(time, EventType.SLO_EXPIRING, desc=request.desc,
                          id=request.id, deadline=request.deadline,
                          predictor=predictor, executor=executor,
                          event_counter=event_counter)
        return
    
    def generate_batch_expiring_event(self, time, request, predictor, executor, event_counter):
        ''' Generate a BATCH_EXPIRING event and insert it into the simulator's
        event queue
        '''
        self.insert_event(time, EventType.BATCH_EXPIRING, desc=request.desc,
                          id=request.id, deadline=request.deadline,
                          predictor=predictor, executor=executor,
                          event_counter=event_counter)
        return

    def process_slo_expiring_event(self, event, clock):
        ''' When an SLO_EXPIRING event is encountered, trigger the appropriate
        callbacks
        '''
        executor = event.executor
        predictor = event.predictor

        predictor.slo_expiring_callback(event, clock)
        return
    
    def process_batch_expiring_event(self, event, clock):
        ''' When a BATCH_EXPIRING event is encountered, trigger the appropriate
        callbacks
        '''
        predictor = event.predictor
        predictor.batch_expiring_callback(event, clock)
        return

    def process_end_event(self, event, clock):
        ''' Process EventType.END_REQUEST
        '''
        isi = event.desc
        executor = self.executors[isi]
        request_finished = executor.finish_request(event, clock)
        if request_finished:
            processing_time = clock - event.event_counter
            self.bump_successful_request_stats(event, processing_time=processing_time)
            self.latency_logfile.write(f'{isi},{event.event_counter},{clock},{processing_time}\n')

            (executingSince, executingQuery) = self.executing_queries[event.id]
            executionTime = clock - executingSince
            print(f'QUERY_FINISHED, clock: {clock}, execution time: {executionTime}, '
                  f'event: {event.getStr()}')
            del self.executing_queries[event.id]

            if event.id not in self.request_times:
                self.request_times[event.id] = {}
            self.request_times[event.id]['execution_time'] = executionTime

            print(f'\n\nwaiting queries: {len(self.waiting_queries)}\n\n')
        else:
            self.bump_failed_request_stats(event, clock)

        return
    
    def process_predictor_enqueued_query_event(self, event, clock):
        ''' Process EventType.PREDICTOR_ENQUEUED_QUERY
        '''
        return
    
    def process_predictor_dequeued_query_event(self, event, clock):
        ''' Process EventType.PREDICTOR_DEQUEUED_QUERY
        '''
        (waitingSince, waitingQuery) = self.waiting_queries[event.id]
        waitingTime = clock - waitingSince
        print(f'PREDICTOR_DEQUEUED_QUERY, clock: {clock}, waiting time: {waitingTime}, '
              f'event: {event.getStr()}')
        
        del self.waiting_queries[event.id]
        if event.id not in self.request_times:
            self.request_times[event.id] = {}
        self.request_times[event.id]['waiting_time'] = waitingTime
        self.request_times[event.id]['model_family'] = event.desc
        return
    
    def pop_expired_request_in_queue(self, event, clock):
        print(f'POPPED_EXPIRED_REQUEST, clock: {clock}, event: {event.getStr()}')
        del self.waiting_queries[event.id]

    def process_scheduling_event(self, event, clock):
        ''' Process EventType.SCHEDULING
        '''
        sched_decision = self.invoke_scheduling_agent()
        if sched_decision is not None:
            self.log.debug(sched_decision)
        else:
            self.log.error('ERROR: The scheduling agent returned an exception. (Time: {})'.format(
                event.start_time))
        self.insert_event(clock + self.sched_interval,
                            EventType.SCHEDULING, event.desc)
        return

    def bump_failed_request_stats(self, event, clock):
        ''' Bump stats for a failed request
        '''
        self.log.debug(f'Failed request: {event.desc}')
        isi = event.desc
        self.failed_requests += 1
        self.total_failed += 1
        
        self.failed_requests_arr[self.isi_to_idx[isi]] += 1
        self.failed_per_model[isi] += 1
        self.slo_timeouts['timeouts'] += 1
        self.slo_timeouts['total'] += 1
        if isi in self.slo_timeouts_per_executor['timeouts']:
            self.slo_timeouts_per_executor['timeouts'][isi] += 1
            self.slo_timeouts_per_executor['total'][isi] += 1
            self.requests_per_executor[isi] += 1
        else:
            self.slo_timeouts_per_executor['timeouts'][isi] = 1
            self.slo_timeouts_per_executor['total'][isi] = 1
            self.requests_per_executor[isi] = 1
        self.bump_qos_unmet_stats(event)

        print(f'QUERY_DROPPED, clock: {clock}, event: {event.getStr()}')
        return

    def bump_qos_unmet_stats(self, event):
        ''' Bump stats for unmet QoS on request
        '''
        isi = event.desc
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2 + 1] += 1
        return
    
    def bump_successful_request_stats(self, event, processing_time):
        ''' Bump stats for a successful request
        '''
        isi = event.desc
        self.completed_requests += 1
        self.successful_requests += 1
        self.successful_requests_arr[self.isi_to_idx[isi]] += 1
        self.successful_per_model[isi] += 1
        self.slo_timeouts['succeeded'] += 1
        self.total_successful += 1
        if isi in self.total_accuracy_per_model:
            self.total_successful_per_model[isi] += 1
        else:
            self.total_successful_per_model[isi] = 1
        self.total_accuracy += event.accuracy_seen
        if isi in self.slo_timeouts_per_executor['succeeded']:
            self.slo_timeouts_per_executor['succeeded'][isi] += 1
        else:
            self.slo_timeouts_per_executor['succeeded'][isi] = 1
        if isi in self.total_accuracy_per_model:
            self.total_accuracy_per_model[isi] += event.accuracy_seen
        else:
            self.total_accuracy_per_model[isi] = event.accuracy_seen

        if isi in self.requests_per_executor:
            self.requests_per_executor[isi] += 1
        else:
            self.requests_per_executor[isi] = 1

        if processing_time > event.deadline:
            # print(f'late event, event.deadline: {event.deadline}, processing_'
            #       f'time: {processing_time}, event: {event.getStr()}')
            # time.sleep(0.1)
            self.slo_timeouts['late'] += 1

            if isi in self.slo_timeouts_per_executor:
                self.slo_timeouts_per_executor['late'][isi] += 1
            else:
                self.slo_timeouts_per_executor['late'][isi] = 1
        return

    def bump_overall_request_stats(self, event):
        ''' Bump overall request stats, i.e., total requests per ISI, requests
        per model, and QoS stats
        '''
        isi = event.desc
        self.total_requests_arr[self.isi_to_idx[isi]] += 1
        self.requests_per_model[isi] += 1
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2] += 1
        return
    
    def get_all_predictors(self):
        ''' Returns list of all existing predictors in the system
        '''
        predictors = []
        for executorKey in self.executors:
            executor = self.executors[executorKey]
            for predictorKey in executor.predictors:
                predictor = executor.predictors[predictorKey]
                predictors.append(predictor)
        return predictors
    
    def get_pending_removal_predictors(self):
        ''' Returns the list of predictors pending removal in the system
            These predictors can process outstanding requests in their queues
            but may not accept new requests
        '''
        predictors = []
        for executorKey in self.executors:
            executor = self.executors[executorKey]
            for predictorKey in executor.pending_removal_predictors:
                predictor = executor.pending_removal_predictors[predictorKey]
                if len(predictor.request_queue) > 0 or predictor.busy:
                    predictors.append(predictor)
                else:
                    predictor.executor.predictor_ready_to_remove(predictor)
        
        # Remove predictors that are in ready_to_remove state
        for executorKey in self.executors:
            executor = self.executors[executorKey]
            executor.remove_ready_to_remove_predictors()

        return predictors

    def add_executor(self, isi, job_sched_algo, runtimes=None, model_variant_runtimes=None, 
                        model_variant_loadtimes=None, max_acc_per_type=0, simulator=None,
                        application=None):
        executor = Executor(isi, job_sched_algo, self.logging_level, self.n_qos_levels,
                            runtimes, model_variant_runtimes, model_variant_loadtimes,
                            max_acc_per_type=max_acc_per_type, simulator=simulator,
                            configured_max_batch_size=self.max_batch_size,
                            application=application)
        self.executors[executor.isi] = executor
        return executor.id

    def insert_event(self, start_time, type, desc, runtime=None, id='', deadline=1000,
                    qos_level=0, accuracy=None, predictor=None, executor=None,
                    event_counter=0, accuracy_seen=None, parent_request_id='',
                    target_predictor_id=None, sequence_num=None, path=None,
                    dynamic_rerouted=False, rerouted_target_predictors={},
                    app_name=None, app_parent_task=None):
        if type == EventType.START_REQUEST:
            if parent_request_id == '':
                # If parent_request_id is not specified, this is a new request
                # We will not generate any subrequest for it
                event = Event(start_time, type, desc, runtime, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy,
                            sequence_num=sequence_num, path=path,
                            app_name=app_name, app_parent_task=app_parent_task)
            else:
                # If parent_request_id is already present, we are generating a
                # subrequest
                event = Event(start_time=start_time, type=type, desc=desc,
                              runtime=runtime, deadline=deadline,
                              qos_level=qos_level, accuracy=accuracy,
                              parent_request_id=parent_request_id,
                              target_predictor_id=target_predictor_id,
                              sequence_num=sequence_num, path=path,
                              app_name=app_name, app_parent_task=app_parent_task)
        elif type == EventType.END_REQUEST:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, event_counter=event_counter,
                          accuracy_seen=accuracy_seen, sequence_num=sequence_num,
                          path=path, parent_request_id=parent_request_id,
                          dynamic_rerouted=dynamic_rerouted,
                          rerouted_target_predictors=rerouted_target_predictors,
                          app_name=app_name, app_parent_task=app_parent_task)
        elif type == EventType.FINISH_BATCH:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, accuracy=accuracy, predictor=predictor,
                          executor=executor)
        elif type == EventType.SLO_EXPIRING:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, accuracy=accuracy, predictor=predictor,
                          executor=executor, event_counter=event_counter)
        elif type == EventType.BATCH_EXPIRING:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, accuracy=accuracy, predictor=predictor,
                          executor=executor, event_counter=event_counter)
        elif type == EventType.PREDICTOR_ENQUEUED_QUERY:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, event_counter=event_counter,
                          parent_request_id=parent_request_id)
        elif type == EventType.PREDICTOR_DEQUEUED_QUERY:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                          qos_level=qos_level, event_counter=event_counter,
                          parent_request_id=parent_request_id)
        else:
            event = Event(start_time, type, desc, runtime, deadline=deadline,
                          qos_level=qos_level, accuracy=accuracy)

        # using binary search to reduce search time complexity
        idx = self.binary_find_index(self.event_queue, start_time)
        self.event_queue.insert(idx, event)
        inserted = True

        return inserted, event

    def linear_find_index(self, arr, number):
        size = len(arr)
        # Traverse the array
        for i in range(size):
            # If number is found
            if arr[i].start_time == number:
                return i
            # If arr[i] exceeds number
            elif arr[i].start_time > number:
                return i
        # If all array elements are smaller
        return size

    def binary_find_index(self, arr, number):
        # Lower and upper bounds
        start = 0
        end = len(arr) - 1

        # Traverse the search space
        while start <= end:
            mid = (start + end) // 2
            if arr[mid].start_time == number:
                return mid
            elif arr[mid].start_time < number:
                start = mid + 1
            else:
                end = mid - 1
        # Return the insert position
        return end + 1

    def invoke_scheduling_agent(self):
        if self.mode == 'debugging':
            # for readability
            time.sleep(2)

        self.log.info('Invoking scheduling event on scheduling_agent')
        print('Invoking scheduling event on scheduling_agent')

        request_url = self.sched_agent_uri + '/scheduling_event'
        request = None
        try:
            request = requests.get(request_url)
            self.log.debug(request.content)
            return request.content
        except requests.exceptions.ConnectionError as connException:
            self.log.error(connException)
            return None

    def print(self):
        for i in range(len(self.event_queue)):
            event = self.event_queue[i]
            self.log.debug('Time: {}, event {} of type {}'.format(
                event.start_time, event.desc, event.type))
    
    def print_allocation(self):
        total_predictors = 0
        total_queued = 0
        total_capacity = 0
        total_served = 0
        print(f'LOG_NUM_SERVERS, clock: {self.clock}, available_predictors: '
              f'{self.available_predictors}, total_available_predictors: '
              f'{sum(self.available_predictors)}, total_predictors: '
              f'{sum(self.predictors_max)}')
        self.log.info('------')
        self.log.info('Printing simulator\'s current allocation..')
        for key in self.executors:
            predictors = self.executors[key].predictors
            total_predictors += len(predictors)
            total_queued += sum(list(map(lambda x: len(x.request_queue), predictors.values())))
            peak_throughputs = list(map(lambda x: x.peak_throughput, predictors.values()))
            total_capacity += sum(peak_throughputs)

            total_served += sum(list(map(lambda x: x.served_requests_per_step, predictors.values())))

            incoming_requests = list(map(lambda x: x.incoming_requests_per_step, predictors.values()))
            incoming_requests_predictors = list(map(lambda x: (AccType(x.acc_type).name, x.variant_name),
                                                    predictors.values()))
            incoming_requests_ratio = [i / j if j > 0 else 0 for i, j in zip(incoming_requests, peak_throughputs)]

            if any(x > 1.0 for x in incoming_requests_ratio):
                self.log.debug(f'executor: {key}, incoming requests ratio: {incoming_requests_ratio}')
                self.log.debug(f'routing table: {self.executors[key].canary_routing_table}')
                self.log.debug(f'predictors available: {incoming_requests_predictors}')
            
            for key in predictors:
                predictor = predictors[key]
                predictor.served_requests_per_step = 0
            predictors = list(map(lambda x: (x.variant_name, f'acc_type: {x.acc_type}',
                                             f'queue length: {len(x.request_queue)}',
                                             f'peak throughput: {x.peak_throughput}',),
                                  predictors.values()))
        self.log.debug(f'Total queued requests: {total_queued}, system serving capacity: '
                      f'{total_capacity:.2f} reqs/sec, total served by predictors since last '
                      f'checked: {total_served}')
        self.log.debug(f'Total predictors: {total_predictors}')
        self.log.debug('------')
