import copy
import logging
import pprint
import time
from collections import OrderedDict
from functools import cmp_to_key
from typing import List
from common.app import App
from common.host import RoutingPolicy
from core.common import RoutingEntry
from core.predictor import Predictor


UNKNOWN_DEMAND_VALUE = 10


class SimLoadBalancer:
    def __init__(self, simulator=None):
        self.simulator = simulator

        self.log = logging.getLogger(__name__)

        # TODO: If simulator's job_sched_algo is most accurate first, we use
        #       MOST_ACCURATE_FIRST policy here. Otherwise, we use EQUAL
        # TODO: make routing policy changeable
        # self.routingPolicy = RoutingPolicy.EQUAL
        self.routingPolicy = RoutingPolicy.MOST_ACCURATE_FIRST

    
    def balanceLoad(self):
        ''' Call the appropriate load balancing function according to the
            specified policy
        '''
        if self.routingPolicy == RoutingPolicy.EQUAL:
            self.balanceLoadEqual()
        elif self.routingPolicy == RoutingPolicy.MOST_ACCURATE_FIRST:
            self.balanceLoadMAF()
        else:
            raise Exception(f'Routing policy {self.routingPolicy} not implemented')
        return
    

    def balanceLoadEqual(self):
        ''' Performs equal load balancing for all applications
        '''

        allPredictors = self.simulator.get_all_predictors()
        logging.debug(f'\n\nload_balancer.balanceLoadEqual(): allPredictors: '
                     f'{list(map(lambda x: (x.id, x.executor.isi), allPredictors))}')
        
        pendingRemovalPredictors = self.simulator.get_pending_removal_predictors()
        # print(f'\n\nload_balancer.balanceLoadEqual(): allPredictors: '
        #              f'{list(map(lambda x: (x.id, x.executor.isi), allPredictors))}')
        # print(f'\n\nload_balancer.balanceLoadEqual(): pendingRemovalPredictors (length '
        #       f'{len(pendingRemovalPredictors)}): '
        #       f'{list(map(lambda x: (x.id, x.executor.isi, len(x.request_queue)), pendingRemovalPredictors))}')

        for appName in self.simulator.apps:
            app = self.simulator.apps[appName]

            logging.debug(f'Balancing load for app: {app}, appID: {app.appID}')
            # app.print()

            activeAppPredictors = list(filter(lambda x: x.application.getName() == app.getName(),
                                        allPredictors))
            pendingRemovalAppPredictors = list(filter(lambda x: x.application.getName() == app.getName(),
                                                      pendingRemovalPredictors))
            appPredictors = activeAppPredictors + pendingRemovalAppPredictors
            logging.debug(f'application: {appName}, activeAppPredictors: '
                          f'{activeAppPredictors}')

            # We want to set routing table for both active predictors in the
            # system and those pending removal
            for predictor in appPredictors:
                # Clear the routing table for now. If no new table is constructed,
                # re-use the old table
                predictor.set_routing_policy(self.routingPolicy)
                oldTable = predictor.clearRoutingTable()
                newTableConstructed = False

                task = predictor.executor.isi
                node = app.findNodeByTask(task)
                
                if node is None:
                    raise Exception('app.findNodeByTask returned None')
                if len(node.children) == 0:
                    if node.getTaskName() != 'sink':
                        logging.warning(f'Encountered node with no outgoing edges: '
                                        f'{node.task}, do not know how to handle it '
                                        f'yet')
                
                for child in node.children:
                    # We only want to add active predictors in the routing tables
                    filteredPredictors = list(filter(lambda x: x.executor.isi == child.task,
                                                     activeAppPredictors))
                    
                    logging.debug(f'node.task: {node.task}, child.task: {child.task}, '
                                 f'filteredPredictors: {filteredPredictors}')
                    
                    if len(filteredPredictors) == 0:
                        logging.warning(f'No workers found for task: {child.task}')
                        continue

                    for filteredPredictor in filteredPredictors:
                        routingEntry = RoutingEntry(predictor=filteredPredictor,
                                                    task=filteredPredictor.executor.isi,
                                                    percentage=1/len(filteredPredictors))
                        predictor.addRoutingEntry(routingEntry)

                    newTableConstructed = True

                # routingTableStr = [entry.getStr() for entry in predictor.routingTable]
                routingTableStr = [entry.predictor.id for entry in predictor.routingTable]

                # Re-use the old table if no new table is constructed
                if not(newTableConstructed):
                    predictor.setRoutingTable(oldTable)
                    if predictor.executor.isi != 'sink':
                        logging.info(f'Predictor {predictor.id}, executor: '
                                     f'{predictor.executor.isi}, routing table: '
                                     f'{routingTableStr}')
        return
                        
    
    def balanceLoadMAF(self):
        ''' Performs load balancing (request routing) with the Most-Accurate-First
            algorithm and returns the effective accuracy
        '''

        for appName in self.simulator.apps:
            app = self.simulator.apps[appName]

            # TODO: Take a snapshot of current configuration and send it to
            #       mostAccurateFirst
            # TODO: Modify mostAccurateFirst to use snapshot of config instead
            #       of current configuration in the Simulator, we will need
            #       this for the online version of our algorithm

            routingTables, effectiveAccuracy = self.mostAccurateFirst(app=app)

            # TODO: Set routingTables['client'] in Executor of first task
            #       There should only be one childTask in routingTables['client']
            #       So essentially we are using routingTables['client'][firstTask]
            #       to setup the Executor

            if routingTables is None:
                # No tables have been set because there
                logging.warning(f'Routing tables have not been set because no '
                                f'workers could be found')
                return
            
            if 'client' in routingTables:
                # logging.warning(f'Implement MAF in executor as well, i.e., setup '
                #                 f'client-to-first task routes in executor')
                logging.warning('Resolve all TODOs in load_balancer.py')
                # time.sleep(0.1)
            else:
                raise Exception('No client-to-first-task routes found')

            logging.warning(f'getBranchingFactor() is using a simple average. '
                            f'We need a better way to get the moving average')

            # demand_since_last = self.simulator.ewma_demand.ravel()
            # demand = demand_since_last / (self.simulator.allocation_window / 1000)
            # firstTaskIdx = self.simulator.isi_to_idx['object_detection']
            # firstTaskEWMADemand = demand[firstTaskIdx]
            # if 'client' in routingTables:
            #     print(f'routingTables[client]: {routingTables["client"]}')
            #     print(f'first task demand: {firstTaskEWMADemand}')
            #     if 'object_detection' in routingTables['client'] and len(routingTables['client']['object_detection']) > 0:
            #         print(f'first worker capacity: {list(routingTables["client"]["object_detection"].keys())[0].peak_throughput}')
            #     # print(f'routingTables: {routingTables}')
            #     time.sleep(0.1)
            # else:
            #     raise Exception('No client-to-first-task routes found')

            # Construct List[RoutingEntries] from routingTables returned by
            # mostAccurateFirst for all workers (not client)
            for worker in routingTables:
                if worker == 'client':
                    continue
                worker.set_routing_policy(self.routingPolicy)
                worker.clearRoutingTable()
                for path in routingTables[worker]:
                    for childTask in routingTables[worker][path]:
                        for childWorker in routingTables[worker][path][childTask]:
                            routingPct = routingTables[worker][path][childTask][childWorker]
                            routingEntry = RoutingEntry(predictor=childWorker,
                                                        task=childTask,
                                                        percentage=routingPct,
                                                        path=path)
                            worker.addRoutingEntry(routingEntry)

        # Probability-based routing should be fine on average

        # What information is needed for this algorithm to run?
        # We need:
        # 1. Capacity of each loaded instance
        # Flow information for each request (we are only modeling here so
        # we know it, but when the actual request routing is done, the
        # predictor will need to be able to track the incoming path. It is
        # simple to do, just keep a path string with each request and stamp
        # the string at the end with the latest model variant that served
        # it)

        # 2. Single-model accuracy of all model variants

        # 3. Branching and multiplicative factor of all hosted instances
        #     -> Per model variant
        return

    
    def mostAccurateFirst(self, app: App) -> None:
        ''' Runs Most-Accurate-First request routing and returns:
            1. Routing tables
            2. Estimated effective accuracy from the routing
        '''
        routingTables = {}
        intermediateRoutingTables = {}
        incomingPathDict = {}

        topSortedTasks = app.getAllTasksTopSorted()

        # -----------------------------------------
        # 1. Routing from client to first task
        # -----------------------------------------
        # TODO: Do client-to-first-task routing and set incoming paths based
        #       on it
        firstTask = topSortedTasks[0]

        # Divide demand by time elapsed since last measurement to get demand in
        # units of requests per second
        demand_since_last = self.simulator.ewma_demand.ravel()
        demand = demand_since_last / (self.simulator.allocation_window / 1000)
        firstTaskIdx = self.simulator.isi_to_idx[firstTask]
        firstTaskEWMADemand = demand[firstTaskIdx]

        # We need to send firstTaskEWMADemand to workers of the first task
        # in order of MostAccurateFirst
        childTask = firstTask

        # -----------------------------------------
        # Repeated code starting
        # -----------------------------------------
        childWorkers = self.getActivePredictorsForTask(task=childTask, app=app)
        childWorkers = self.sortWorkersBySingleModelAccuracy(workers=childWorkers)

        # There are no worker for the first task, so no further request routing
        # can be done
        if len(childWorkers) == 0:
            return None, None
        
        # Every childWorker should start with maximum capacity
        for childWorker in childWorkers:
            childWorker.resetRemainingCapacity()

        # print(f'task: client, childTask: {childTask}, childWorkers: {childWorkers}')

        idx = 0
        childWorkersOverloaded = False

        # -----------------------------------------
        # Repeated code change starts
        # -----------------------------------------

        # Outgoing path from client is empty
        outgoingPath = ''

        # Initializing routing tables
        intermediateRoutingTables['client'] = {}
        routingTables['client'] = {}
        intermediateRoutingTables['client'][childTask] = {}
        routingTables['client'][childTask] = {}

        outgoing = firstTaskEWMADemand
        totalOutgoing = copy.deepcopy(outgoing)

        # Route requests by Most-Accurate-First as long as there is capacity
        while outgoing > 0 and not(childWorkersOverloaded):
            childWorker = childWorkers[idx]
            routed = min(childWorker.remaining_capacity, outgoing)

            childWorker.remaining_capacity -= routed
            outgoing -= routed

            # Add routing entry to worker's routing table, keyed
            # by incoming path
            intermediateRoutingTables['client'][childTask][childWorker] = \
                intermediateRoutingTables['client'][childTask].get(childWorker, 0) + routed

            # Updating incomingPathDict of childWorker
            if childWorker not in incomingPathDict:
                incomingPathDict[childWorker] = {}
            incomingPathDict[childWorker][outgoingPath] = \
                incomingPathDict[childWorker].get(outgoingPath, 0) + routed

            # The current childWorker is filled to capacity, so move to the next
            # one
            if childWorker.remaining_capacity == 0:
                idx += 1

            # All childWorkers are filled to capacity
            if idx == len(childWorkers):
                childWorkersOverloaded = True
        
        # If there is no more capacity left across all childWorkers but there
        # are still requests to route, we need to start overloading them
        # TODO: Effective accuracy should not consider this over
        #       loaded capacity. Or should it?
        if childWorkersOverloaded and outgoing > 0:
            # We divide remaining outgoing requests equally
            # TODO: We could also potentially overload in  proportion to capacity
            routed = outgoing / len(childWorkers)
            for childWorker in childWorkers:
                intermediateRoutingTables['client'][childTask][childWorker] = \
                    intermediateRoutingTables['client'][childTask].get(childWorker, 0) + routed

                childWorker.remaining_capacity -= routed
                outgoing -= routed
                
                # childWorker.addIncomingPath(outgoingPath, routed)
                if childWorker not in incomingPathDict:
                    incomingPathDict[childWorker] = {}
                incomingPathDict[childWorker][outgoingPath] = \
                    incomingPathDict[childWorker].get(outgoingPath, 0) + routed
        
        # Finally, create routingTables based from intermediateRoutingTables
        # by doing routed / totalOutgoing
        for childWorker in intermediateRoutingTables['client'][childTask]:
            routingTables['client'][childTask][childWorker] = intermediateRoutingTables['client'][childTask][childWorker] / totalOutgoing

            if routingTables['client'][childTask][childWorker] > 1.05:
                self.log.error(f'routingTables[client][{childTask}][{childWorker}]: '
                               f'{routingTables["client"][childTask][childWorker]}')
                raise Exception('Routing table does not add up to 1')
        if sum(routingTables['client'][childTask].values()) > 1.05:
            self.log.error(f'routingTables[client][{childTask}]: '
                           f'{routingTables["client"][childTask]}')
            # pprint.pprint(routingTables['client'][childTask])
            self.log.error(f'sum: {sum(routingTables["client"][childTask].values())}')
            raise Exception('Routing table adds up to > 1.05')
        if sum(routingTables['client'][childTask].values()) < 0.95 and firstTaskEWMADemand > 0:
            self.log.error(f'routingTables[client][{childTask}]: '
                           f'{routingTables["client"][childTask]}')
            raise Exception('Routing table adds up to < 0.95')

        # -----------------------------------------
        # Repeated code change ends
        # -----------------------------------------

        # -----------------------------------------
        # Repeated code ending
        # -----------------------------------------

        # -----------------------------------------
        # 2. Routing from first task till last task
        # -----------------------------------------
        for task in topSortedTasks:
            # TODO: Do we need only active predictors? We might need pending-
            #       removal predictors as well (only for workers, not childWorkers)
            workers = self.getActivePredictorsForTask(task=task, app=app)
            workers = self.sortWorkersBySingleModelAccuracy(workers=workers)

            for worker in workers:
                worker.set_routing_policy(routingPolicy=self.routingPolicy)

            node = app.findNodeByTask(task)
            for childNode in node.children:
                childTask = childNode.task
                childWorkers = self.getActivePredictorsForTask(task=childTask,
                                                               app=app)
                childWorkers = self.sortWorkersBySingleModelAccuracy(workers=childWorkers)

                # print(f'task: {task}, childTask: {childTask}, childWorkers: '
                #       f'{childWorkers}')

                # If there are no childWorkers available, there is nothing we
                # can do here
                if len(childWorkers) == 0:
                    continue
                
                # Every childWorker should start with maximum capacity
                for childWorker in childWorkers:
                    childWorker.resetRemainingCapacity()

                # We start counting from the first childWorker and assume there
                # is capacity to add requests
                idx = 0
                childWorkersOverloaded = False

                for worker in workers:
                    if worker not in intermediateRoutingTables:
                        intermediateRoutingTables[worker] = {}
                        routingTables[worker] = {}

                    incomingPaths = incomingPathDict.get(worker, {})
                    sortedPaths = self.sortPaths(incomingPaths)

                    if len(sortedPaths) == 0:
                        if task == topSortedTasks[0]:
                            sortedPaths[''] = UNKNOWN_DEMAND_VALUE
                            logging.warning(f'No demand found for first task '
                                            f'worker. Assuming some demand '
                                            f'to route requests')
                        else:
                            # TODO: is this correct?
                            continue

                    for path in sortedPaths:
                        if path not in intermediateRoutingTables[worker]:
                            intermediateRoutingTables[worker][path] = {}
                            routingTables[worker][path] = {}
                        
                        if childTask not in intermediateRoutingTables[worker][path]:
                            intermediateRoutingTables[worker][path][childTask] = {}
                            routingTables[worker][path][childTask] = {}

                        incomingRequestsForPath = sortedPaths[path]
                        branchingFactor = self.getBranchingFactor(worker, childTask)
                        # If there is no profiled branching information present,
                        # we will assume the branching factor is 1. As time goes
                        # on, this will get updated
                        if branchingFactor == 0:
                            branchingFactor = 1

                        outgoing = incomingRequestsForPath * branchingFactor
                        totalOutgoing = copy.deepcopy(outgoing)

                        if outgoing == 0:
                            continue

                        # print(f'Pre-first phase, intermediateRoutingTables[{worker.id}][{path}][{childTask}]:')
                        # pprint.pprint(intermediateRoutingTables[worker][path][childTask])
                        # print(f'outgoing remaining: {outgoing}, total outgoing: {totalOutgoing}')

                        # Route requests by Most-Accurate-First as long as there
                        # is capacity
                        while outgoing > 0 and not(childWorkersOverloaded):
                            # print(f'worker: {worker.id}, isi: {worker.executor.isi}, '
                            #       f'outgoing: {outgoing}')
                            childWorker = childWorkers[idx]
                            routed = min(childWorker.remaining_capacity, outgoing)

                            childWorker.remaining_capacity -= routed
                            outgoing -= routed

                            # Add routing entry to worker's routing table, keyed
                            # by incoming path
                            intermediateRoutingTables[worker][path][childTask][childWorker] = \
                                intermediateRoutingTables[worker][path][childTask].get(childWorker, 0) + routed

                            # Add outgoingPath and requests from incoming path to
                            # childWorker
                            if path == '':
                                outgoingPath = worker.variant_name
                            else:
                                outgoingPath = path + ',' + worker.variant_name
                            
                            # Updating incomingPathDict for childWorker
                            if childWorker not in incomingPathDict:
                                incomingPathDict[childWorker] = {}
                            incomingPathDict[childWorker][outgoingPath] = \
                                incomingPathDict[childWorker].get(outgoingPath, 0) + routed

                            # The current childWorker is filled to capacity, so
                            # move to the next one
                            if childWorker.remaining_capacity == 0:
                                idx += 1

                            # All childWorkers are filled to capacity
                            if idx == len(childWorkers):
                                childWorkersOverloaded = True

                        #     print(f'outgoing left: {outgoing}, totalOutgoing: {totalOutgoing}')
                        
                        # print(f'First phase, intermediateRoutingTables[{worker.id}][{path}][{childTask}]:')
                        # pprint.pprint(intermediateRoutingTables[worker][path][childTask])
                        # print(f'outgoing remaining: {outgoing}, total outgoing: {totalOutgoing}')
                        
                        # If there is no more capacity left across all childWorkers
                        # but there are still requests to route, we need to start
                        # overloading them
                        # TODO: Effective accuracy should not consider this over
                        #       loaded capacity. Or should it?
                        if childWorkersOverloaded and outgoing > 0:
                            # We divide remaining outgoing requests equally
                            # TODO: We could also potentially overload in
                            #       proportion to capacity
                            routed = outgoing / len(childWorkers)
                            for childWorker in childWorkers:
                                intermediateRoutingTables[worker][path][childTask][childWorker] = \
                                    intermediateRoutingTables[worker][path][childTask].get(childWorker, 0) + routed

                                childWorker.remaining_capacity -= routed
                                outgoing -= routed
                                
                                # TODO: We should update the incoming path dict of
                                # childWorker here as well
                                # TODO: This is a lot of repeated code, perhaps create
                                #       a function for it
                                if path == '':
                                    outgoingPath = worker.variant_name
                                else:
                                    outgoingPath = path + ',' + worker.variant_name

                                # Add incoming path for childWorker
                                if childWorker not in incomingPathDict:
                                    incomingPathDict[childWorker] = {}
                                incomingPathDict[childWorker][outgoingPath] = \
                                    incomingPathDict[childWorker].get(outgoingPath, 0) + routed

                        # print(f'Second phase, intermediateRoutingTables[{worker.id}][{path}][{childTask}]:')
                        # pprint.pprint(intermediateRoutingTables[worker][path][childTask])
                        # print(f'outgoing remaining: {outgoing}, total outgoing: {totalOutgoing}')
                        
                        # Finally, create routingTables based from intermediateRoutingTables
                        # by doing routed / totalOutgoing
                        for childWorker in intermediateRoutingTables[worker][path][childTask]:
                            routingTables[worker][path][childTask][childWorker] = intermediateRoutingTables[worker][path][childTask][childWorker] / totalOutgoing

                            if routingTables[worker][path][childTask][childWorker] > 1.05:
                                raise Exception('Routing table does not add up to 1')
                        if sum(routingTables[worker][path][childTask].values()) > 1.05:
                            print(f'routingTables[{worker.id}][{path}][{childTask}]:')
                            pprint.pprint(routingTables[worker][path][childTask])
                            print(f'sum: {sum(routingTables[worker][path][childTask].values())}')
                            raise Exception('Routing table adds up to > 1.05')
                        if sum(routingTables[worker][path][childTask].values()) < 0.95:
                            print(f'routingTables[{worker.id}][{path}][{childTask}]: '
                                  f'{routingTables[worker][path][childTask]}')
                            raise Exception('Routing table adds up to < 0.95')
                        
        #                 print()
        #                 # print('intermediateRoutingTables:')
        #                 print(f'outgoing for worker: {totalOutgoing}')
        #                 print(f'sum(intermediateRoutingTables[worker][path][childTask].values()): {sum(intermediateRoutingTables[worker][path][childTask].values())}')
        #                 for childWorker in intermediateRoutingTables[worker][path][childTask]:
        #                     print(f'intermediateRoutingTables[{worker.id}][{path}][{childTask}][{childWorker.id}]: {intermediateRoutingTables[worker][path][childTask][childWorker]}')
        #                     print(f'routingTables[{worker.id}][{path}][{childTask}][{childWorker.id}]: {routingTables[worker][path][childTask][childWorker]}')
        #                 # pprint.pprint(intermediateRoutingTables)
        #                 # print('routingTables:')
        #                 # pprint.pprint(routingTables)
        #                 print()
        # time.sleep(0.1)

        # print('incomingPathDict:')
        # # pprint.pprint(incomingPathDict)
        # for worker in incomingPathDict:
        #     for incomingPath in incomingPathDict[worker]:
        #         print(f'incomingPathDict[{worker.id},{worker.executor.isi}][{incomingPath}]: {incomingPathDict[worker][incomingPath]}')
        # print(f'all active predictors: {list(map(lambda x: (x.id, x.executor.isi), self.simulator.get_all_predictors()))}')
        # time.sleep(0.5)

        # TODO: set effective accuracy
        effectiveAccuracy = 0
        return routingTables, effectiveAccuracy
    

    def sortWorkersBySingleModelAccuracy(self, workers: List[Predictor]) -> List[Predictor]:
        ''' Sorts workers by their single-model accuracies and returns
        '''
        return sorted(workers, key=lambda x: x.profiled_accuracy)
    

    def sortPaths(self, incomingPaths: dict) -> OrderedDict:
        ''' Sorts a dictionary of incoming paths (mapping path -> requests)
            in non-increasing order of accuracy. If any pairs are found that
            do not obey ordering property, raises Exception
        '''
        # TODO: Raise Exception if strict ordering cannot be established, e.g.,
        # a1b2 and a2b1
        return OrderedDict(sorted(incomingPaths.items(),
                                  key=cmp_to_key(self.comparePaths),
                                  reverse=True))
    

    def getActivePredictorsForTask(self, task: str, app: App) -> List[Predictor]:
        ''' Get the list of active predictors in the system for a given task
            of an App
        '''
        allPredictors = self.simulator.get_all_predictors()

        activeAppPredictors = list(filter(lambda x: x.application.getName() == app.getName(),
                                        allPredictors))
        filteredPredictors = list(filter(lambda x: x.executor.isi == task,
                                         activeAppPredictors))
        return filteredPredictors
    

    def comparePaths(self, p1: str, p2: str):
        ''' Compare two paths and return the one with higher accuracy according
            to the strict ordering property. If no order can be established,
            raise an Exception
        '''
        modelsInP1 = p1[0].split(',')
        modelsInP2 = p2[0].split(',')
        if len(modelsInP1) != len(modelsInP2):
            return None

        p1Greater = False
        p2Greater = False
        for i in range(len(modelsInP1)):
            p1Model = modelsInP1[i]
            p2Model = modelsInP2[i]

            if self.getVariantAccuracy(p1Model) > self.getVariantAccuracy(p2Model):
                p1Greater = True
            if self.getVariantAccuracy(p2Model) > self.getVariantAccuracy(p1Model):
                p2Greater = True
        
        if p1Greater and p2Greater:
            raise Exception(f'Paths do not obey ordering policy for Most-Accurate'
                            f'-First. Path 1: {p1[0]}. Path 2: {p2[0]}.')
        elif p1Greater:
            return 1
        elif p2Greater:
            return -1
        else:
            return 0
        
    
    def getVariantAccuracy(self, modelVariant: str):
        ''' Find the single-model accuracy of a model variant
        '''
        accuracies = self.simulator.model_variant_accuracies
        
        for key in accuracies:
            if modelVariant == key[1]:
                return accuracies[key]
        return None
    
    
    def getBranchingFactor(self, worker: Predictor, childTask: str) -> float:
        ''' Given a worker and a childTask, returns the average requests
            sent to childTask during recent history
        '''
        # If there is no profiled history, return 0
        if worker not in self.simulator.profiled_branching:
            return 0
        if childTask not in self.simulator.profiled_branching[worker]:
            return 0
        
        # Get the recent history of requests sent to this branch
        branchingInfo = self.simulator.profiled_branching[worker][childTask]

        # Calculate a smoothed average of history
        avgFactor = sum(branchingInfo) / len(branchingInfo)

        return avgFactor
    
