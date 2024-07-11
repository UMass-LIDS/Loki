from enum import Enum
import uuid


class EventType(Enum):
    START_REQUEST = 1
    SCHEDULING = 2
    END_REQUEST = 3
    FINISH_BATCH = 4
    SLO_EXPIRING = 5
    BATCH_EXPIRING = 6
    PREDICTOR_ENQUEUED_QUERY = 7
    PREDICTOR_DEQUEUED_QUERY = 8


class Event:
    def __init__(self, start_time, type, desc, runtime=None, deadline=1000, id='',
                qos_level=0, accuracy=100.0, predictor=None, executor=None,
                event_counter=0, accuracy_seen=None, late=None, parent_request_id='',
                target_predictor_id=None, sequence_num=None, path=None,
                dynamic_rerouted=False, rerouted_target_predictors={},
                app_name=None, app_parent_task=None):
        if id == '':
            # If this is a new request (at first task of an app), generate a new
            # request ID 
            id = uuid.uuid4().hex

        if parent_request_id == '':
            parent_request_id = uuid.uuid4().hex

            excluded_events = [EventType.FINISH_BATCH, EventType.BATCH_EXPIRING,
                               EventType.SLO_EXPIRING]

            if desc != app_parent_task and type not in excluded_events:
                raise Exception(f'No parent request ID passed for child task: {desc}')

        self.id = id
        self.parent_request_id = parent_request_id
        self.sequence_num = sequence_num
        self.type = type
        self.start_time = start_time
        self.desc = desc
        self.runtime = runtime
        self.deadline = deadline
        self.qos_level = qos_level
        self.accuracy = accuracy

        # parameters needed for batch processing
        self.predictor = predictor
        self.executor = executor
        # event counter is only set if event is SLO_EXPIRING
        self.event_counter = event_counter
        self.accuracy_seen = accuracy_seen
        self.late = late

        self.target_predictor_id = target_predictor_id

        self.path = path

        self.dynamic_rerouted = dynamic_rerouted
        self.rerouted_target_predictors = rerouted_target_predictors

        self.app_name = app_name
        self.app_parent_task = app_parent_task

        return

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.start_time < other.start_time
        else:
            return NotImplemented
        
    def getStr(self):
        if self.path is None:
            path = ''
        else:
            path = self.path.replace(',', ';')

        str = (f'id: {self.id}, type: {self.type}, start_time: {self.start_time}, ' 
               f'desc: {self.desc}, runtime: {self.runtime}, deadline: {self.deadline}, '
               f'qos_level: {self.qos_level}, accuracy: {self.accuracy}, predictor: '
               f'{self.predictor}, executor: {self.executor}, event_counter: '
               f'{self.event_counter}, accuracy_seen: {self.accuracy_seen}, late: '
               f'{self.late}, target_predictor_id: {self.target_predictor_id}, '
               f'sequence_num: {self.sequence_num}, parent_request_id: '
               f'{self.parent_request_id}, path: {path}, dynamically rerouted: '
               f'{self.dynamic_rerouted}, app_name: {self.app_name}, app_parent_'
               f'task: {self.app_parent_task}')
        return str

        
class Behavior(Enum):
    BESTEFFORT = 1
    STRICT = 2


class TaskAssignment(Enum):
    RANDOM = 1
    ROUND_ROBIN = 2
    EARLIEST_FINISH_TIME = 3
    LATEST_FINISH_TIME = 4
    INFAAS = 5
    CANARY = 6
    MOST_ACCURATE_FIRST = 7


class RoutingEntry:
    def __init__(self, predictor, task: str, percentage: float, path: str = None):
        self.predictor = predictor
        self.task = task
        self.percentage = percentage
        self.path = path

    def getStr(self) -> str:
        entryStr = (f'predictor id: {self.predictor.id}, path: {self.path}, '
                    f'task: {self.task}, percentage: {self.percentage}')
        return entryStr
    