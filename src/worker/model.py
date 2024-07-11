import os
import logging
import pickle
import time
import onnxruntime as ort
from enum import Enum
from multiprocessing import Process
from common.query import Query
from utils.vision import prepare_data_batch
from utils.vision import yolo_preprocess, ort_yolo_predict, yolo_postprocess
from utils.vision import eb_car_preprocess, ort_eb_car_predict, eb_car_postprocess
from utils.vision import sink_preprocess, sink_process, sink_postprocess


USECONDS_IN_SEC = 1000 * 1000

class ModelState(Enum):
    READY = 1
    NO_MODEL_LOADED = 2

# This class is responsible for loading a model variant and running inference
# on it
class LoadedModel:
    def __init__(self, pipe1, pipe2, dataset):
        self.model = None
        self.modelName = None
        self.preprocess = None
        self.ort_predict = None
        self.postprocess = None
        # Make sure that LoadedModel's appID and worker's appID are always in sync
        # i.e., there are no methods that modify LoadedModel's appID but do not
        # change worker's appID, and vice versa
        self.appID = None
        self.task = None
        self.queue = []
        self.modelDir = '../../models'

        self.dataset = dataset

        self.label_to_task = None

        self.state = ModelState.NO_MODEL_LOADED

        # TODO: should these be hard-coded?
        self.modelPaths = {
                           'eb0': 'eb0_checkpoint_150epochs.onnx',
                           'eb1': 'eb1_checkpoint_150epochs.onnx',
                           'eb2': 'eb2_checkpoint_150epochs.onnx',
                           'eb3': 'eb3_checkpoint_150epochs.onnx',
                           'eb4': 'eb4_checkpoint_150epochs.onnx',
                           'eb5': 'eb5_checkpoint_150epochs.onnx',
                           'eb6': 'eb6_checkpoint_150epochs.onnx',
                           'yolov5n': 'yolov5n.onnx',
                           'yolov5s': 'yolov5s.onnx',
                           'yolov5m': 'yolov5m.onnx',
                           'yolov5l': 'yolov5l.onnx',
                           'yolov5x': 'yolov5x.onnx',
                           'sink': 'sink'
                           }
        
        self.readPipe, _ = pipe1
        _, self.writePipe = pipe2
        self.pipeProcess = Process(target=self.readIPCMessages, args=((pipe1, pipe2,)))
        self.pipeProcess.start()


    # Load a new model
    def load(self, modelName, appID, task):
        previousModel = self.model
        loadedFrom = None

        modelPath = os.path.join(self.modelDir, self.modelPaths[modelName])

        try:
            if 'sink' in modelPath:
                loadedFrom, loadingTime = 'storage', 0
            elif os.path.exists(modelPath):
                loadedFrom, loadingTime = self.loadFromStorage(modelPath)
            else:
                loadedFrom, loadingTime = self.loadFromNetwork(None)

            self.loadOrtFunctionPtr(modelName)
            
            # TODO: make this asynchronous, we do not want to wait on model unloading
            # TODO: or should it be synchronous and we wait for requests of currently
            #       loaded model to finish before we load new model?
            if previousModel is not None:
                self.unload(previousModel)

            self.appID = appID
            self.task = task
            self.state = ModelState.READY

            logging.info(f'self.state: {self.state}')
            logging.info(f'self.model: {self.model}')
            
            return loadedFrom, loadingTime
        except Exception as e:
            raise e


    def loadFromStorage(self, modelPath):
        # TODO: is there anything else to do?
        loadingTimeStart = time.time()
        self.model = ort.InferenceSession(modelPath, providers=['CUDAExecutionProvider',
                                                                'CPUExecutionProvider'])
        loadingTime = int((time.time() - loadingTimeStart) * USECONDS_IN_SEC)

        return 'storage', loadingTime
    

    def loadFromNetwork(self, parameters):
        # TODO: download model and place in self.modelDir
        # TODO: if not enough disk space to download, evict least recently used model
        loadingTimeStart = time.time()
        loadingTime = int((time.time() - loadingTimeStart) * USECONDS_IN_SEC)

        raise Exception('loadFromNetwork is not yet implemented')
    
        return 'network', loadingTime
    

    def loadOrtFunctionPtr(self, modelName):
        self.modelName = modelName
        if 'yolo' in modelName:
            self.preprocess = yolo_preprocess
            self.ort_predict = ort_yolo_predict
            self.postprocess = yolo_postprocess
        elif 'eb' in modelName:
            self.preprocess = eb_car_preprocess
            self.ort_predict = ort_eb_car_predict
            self.postprocess = eb_car_postprocess
        elif 'sink' in modelName:
            self.preprocess = sink_preprocess
            self.ort_predict = sink_process
            self.postprocess = sink_postprocess
        else:
            raise Exception(f'Model {modelName} not implemented by loadOrtFunctionPtr')


    # Unload the currently loaded model
    def unload(self, model):
        # TODO: Stop its execution thread and remove model from GPU memory
        # TODO: Should we wait for its current requests to complete? (empty queue)
        #       If yes, should this block before the next model is loaded? Otherwise
        #       GPU might automatically unload this model to load new one, load this one
        #       again to execute its requests, resulting in thrashing
        pass
        # Join will not work if the queueProcess does not finish on its own,
        # we may have to interrupt it
        self.queueProcess.join()
        raise Exception('unload is not yet implemented')
    

    def serviceQueue(self):
        # TODO: this should only be called from the readQueue process to avoid
        #       self.queue synchronization issues
        # Should we use semaphore on self.queue anyway? What about callbacks to this
        # function? Which process do they execute in?
        # Possibile options:
        # 1. Not enough requests in queue, return
        # 2. Pop requests from queue and serve (what does the callback do?)
        # 3. For other algorithms, perhaps set an interrupt timer to this function

        if len(self.queue) == 0:
            # Nothing to do
            return
        
        # If there are any requests in queue, serve each of them one-by-one
        # with batch size of 1
        while len(self.queue) > 1:
            popped = self.queue.pop(0)

            event = {'event': 'WORKER_DEQUEUED_QUERY',
                    'requestID': popped.requestID, 'queryID': popped.queryID,
                    'userID': popped.userID, 'appID': popped.applicationID,
                    'task': self.task, 'sequenceNum': popped.sequenceNum,
                    'timestamp': time.time()}
            logging.info(f'EVENT,{str(event)}')

            self.writePipe.send(f'QUEUE_SIZE,{len(self.queue)}')

            # Batch size of 1
            popped = [popped]

            self.executeBatch(popped)
            pass

        return
    

    def executeBatch(self, queries):
        # Check if model is ready to execute
        if self.state == ModelState.NO_MODEL_LOADED:
            logging.error(f'\texecuteBatch: no model is currently loaded, cannot '
                          f'execute request')
            return
        elif self.state == ModelState.READY:
            pass
        else:
            logging.error(f'Model state {self.state} not handled by executeBatch()')
            return

        # Extract data from list of Query objects
        data_array = list(map(lambda x: x.data, queries))

        # Prepare batched tensor
        data_batch = prepare_data_batch(data_array)

        # Run the inference
        start_time = time.time()
        results = self.ort_predict(self.model, data_batch)
        print(f'\tProcess 2, inference time: {(time.time() - start_time):.6f}')
        # print(f'\tProcess 2, len(results): {len(results)} shape of [0]: {results[0].shape}')

        if 'yolo' in self.modelName:
            results['label_to_task'] = self.label_to_task

        # Post-processing
        start_time = time.time()
        results = self.postprocess(results=results)
        print(f'\tProcess 2, post-processing time: {(time.time() - start_time):.6f}')
        print(f'\tPost-processed results: {results}')

        # TODO: this post-processing time is in order of 1-10ms, however could it be
        #       much more when doing a batch of requests? (since it is done sequentially)
        #       if so, we could look into parallelizing batch post-processing
        
        # Return output to worker daemon
        # TODO: use appropriate request ID
        # TODO: use appropriate task ID and separate results (implement in utils.vision
        #       by using some version of extract_object_tensors, but don't extract tensors)
        # TODO: how to get taskID from label? this interface should be generic like
        #       ort_predict, and then every model should implement its own version
        #       This is somewhat like polymorphism
        print(f'\tProcess 2, sending completed inference at {time.time()}')
        self.writePipe.send('COMPLETED_INFERENCE')
        # self.writePipe.send(f'{queries[0].requestID},0,{serialized_results}')
        # self.writePipe.send(f'{queries[0].requestID},1,{results}')
        self.writePipe.send(f'{queries[0].queryID},{results}')
        self.writePipe.send('DONE_SENDING')

        for query in queries:
            event = {'event': 'WORKER_COMPLETED_QUERY',
                    'requestID': query.requestID, 'queryID': query.queryID,
                    'userID': query.userID, 'appID': query.applicationID,
                    'task': self.task, 'sequenceNum': query.sequenceNum,
                    'timestamp': time.time()}
            logging.info(f'EVENT,{str(event)}')
        
        # # Not needed, blows up data size too much
        # extracted_tensors = extract_object_tensors(img_tensor=data_batch[0],
        #                                            obj_det_tensors=results,
        #                                            label=None)
        # serialized_results = pickle.dumps(extracted_tensors)
        # print(f'\tProcess 2, extracted tensors.. time to serialize: '
        #       f'{(time.time() - start_time):.6f},  serialized message length: '
        #       f'{len(serialized_results)}')

        # TODO: report queuing time

        return
      

    def readIPCMessages(self, pipe1, pipe2):
        logfile_name = f'../../logs/model_{time.time()}.log'
        logging.basicConfig(filename=logfile_name, level=logging.INFO, encoding='utf-8',
                            format='%(asctime)s %(levelname)-8s %(message)s')
        
        readPipe, _ = pipe1
        _, writePipe = pipe2
        # TODO: This is busy waiting. Is there a better way to do this?
        while True:
            message = readPipe.recv()

            logging.info(f'\tProcess 2, readQueue: message: {message}')

            if message == 'QUERY':
                query = readPipe.recv()
                self.queue.append(query)

                logging.info(f'\tProcess 2, readQueue: Appended query to queue from '
                             f'readQueue, time: {time.time()}')
                
                self.serviceQueue()
            
            elif message == 'REQUEST':
                request = readPipe.recv()
                # TODO: construct queries from request and put them in queue
                #       it is better to do that here than in the worker daemon process

                # TODO: Initial request has task ID 0
                # TODO: replace this application's defined task
                # TODO: for intermediate task, it should use that task information
                # TODO: Perhaps this task information should be passed as part of
                #       the request


                print(f'before preprocessing,request: {request}')
                logging.info(f'before preprocessing, request: {request}')
                print(f'before preprocessing, request.data: {request.data}')
                logging.info(f'before preprocessing, request.data: {request.data}')
                queries = self.preprocess(request, self.dataset)
                for query in queries:
                    self.queue.append(query)

                    event = {'event': 'WORKER_ENQUEUED_QUERY',
                            'requestID': query.requestID, 'queryID': query.queryID,
                            'userID': query.userID, 'appID': query.applicationID,
                            'task': self.task, 'modelVariant': self.modelName,
                            'sequenceNum': query.sequenceNum,
                            'timestamp': time.time()}
                    logging.info(f'EVENT,{str(event)}')

                    writePipe.send(f'QUEUED_QUERY,{len(self.queue)}')
                    writePipe.send(query)

                logging.info(f'\tProcess 2, readQueue: Appended query to queue from '
                             f'readQueue, time: {time.time()}')

                self.serviceQueue()

            elif message == 'LOAD_MODEL':
                load_model_message = readPipe.recv()
                modelName, appID, task= load_model_message.split(',')
                self.childrenTasks = readPipe.recv()
                self.label_to_task = readPipe.recv()
                print(f'\tchildrenTasks: {self.childrenTasks}')
                print(f'\tlabel_to_task: {self.label_to_task}')
                logging.info(f'\tchildrenTasks: {self.childrenTasks}')
                logging.info(f'\tlabel_to_task: {self.label_to_task}')

                loadedFrom, loadingTime = self.load(modelName, appID, task)

                logging.info(f'\tProcess 2, readQueue: loaded model {modelName} from '
                             f'{loadedFrom} in time {loadingTime} micro-seconds')
                
                writePipe.send('LOAD_MODEL_RESPONSE')
                writePipe.send(f'{modelName},{loadedFrom},{loadingTime}')
    

    def inference(self):
        # TODO: If a query in the queue does not belong to the appID, remove it
        raise Exception('inference is not yet implemented')

