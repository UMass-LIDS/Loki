# NOTE: Old script for profiling runtimes. It has been replaced by
#       profile_runtimes.py which is more modular and user-friendly

import os
import time
import tensorrt
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort


model_base_dir = '/work/pi_rsitaram_umass_edu/sohaib/models/'
tensorrt_cache = os.path.join(model_base_dir, 'tensorrt_cache')

models = {
          'yolov5': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
          'efficientnet_car': ['eb0_checkpoint_150epochs', 'eb1_checkpoint_150epochs',
                               'eb2_checkpoint_150epochs', 'eb3_checkpoint_150epochs',
                               'eb4_checkpoint_150epochs', 'eb5_checkpoint_150epochs',
                               'eb6_checkpoint_150epochs']
         }

# models = {
#           'yolov5': ['yolov5n'],
#          }

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
# batch_sizes = [1]

max_iterations = 1000

columns = ['model', 'accelerator', 'batch_size', 'avg_runtime', '50th_pct_runtime',
           '90th_pct_runtime', '95th_pct_runtime']
df = pd.DataFrame(columns=columns)

for model_family in models:
    for model_name in models[model_family]:
        model_path = os.path.join(model_base_dir, model_family, 'onnx', model_name)

        start_time = time.time()
        ort_sess = ort.InferenceSession(f'{model_path}.onnx',
                                        # providers=['CUDAExecutionProvider'])
                                        providers=[('TensorrtExecutionProvider',
                                                   {
                                                    'trt_engine_cache_enable': True,
                                                    'trt_engine_cache_path': tensorrt_cache
                                                    })])
        end_time = time.time()
        print(f'model loading time: {end_time-start_time}')

        model = os.path.basename(model_name)

        # Warming up the model with 10 iterations
        for warmup_iteration in range(10):
            # TODO: repeated code, refactor into a function
            if 'yolov5' in model_name:
                data = torch.rand(1, 3, 640, 640).numpy()
            elif 'eb' in model_name:
                data = torch.rand(1, 3, 224, 224).numpy()
            input_name = ort_sess.get_inputs()[0].name
            ort_inputs = {input_name: data}
            outputs = ort_sess.get_outputs()
            output_names = list(map(lambda output: output.name, outputs))
            ort_outs = ort_sess.run(output_names, ort_inputs)

        # Starting the profiling
        for batch_size in batch_sizes:
            runtimes = []

            for iteration in range(max_iterations):
                if iteration % 100 == 0:
                    print(f'model: {model}, batch size: {batch_size}, '
                          f'iteration: {iteration}/{max_iterations}')
                # input_shape = ort_sess.get_inputs()[0].shape
                # data = torch.rand(input_shape)
                if 'yolov5' in model_name:
                    data = torch.rand(batch_size, 3, 640, 640).numpy()
                elif 'eb' in model_name:
                    data = torch.rand(batch_size, 3, 224, 224).numpy()

                input_name = ort_sess.get_inputs()[0].name
                ort_inputs = {input_name: data}

                outputs = ort_sess.get_outputs()
                output_names = list(map(lambda output: output.name, outputs))

                start_time = time.time()
                ort_outs = ort_sess.run(output_names, ort_inputs)
                end_time = time.time()
                runtime = end_time - start_time
                runtimes.append(runtime)
                # print(f'runtime: {runtime}')

            avg = np.average(runtimes)
            pct_50 = np.percentile(runtimes, 50)
            pct_90 = np.percentile(runtimes, 90)
            pct_95 = np.percentile(runtimes, 95)

            # TODO: do not use hard-coded accelerator value
            new_row = {'model': model, 'accelerator': '1080ti', 'batch_size': batch_size,
                    'avg_runtime': avg, '50th_pct_runtime': pct_50,
                    '90th_pct_runtime': pct_90, '95th_pct_runtime': pct_95}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(df)
    df.to_csv(f'profiling/profiled/throughput_{model_family}.csv')
