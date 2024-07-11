import os
import argparse
import math
import time
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
from src.utils.logging import query_yes_no
from src.utils.vision import ort_yolo_predict, ort_eb_car_predict, ort_vgg_gender_predict


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-i', required=False, default='models',
                        dest='model_dir', help='Directory where models are stored')
    parser.add_argument('--model', '-m', required=True, dest='model',
                        help='Model to profile')
    parser.add_argument('--max_batch_size', '-bs', required=False, default='64',
                        dest='max_batch_size', help=f'Maximum batch size to use. '
                        f'Default is 64')
    parser.add_argument('--output_dir', '-o', required=False, default='profiling/runtimes',
                        dest='output_dir', help='Directory to write profile to')
    parser.add_argument('--trials', '-t', required=False, default='1000',
                        dest='trials', help='Number of trials to conduct')
    return parser.parse_args()


def log2(x):
    return math.log10(x) / math.log10(2)


def isPowerOfTwo(x):
    return math.ceil(log2(x)) == math.floor(log2(x))


def main(args):
    model_dir = args.model_dir
    model = args.model
    max_batch_size = int(args.max_batch_size)
    output_dir = args.output_dir
    trials = int(args.trials)

    if not(isPowerOfTwo(max_batch_size)):
        print('Maximum batch size needs to be a power of 2.')
        exit(1)
    
    model_path = os.path.join(model_dir, model)
    if not(os.path.exists(model_path)):
        print(f'Model does not exist: {model_path}\nExiting..')
        exit(1)

    if not(os.path.exists(output_dir)):
        create_dir = query_yes_no(f'Output directory does not exist: {output_dir}'
                                  f'\nCreate directory and proceed?')
        if create_dir:
            os.makedirs(output_dir)
            print(f'Created output directory: {output_dir}')
        else:
            print(f'Directory not created, exiting..')
            exit(1)

    # Starting ORT session
    ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    # Adjust session options
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    # Setting input size based on model family used
    input_sizes = {'yolo': [640, 640], 'eb': [224, 224], 'resnet': [224, 224],
                   'senet': [224, 224], 'vgg': [224, 224], 'gender': [224, 224]}
    input_size = None
    for model_family in input_sizes:
        if model_family in model:
            input_size = input_sizes[model_family]
    if input_size is None:
        print(f'Input size could not be found from pre-defined model families.')
        exit(1)
    else:
        print(f'Input size: {input_size}')

    # Setting function pointer for prediction based on model family used
    pred_functions = {'yolo': ort_yolo_predict, 'eb': ort_eb_car_predict,
                      'gender': ort_vgg_gender_predict}
    pred_fn_ptr = None
    for model_family in pred_functions:
        if model_family in model:
            pred_fn_ptr = pred_functions[model_family]
    if pred_fn_ptr is None:
        print(f'Function pointer could not be found from pre-defined model '
                f'families.')
        exit(1)

    # Creating DataFrame for profiled data
    columns = ['model', 'accelerator', 'batch_size', 'avg_runtime',
               '50th_pct_runtime', '90th_pct_runtime', '95th_pct_runtime']
    df = pd.DataFrame(columns=columns)

    # Profiling time
    for batch_size in [2**i for i in range(0, int(log2(max_batch_size))+1)]:
        print(f'Testing with batch size: {batch_size}')

        runtimes = []
        for trial in range(trials):
            if trial % 100 == 0:
                print(f'model: {model}, batch size: {batch_size}, '
                        f'trial: {trial}/{trials}')
                
            input_data = torch.rand(batch_size, 3, input_size[0], input_size[1])
            start_time = time.time()
            pred_fn_ptr(ort_session=ort_sess, data=input_data)
            end_time = time.time()
            processing_time = end_time - start_time
            runtimes.append(processing_time)
        
        avg = np.average(runtimes)
        pct_50 = np.percentile(runtimes, 50)
        pct_90 = np.percentile(runtimes, 90)
        pct_95 = np.percentile(runtimes, 95)

        new_row = {'model': model, 'accelerator': '1080ti', 'batch_size': batch_size,
                'avg_runtime': avg, '50th_pct_runtime': pct_50,
                '90th_pct_runtime': pct_90, '95th_pct_runtime': pct_95}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print(df)
    filename = model.split('.')[0]+'.csv'
    df.to_csv(os.path.join(output_dir, filename))

    return


if __name__=='__main__':
    main(getargs())
