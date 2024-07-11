# In this file, we want to measure the average true-positive ratio of the object
# detection models (i.e., TP / (TP + FP + FN)). This is the metric that is multiplied
# by the 2nd stage accuracy in Traffic Analysis/Monitoring and Sports Analysis
# pipelines to get the end-to-end accuracy

import os
import pandas as pd


data_dir = 'profiling/accuracy/sports_analysis/'

obj_det_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

for obj_det_model in obj_det_models:
    # if obj_det_model != 'yolov5x': continue

    # Open the CSV
    filename = os.path.join(data_dir, f'{obj_det_model}_confusion_matrix.csv')
    df = pd.read_csv(filename)

    tp = df['tp'].values
    fp = df['fp'].values
    fn = df['fn'].values

    tpr = tp / (tp + fp + fn)

    print(f'model: {obj_det_model}, tpr: {tpr}')

    average_tpr = sum(tpr) / len(tpr)

    print(f'average_tpr: {average_tpr}')

    # exit(0)
