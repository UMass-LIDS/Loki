import os
import pandas as pd
import numpy as np
from utils.detection import read_output_from_df, confusion_matrix


data_dir = 'output'
profiling_dir = 'profiling/accuracy/traffic_analysis'

# # ----------------------------
# # 1 stage pipeline
# # ----------------------------

# # ground_truth = 'Bellevue_Bellevue_NE8th__2017-09-10_18-08-23_detections'
# obj_det_ground_truth = 'yolov5x'
# obj_det_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# # models = ['yolov5x']
# # models = ['yolov5l']

# for obj_det_model in obj_det_models:
#     obj_det_gt_df = pd.read_csv(os.path.join(data_dir, f'{obj_det_ground_truth}.csv'))
#     obj_det_model_df = pd.read_csv(os.path.join(data_dir, f'{obj_det_model}.csv'))

#     if 'yolo' in obj_det_ground_truth:
#         obj_det_gt_df['class'] = obj_det_gt_df['class'] + 1

#     if 'yolo' in obj_det_model:
#         obj_det_model_df['class'] = obj_det_model_df['class'] + 1

#     timestamps = np.sort(np.unique(obj_det_gt_df['timestamp'].to_numpy()))

#     target = read_output_from_df(df=obj_det_gt_df, timestamps=timestamps, x1_label='xmin',
#                                  x2_label='xmax', y1_label='ymin', y2_label='ymax',
#                                  stage_1_score='score', stage_1_class='class', target=True)

#     preds = read_output_from_df(df=obj_det_model_df, timestamps=timestamps, x1_label='xmin',
#                                 x2_label='xmax', y1_label='ymin', y2_label='ymax',
#                                 stage_1_score='score', stage_1_class='class', target=False)

#     obj_det_cmatrix = {'tp': [], 'fp': [], 'fn': []}
#     for timestamp_idx in range(len(timestamps)):
#         # print(f'timestamp_idx: {timestamp_idx}, preds: {len(preds[timestamp_idx]["labels"])},'
#         #       f' gts: {len(target[timestamp_idx]["labels"])}')
#         cmatrix_t = confusion_matrix(target=target[timestamp_idx],
#                                      preds=preds[timestamp_idx], 
#                                      iou_threshold=0.5)
#         # print(f'cmatrix: {cmatrix}')

#         obj_det_cmatrix['tp'].append(cmatrix_t['tp'])
#         obj_det_cmatrix['fp'].append(cmatrix_t['fp'])
#         obj_det_cmatrix['fn'].append(cmatrix_t['fn'])

#     obj_det_cmatrix['timestamp'] = timestamps

#     obj_det_cmatrix_df = pd.DataFrame(data=obj_det_cmatrix)
    
#     savefile_name = os.path.join(profiling_dir, f'{obj_det_model}_confusion_matrix.csv')
#     obj_det_cmatrix_df.to_csv(savefile_name)
#     print(f'Saved profiled confusion matrix to: {savefile_name}')


# ----------------------------
# 2 stage pipeline
# ----------------------------

# TODO: use different facial recognition models as well
#       should be straight-forward to compute their confusion matrices
#       once car classification is done
#       but they need a separate inference interface

obj_det_ground_truth = 'yolov5x'
# obj_det_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# obj_det_models = ['yolov5x', 'yolov5l', 'yolov5m', 'yolov5n']
obj_det_models = ['yolov5x']

# car_cls_ground_truth = 'eb6_checkpoint_150epochs'
# car_cls_models = ['eb6_checkpoint_150epochs', 'eb5_checkpoint_150epochs',
#                   'eb4_checkpoint_150epochs', 'eb3_checkpoint_150epochs',
#                   'eb2_checkpoint_150epochs', 'eb1_checkpoint_150epochs',
#                   'eb0_checkpoint_150epochs']

face_rec_ground_truth = 'genderNet_16'
face_rec_models = ['genderNet', 'genderNet_11', 'genderNet_16', 'genderNet_19']

# for car_cls_model in car_cls_models:
#     for obj_det_model in obj_det_models:
#         gt_filename = os.path.join(data_dir, f'{car_cls_ground_truth}_{obj_det_ground_truth}.csv')
#         model_filename = os.path.join(data_dir, f'{car_cls_model}_{obj_det_model}.csv')

#         if not(os.path.isfile(model_filename)):
#             print(f'File not found, skipping: {model_filename}')
#             continue

#         gt_df = pd.read_csv(gt_filename)
#         model_df = pd.read_csv(model_filename)
        
#         timestamps = np.sort(np.unique(gt_df['timestamp'].to_numpy()))
        
#         target = read_output_from_df(df=gt_df, timestamps=timestamps, x1_label='xmin',
#                                      x2_label='xmax', y1_label='ymin', y2_label='ymax',
#                                      stage_1_score='obj_score', stage_1_class='obj_class',
#                                      target=True, stage_2_score='score',
#                                      stage_2_class='class')

#         preds = read_output_from_df(df=model_df, timestamps=timestamps, x1_label='xmin',
#                                     x2_label='xmax', y1_label='ymin', y2_label='ymax',
#                                     stage_1_score='obj_score', stage_1_class='obj_class',
#                                     target=False, stage_2_score='score',
#                                     stage_2_class='class')
        
#         cmatrix = {'tp': [], 'fp': [], 'fn': [], 'tp_tp': [], 'tp_fp': []}
#         for timestamp_idx in range(len(timestamps)):
#             cmatrix_t = confusion_matrix(target=target[timestamp_idx],
#                                          preds=preds[timestamp_idx], 
#                                          iou_threshold=0.5)

#             cmatrix['tp'].append(cmatrix_t['tp'])
#             cmatrix['fp'].append(cmatrix_t['fp'])
#             cmatrix['fn'].append(cmatrix_t['fn'])
#             cmatrix['tp_tp'].append(cmatrix_t['tp_tp'])
#             cmatrix['tp_fp'].append(cmatrix_t['tp_fp'])

#         cmatrix['timestamp'] = timestamps

#         cmatrix_df = pd.DataFrame(data=cmatrix)
        
#         savefile_name = os.path.join(profiling_dir, f'{car_cls_model}_{obj_det_model}'
#                                                     f'_confusion_matrix.csv')
#         cmatrix_df.to_csv(savefile_name)
#         print(f'Saved profiled confusion matrix to: {savefile_name}')


for face_rec_model in face_rec_models:
    for obj_det_model in obj_det_models:
        gt_filename = os.path.join(data_dir, f'{face_rec_ground_truth}_{obj_det_ground_truth}.csv')
        model_filename = os.path.join(data_dir, f'{face_rec_model}_{obj_det_model}.csv')

        if not(os.path.isfile(model_filename)):
            print(f'File not found, skipping: {model_filename}')
            continue

        gt_df = pd.read_csv(gt_filename)
        model_df = pd.read_csv(model_filename)
        
        timestamps = np.sort(np.unique(gt_df['timestamp'].to_numpy()))
        
        target = read_output_from_df(df=gt_df, timestamps=timestamps, x1_label='xmin',
                                     x2_label='xmax', y1_label='ymin', y2_label='ymax',
                                     stage_1_score='obj_score', stage_1_class='obj_class',
                                     target=True, stage_2_score='score',
                                     stage_2_class='class')

        preds = read_output_from_df(df=model_df, timestamps=timestamps, x1_label='xmin',
                                    x2_label='xmax', y1_label='ymin', y2_label='ymax',
                                    stage_1_score='obj_score', stage_1_class='obj_class',
                                    target=False, stage_2_score='score',
                                    stage_2_class='class')
        
        cmatrix = {'tp': [], 'fp': [], 'fn': [], 'tp_tp': [], 'tp_fp': []}
        for timestamp_idx in range(len(timestamps)):
            cmatrix_t = confusion_matrix(target=target[timestamp_idx],
                                         preds=preds[timestamp_idx], 
                                         iou_threshold=0.5)

            cmatrix['tp'].append(cmatrix_t['tp'])
            cmatrix['fp'].append(cmatrix_t['fp'])
            cmatrix['fn'].append(cmatrix_t['fn'])
            cmatrix['tp_tp'].append(cmatrix_t['tp_tp'])
            cmatrix['tp_fp'].append(cmatrix_t['tp_fp'])

        cmatrix['timestamp'] = timestamps

        cmatrix_df = pd.DataFrame(data=cmatrix)
        
        savefile_name = os.path.join(profiling_dir, f'{face_rec_model}_{obj_det_model}'
                                                    f'_confusion_matrix.csv')
        cmatrix_df.to_csv(savefile_name)
        print(f'Saved profiled confusion matrix to: {savefile_name}')
