from typing import List
import pandas as pd
import torch
from torchvision.ops import box_iou


def confusion_matrix(target: dict, preds: dict, iou_threshold: int) -> dict:
    """ Generates the confusion matrix

    Args:
        target (torch.tensor):  target dict containing 'boxes'
                                and 'labels'
        preds (torch.tensor):   prediction dict containing 'boxes',
                                'scores' and 'labels'
        iou_threshold (int):    iou threshold for detecting objects

    Raises:
        Exception: if ground truths don't add up (tp + fn)
        Exception: if detections don't add up (tp + fp)

    Returns:
        dict: confusion matrix
    """    
    cmatrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tp_tp': 0, 'tp_fp': 0}

    ious = box_iou(preds['boxes'], target['boxes'])
    # print(f'ious: {ious}')

    stage_2_exists = True

    iou_idx = 0
    gt_counted = 0
    for iou_idx in range(len(ious)):
        iou = ious[iou_idx]

        stage_1_max_overlap = torch.max(iou)
        stage_1_argmax_overlap = torch.argmax(iou)

        # print(f'overlap: {iou}')

        # TODO: some ground truths could be counted multiple times
        #       if they overlap with multiple detections
        #       however, removing ground truths from ious is not a solution
        #       since that messes up the indices
        # TODO: we can actually see the effect of this, it has artifically
        #       raised the average precision of stage 1 significantly
        if stage_1_max_overlap > iou_threshold:
            # print(f'iou_idx: {iou_idx}, stage_1_argmax_overlap: {stage_1_argmax_overlap}')
            # print(f'preds[stage_1_labels]: {preds["stage_1_labels"]}')
            # print(f'target[stage_1_labels]: {target["stage_1_labels"]}')
            # print(f'preds[stage_2_labels]: {preds["stage_2_labels"]}')
            # print(f'target[stage_2_labels]: {target["stage_2_labels"]}')
            if preds['stage_1_labels'][iou_idx] == target['stage_1_labels'][stage_1_argmax_overlap]:
                # object detection (stage 1) matched
                cmatrix['tp'] += 1
                gt_counted += 1

                if not(stage_2_exists) or 'stage_2_labels' not in preds or \
                    'stage_2_labels' not in target or  len(preds['stage_2_labels']) == 0 \
                    or len(target['stage_2_labels']) == 0:
                    stage_2_exists = False
                    continue
                
                # now check whether the stage 2 labels match
                # TODO: the tp_tp and tp_fp ratios are not adding up to 1 (tp)
                #       how could this be? is tp being updated somewhere else?
                if preds['stage_2_labels'][iou_idx] == target['stage_2_labels'][stage_1_argmax_overlap]:
                    cmatrix['tp_tp'] += 1
                else:
                    cmatrix['tp_fp'] += 1

                # print(f'preds[stage_1_labels][iou_idx]: {preds["stage_1_labels"][iou_idx]}, '
                #       f'target[stage_1_labels][stage_1_argmax_overlap]: {target["stage_1_labels"][stage_1_argmax_overlap]}, '
                #       f'preds[stage_2_labels][iou_idx]: {preds["stage_2_labels"][iou_idx]}, '
                #       f'target[stage_2_labels][stage_1_argmax_overlap]: {target["stage_2_labels"][stage_1_argmax_overlap]}')
                # print()

                # target['stage_1_labels'] = torch.cat((target['stage_1_labels']
                #                                             [:stage_1_argmax_overlap],
                #                                       target['stage_1_labels']
                #                                             [stage_1_argmax_overlap+1:]))
                # ious = torch.cat((ious[:, :stage_1_argmax_overlap],
                #                   ious[:, stage_1_argmax_overlap+1:]), axis=1)
            else:
                cmatrix['fp'] += 1
        else:
            cmatrix['fp'] += 1

        # if there are no more ground truths left to compare against, end loop
        if ious.shape[1] == 0:
            break

    # Any leftover ground truths that did not match are false negatives
    leftover_gt = ious.shape[1] - gt_counted
    cmatrix['fn'] += leftover_gt

    # Any leftover predictions that did not match are false positives
    if len(preds['stage_1_labels']) == 0:
        leftover_preds = 0
    else:
        leftover_preds = (len(preds['stage_1_labels'])-1) - iou_idx

    cmatrix['fp'] += leftover_preds
    
    if cmatrix['tp'] + cmatrix['fn'] != target['boxes'].shape[0]:
        raise Exception('Ground truths don\'t add up (TP + FN)')
    
    if cmatrix['tp'] + cmatrix['fp'] != preds['boxes'].shape[0]:
        raise Exception('Predictions don\'t add up (TP + FP)')

    if cmatrix['tp_tp'] + cmatrix['tp_fp'] != cmatrix['tp'] and stage_2_exists:
        raise Exception('Second stage predictions don\'t add up (TP_TP + TP_FP)')
    
    return cmatrix


def read_output_from_df(df: pd.DataFrame, timestamps: pd.Series, x1_label: str,
                        x2_label: str, y1_label: str, y2_label: str,
                        stage_1_score: str, stage_1_class: str, target: bool,
                        stage_2_score: str = None, stage_2_class: str = None) -> List[dict]:
    """ Returns the object detection bounding boxes from a Pandas dataframe

    Args:
        df (pd.DataFrame): the dataframe
        timestamps (pd.Series): a list of timestamps
        x1_label (str): the xmin column name in the dataframe
        x2_label (str): the xmax column name in the dataframe
        y1_label (str): the ymin column name in the dataframe
        y2_label (str): the ymax column name in the dataframe
        score_label (str): the scores column name in the dataframe
        class_label (str): the labels/class column name in the dataframe
        target (bool): whether data is ground truth or not

    Returns:
        List[dict]: list of dicts for each timestamp
    """       
    return_dict = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    for timestamp in timestamps:
        sliced_df = df[df['timestamp'] == timestamp]
        x1 = torch.reshape(torch.tensor(sliced_df[x1_label].to_numpy(), device=device),
                           (-1, 1))
        x2 = torch.reshape(torch.tensor(sliced_df[x2_label].to_numpy(), device=device),
                           (-1, 1))
        y1 = torch.reshape(torch.tensor(sliced_df[y1_label].to_numpy(), device=device),
                           (-1, 1))
        y2 = torch.reshape(torch.tensor(sliced_df[y2_label].to_numpy(), device=device),
                           (-1, 1))
        
        boxes = torch.hstack((x1, y1, x2, y2))
        obj_det_labels = torch.tensor(sliced_df[stage_1_class].to_numpy(),
                                      device=device).to(torch.int)
        obj_det_scores = torch.tensor(sliced_df[stage_1_score].to_numpy(), device=device)

        stage_2_scores = []
        stage_2_labels = []
        if stage_2_score is not None and stage_2_class is not None:
            stage_2_scores = torch.tensor(sliced_df[stage_2_score].to_numpy(),
                                          device=device)
            stage_2_labels = torch.tensor(sliced_df[stage_2_class].to_numpy(),
                                          device=device).to(torch.int)

        if target is True:
            return_dict.append(dict(boxes=boxes, stage_1_labels=obj_det_labels,
                                    stage_2_labels=stage_2_labels))
        else:
            return_dict.append(dict(boxes=boxes, stage_1_scores=obj_det_scores,
                                    stage_1_labels=obj_det_labels,
                                    stage_2_scores=stage_2_scores,
                                    stage_2_labels=stage_2_labels))

    return return_dict
