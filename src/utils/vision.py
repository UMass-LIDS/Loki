import os
import time
import natsort
import logging
import pickle
import torch
import torchvision
import uuid
import numpy as np
from typing import List
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.ops import box_iou
from common.query import Query
from protos import worker_pb2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        
        print(f'img_loc: {img_loc}')
        if img_loc.split('/')[-1] == '.DS_Store' or os.path.isdir(img_loc):
            del self.total_imgs[idx]
            return self.__getitem__(idx)
        
        # image = Image.open(img_loc).convert("RGB").resize((640, 640))
        image = Image.open(img_loc).convert("RGB")
        if self.transform != None:
            tensor_image = self.transform(image)
        else:
            transform = transforms.Compose([
                # you can add other transformations in this list
                transforms.ToTensor()
            ])
            tensor_image = transform(image)
        return tensor_image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def ort_yolo_predict(ort_session, data):
    # If data is not batched, unsqueeze to use batch size of 1
    if len(data.shape) < 4:
        data = torch.unsqueeze(data, 0)

    if data.shape[-1] != data.shape[-2]:
        input_size = [data.shape[-1], data.shape[-1]]
        data = transforms.Resize(size=input_size)(data)

    numpy_data = data.numpy()

    ort_inputs = {'images': numpy_data}

    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    ort_outs = ort_session.run(output_names, ort_inputs)

    logging.warning(f'ort_yolo_predict(): original_img and downscaled_size are '
                    'hard-coded')
    results = {'ort_outs': ort_outs, 'original_img': data,
               'downscaled_size': [720, 1280]}

    return results


def yolo_preprocess(request: worker_pb2.InferenceRequest, dataset: Dataset):
    ''' For a given InferenceRequest, extracts the image data from it,
        constructs a Query object and returns it
    '''

    data = dataset[request.sequenceNum]
    queryID = uuid.uuid4()

    query = Query(requestID=request.requestID, queryID=queryID,
                  userID=request.userID, applicationID=request.applicationID,
                  taskID=0, data=data, startTimestamp=request.timestamp,
                  queuedTimeStamp=request.timestamp,
                  latencyBudget=request.latencySLOInUSec,
                  sequenceNum=request.sequenceNum)
    
    return [query]


def yolo_postprocess(results: dict):
    # TODO: only works for a batch size of 1 currently
    outputs = results['ort_outs']
    original_img = results['original_img']
    downscaled_size = results['downscaled_size']
    label_to_task = results['label_to_task']
    
    for output in outputs[0]:
        single_output = np.expand_dims(output, axis=0)
        # TODO: data copying overhead could be reduced, but seems like it is
        #       not practically making any difference. Perhaps data is already
        #       on GPU?
        # TODO: use appropriate device
        # nms_pred = non_max_suppression(torch.tensor(np.array(single_output), device=device))
        nms_pred = non_max_suppression(torch.tensor(np.array(single_output)))
        scaled_objdet_pred = scale_bboxes(preds=nms_pred, original_img=original_img,
                                          downscaled_size=downscaled_size)
        
        # TODO: use appropriate device
        # scaled_objdet_tensors = torch.tensor(np.zeros((1, 8)), device=device)
        scaled_objdet_tensors = torch.tensor(np.zeros((1, 8)))
        separated_by_task_tensors = {}

        for _pred in scaled_objdet_pred:
            # for every image in the batch
            for det in _pred:
                # TODO: instead of image_id and timestamp values, perhaps
                #       we should have request id (and maybe user id) instead
                logging.warning(f'yolo_postprocess(): Hard-coded values for image_id '
                                f'and frames_per_sec')
                image_id = 1
                frames_per_sec = 1
                # for every object detected in the image
                timestamp = image_id / frames_per_sec

                logging.warning(f'yolo_postprocess(): no device specified')
                # concat_tensor = torch.tensor([image_id, det[5], det[4], det[0], det[1],
                #                             det[2], det[3], timestamp],
                #                             device=device)

                # det[5] is the label and det[4] is the score
                concat_tensor = torch.tensor([image_id, det[5], det[4], det[0], det[1],
                                            det[2], det[3], timestamp])
                concat_tensor = concat_tensor[None, :]

                label = int(det[5])

                # If we encounter a label that is not defined by worker.py, skip it
                if label not in label_to_task:
                    continue

                task = label_to_task[label]
                if task in separated_by_task_tensors:
                    separated_by_task_tensors[task] = torch.cat((separated_by_task_tensors[task],
                                                                   concat_tensor))
                else:
                    separated_by_task_tensors[task] = concat_tensor

                scaled_objdet_tensors = torch.cat((scaled_objdet_tensors, concat_tensor))
        scaled_objdet_tensors = scaled_objdet_tensors[1:, :]
        
        # TODO: should we be returning outside for loop?
        return separated_by_task_tensors
        


def prepare_data_batch(tensor_list: List[torch.tensor]):
    if len(tensor_list) == 0:
        return None
    
    unsqueezed_tensors = []
    for tensor in tensor_list:
        unsqueezed_tensor = torch.unsqueeze(tensor, 0)
        unsqueezed_tensors.append(unsqueezed_tensor)
    data_batch = torch.cat(unsqueezed_tensors, dim=0)
    return data_batch


def ort_resnet_face_predict(ort_session, data):
    # TODO: does not support batched execution as of yet
    # print(f'number of expected inputs: {len(ort_session.get_inputs())}')
    # print(f'expected inputs: {ort_session.get_inputs()}')

    # print(f'ort_resnet_predict -- data.shape: {data.shape}')

    # TODO: we unsqueeze because we have non-batched data and we need
    #       convert it into shape [1, x, y, z] instead of [x, y, z]
    print(f'pre data.shape: {data.shape}')
    data = torch.unsqueeze(data, 0)
    data = data.permute(0, 2, 3, 1)
    data = data.numpy()

    # print(f'ort_resnet_predict -- data.shape: {data.shape}')

    # input_names = list(map(lambda input: input.name, ort_session.get_inputs()))
    # print(f'input_names: {input_names}')

    print(ort_session.get_inputs()[0])

    print(f'data.shape: {data.shape}')
    exit()

    ort_inputs = {'input_2': data}

    # first_input_shape = ort_session.get_inputs()[0].shape
    # print(f'first input shape: {first_input_shape}')

    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    # print(f'output names: {output_names}')

    ort_outs = ort_session.run(output_names, ort_inputs)
    return ort_outs


def ort_vgg_gender_predict(ort_session, data):
    # TODO: does not support batched execution as of yet

    # TODO: we unsqueeze because we have non-batched data and we need
    #       convert it into shape [1, x, y, z] instead of [x, y, z]
    if len(data.shape) < 4:
        data = torch.unsqueeze(data, 0)

    data = data.numpy()

    ort_inputs = {'input': data}

    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    ort_outs = ort_session.run(output_names, ort_inputs)
    return ort_outs


def ort_senet_face_predict(ort_session, data):
    # TODO: does not support batched execution as of yet

    # TODO: we unsqueeze because we have non-batched data and we need
    #       convert it into shape [1, x, y, z] instead of [x, y, z]
    data = torch.unsqueeze(data, 0)
    data = data.permute(0, 2, 3, 1)
    data = data.numpy()

    print(f'ort_senet_face_predict -- data.shape: {data.shape}')

    input_names = list(map(lambda input: input.name, ort_session.get_inputs()))
    print(f'input_names: {input_names}')

    ort_inputs = {'input_3': data}

    first_input_shape = ort_session.get_inputs()[0].shape
    print(f'first input shape: {first_input_shape}')

    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    print(f'output names: {output_names}')

    ort_outs = ort_session.run(output_names, ort_inputs)
    return ort_outs



def get_ort_face_function(face_rec_model):
    if 'resnet' in face_rec_model:
        return ort_resnet_face_predict
    elif 'senet' in face_rec_model:
        return ort_senet_face_predict
    elif 'genderNet' in face_rec_model:
        return ort_vgg_gender_predict
    else:
        raise Exception(f'face_rec_model {face_rec_model} not recognized')
    

def eb_car_preprocess(request: worker_pb2.InferenceRequest, dataset: Dataset):
    ''' For a given InferenceRequest, extract the queries by converting the
        xmin, xmax, ymin, ymax values to image tensors for the appropriate
        image. This will generate multiple queries for the given request
    '''
    queries = []
    unpickledData = pickle.loads(request.data)
    image = dataset[request.sequenceNum]
    tensors = extract_object_tensors(img_tensor=image,
                                     obj_det_tensors=unpickledData)
    # TODO: remove hard-coded value
    resizedTensors = resize_tensors(tensors=tensors, size=[224, 224])
    
    for tensorData in resizedTensors:
        queryID = str(uuid.uuid4())
        query = Query(requestID=request.requestID, queryID=queryID,
                      userID=request.userID, applicationID=request.applicationID,
                      taskID=0, data=tensorData, startTimestamp=request.timestamp,
                      queuedTimeStamp=request.timestamp,
                      latencyBudget=request.latencySLOInUSec,
                      sequenceNum=request.sequenceNum)
        queries.append(query)

    return queries


def ort_eb_car_predict(ort_session, data):
    # TODO: does not support batched execution as of yet

    # We unsqueeze because we have non-batched data and we need to convert
    # it into shape [1, x, y, z] instead of [x, y, z]
    if len(data.shape) < 4:
        data = torch.unsqueeze(data, 0)

    # data = data.permute(0, 2, 3, 1)
    data = data.numpy()

    ort_inputs = {'input_1': data}

    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    ort_outs = ort_session.run(output_names, ort_inputs)

    results = {'ort_outs': ort_outs}

    return results


def eb_car_postprocess(results: dict):
    ''' Given the results from the processing stage, extracts the car label
        and score
    '''
    # TODO: only works for a batch size of 1 currently
    outputs = results['ort_outs']
    
    label = np.argmax(outputs[0])
    score = np.max(outputs[0])

    postprocessed_results = {'sink': torch.tensor([label, score])}

    return postprocessed_results


def sink_preprocess(request: worker_pb2.InferenceRequest, dataset: Dataset):
    queryID = uuid.uuid4()
    unpickledData = pickle.loads(request.data)
    tensorData = torch.tensor(unpickledData)
    query = Query(requestID=request.requestID, queryID=queryID,
                  userID=request.userID, applicationID=request.applicationID,
                  taskID=0, data=tensorData, startTimestamp=request.timestamp,
                  queuedTimeStamp=request.timestamp,
                  latencyBudget=request.latencySLOInUSec,
                  sequenceNum=request.sequenceNum)
    return [query]


def sink_process(ort_session, data):
    results = {'ort_outs': data}
    return results


def sink_postprocess(results: dict):
    ort_outs = results['ort_outs']
    resultsByTask = {'none': ort_outs}
    return resultsByTask


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_bboxes(preds, original_img, downscaled_size):
    x_scale_factor = original_img.shape[-1] / downscaled_size[-1]
    y_scale_factor = original_img.shape[-2] / downscaled_size[-2]
    for bboxes in preds:
        # print(f'pre-scaling bboxes: {bboxes}')
        # print(f'original_img shape: {original_img.shape}')
        # print(f'downscaled_size: {downscaled_size}')
        bboxes[:, :1] = bboxes[:, :1] * x_scale_factor
        bboxes[:, 1:2] = bboxes[:, 1:2] * y_scale_factor
        bboxes[:, 2:3] = bboxes[:, 2:3] * x_scale_factor
        bboxes[:, 3:4] = bboxes[:, 3:4] * y_scale_factor
        # print(f'post-scaling bboxes: {bboxes}')
    return preds


def save_images(tensors, path, name, last_idx):
    for tensor in tensors:
        filepath = os.path.join(path, f'{name}_{last_idx}.png')
        print(f'filepath: {filepath}')
        print(f'tensor shape: {tensor.shape}')
        save_image(tensor, filepath)
        print(f'Saved image: {filepath}')
        last_idx += 1
    return last_idx


def extract_object_tensors(img_tensor: torch.tensor,
                           obj_det_tensors: List[torch.tensor],
                           label: int=None):
    # TODO: only supports batch size of 1

    # TODO: CPU-bound operation because extracted_tensors is a list of tensors
    #       of different shapes

    # Filter tensors if label is given
    if label is not None:
        obj_det_tensors = obj_det_tensors[obj_det_tensors[:, 1] == label]

    # obj_det_tensor contains xmin, xmax, ymin, ymax co-ordinates
    # extracted tensor contains the image data within those co-ordinates
    extracted_tensors = []
    for bbox in obj_det_tensors:
        extracted_tensor = img_tensor[:, torch.floor(bbox[4]).type(torch.int64):
                                      torch.ceil(bbox[6]).type(torch.int64),
                                      torch.floor(bbox[3]).type(torch.int64):
                                      torch.ceil(bbox[5]).type(torch.int64)]

        if extracted_tensor.shape[-1] == 0 or extracted_tensor.shape[-2] == 0:
            continue

        extracted_tensors.append(extracted_tensor)
    
    return extracted_tensors


def extract_object_tensor_pairs(img_tensor, obj_det_tensors, label=None):
    # TODO: only supports batch size of 1

    # TODO: CPU-bound operation because new_tensors is a list of tensors
    #       of different shapes

    # Filter tensors if label is given
    if label is not None:
        obj_det_tensors = obj_det_tensors[obj_det_tensors[:, 1] == label]
    
    # First element in each pair is the extracted tensor and the second element
    # is the previous stage tensor (obj_det_tensor)
    # obj_det_tensor contains xmin, xmax, ymin, ymax co-ordinates
    # extracted tensor contains the image data within those co-ordinates
    extracted_tensors_pairs = []
    for bbox in obj_det_tensors:
        previous_tensor = bbox
        extracted_tensor = img_tensor[:, torch.floor(bbox[4]).type(torch.int64):
                                      torch.ceil(bbox[6]).type(torch.int64),
                                      torch.floor(bbox[3]).type(torch.int64):
                                      torch.ceil(bbox[5]).type(torch.int64)]

        if extracted_tensor.shape[-1] == 0 or extracted_tensor.shape[-2] == 0:
            continue

        extracted_tensor_pair = (extracted_tensor, previous_tensor)
        extracted_tensors_pairs.append(extracted_tensor_pair)

    return extracted_tensors_pairs


def resize_tensors(tensors: List[torch.tensor], size: List[int]):
    """ Resizes a list of tensors to a given size

    Args:
        tensors (List[torch.tensor]): a list of tensors
        size (List[int]): the size to resize the tensors to

    Returns:
        List[torch.tensor]: the list of resized tensors
    """    
    resized_tensors = []
    for tensor in tensors:
        resized_tensor = transforms.Resize(size=size)(tensor)
        resized_tensors.append(resized_tensor)
    return resized_tensors


def resize_tensor_pairs(tensor_pairs, size):
    """ Resizes a list of tensors to a given size

    Args:
        tensor_pairs (_type_): a list of tensor pairs; each pair is a tuple
                               of (new tensor, previous stage tensor)
                               previous stage tensor is the tensor
                               from the previous stage
        size (_type_): the size to resize the tensors to

    Returns:
        _type_: returns the list of resized tensor pairs
    """    
    resized_tensor_pairs = []
    for (tensor, previous_tensor) in tensor_pairs:
        resized_tensor = transforms.Resize(size=size)(tensor)
        resized_tensor_pairs.append((resized_tensor, previous_tensor))
    return resized_tensor_pairs


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    # print(f'device: {device}')
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 2.0 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            logging.log(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

