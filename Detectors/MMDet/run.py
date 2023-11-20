import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import json
from tqdm import tqdm
import sys
import torch
import mmcv
from sahi.slicing import slice_image
from mmdet.utils.large_image import merge_results_by_nms, shift_predictions

def preds_from_result(result):
    if 'pred_instances' in result:
        instances = result.pred_instances
        scores = instances.scores
        bboxes = instances.bboxes # x1, y1, x2, y2
        labels = instances.labels
    return bboxes, labels, scores

@torch.no_grad
def get_sahi_result(frame, model, test_pipeline, args):
    height, width = frame.shape[:2]
    sliced_image_object = slice_image(
        frame,
        slice_height=args["SahiPatchSize"],
        slice_width=args["SahiPatchSize"],
        auto_slice_resolution=False,
        overlap_height_ratio=args["SahiPatchOverlapRatio"],
        overlap_width_ratio=args["SahiPatchOverlapRatio"],
    )

    slice_results = []
    start = 0
    while True:
        # prepare batch slices
        end = min(start + args["SahiPatchBatchSize"], len(sliced_image_object))
        images = []
        for sliced_image in sliced_image_object.images[start:end]:
            images.append(sliced_image)

        # forward the model
        slice_results.extend(inference_detector(model, images, test_pipeline=test_pipeline))

        if end >= len(sliced_image_object):
            break
        start += args.batch_size

    # shifted_instances = shift_predictions(
    #     slice_results,
    #     sliced_image_object.starting_pixels,
    #     src_image_shape=(height, width))
    # merged_result = slice_results[0].clone()
    # merged_result.pred_instances = shifted_instances

    image_result = merge_results_by_nms(
    slice_results,
    sliced_image_object.starting_pixels,
    src_image_shape=(height, width),
    nms_cfg={
        'type': 'nms',
        'iou_threshold': args["SahiNMSTh"]
    })

    result = image_result.cpu()
    return result

@torch.no_grad
def get_full_frame_result(frame, model, test_pipeline, args):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    result = result.cpu()
    return result

if __name__ == "__main__":
    args = json.loads(sys.argv[-1])
    # config and check point will come from args
    config_file = args["MMDetConfig"]
    checkpoint_file = args["MMDetCheckPoint"]
    # select compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')
    # set video and result path
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"] 

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=device_name)

    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    video_reader = mmcv.VideoReader(video_path)
    fn=0
    with open (text_result_path,"w") as f: 
        for frame in tqdm(video_reader):
            if args["SAHI"]:
                result = get_sahi_result(frame, model, test_pipeline, args)
            else:
                result = get_full_frame_result(frame, model, test_pipeline, args)

            bboxes, labels, scores = preds_from_result(result)
            for box, label, score in zip(bboxes,labels, scores):
                    r=box
                    f.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            fn+=1