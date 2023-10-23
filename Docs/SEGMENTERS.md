# Segmentation Guide
This pipeline supports:
  * detectors from [mmsegment](https://github.com/open-mmlab/mmsegmentation)
  * custom detectors added by user/source

**NOTE**: before running any functionality of the pipeline makesure to format data according to [our requirements](./).

**NOTE**: there is a full-functionality template in `run.py`.

## Run segmentation on video
```
python3 main.py --Dataset={src} --Segmenter={seg} --Segment
```
Results will be stored under src/Results/Segmentation/*.[pkl/txt]

## Visualize segmentation
```
python3 main.py --Dataset={src} --Segmenter={seg} --VisSegment
```
Results will be stored under src/Results/Visualization/*.mp4


## Segment post processing
* `--SegTh` to maskout detection.
* `--classes_to_kee` to select class object to keep.
```
python3 main.py --Dataset={src} --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7
```
## Supported segmenters
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="https://github.com/OpenGVLab/InternImage">InternImage</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## Add custom segmenter
Create a folder with the name of `<SegmenterName>` under `./Segmenters/`.

Then create two python files `./Segmenters/SegmenterName/segment.py` and `./Segmenters/SegmenterName/run.py`.

The first file is where our pipeline calls `run.py` with `conda_env=SegmenterName` which will get detections and store them as txt under Results forlder.

A template for these two files are provided below.

`segment.py`
```
def segment(args):
    setup(args)
    env_name = args.Segmenter
    exec_path = "./Segmenters/InternImage/run.py"
    conda_pyrun(env_name, exec_path, args)

def setup(args):
    env_name = args.Segmenter
    src_url = "https://github.com/OpenGVLab/InternImage.git"
    rep_path = "./Segmenters/InternImage/InternImage"
    
    if not "InternImage" in os.listdir("./Segmenters/InternImage/"):
      os.system(f"git clone {src_url} {rep_path}")

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        os.system(f"conda run --live-stream -n {env_name} conda install pip")
```

`run.py`
```
if __name__ == "__main__":
    config_file = './Segmenters/InternImage/InternImage/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py'
    checkpoint_file = './Segmenters/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_3x_coco.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')

    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    results_path = args["SegmentPkl"]
    results_path_bu = args["SegmentPklBackUp"]

    model = init_detector(config_file, checkpoint_file, device=device_name)
    video = mmcv.VideoReader(video_path)

    fn=0
    for frame in tqdm(video):
        data_dict = make_data_dict()
        results_path_fn = get_results_path_with_frame(results_path, fn)
        results_path_bu_fn = get_results_path_with_frame(results_path_bu, fn)

        with torch.no_grad():
            result = inference_detector(model, frame)
        bboxes, labels, segms= seperate_results(result)

        for bbox, label, segm in zip(bboxes, labels, segms):
            x1, y1, x2, y2, score = bbox
            cropped_mask = segm[int(y1):int(y2), int(x1):int(x2)]
            data_dict["x1"].append(x1)
            data_dict["y1"].append(y1)
            data_dict["x2"].append(x2)
            data_dict["y2"].append(y2)
            data_dict["score"].append(score)
            data_dict["class"].append(label)
            data_dict["fn"].append(fn)
            data_dict["mask"].append(cropped_mask)

        df = pd.DataFrame.from_dict(data_dict)
        df.to_pickle(results_path_fn, protocol=4, compression=None)
        df.to_pickle(results_path_bu_fn, protocol=4, compression=None)
        fn += 1
```
