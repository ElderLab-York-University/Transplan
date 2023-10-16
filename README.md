# Introduction

Welcome to the TransPlan Pipeline, a comprehensive open-source solution for detecting, tracking, segmenting, and counting objects in video streams. This repository offers a powerful suite of computer vision tools and algorithms, designed to assist researchers, developers, and enthusiasts in analyzing video data with unprecedented precision and efficiency.

## Key Features

* **Detection**: Our detection module is based on [mmdetection](https://mmdetection.readthedocs.io/en/latest/).
  
  Beside supporting all the detectors from mmdetection, there is support for other detectors as well.
  
  You have the option to add your custom detectors as well(from a github repository or any other sources).
  
  See [Detectors Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)

* **Segmentation**: Currently the segmentation module only supports InternImage.
  
  Adding more segmentaion models from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) is in progress.
  
  You can also add your own segmentation module.

  See [Segmenter Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)

* **Tracking**: This pipeline focuses on tracking by detection methodology but it also has the capability to support off-line tracking models.

  You can also add your own tracker.

  See [Tracker Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)

* **Benchmarking**: There is support for detection and tracking evaluation if a grand-truch is provided.
  
    See [Benchmarking Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)

* **Reprojection**: There is support for reprojecting points on video to real-world coordinates.
  * If camera extrinsics are given this can be done with DSM terrain models.
  * In other cases we have GUI for solving homographies.
    
  See [Reprojection Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)

## Task Specific Features

* **Object Counting**: There is support to categorize objects' movements if their entry and exit spots form a structured environemnt such as an InterSection.

  See [Counting Guide](https://github.com/ElderLab-York-University/Transplan/edit/booklet/README.md)


## Install
  Most of the code is based on `Python=3.8`. We use conda virtural environments to mange model specific dependancies internally.

  To install see [Installation Guide](./)

## Getting Started
  We provide a typical usage flow of the pipeline in `run.py`.
  
  There is also task specific user instructions in [User Guide](./)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## License

This project is released under the [MIT license](LICENSE).
