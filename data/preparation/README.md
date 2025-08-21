# Pre-processing

We provide a pre-processing pipeline in this repository for detecting and cropping mouth regions of interest (ROIs) as well as corresponding audio waveforms for CMLR


## Setup

1. Install all dependency-packages.

```Shell
pip install -r requirements.txt
```

2. Install [retinaface](./tools) tracker,you can put another detector in `/detectors`:

- `cd ../data/preparation/detectors/retinaface/`
- Install [ibug.face_detection](https://github.com/hhj1897/face_detection)

```Shell
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
pip install -e .
cd ..
```
Recommendation: manually download to the specified directory, since errors frequently occur with `ibug/face_detection/retina_face/weights/Resnet50_Final.pth`.

- Install [*`ibug.face_alignment`*](https://github.com/hhj1897/face_alignment)

```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```
- Reference mean face download from: [Line]( https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)


## Pre-processing CMLR

To pre-process the CMLR dataset, plrase follow these steps:

1. Download the CMLR dataset from the official website.

2. Download pre-computed landmarks below. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

| File Name              | Source URL                                                                              | File Size  |
|------------------------|-----------------------------------------------------------------------------------------|------------|
| CMLR_landmarks.zip     |[GoogleDrive](https://bit.ly/) or [BaiduDrive](https://bit.lyh)(key: mi3c) |     18GB   |


3. Run the following command to pre-process dataset:

```Shell
python preprocess_lrs2lrs3.py \
    --data-dir [data_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --gpu_type [gpu_type] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```

### Arguments
- `data-dir`: Directory of original dataset.
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid values are: `lrs2` and `lrs3`.
- `gpu_type`: Type of GPU to use. Valid values are `cuda` and `mps`. Default: `cuda`.
- `subset`: Subset of dataset. For `lrs2`, the subset can be `train`, `val`, and `test`. For `lrs3`, the subset can be `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `16`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

### Steps
Several steps are defined through the option `--step`, and each step can be executed independently.

`Step 0.` To perform bulk decompression of the CMLR corpus, set `--step 0` to complete this step:

    '''
    Make sure you have the following directory structure:

    /.../.../CMLR-CORPUS/
    ├── audio
    │   ├── s1.zip
    ├── video
    │   ├── s1.zip
    ├── text.zip
    ├── train.csv
    ├── val.csv
    ├── test.csv

    '''

`step 1.` To split dataset to `train, val, test` in `../datasets`

`step 2.`
