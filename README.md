# OccludedPASCAL3D+
The OccludedPASCAL3D+ is a dataset is designed to evaluate the robustness to occlusion for a number of computer vision tasks, such as object detection, keypoint detection and pose estimation. 
In the OccludedPASCAL3D+ dataset, we simulate partial occlusion by superimposing objects cropped from the MS-COCO dataset on top of objects from the PASCAL3D+ dataset. We only use ImageNet subset in PASCAL3D+, which has 10812 testing images. **Note:** The OccludedPASCAL3D+ dataset is designed for evaluating out distribution robustness toward unsign occlusion. Thus, the training set with occlusion is only for abilation usage. 
![Figure of Car in OccludedPASCAL3D+ in 9 occlussion levels](https://github.com/Angtian/OccludedPASCAL3D/blob/master/Example.JPEG)

The OccludedPASCAL3D+ has 9 occlusion levels in total, with three foreground occlusion levels (FGL1, FGL2, FGL3) and three background occlusion levels (BGL1, BGL2, BGL3). Note that the amount of occlusion is compuated as the number of occluded pixels on the object mask. 
The occlusion ratio of the foreground:  
| Occlusion Level | FGL1    | FGL2    | FGL3    |
|-----------------|---------|---------|---------|
| Occlusion Ratio | 20%-40% | 40%-60% | 60%-80% |   

The occlusion ratio of the background:  
| Occlusion Level | BGL1    | BGL2    | BGL3    |
|-----------------|---------|---------|---------|
| Occlusion Ratio | 0%-20%  | 20%-40% | 40%-60% |  

Number of images for each level:
|      | FGL1  | FGL2  | FGL3  |
|------|-------|-------|-------|
| BGL1 | 10421 | 10270 | 9965  |
| BGL2 | 10304 | 10219 | 10056 |
| BGL3 | 9143  | 10125 | 9983  |  

## Download dataset
We provide two scripts for downloading either the full dataset or the foreground only dataset (FGL1_BGL1, FGL2_BGL2, FGL3_BGL3). The foreground only dataset is designed for computer vision tasks that assume a given bounding box during inference, such as keypoint detection and pose estimation.  
1. Clone this repo
2. Run the script to download full dataset:

```
chmod +x download_FG_and_BG.sh
./download_FG_and_BG.sh
```

Or run the script to download foreground only dataset:

```
chmod +x download_FG.sh
./download_FG.sh
```

3. After running the above commands, you should see following folders:  
**images**: contains occluded images.  
**annotations**: annotations for each images.  
**lists**: lists indicate the names of available images.  

## Use the annotations
Inside the annotations folder you find folders named in the format "%sFGL%d_BGL%d" % (cate, fg_occ_lv, bg_occ_lv). In each folder, there are npz files containing the annotations for each individual image.
To load the annotations:

```
import numpy as np

annos = np.load('IMG_NAME.npz', allow_pickle=True)
```

The variable annos will contain the following attributes:
1. 'source': name of the image.  
2. 'occluder_mask': a binary mask indicating the occluder.  
3. 'mask': a binary mask indicating the object.  
4. 'box': the bounding box of the object, in the format \[ y0, y1, x0, x1, img_h, img_h \].  
5. 'occluder_box': a list of bounding boxes of each occluder respectively, in the format \[ \[ y0, y1, x0, x1, img_h, img_h \], \[ y0, y1, x0, x1, img_h, img_h \] ... \].  

## Create your own version of the OccludedPASCAL3D+ dataset
If you are not satisfied with the version we provide, you can also create the dataset using code we provide in the code folder. To create the dataset:
1. Install the BboxTools (a python lib for bounding boxing operations).

```
git clone https://github.com/Angtian/BboxTools.git
python ./BboxTools/setup.py install
```

2. Download the occluder library cropped from the MS-COCO dataset.

```
cd code
chmod +x download_occluder_lib.sh
./download_occluder_lib.sh
```

3. Change the path in CreateOccludedDataset.py and Process_anno.py
4. Run these python scripts:

```
python CreateOccludedDataset.py
python Process_anno.py
```

## Citation
If you find this dataset is useful in your research, please cite:

```
@inproceedings{wang2020robust,
  title={Robust Object Detection Under Occlusion With Context-Aware CompositionalNets},
  author={Wang, Angtian and Sun, Yihong and Kortylewski, Adam and Yuille, Alan L},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12645--12654},
  year={2020}
}
```




