# OccludedPASCAL3D+
The OccludedPASCAL3D+ is a dataset design for evaluate robustness of algorithms for detection, keypoint detection and pose estimation. 
The OccludedPASCAL3D+ is built via superimpose occluder cropped from MSCOCO dataset to PASCAL3D+ dataset. Specifically, we only use ImageNet subset in PASCAL3D+, which have 10812 testing images.  
![Figure of Car in OccludedPASCAL3D+ in 9 occlussion levels](https://github.com/Angtian/OccludedPASCAL3D/blob/master/Example.JPEG)

The OccludedPASCAL3D+ has totally 9 occlusion levels, which 3 foreground occlusion levels and 3 background occlusion levels. Note the occlusion ratio is measured via occluded pixels on the object mask. 
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
We provide two scripts to download full dataset or foreground only dataset (FGL1_BGL1, FGL2_BGL2, FGL3_BGL3). The foreground only dataset is designed for tasks that given bounding box during inference such as keypoint detection and pose estimation.  
1. Clone this repo
2. Run the script to download full dataset:
'''
chmod +x download_FG_and_BG.sh
./download_FG_and_BG.sh
'''
Or run the script to download foreground only dataset:
'''
chmod +x download_FG.sh
./download_FG.sh
'''
3. After run above commands, you should see following folders:  
**images**: contains occlude images.  
**annotations**: annotations for each images.
**lists**: lists indicate names of available images.

## Use the annotations
Inside the annotations folder you could see folders named in format "%sFGL%d_BGL%d" % (cate, fg_occ_lv, bg_occ_lv). In each folder, there are npz files contains annotations for each individual image.
To load the annotations:
'''
import numpy as np

annos = np.load('IMG_NAME.npz', allow_pickle=True)
'''
The annos will contain following attributes:
1. 'source': name of the image.  
2. 'occluder_mask': a binary mask indicates occluder.  
3. 'mask': a binary mask indicates the object.  
4. 'box': the bounding box of the object, in format \[ y0, y1, x0, x1, img_h, img_h \].  
5. 'occluder_box': a list of bounding boxes of the each occluder respectively, in format \[ \[ y0, y1, x0, x1, img_h, img_h \], \[ y0, y1, x0, x1, img_h, img_h \] ... \].  

## Create you own version of OccludedPASCAL3D+ dataset
If you are not satisfied with the version we provide, you can also create the dataset using code we provide in the code folder. To create the dataset:
1. Install the BboxTools (a python lib for bounding boxing operations).
'''
git clone https://github.com/Angtian/BboxTools.git
python ./BboxTools/setup.py install
'''
2. Download the occluder lib cropped from MSCOCO dataset (you can create your own lib, but unfortunable we lose the code for cropping occluder from MSCOCO).
'''
cd code
chmod +x download_occluder_lib.sh
./download_occluder_lib.sh
'''
3. Change the path in CreateOccludedDataset.py and Process_anno.py
4. Run these python scripts:
'''
python CreateOccludedDataset.py
python Process_anno.py
'''

## Citation
If you find this dataset is useful in your research, please cite:
'''
@inproceedings{wang2020robust,
  title={Robust Object Detection Under Occlusion With Context-Aware CompositionalNets},
  author={Wang, Angtian and Sun, Yihong and Kortylewski, Adam and Yuille, Alan L},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12645--12654},
  year={2020}
}
'''




