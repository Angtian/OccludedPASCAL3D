# OccludedPASCAL3D+
The OccludedPASCAL3D+ is a dataset design for evaluate robustness of algorithms for detection, keypoint detection and pose estimation. 
The OccludedPASCAL3D+ is built via superimpose occluder cropped from MSCOCO dataset to PASCAL3D+ dataset. Specifically, we only use ImageNet subset in PASCAL3D+, which have 10812 testing images.   

The OccludedPASCAL3D+ has totally 9 occlusion levels, which 3 foreground occlusion levels and 3 background occlusion levels.  
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

