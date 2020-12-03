import numpy as np
import os
import time
source_anno_path = '../PASCAL3D+/PASCAL_NEWx/annotations_grouped'
save_anno_path = '../PASCAL3D+/PASCAL_NEWx/annotations'

categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train', 'boat', 'bottle', 'chair', 'diningtable',
              'sofa', 'tvmonitor']

# categories = ['bicycle']
# categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']
# folder_name_list = ['%sFGL0_BGL0'] + ['%sFGL%d_BGL%d' % ('%s', j, i) for i in range(1, 4) for j in range(1, 4)]
folder_name_list = ['%sFGL%d_BGL%d' % ('%s', j, i) for i in range(1, 4) for j in range(1, 4)]
# folder_name_list = ['%sFGL0_BGL0']
# folder_name_list = ['%s_raw', '%s_occluded']

for cate in categories:
    for name_ in folder_name_list:
        print(name_ % cate)
        this_folder = os.path.join(save_anno_path, name_) % cate
        os.makedirs(this_folder, exist_ok=True)
        data = np.load(os.path.join(source_anno_path, (name_ % cate) + '.npz'), allow_pickle=True)

        source_list = data['source']
        mask_list = data['mask']
        box_list = data['box']
        occluder_box_list = data['occluder_box']
        occluder_mask = data['occluder_mask']

        for i in range(data['mask'].size):
            this_name = source_list[i].split('/')[-1].split('.')[0]
            np.savez(os.path.join(this_folder, this_name + '.npz'), source=source_list[i], mask=mask_list[i], box=box_list[i], occluder_mask=occluder_mask[i], occluder_box=occluder_box_list[i], category=cate, occluder_level=name_.strip('%s')[1])
