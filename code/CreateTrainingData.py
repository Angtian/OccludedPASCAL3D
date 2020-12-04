import numpy as np
import BboxTools as bbt
import os
import scipy.io
from PIL import Image
import cv2
from CreateOccludedDataset import mix_imgs, mix_masks, apply_n_occluder, get_occ, merge_occ_image, load_one_annotation


if_occ = True

occ_libs_dir = './occluder_libs_train_%s.npz'
occ_libs_name = ['large', 'medium', 'small']

path_save = '../OccludedPASCAL3D_Train/'
path_to_original_pascal3dp = '../../PASCAL3D+/PASCAL3D+_release1.1/'

categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train', 'boat', 'bottle', 'chair', 'diningtable',
              'sofa', 'tvmonitor']
# categories = ['boat', 'bottle', 'chair', 'diningtable', 'sofa', 'tvmonitor']
save_anno_path = path_save + 'annotations_grouped'
save_img_path = path_save + 'images'
save_list_path = path_save + 'lists'

if if_occ:
    sub_name = '%s_occluded'
else:
    sub_name = '%s_raw'

source_list_path = path_to_original_pascal3dp + 'Image_sets/%s_imagenet_train.txt'
source_image_path = path_to_original_pascal3dp + 'Images/%s_imagenet'
source_anno_path = path_to_original_pascal3dp + 'Annotations/%s_imagenet'

# 0: only start randomly, 1: only start in box, 2: using both mode
l_s_thr = 150000
occluding_modes_l = ['s', 'm', 'l', 'lm', 'll', 'ml', 'mm', 'lms', 'lls', 'lmm', 'lll']
start_mode_l =       [1,   2,   2,    0,    2,   1,    1,    0,     1,    0,      0]
occluding_modes_s = ['s', 'm', 'l', 'ms', 'ms', 'mm', 'mms', ]
start_mode_s =       [1,   2,   2,   2,    1,    2,    1, ]

num_per_image = 1

start_off_box_l = [md for md, sd in zip(occluding_modes_l, start_mode_l) if sd == 0 or sd == 2]
start_in_box_l = [md for md, sd in zip(occluding_modes_l, start_mode_l) if sd == 1 or sd == 2]
start_off_box_s = [md for md, sd in zip(occluding_modes_s, start_mode_s) if sd == 0 or sd == 2]
start_in_box_s = [md for md, sd in zip(occluding_modes_s, start_mode_s) if sd == 1 or sd == 2]


def generate_one_img(img, box_anno, occ_libs, seg_anno):
    img_size = img.shape[0] * img.shape[1]
    if img_size > l_s_thr:
        using_start_off_box = start_off_box_l
        using_start_in_box = start_in_box_l
    else:
        using_start_off_box = start_off_box_s
        using_start_in_box = start_in_box_s

    image_out = []
    occluder_mask = []
    occluder_box = []
    for n in range(num_per_image):
        using_idx = np.random.randint(0, len(using_start_off_box) + len(using_start_in_box))

        if using_idx > len(using_start_off_box):
            using_idx -= len(using_start_off_box)

            working_mode = using_start_in_box[using_idx]

            t_boxes, t_masks, t_images = get_occ(working_mode, occ_libs)
            t_boxes, t_process = apply_n_occluder(t_boxes, img_shape=img.shape[0:2], in_box=box_anno)

        else:
            working_mode = using_start_off_box[using_idx]

            t_boxes, t_masks, t_images = get_occ(working_mode, occ_libs)
            t_boxes, t_process = apply_n_occluder(t_boxes, img_shape=img.shape[0:2], in_box=None)

        for i, proc in enumerate(t_process):
            if proc:
                t_masks[i] = proc.apply(t_masks[i])
                t_images[i] = proc.apply(t_images[i])

        mask_map = mix_masks(t_masks, t_boxes)
        occluder_map = mix_imgs(t_masks, t_boxes, t_images)

        image_out.append(merge_occ_image(mask_map, occluder_map, img.copy()))
        occluder_mask.append(mask_map)
        occluder_box.append(t_boxes)

    return image_out, occluder_mask, occluder_box


def generate_dataset(cate, file_list, img_dir, anno_dir, mask_dir, save_img_dir, save_list_dir, save_anno_dir, occ_lib_dir, occ_lib_names, record_file):
    occ_libs = {}
    annotation = {'source': [], 'mask': [], 'box': [], 'occluder_box': [], 'occluder_mask': []}
    img_list_ = ''

    for k in occ_lib_names:
        occ_libs[k] = dict(np.load(occ_lib_dir % k, allow_pickle=True))
        # occ_libs[k] = dict(np.load('tem_lib.npz', allow_pickle=True))
        occ_libs[k]['boxes'] = bbt.bbox_list_from_dump(occ_libs[k]['boxes'])

    save_img_dir = os.path.join(save_img_dir, sub_name % cate)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_list_dir, exist_ok=True)
    os.makedirs(save_anno_dir, exist_ok=True)

    for file_name in file_list:
        print(file_name)
        try:
            # if True:
            anno, flag_ = load_one_annotation(os.path.join(anno_dir, file_name + '.mat'))
            if flag_:
                record_file.write('Skipped %s for multi objects\n' % file_name)
                continue
            img = np.array(Image.open(os.path.join(img_dir, file_name + '.JPEG')))

            box = bbt.from_numpy(anno, image_boundary=img.shape[0:2], sorts=('y0', 'x0', 'y1', 'x1'))
            if if_occ:
                images_, masks_, boxes_ = generate_one_img(img, box, occ_libs, None)
            else:
                images_ = np.array([img])
                masks_ = np.zeros_like(np.array([img]), dtype=bool)
                boxes_ = [[]]

        except:
            print('Unknown Expectations at %s' % file_name)
            record_file.write('Unknown Expectations at %s\n' % file_name)

            continue

        i = 0
        if not num_per_image == 1:
            raise Exception('Currently not support more than one output per image')

        Image.fromarray(images_[i].astype(np.uint8)).save(
            os.path.join(save_img_dir, file_name + '.JPEG'))
        annotation['source'].append(os.path.join(img_dir, file_name + '.JPEG'))
        annotation['occluder_mask'].append(masks_[i])
        annotation['mask'].append(None)
        annotation['box'].append(bbt.dump_bbox_list([box]).ravel())
        annotation['occluder_box'].append(bbt.dump_bbox_list(boxes_[i]))

        img_list_ += file_name + '.JPEG' + '\n'

    name_ = sub_name
    np.savez(os.path.join(save_anno_dir, (name_ % cate) + '.npz'), **annotation)

    with open(os.path.join(save_list_dir, (name_ % cate) + '.txt'), 'w') as file:
            file.write(img_list_)

    return


if __name__ == '__main__':
    for cate in categories:
        print('Start cate: ', cate)
        tem = open('generating_record_%s_1031.txt' % cate, 'w')
        file_list_ = open(source_list_path % cate).readlines()
        file_list_ = [tem.strip('\n') for tem in file_list_]
        source_image_path_ = source_image_path % cate
        source_anno_path_ = source_anno_path % cate
        generate_dataset(cate, file_list_, source_image_path_, source_anno_path_, '', save_img_path, save_list_path,
                         save_anno_path, occ_libs_dir, occ_libs_name, tem)
        tem.close()

