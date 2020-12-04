import numpy as np
import BboxTools as bbt
import os
import scipy.io
from PIL import Image
import cv2


categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train', 'boat', 'bottle', 'chair', 'diningtable',
              'sofa', 'tvmonitor']

path_save = '../OccludedPASCAL3D/'
path_to_original_pascal3dp = '../../PASCAL3D+/PASCAL3D+_release1.1/'

save_anno_path = path_save + 'annotations_grouped'
save_img_path = path_save + 'images'
save_list_path = path_save + 'lists'

source_list_path = path_to_original_pascal3dp + 'Image_sets/%s_imagenet_val.txt'
source_image_path = path_to_original_pascal3dp + 'Images/%s_imagenet'
source_anno_path = path_to_original_pascal3dp + 'Annotations/%s_imagenet'
source_mask_path = path_to_original_pascal3dp + 'obj_mask/%s'


folder_name_list = ['%sFGL0_BGL0']


def load_one_annotation(anno_path):
    a = scipy.io.loadmat(anno_path)
    bbox_ = a['record'][0][0][1][0][0][1][0]
    num_obj = len(a['record'][0][0][1][0])
    return bbox_, num_obj != 1


def generate_dataset(cate, file_list, img_dir, anno_dir, mask_dir, save_img_dir, save_list_dir, save_anno_dir, occ_lib_dir,
                     occ_lib_names, record_file):

    annotations = [{'source': [], 'mask': [], 'box': [], 'occluder_box': [], 'occluder_mask': []} for _ in range(len(folder_name_list))]
    img_list_ = ['' for _ in range(len(folder_name_list))]

    save_img_dir_list = [os.path.join(save_img_dir, folder_name % cate) for folder_name in folder_name_list]
    for folder_name in save_img_dir_list:
        os.makedirs(folder_name, exist_ok=True)
    os.makedirs(save_list_dir, exist_ok=True)
    os.makedirs(save_anno_dir, exist_ok=True)

    for file_name in file_list:
        print(file_name)
        try:
            anno, flag_ = load_one_annotation(os.path.join(anno_dir, file_name + '.mat'))
            if flag_:
                record_file.write('Skipped %s for multi objects\n' % file_name)
                continue
            img = np.array(Image.open(os.path.join(img_dir, file_name + '.JPEG')))
            mask = np.array(Image.open(os.path.join(mask_dir, file_name + '.JPEG')))

            if not mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            box = bbt.from_numpy(anno, image_boundary=img.shape[0:2], sorts=('y0', 'x0', 'y1', 'x1'))
            filled_ = np.array([True])
            images_ = np.array([img])
            masks_ = np.zeros_like(np.array([img]), dtype=bool)
            boxes_ = [[]]

        except:
            print('Unknown Expectations at %s' % file_name)
            record_file.write('Unknown Expectations at %s\n' % file_name)

            continue
        if not np.all(filled_):
            record_file.write('Unfill %s: ' % file_name)

        for i in range(filled_.size):
            if filled_[i]:
                Image.fromarray(images_[i].astype(np.uint8)).save(
                    os.path.join(save_img_dir_list[i], file_name + '.JPEG'))
                annotations[i]['source'].append(os.path.join(img_dir, file_name + '.JPEG'))
                annotations[i]['occluder_mask'].append(masks_[i])
                annotations[i]['mask'].append(mask)
                annotations[i]['box'].append(bbt.list_box_to_numpy([box], save_image_boundary=True).ravel())
                annotations[i]['occluder_box'].append(bbt.list_box_to_numpy(boxes_[i], save_image_boundary=True))

                img_list_[i] += file_name + '.JPEG' + '\n'
            else:
                record_file.write(' %d' % i)

        if not np.all(filled_):
            record_file.write('\n')

    for name_, anno_ in zip(folder_name_list, annotations):
        np.savez(os.path.join(save_anno_dir, (name_ % cate) + '.npz'), **anno_)

    for name_, list_ in zip(folder_name_list, img_list_):
        with open(os.path.join(save_list_dir, (name_ % cate) + '.txt'), 'w') as file:
            file.write(list_)

    return


if __name__ == '__main__':
    for cate in categories:
        print('Start cate: ', cate)
        tem = open('generating_record_%s_1030.txt' % cate, 'w')
        file_list_ = open(source_list_path % cate).readlines()
        file_list_ = [tem.strip('\n') for tem in file_list_]
        source_image_path_ = source_image_path % cate
        source_anno_path_ = source_anno_path % cate
        source_mask_path_ = source_mask_path % cate
        generate_dataset(cate, file_list_, source_image_path_, source_anno_path_, source_mask_path_, save_img_path, save_list_path,
                         save_anno_path, '', '', tem)
        tem.close()
