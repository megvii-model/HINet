import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reds():
    folder_path = './datasets/REDS/val/sharp_300'
    lmdb_path = './datasets/REDS/val/sharp_300.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    folder_path = './datasets/REDS/val/blur_300'
    lmdb_path = './datasets/REDS/val/blur_300.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_gopro():
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/test/target'
    lmdb_path = './datasets/GoPro/test/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/test/input'
    lmdb_path = './datasets/GoPro/test/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

