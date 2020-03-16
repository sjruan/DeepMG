import argparse
from PIL import Image
from skimage.morphology import skeletonize
import numpy as np
import cv2
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', help='the configuration file of the dateset')
    parser.add_argument('--tile_path', help='the directory of predicted tiles')
    parser.add_argument('--results_path', help='the path to results')

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.results_path, exist_ok=True)
    with open(opt.conf_path, 'r') as f:
        conf = json.load(f)
    tile_height, tile_width = conf['feature_extraction']['tile_pixel_size'], \
                              conf['feature_extraction']['tile_pixel_size']
    row_min, row_max, col_min, col_max = conf['feature_extraction']['test_tile_row_min'], \
                                         conf['feature_extraction']['test_tile_row_max'], \
                                         conf['feature_extraction']['test_tile_col_min'], \
                                         conf['feature_extraction']['test_tile_col_max']

    img_tmp = '{}_{}_pred_rn_img.png'
    pred_eval_region_data = np.zeros(((row_max - row_min) * tile_height, (col_max - col_min) * tile_width),
                                     dtype=np.uint8)
    for i in range(row_min, row_max):
        for j in range(col_min, col_max):
            filename = opt.tile_path + img_tmp.format(i, j)
            tile_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            sl = (slice((i - row_min) * tile_height, (i + 1 - row_min) * tile_height),
                  slice((j - col_min) * tile_width, (j + 1 - col_min) * tile_width))
            pred_eval_region_data[sl] = tile_data
    cv2.imwrite(opt.results_path + 'pred.png', pred_eval_region_data)

    inp = cv2.imread(opt.results_path + 'pred.png', cv2.IMREAD_GRAYSCALE)
    inp[inp > 0] = 1
    skeleton = skeletonize(inp)
    skeleton[skeleton > 0] = 255
    skeleton = Image.fromarray(skeleton).convert('RGB')
    skeleton.save(opt.results_path + 'pred_thinned.png')
