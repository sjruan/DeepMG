import os
import sys
sys.path.append('../tptk/')
from tptk.common.road_network import load_rn_shp
from tptk.common.trajectory import parse_traj_file
from tptk.common.grid import Grid
from tptk.common.mbr import MBR
from tptk.common.spatial_func import distance, bearing, LAT_PER_METER
import cv2
from tqdm import tqdm
import numpy as np
import json
import shutil


def generate_point_image(traj_dir, grid_idx, feature_path):
    pt_cnt = np.zeros((grid_idx.row_num, grid_idx.col_num))
    for filename in tqdm(os.listdir(traj_dir)):
        if not filename.endswith('.txt'):
            continue
        trajs = parse_traj_file(os.path.join(traj_dir, filename))
        for traj in trajs:
            for cur_pt in traj.pt_list:
                try:
                    row_idx, col_idx = grid_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                    pt_cnt[row_idx, col_idx] += 1
                except IndexError:
                    continue
    pt_cnt = pt_cnt / 2 * 255
    pt_cnt[pt_cnt > 255] = 255
    cv2.imwrite(os.path.join(feature_path, 'point.png'), pt_cnt)


def generate_line_image(traj_dir, grid_idx, feature_path):
    MIN_DISTANCE_IN_METER = 5
    MAX_DISTANCE_IN_METER = 300
    traj_line_img = np.zeros((grid_idx.row_num, grid_idx.col_num), dtype=np.uint8)
    for filename in tqdm(os.listdir(traj_dir)):
        if not filename.endswith('.txt'):
            continue
        traj_list = parse_traj_file(os.path.join(traj_dir, filename))
        for traj in traj_list:
            one_traj_line_img = np.zeros((grid_idx.row_num, grid_idx.col_num), dtype=np.uint8)
            for j in range(len(traj.pt_list) - 1):
                cur_pt, next_pt = traj.pt_list[j], traj.pt_list[j + 1]
                if MIN_DISTANCE_IN_METER < distance(cur_pt, next_pt) < MAX_DISTANCE_IN_METER:
                    try:
                        y1, x1 = grid_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                        y2, x2 = grid_idx.get_matrix_idx(next_pt.lat, next_pt.lng)
                        cv2.line(one_traj_line_img, (x1, y1), (x2, y2), 16, 1, lineType=cv2.LINE_AA)
                    except IndexError:
                        continue
            traj_line_img = cv2.add(traj_line_img, one_traj_line_img)
    cv2.imwrite(os.path.join(feature_path, 'line.png'), traj_line_img)


def generate_speed_data(traj_dir, grid_idx, feature_path):
    MIN_DISTANCE_IN_METER = 5
    MAX_DISTANCE_IN_METER = 300
    speed_data = np.zeros((grid_idx.row_num, grid_idx.col_num, 1), dtype=np.float)
    cnt_data = np.zeros((grid_idx.row_num, grid_idx.col_num, 1), dtype=np.float)
    for filename in tqdm(os.listdir(traj_dir)):
        if not filename.endswith('.txt'):
            continue
        traj_list = parse_traj_file(os.path.join(traj_dir, filename))
        for traj in traj_list:
            for i in range(len(traj.pt_list) - 1):
                cur_pt, next_pt = traj.pt_list[i], traj.pt_list[i + 1]
                delta_time = (next_pt.time - cur_pt.time).total_seconds()
                if MIN_DISTANCE_IN_METER < distance(cur_pt, next_pt) < MAX_DISTANCE_IN_METER:
                    try:
                        row_idx, col_idx = grid_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                        speed = distance(next_pt, cur_pt) / delta_time
                        # 120 km/h
                        if speed > 34:
                            continue
                        speed_data[row_idx, col_idx, 0] += speed
                        cnt_data[row_idx, col_idx, 0] += 1
                    except IndexError:
                        continue
    speed_data = np.divide(speed_data, cnt_data, out=np.zeros_like(speed_data), where=cnt_data != 0)
    np.save(os.path.join(feature_path, 'speed.npy'), speed_data)


def generate_dir_dist_data(traj_dir, grid_idx, feature_path):
    MIN_DISTANCE_IN_METER = 5
    MAX_DISTANCE_IN_METER = 300
    dir_data = np.zeros((grid_idx.row_num, grid_idx.col_num, 8), dtype=np.uint8)
    for filename in tqdm(os.listdir(traj_dir)):
        if not filename.endswith('.txt'):
            continue
        traj_list = parse_traj_file(os.path.join(traj_dir, filename))
        for traj in traj_list:
            for i in range(len(traj.pt_list) - 1):
                cur_pt, next_pt = traj.pt_list[i], traj.pt_list[i+1]
                if MIN_DISTANCE_IN_METER < distance(cur_pt, next_pt) < MAX_DISTANCE_IN_METER:
                    try:
                        row_idx, col_idx = grid_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                        direction = int(((bearing(cur_pt, next_pt) + 22.5) % 360) // 45)
                        dir_data[row_idx, col_idx, direction] += 1
                    except IndexError:
                        continue
    np.save(os.path.join(feature_path, 'direction.npy'), dir_data)


def generate_spatial_view(traj_dir, grid_idx, feature_path):
    generate_point_image(traj_dir, grid_idx, feature_path)
    generate_line_image(traj_dir, grid_idx, feature_path)
    generate_speed_data(traj_dir, grid_idx, feature_path)
    generate_dir_dist_data(traj_dir, grid_idx, feature_path)


def generate_transition_view(traj_dir, grid_idx, nbhd_size, nbhd_dist, feature_path):
    MIN_DISTANCE_IN_METER = 5
    MAX_DISTANCE_IN_METER = 300
    meters_per_grid = grid_idx.lat_interval / LAT_PER_METER
    radius = int(nbhd_dist / meters_per_grid)
    transit_data = np.zeros((grid_idx.row_num, grid_idx.col_num, nbhd_size, nbhd_size, 2),
                            dtype=np.uint8)
    for filename in tqdm(os.listdir(traj_dir)):
        if not filename.endswith('.txt'):
            continue
        traj_list = parse_traj_file(os.path.join(traj_dir, filename))
        for traj in traj_list:
            for idx in range(len(traj.pt_list) - 1):
                cur_pt = traj.pt_list[idx]
                next_pt = traj.pt_list[idx + 1]
                if MIN_DISTANCE_IN_METER < distance(cur_pt, next_pt) < MAX_DISTANCE_IN_METER:
                    try:
                        global_cur_i, global_cur_j = grid_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                        local_idx = get_local_idx(global_cur_i, global_cur_j, radius, grid_idx, nbhd_dist)
                        local_next_i, local_next_j = local_idx.get_matrix_idx(next_pt.lat, next_pt.lng)
                        transit_data[global_cur_i, global_cur_j, local_next_i, local_next_j, 0] = 1

                        global_next_i, global_next_j = grid_idx.get_matrix_idx(next_pt.lat, next_pt.lng)
                        local_idx = get_local_idx(global_next_i, global_next_j, radius, grid_idx, nbhd_dist)
                        local_cur_i, local_cur_j = local_idx.get_matrix_idx(cur_pt.lat, cur_pt.lng)
                        transit_data[global_next_i, global_next_j, local_cur_i, local_cur_j, 1] = 1
                    except IndexError:
                        continue
    np.save(os.path.join(feature_path, 'transition.npy'), transit_data)


def get_local_idx(i, j, radius, grid_idx, target_region_size):
    min_i = i - radius
    max_i = i + radius
    min_j = j - radius
    max_j = j + radius
    local_lower_left_mbr = grid_idx.get_mbr_by_matrix_idx(max_i, min_j)
    local_upper_right_mbr = grid_idx.get_mbr_by_matrix_idx(min_i, max_j)
    local_mbr = MBR(local_lower_left_mbr.min_lat, local_lower_left_mbr.min_lng,
                    local_upper_right_mbr.max_lat, local_upper_right_mbr.max_lng)
    local_idx = Grid(local_mbr, target_region_size, target_region_size)
    return local_idx


def generate_features(traj_dir, grid_idx, nbhd_size, nbhd_dist, feature_path):
    os.makedirs(feature_path, exist_ok=True)
    generate_spatial_view(traj_dir, grid_idx, feature_path)
    generate_transition_view(traj_dir, grid_idx, nbhd_size, nbhd_dist, feature_path)


def generate_road_centerline_label(rn_path, grid_idx, label_path):
    rn = load_rn_shp(rn_path)
    centerline_img = np.zeros((grid_idx.row_num, grid_idx.col_num), dtype=np.uint8)
    for edge_key in tqdm(rn.edges):
        coords = rn.edges[edge_key]['coords']
        for i in range(len(coords)-1):
            start_node, end_node = coords[i], coords[i+1]
            try:
                y1, x1 = grid_idx.get_matrix_idx(start_node.lat, start_node.lng)
                y2, x2 = grid_idx.get_matrix_idx(end_node.lat, end_node.lng)
                cv2.line(centerline_img, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_8)
            except IndexError:
                continue
    cv2.imwrite(os.path.join(label_path, 'centerline.png'), centerline_img)


def generate_road_region_label(centerline_path, radius, label_path):
    centerline_img = cv2.imread(centerline_path, cv2.IMREAD_GRAYSCALE)
    centerline_pixels = np.where(centerline_img == 255)
    H, W = centerline_img.shape
    region_img = np.zeros(centerline_img.shape, dtype=np.uint8)
    for i, j in tqdm(list(zip(centerline_pixels[0], centerline_pixels[1]))):
        for y in range(max(i-radius, 0), min(i+radius+1, H)):
            for x in range(max(j-radius, 0), min(j+radius+1, W)):
                region_img[y, x] = 255
    cv2.imwrite(os.path.join(label_path, 'region.png'), region_img)


def generate_labels(rn_path, grid_idx, label_path):
    os.makedirs(label_path, exist_ok=True)
    generate_road_centerline_label(rn_path, grid_idx, label_path)
    generate_road_region_label(label_path + 'centerline.png', 2, label_path)


def generate_samples(feature_path, label_path, grid_idx, tile_pixel_size, dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    point = cv2.imread(os.path.join(feature_path, 'point.png'))
    line = cv2.imread(os.path.join(feature_path, 'line.png'))
    speed = np.load(os.path.join(feature_path, 'speed.npy'))
    direction = np.load(os.path.join(feature_path, 'direction.npy'))
    transition = np.load(os.path.join(feature_path, 'transition.npy'))
    centerline = cv2.imread(os.path.join(label_path, 'centerline.png'))
    region = cv2.imread(os.path.join(label_path, 'region.png'))
    for i in tqdm(range(grid_idx.row_num // tile_pixel_size)):
        for j in range(grid_idx.col_num // tile_pixel_size):
            slices = (slice(i * tile_pixel_size, (i + 1) * tile_pixel_size),
                      slice(j * tile_pixel_size, (j + 1) * tile_pixel_size))
            image_sample = np.concatenate((point[slices], line[slices], centerline[slices], region[slices]), axis=1)
            cv2.imwrite(os.path.join(dataset_path, '{}_{}.png'.format(i, j)), image_sample)
            np.save(os.path.join(dataset_path, '{}_{}_speed.npy'.format(i, j)), speed[slices])
            np.save(os.path.join(dataset_path, '{}_{}_direction.npy'.format(i, j)), direction[slices])
            np.save(os.path.join(dataset_path, '{}_{}_transition.npy'.format(i, j)), transition[slices])


def split_train_val_test(dataset_path, test_row_min, test_row_max, test_col_min, test_col_max, learning_path):
    test_tiles = set()
    for row in range(test_row_min, test_row_max):
        for col in range(test_col_min, test_col_max):
            test_tiles.add('{}_{}'.format(row, col))
    samples = set([name[:-4] for name in os.listdir(dataset_path) if name.endswith('.png')])
    train_val_samples = samples - test_tiles
    val_split = 0.1
    train_val_samples = list(train_val_samples)
    nb_samples = len(train_val_samples)
    seed = 2017
    idxes = np.random.RandomState(seed=seed).permutation(nb_samples)
    train_split = 1 - val_split
    train_size = int(nb_samples * train_split)
    train_idxes = idxes[:train_size]
    val_idxes = idxes[train_size:]
    # create train set
    train_path = os.path.join(learning_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    for train_idx in train_idxes:
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[train_idx]] + '.png'), train_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[train_idx]] + '_direction.npy'), train_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[train_idx]] + '_speed.npy'), train_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[train_idx]] + '_transition.npy'), train_path)
    # create val set
    val_path = os.path.join(learning_path, 'val')
    os.makedirs(val_path, exist_ok=True)
    for val_idx in val_idxes:
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[val_idx]] + '.png'), val_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[val_idx]] + '_direction.npy'), val_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[val_idx]] + '_speed.npy'), val_path)
        shutil.move(os.path.join(dataset_path, train_val_samples[idxes[val_idx]] + '_transition.npy'), val_path)
    # create test set
    test_path = os.path.join(learning_path, 'test')
    os.makedirs(test_path, exist_ok=True)
    for test_tile in test_tiles:
        shutil.move(os.path.join(dataset_path, test_tile + '.png'), test_path)
        shutil.move(os.path.join(dataset_path, test_tile + '_direction.npy'), test_path)
        shutil.move(os.path.join(dataset_path, test_tile + '_speed.npy'), test_path)
        shutil.move(os.path.join(dataset_path, test_tile + '_transition.npy'), test_path)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        conf = json.load(f)
    traj_dir = '../data/{}/traj/'.format(conf['dataset']['dataset_name'])
    rn_path = '../data/{}/rn/'.format(conf['dataset']['dataset_name'])
    feature_path = '../data/{}/feature/'.format(conf['dataset']['dataset_name'])
    label_path = '../data/{}/label/'.format(conf['dataset']['dataset_name'])
    dataset_path = '../data/{}/dataset/'.format(conf['dataset']['dataset_name'])
    learning_path = '../data/{}/learning/'.format(conf['dataset']['dataset_name'])
    mbr = MBR(conf['dataset']['min_lat'], conf['dataset']['min_lng'],
              conf['dataset']['max_lat'], conf['dataset']['max_lng'])
    grid_idx = Grid(mbr, conf['dataset']['nb_rows'], conf['dataset']['nb_cols'])
    generate_features(traj_dir, grid_idx, conf['feature_extraction']['nbhd_size'],
                      conf['feature_extraction']['nbhd_dist'], feature_path)
    generate_labels(rn_path, grid_idx, label_path)
    generate_samples(feature_path, label_path, grid_idx, conf['feature_extraction']['tile_pixel_size'], dataset_path)
    split_train_val_test(dataset_path, conf['feature_extraction']['test_tile_row_min'],
                         conf['feature_extraction']['test_tile_row_max'],
                         conf['feature_extraction']['test_tile_col_min'],
                         conf['feature_extraction']['test_tile_col_max'], learning_path)
