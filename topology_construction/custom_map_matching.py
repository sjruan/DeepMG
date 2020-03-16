import os
from tptk.map_matching.hmm.hmm_map_matcher import TIHMMMapMatcher
from tptk.common.trajectory import parse_traj_file
from tptk.common.path import store_path_file
from tptk.common.road_network import load_rn_shp


class CustomMapMatching:
    def __init__(self, rn_path, alpha):
        self.rn_path = rn_path
        self.alpha = alpha

    def execute(self, filename, traj_path, mm_result_path):
        rn = load_rn_shp(self.rn_path, is_directed=True)
        for u, v, data in rn.edges(data=True):
            if data['type'] == 'virtual':
                data['weight'] = data['length'] * self.alpha
            else:
                data['weight'] = data['length']
        map_matcher = TIHMMMapMatcher(rn, routing_weight='weight')
        traj_list = parse_traj_file(os.path.join(traj_path, filename))
        all_paths = []
        for traj in traj_list:
            paths = map_matcher.match_to_path(traj)
            all_paths.extend(paths)
        store_path_file(all_paths, os.path.join(mm_result_path, filename))
