from tptk.common.road_network import store_rn_shp
from tptk.common.path import parse_path_file
import os
import copy


class MapRefiner:
    def __init__(self, min_sup):
        self.min_sup = min_sup
        self.pred_min_sup = 1

    def refine(self, linked_rn, mm_traj_path, final_rn_path):
        edge2cnt = self.get_edge2cnt(linked_rn, mm_traj_path)
        edges_to_del = []
        for u, v, type in linked_rn.edges(data='type'):
            if type == 'pred':
                if (u, v) not in edge2cnt or edge2cnt[(u, v)] < self.pred_min_sup:
                    edges_to_del.append((u, v))
            elif type == 'virtual':
                if (u, v) not in edge2cnt or edge2cnt[(u, v)] < self.min_sup:
                    edges_to_del.append((u, v))
        final_rn = copy.deepcopy(linked_rn)
        print('edges&links to delete:{}'.format(len(edges_to_del)))
        for u, v in edges_to_del:
            final_rn.remove_edge(u, v)
        self.final_refine(final_rn)
        store_rn_shp(final_rn, final_rn_path)

    def get_edge2cnt(self, linked_rn, mm_result_path):
        edge2cnt = {}
        for filename in (os.listdir(mm_result_path)):
            paths = parse_path_file(os.path.join(mm_result_path, filename))
            for path in paths:
                for path_entity in path.path_entities:
                    edge = linked_rn.edge_idx[path_entity.eid]
                    if edge not in edge2cnt:
                        edge2cnt[edge] = 1
                    else:
                        edge2cnt[edge] += 1
        return edge2cnt

    def final_refine(self, rn):
        # make sure each vertex will not exceed 2 virtual links
        rn_undir = rn.to_undirected()
        link_cnt = 0
        links_to_delete = []
        for u, v, data in rn_undir.edges(data=True):
            if data['type'] != 'virtual':
                continue
            link_cnt += 1
            candi_links = []
            for x in rn_undir[u]:
                if rn_undir[x][u]['type'] == 'virtual':
                    candi_links.append((x, u))
            if len(candi_links) > 2:
                candi_links = sorted(candi_links, key=lambda k: rn_undir[k[0]][k[1]]['length'])
                links_to_delete.extend(candi_links[2:])
            candi_links = []
            for y in rn_undir[v]:
                if rn_undir[y][v]['type'] == 'virtual':
                    candi_links.append((y, v))
            if len(candi_links) > 2:
                candi_links = sorted(candi_links, key=lambda k: rn_undir[k[0]][k[1]]['length'])
                links_to_delete.extend(candi_links[2:])
        print('final links to delete:{}'.format(len(links_to_delete)))
        for link in links_to_delete:
            if rn.has_edge(link[0], link[1]):
                rn.remove_edge(link[0], link[1])
            if rn.has_edge(link[1], link[0]):
                rn.remove_edge(link[1], link[0])
