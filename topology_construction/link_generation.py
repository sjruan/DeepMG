import copy
from tptk.common.spatial_func import LAT_PER_METER, LNG_PER_METER, SPoint, distance, project_pt_to_line, cal_loc_along_line, angle
from tptk.common.mbr import MBR
from tptk.common.road_network import store_rn_shp
from topology_construction.topo_utils import line_ray_intersection_test, is_line_line_intersected, angle_between
import numpy as np


class VirtualLink:
    def __init__(self, end_node, target_segment, split_edge_idx, split_edge_offset):
        self.end_node = end_node
        self.target_segment = target_segment
        self.split_edge_idx = split_edge_idx
        self.split_edge_offset = split_edge_offset

    def __repr__(self):
        return '(index:{},offset:{})'.format(self.split_edge_idx, self.split_edge_offset)


class LinkGenerator:
    def __init__(self, radius):
        self.radius = radius
        self.NO_NEW_VERTEX_OFFSET = 15
        self.SIMILAR_DIRECTION_THRESHOLD = 20

    def generate_pt_to_link(self, extracted_rn, last_edge_of_dead_end, target_coords, dead_segment, target_segment):
        """
        
        :param extracted_rn: 
        :param last_edge_of_dead_end: the last edge of the road segment containing the dead end
        :param target_coords: the coords of the target segment
        :param dead_segment: key
        :param target_segment: key
        :return: 
        """
        f, o = last_edge_of_dead_end
        opposite_of_o = dead_segment[1] if dead_segment[0][0] == o.lng and dead_segment[0][1] == o.lat else dead_segment[0]
        opposite_of_o = SPoint(opposite_of_o[1], opposite_of_o[0])
        # short isolated segment, the direction might be unreliable, we add the perpendicular edge to the neighborhood
        if extracted_rn.edges[dead_segment]['length'] < self.NO_NEW_VERTEX_OFFSET and \
                extracted_rn.degree(dead_segment[0]) == 1 and extracted_rn.degree(dead_segment[1]) == 1:
            return self.perpendicular_intersection(o, target_coords, extracted_rn, opposite_of_o, target_segment)
        return self.extension_intersection(o, f, target_coords)

    def cal_projection(self, pt, target_coords):
        split_edge_idx = float('inf')
        split_edge_offset = float('inf')
        min_dist = float('inf')
        candidates = [project_pt_to_line(target_coords[i], target_coords[i + 1], pt) for i in
                      range(len(target_coords) - 1)]
        if len(candidates) > 0:
            for i in range(len(candidates)):
                if candidates[i][1] <= 0.0 or candidates[i][1] >= 1.0:
                    continue
                if candidates[i][2] < min_dist and candidates[i][2] < self.radius:
                    min_dist = candidates[i][2]
                    split_edge_idx = i
                    split_edge_offset = candidates[i][1]
        return min_dist, split_edge_idx, split_edge_offset

    def extension_intersection(self, o, f, target_coords):
        min_dist = float('inf')
        split_edge_idx = float('inf')
        split_edge_offset = float('inf')
        # check whether internal edge has intersection (if multiple intersections, select the shortest)
        for i in range(0, len(target_coords) - 1):
            a = target_coords[i]
            b = target_coords[i + 1]
            result = line_ray_intersection_test(o, f, a, b)
            if result is None or result[0] < 0 or result[0] > 1:
                continue
            else:
                dist_tmp = distance(o, result[1])
                if dist_tmp < self.radius and dist_tmp < min_dist:
                    nearest_node_with_offset = (target_coords[i], 0.0) if result[0] < 0.5 else \
                        (target_coords[i + 1], 1.0)
                    min_dist = dist_tmp
                    split_edge_idx = i
                    # prefer link to existing nodes if too short
                    if distance(nearest_node_with_offset[0], result[1]) < self.NO_NEW_VERTEX_OFFSET:
                        split_edge_offset = nearest_node_with_offset[1]
                    else:
                        split_edge_offset = result[0]
        # doesn't have internal intersection, check whether has smooth transition
        if split_edge_idx == float('inf'):
            # check start node
            tmp_dist = distance(target_coords[0], o)
            if tmp_dist < self.radius and \
                    angle_between((o.lng - f.lng, o.lat - f.lat),
                                  (target_coords[0].lng - o.lng, target_coords[0].lat - o.lat)) <= 0.5 * np.pi:
                min_dist = tmp_dist
                split_edge_idx = 0
                split_edge_offset = 0.0
            # check end node
            tmp_dist = distance(target_coords[-1], o)
            if tmp_dist < self.radius and \
                    angle_between((o.lng - f.lng, o.lat - f.lat),
                                  (target_coords[-1].lng - o.lng, target_coords[-1].lat - o.lat)) <= 0.5 * np.pi:
                if tmp_dist < min_dist:
                    split_edge_idx = len(target_coords) - 2
                    split_edge_offset = 1.0
        return split_edge_idx, split_edge_offset

    def perpendicular_intersection(self, o, target_coords, rn, opposite_of_o, target_segment):
        split_edge_idx = float('inf')
        split_edge_offset = float('inf')
        o_min_dist, o_split_edge_idx, o_split_edge_offset = self.cal_projection(o, target_coords)
        other_min_dist, other_split_edge_idx, other_split_edge_offset = self.cal_projection(opposite_of_o,
                                                                                            target_coords)
        if o_min_dist < other_min_dist:
            split_edge_idx = o_split_edge_idx
            split_edge_offset = o_split_edge_offset
        # no internal intersection
        if split_edge_idx == float('inf'):
            # the target segment is also a short isolated one
            if rn.edges[target_segment]['length'] < self.NO_NEW_VERTEX_OFFSET and \
                    rn.degree(target_segment[0]) == 1 and rn.degree(target_segment[1]) == 1:
                a = SPoint(target_segment[0][1], target_segment[0][0])
                b = SPoint(target_segment[1][1], target_segment[1][0])
                dead_end_to_target_dist = min(distance(o, a), distance(o, b))
                opposite_to_target_dist = min(distance(opposite_of_o, a), distance(opposite_of_o, b))
                # current dead end is shorter to the target than opposite, link with the nearest vertex
                if dead_end_to_target_dist < opposite_to_target_dist and dead_end_to_target_dist < self.radius:
                    target = a if distance(o, a) < distance(o, b) else b
                    if target == target_coords[0]:
                        split_edge_idx = 0
                        split_edge_offset = 0.0
                    else:
                        split_edge_idx = len(target_coords) - 2
                        split_edge_offset = 1.0
        return split_edge_idx, split_edge_offset

    def divide_segment(self, ori_coords, virtual_links):
        """
        :param ori_coords:
        :param virtual_links:
        :return: list of edges to add (start, end, coords)
        """
        splitted_segments = []
        link_segments = []
        # aggregate by edge_idx, and offset, i.e., new nodes
        loc2virtual_links = {}
        for virtual_link in virtual_links:
            split_edge_idx = virtual_link.split_edge_idx
            split_edge_offset = virtual_link.split_edge_offset
            # (big_edge_index, 0.0):will not appear, distance is only updated if next edge distance is smaller
            if (split_edge_idx, split_edge_offset) not in loc2virtual_links:
                loc2virtual_links[(split_edge_idx, split_edge_offset)] = []
            loc2virtual_links[(split_edge_idx, split_edge_offset)].append(virtual_link)
        # (edge_idx,offset), pt, virtual link list
        node_seq = []
        for loc in loc2virtual_links:
            edge_idx, edge_offset_rate = loc
            # if edge_offset_rate == 0.0:
            if edge_offset_rate <= 0.0:
                new_node = ori_coords[edge_idx]
            # elif edge_offset_rate == 1.0:
            elif edge_offset_rate >= 1.0:
                new_node = ori_coords[edge_idx + 1]
            else:
                new_node = cal_loc_along_line(ori_coords[edge_idx], ori_coords[edge_idx + 1], edge_offset_rate)
            node_seq.append((new_node, loc, loc2virtual_links[loc]))
        node_seq = sorted(node_seq, key=lambda data: (data[1][0], data[1][1]))
        # add the ori start node if not added
        if node_seq[0][1][0] != 0 or node_seq[0][1][1] != 0.0:
            node_seq = [(ori_coords[0], (0, 0.0), [])] + node_seq
        # add the ori end node if not added
        if node_seq[-1][1][0] != len(ori_coords) - 2 or node_seq[-1][1][1] != 1.0:
            node_seq.append((ori_coords[-1], (len(ori_coords) - 2, 1.0), []))
        # add splitted segment
        for i in range(len(node_seq) - 1):
            from_node, (from_edge_idx, from_offset), _ = node_seq[i]
            to_node, (to_edge_idx, to_offset), _ = node_seq[i + 1]
            shape = []
            shape.append(from_node)
            if from_edge_idx != to_edge_idx:
                for j in range(from_edge_idx + 1, to_edge_idx + 1):
                    shape.append(ori_coords[j])
            shape.append(to_node)
            splitted_segments.append((from_node, to_node, shape))
        # add links
        for node in node_seq:
            node_pt, _, node_virtual_links = node
            for node_virtual_link in node_virtual_links:
                shape = [node_pt, node_virtual_link.end_node]
                link_segments.append((node_pt, node_virtual_link.end_node, shape))
        return splitted_segments, link_segments

    def update_link(self, linked_rn, from_pt, to_pt, coords, avail_eid):
        """
        make sure two link will not have too similar direction
        """
        is_valid = True
        link_dist = distance(from_pt, to_pt)
        # if the new edge is shorter, add new edge and delete old edge
        # check from pt
        links_with_from = [edge for edge in list(linked_rn.edges((from_pt.lng, from_pt.lat))) if
                           linked_rn.edges[edge]['type'] == 'virtual']
        edges_to_delete = []
        for u, v in links_with_from:
            other_node = v if u[0] == from_pt.lng and u[1] == from_pt.lat else u
            ang = angle(from_pt, to_pt, from_pt, SPoint(other_node[1], other_node[0]))
            if ang < self.SIMILAR_DIRECTION_THRESHOLD:
                if link_dist >= linked_rn[u][v]['length']:
                    is_valid = False
                    break
                else:
                    edges_to_delete.append((u, v))
        # check to pt
        links_with_to = [edge for edge in list(linked_rn.edges((to_pt.lng, to_pt.lat))) if
                         linked_rn.edges[edge]['type'] == 'virtual']
        for u, v in links_with_to:
            other_node = v if u[0] == to_pt.lng and u[1] == to_pt.lat else u
            ang = angle(to_pt, from_pt, to_pt, SPoint(other_node[1], other_node[0]))
            if ang < self.SIMILAR_DIRECTION_THRESHOLD:
                if link_dist >= linked_rn[u][v]['length']:
                    is_valid = False
                    break
                else:
                    edges_to_delete.append((u, v))
        if is_valid:
            linked_rn.add_edge((from_pt.lng, from_pt.lat), (to_pt.lng, to_pt.lat), coords=coords, eid=avail_eid,
                               type='virtual')
            for u, v in edges_to_delete:
                if linked_rn.has_edge(u, v):
                    # didn't destroy the connectivity
                    if linked_rn.degree(u) == 2 or linked_rn.degree(v) == 2:
                        continue
                    linked_rn.remove_edge(u, v)
        # linked_rn.add_edge((from_pt.lng, from_pt.lat), (to_pt.lng, to_pt.lat), coords=coords, eid=avail_eid, type='virtual')

    def is_intersected_with_existing_edges(self, rn, virtual_link, candidates):
        is_intersected = False
        f = virtual_link.end_node
        edge_idx = virtual_link.split_edge_idx
        edge_offset_rate = virtual_link.split_edge_offset
        ori_segment = virtual_link.target_segment
        ori_coords = rn[ori_segment[0]][ori_segment[1]]['coords']
        o = cal_loc_along_line(ori_coords[edge_idx], ori_coords[edge_idx + 1], edge_offset_rate)
        for candidate in candidates:
            if candidate == virtual_link.target_segment:
                continue
            u, v = candidate
            coords = rn[u][v]['coords']
            for i in range(len(coords) - 1):
                if is_line_line_intersected(f, o, coords[i], coords[i + 1]):
                    is_intersected = True
                    break
            if is_intersected:
                break
        return is_intersected

    def remove_similar_links(self, rn, edge_virtual_links):
        new_edge_virtual_links = copy.copy(edge_virtual_links)
        o = edge_virtual_links[0].end_node
        stable = False
        while not stable:
            stable = True
            for i in range(len(new_edge_virtual_links) - 1):
                link_a = new_edge_virtual_links[i]
                a = cal_loc_along_line(rn.edges[link_a.target_segment]['coords'][link_a.split_edge_idx],
                                       rn.edges[link_a.target_segment]['coords'][link_a.split_edge_idx + 1],
                                       link_a.split_edge_offset)
                for j in range(i + 1, len(new_edge_virtual_links)):
                    link_b = new_edge_virtual_links[j]
                    b = cal_loc_along_line(rn.edges[link_b.target_segment]['coords'][link_b.split_edge_idx],
                                           rn.edges[link_b.target_segment]['coords'][link_b.split_edge_idx + 1],
                                           link_b.split_edge_offset)
                    # if small angle
                    if angle(o, a, o, b) < self.SIMILAR_DIRECTION_THRESHOLD:
                        # delete longer edge
                        if distance(o, a) < distance(o, b):
                            new_edge_virtual_links.remove(new_edge_virtual_links[j])
                        else:
                            new_edge_virtual_links.remove(new_edge_virtual_links[i])
                        stable = False
                        break
                if not stable:
                    break
        return new_edge_virtual_links

    def generate(self, init_rn, linked_rn_path):
        """
        :param init_rn: it must be undirected
        :param linked_rn_path: the output is directed
        :return:
        """
        linked_rn = copy.deepcopy(init_rn)
        HALF_DELTA_LAT = LAT_PER_METER * self.radius
        HALF_DELTA_LNG = LNG_PER_METER * self.radius
        dead_end_cnt = 0
        virtual_links = []
        for node, degree in init_rn.degree():
            if degree == 1:
                lng, lat = node
                dead_end_pt = SPoint(lat, lng)
                # get opposite node
                u = list(init_rn.adj[node])[0]
                v = node
                seg_coords = init_rn[u][v]['coords']
                if seg_coords[0].lat == lat and seg_coords[0].lng == lng:
                    last_edge_of_dead_end = (seg_coords[1], dead_end_pt)
                elif seg_coords[-1].lat == lat and seg_coords[-1].lng == lng:
                    last_edge_of_dead_end = (seg_coords[-2], dead_end_pt)
                else:
                    raise Exception('error, coords ends is not consistent with node')
                dead_end_cnt += 1
                query_mbr = MBR(lat - HALF_DELTA_LAT, lng - HALF_DELTA_LNG, lat + HALF_DELTA_LAT, lng + HALF_DELTA_LNG)
                # get nearby candidate road segments to be linked (except for the self)
                candidates = init_rn.range_query(query_mbr)
                candidates = [candidate for candidate in candidates if
                              not (candidate[0] in [u, v] and candidate[1] in [u, v])]
                if len(candidates) == 0:
                    continue
                edge_virtual_links = []
                # calculate the linking position
                for candidate in candidates:
                    split_edge_idx, split_edge_offset = self.generate_pt_to_link(init_rn, last_edge_of_dead_end,
                                                                                 init_rn[candidate[0]][candidate[1]]['coords'],
                                                                                 (u, v), candidate)
                    if split_edge_idx == float('inf'):
                        continue
                    virtual_link = VirtualLink(dead_end_pt, candidate, split_edge_idx, split_edge_offset)
                    # not intersect with existing roads
                    if not self.is_intersected_with_existing_edges(init_rn, virtual_link, candidates):
                        edge_virtual_links.append(virtual_link)
                if len(edge_virtual_links) > 0:
                    # remove links from the same dead end that have similar direction (only the shortest link is reserved)
                    edge_virtual_links = self.remove_similar_links(init_rn, edge_virtual_links)
                    virtual_links.extend(edge_virtual_links)
        print('number of dead ends:{}'.format(dead_end_cnt))
        segment2infos = {}
        for virtual_link in virtual_links:
            segment = virtual_link.target_segment
            if segment not in segment2infos:
                segment2infos[segment] = []
            segment2infos[segment].append(virtual_link)
        avail_eid = max([eid for u, v, eid in init_rn.edges.data(data='eid')]) + 1
        for segment in segment2infos:
            ori_coords = init_rn[segment[0]][segment[1]]['coords']
            virtual_links = segment2infos[segment]
            splitted_segments, link_segments = self.divide_segment(ori_coords, virtual_links)
            for from_pt, to_pt, coords in splitted_segments:
                linked_rn.add_edge((from_pt.lng, from_pt.lat), (to_pt.lng, to_pt.lat), coords=coords, eid=avail_eid,
                                   type=init_rn[segment[0]][segment[1]]['type'])
                avail_eid += 1
            for from_pt, to_pt, coords in link_segments:
                # check whether this link should be added, and other links should be removed (similar direction, but the shortest)
                self.update_link(linked_rn, from_pt, to_pt, coords, avail_eid)
                avail_eid += 1
            # if a segment is splitted to multiple segments, remove the original segment
            if len(splitted_segments) > 1:
                linked_rn.remove_edge(segment[0], segment[1])
        linked_rn_directed = linked_rn.to_directed()
        store_rn_shp(linked_rn_directed, linked_rn_path)
