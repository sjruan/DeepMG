import networkx as nx
from osgeo import ogr
import numpy as np
import sys
import itertools
from collections import deque
from rtree import Rtree
from tptk.common.douglas_peucker import DouglasPeucker
from tptk.common.spatial_func import SPoint
from tptk.common.mbr import MBR
from tptk.common.grid import Grid
from tptk.common.spatial_func import distance


class CenterNodePixel:
    def __init__(self, node_pixels):
        self.node_pixels = node_pixels
        self.i = sum([node_pixel[0] for node_pixel in node_pixels]) / len(node_pixels)
        self.j = sum([node_pixel[1] for node_pixel in node_pixels]) / len(node_pixels)

    def center_pixel(self):
        return int(self.i), int(self.j)


class GraphExtractor:
    INVALID = -1
    BLANK = 0
    EDGE = 1
    NODE = 2
    VISITED_NODE = 3

    def __init__(self, epsilon, min_road_dist):
        self.segment_simplifier = DouglasPeucker(epsilon)
        self.min_road_dist = min_road_dist

    def extract(self, skeleton, mbr, target_path):
        assert skeleton.ndim == 2, 'the grey scale skeleton should only have 1 channel'

        # pad zero for safety
        grid = Grid(mbr, skeleton.shape[0], skeleton.shape[1])
        eval_mbr_padded = MBR(mbr.min_lat - 2 * grid.lat_interval, mbr.min_lng - 2 * grid.lng_interval,
                              mbr.max_lat + 2 * grid.lat_interval, mbr.max_lng + 2 * grid.lng_interval)
        mbr = eval_mbr_padded
        skeleton_padded = np.zeros((skeleton.shape[0] + 4, skeleton.shape[1] + 4))
        skeleton_padded[2:skeleton.shape[0] + 2, 2:skeleton.shape[1] + 2] = skeleton
        skeleton = skeleton_padded

        nb_rows, nb_cols = skeleton.shape
        yscale = nb_rows / (mbr.max_lat - mbr.min_lat)
        xscale = nb_cols / (mbr.max_lng - mbr.min_lng)

        sys.stdout.write("Identifying road nodes pixels... ")
        sys.stdout.flush()
        binary_skeleton = (skeleton > 0).astype('int32')
        status_matrix = self.identify_node_pixels(binary_skeleton)
        sys.stdout.write("done.\n")
        sys.stdout.flush()

        sys.stdout.write("Detecting road network component... ")
        nodes, segments = self.detect_rn_component(status_matrix)
        sys.stdout.write("done.\n")
        sys.stdout.flush()

        sys.stdout.write("Constructing and saving road network... ")
        sys.stdout.flush()
        self.construct_undirected_rn(nodes, segments, target_path, nb_rows, xscale, yscale, mbr)
        sys.stdout.write("done.\n")
        sys.stdout.flush()

    def identify_node_pixels(self, skeleton):
        """
        22 23 08 09 10
        21 07 00 01 11
        20 06 -1 02 12
        19 05 04 03 13
        18 17 16 15 14
        :param skeleton:
        :return:
        """
        status_matrix = np.copy(skeleton)
        road_pixels = np.where(status_matrix == GraphExtractor.EDGE)
        nb_road_pixels = len(road_pixels[0])
        print('\n# of road pixels:{}'.format(nb_road_pixels))
        cnt = 1
        for i, j in zip(road_pixels[0], road_pixels[1]):
            if (cnt % 100 == 0) or (cnt == nb_road_pixels):
                sys.stdout.write("\r" + str(cnt) + "/" + str(nb_road_pixels) + "... ")
                sys.stdout.flush()
            cnt += 1
            # skip boundary
            if i < 2 or i >= status_matrix.shape[0] - 2 or j < 2 or j >= status_matrix.shape[1] - 2:
                continue
            p = [skeleton[i - 1][j], skeleton[i - 1][j + 1], skeleton[i][j + 1], skeleton[i + 1][j + 1],
                 skeleton[i + 1][j], skeleton[i + 1][j - 1], skeleton[i][j - 1], skeleton[i - 1][j - 1],
                 skeleton[i - 2][j], skeleton[i - 2][j + 1], skeleton[i - 2][j + 2], skeleton[i - 1][j + 2],
                 skeleton[i][j + 2], skeleton[i + 1][j + 2], skeleton[i + 2][j + 2], skeleton[i + 2][j + 1],
                 skeleton[i + 2][j], skeleton[i + 2][j - 1], skeleton[i + 2][j - 2], skeleton[i + 1][j - 2],
                 skeleton[i][j - 2], skeleton[i - 1][j - 2], skeleton[i - 2][j - 2], skeleton[i - 2][j - 1]]
            fringe = [bool(p[8] and bool(p[7] or p[0] or p[1])),
                      bool(p[9] and bool(p[0] or p[1])),
                      bool(p[10] and p[1]),
                      bool(p[11] and bool(p[1] or p[2])),
                      bool(p[12] and bool(p[1] or p[2] or p[3])),
                      bool(p[13] and bool(p[2] or p[3])),
                      bool(p[14] and p[3]),
                      bool(p[15] and bool(p[3] or p[4])),
                      bool(p[16] and bool(p[3] or p[4] or p[5])),
                      bool(p[17] and bool(p[4] or p[5])),
                      bool(p[18] and p[5]),
                      bool(p[19] and bool(p[5] or p[6])),
                      bool(p[20] and bool(p[5] or p[6] or p[7])),
                      bool(p[21] and bool(p[6] or p[7])),
                      bool(p[22] and p[7]),
                      bool(p[23] and bool(p[7] or p[0]))]
            connected_component_cnt = 0
            for k in range(0, len(fringe)):
                connected_component_cnt += int(not bool(fringe[k]) and bool(fringe[(k + 1) % len(fringe)]))
            if connected_component_cnt == 0:
                status_matrix[i][j] = GraphExtractor.BLANK
            elif (connected_component_cnt == 1) or (connected_component_cnt > 2):
                status_matrix[i][j] = GraphExtractor.NODE
            # if connected_component_cnt == 2, we think it is a normal internal node
        return status_matrix

    def detect_rn_component(self, status_matrix):
        node_pixels = np.where(status_matrix == GraphExtractor.NODE)
        nb_node_pixels = len(node_pixels[0])
        print('\n# of node pixels:{}'.format(nb_node_pixels))
        neighbor_deltas = [dxdy for dxdy in itertools.product([-1, 0, 1], [-1, 0, 1])
                           if dxdy[0] != 0 or dxdy[1] != 0]
        # node pixel -> center node
        nodes = {}
        node_pixel_spatial_index = Rtree()
        # [node pixel sequence (start and end must be node pixel)]
        connected_segments = []
        cnt = 1
        center_nodes = []
        node_pixel_id = 0
        for i, j in zip(node_pixels[0], node_pixels[1]):
            if (cnt % 100 == 0) or (cnt == nb_node_pixels):
                sys.stdout.write("\r" + str(cnt) + "/" + str(nb_node_pixels) + "... ")
                sys.stdout.flush()
            cnt += 1
            if status_matrix[i][j] == GraphExtractor.VISITED_NODE:
                continue
            # region merge neighbor node pixels
            status_matrix[i][j] = GraphExtractor.VISITED_NODE
            candidates = [(i, j)]
            node_pixels = []
            while len(candidates) > 0:
                node_pixel = candidates.pop()
                node_pixels.append(node_pixel)
                m, n = node_pixel
                for dm, dn in neighbor_deltas:
                    if status_matrix[m + dm][n + dn] == GraphExtractor.NODE:
                        status_matrix[m + dm][n + dn] = GraphExtractor.VISITED_NODE
                        candidates.append((m + dm, n + dn))
            center_node = CenterNodePixel(node_pixels)
            center_nodes.append(center_node)
            for node_pixel in node_pixels:
                nodes[node_pixel] = center_node
                node_pixel_spatial_index.insert(node_pixel_id, node_pixel, obj=node_pixel)
                node_pixel_id += 1
            # endregion

            # region find neighbor segments
            # mask current nodes, make sure the edge doesn't return to itself
            for m, n in node_pixels:
                status_matrix[m][n] = GraphExtractor.INVALID
            # find new road segment of the current node in each possible direction
            for node_pixel in node_pixels:
                connected_segment = self.detect_connected_segment(status_matrix, node_pixel)
                if connected_segment is not None:
                    connected_segments.append(connected_segment)
            # restore masked nodes
            for m, n in node_pixels:
                status_matrix[m][n] = GraphExtractor.VISITED_NODE
            # endregion
        print('\n# of directly connected segments:{}'.format(len(connected_segments)))

        # there might be few edge pixels left, that should be fine
        nb_unprocessed_edge_pixels = np.sum(status_matrix[status_matrix == GraphExtractor.EDGE])
        print('unprocessed edge pixels:{}'.format(nb_unprocessed_edge_pixels))

        print('# of nodes:{}'.format(len(center_nodes)))
        print('# of segments:{}'.format(len(connected_segments)))
        return nodes, connected_segments

    def detect_connected_segment(self, status_matrix, start_node_pixel):
        """
        find a path ended with node pixel
        :param status_matrix: status
        :param start_node_pixel: start node pixel
        :return: [start_node_pixel, edge_pixel,...,end_node_pixel]
        """
        # for current implementation, we assume edge pixel has only two arcs
        # but it is possible that edge pixel has multiple connected component rather than 2,
        # because crossing are detected using outer pixels
        s = deque()
        neighbor_deltas = [dxdy for dxdy in itertools.product([-1, 0, 1], [-1, 0, 1])
                           if dxdy[0] != 0 or dxdy[1] != 0]
        # add candidates to stack
        m, n = start_node_pixel
        for dm, dn in neighbor_deltas:
            if status_matrix[m + dm][n + dn] == GraphExtractor.EDGE:
                s.appendleft(((m + dm, n + dn), [start_node_pixel]))
        while len(s) > 0:
            (m, n), path = s.popleft()
            # end node pixel
            if status_matrix[m][n] == GraphExtractor.NODE or \
                    status_matrix[m][n] == GraphExtractor.VISITED_NODE:
                path.append((m, n))
                return path
            # internal edge pixel
            elif status_matrix[m][n] == GraphExtractor.EDGE:
                # mark the edge as visited
                status_matrix[m][n] = GraphExtractor.BLANK
                new_path = path.copy()
                new_path.append((m, n))
                for dm, dn in neighbor_deltas:
                    s.appendleft(((m + dm, n + dn), new_path))
        return None

    def construct_undirected_rn(self, nodes, segments, target_path, nb_rows, xscale, yscale, mbr):
        # node pixel -> road node
        road_nodes = {}
        eid = 0
        # we use coordinate tuples as key, consistent with networkx
        # !!! Though we construct DiGraph (compatible with networkx interface, one segment will only add once)
        # loading this data, we should call g.to_undirected()
        rn = nx.DiGraph()
        for segment in segments:
            coords = []
            # start node
            start_node_pixel = nodes[segment[0]].center_pixel()
            if start_node_pixel not in road_nodes:
                lat, lng = self.pixels_to_latlng(start_node_pixel, mbr, nb_rows, xscale, yscale)
                road_node = SPoint(lat, lng)
                geo_pt = ogr.Geometry(ogr.wkbPoint)
                geo_pt.AddPoint(lng, lat)
                rn.add_node((lng, lat))
                road_nodes[start_node_pixel] = road_node
            start_node = road_nodes[start_node_pixel]
            coords.append(start_node)
            start_node_key = (start_node.lng, start_node.lat)
            # internal nodes, we didn't create id for them
            for coord in segment[1:-1]:
                lat, lng = self.pixels_to_latlng(coord, mbr, nb_rows, xscale, yscale)
                coords.append(SPoint(lat, lng))
            # end node
            end_node_pixel = nodes[segment[-1]].center_pixel()
            if end_node_pixel not in road_nodes:
                lat, lng = self.pixels_to_latlng(end_node_pixel, mbr, nb_rows, xscale, yscale)
                road_node = SPoint(lat, lng)
                geo_pt = ogr.Geometry(ogr.wkbPoint)
                geo_pt.AddPoint(lng, lat)
                rn.add_node((lng, lat))
                road_nodes[end_node_pixel] = road_node
            end_node = road_nodes[end_node_pixel]
            coords.append(end_node)
            end_node_key = (end_node.lng, end_node.lat)
            # region add segment
            # skip loop
            if start_node_key == end_node_key:
                continue
            simplified_coords = self.segment_simplifier.simplify(coords)
            # skip too short segment
            if not self.is_valid(simplified_coords):
                continue
            # add forward segment
            geo_line = ogr.Geometry(ogr.wkbLineString)
            for simplified_coord in simplified_coords:
                geo_line.AddPoint(simplified_coord.lng, simplified_coord.lat)
            rn.add_edge(start_node_key, end_node_key, eid=eid, Wkb=geo_line.ExportToWkb(), type='pred')
            eid += 1
            # endregion
        rn.remove_nodes_from(list(nx.isolates(rn)))
        print('\n# of nodes:{}'.format(rn.number_of_nodes()))
        print('# of edges:{}'.format(rn.number_of_edges()))
        nx.write_shp(rn, target_path)
        return rn

    def is_valid(self, coords):
        dist = 0.0
        for i in range(len(coords) - 1):
            dist += distance(coords[i], coords[i + 1])
        return dist > self.min_road_dist

    def pixels_to_latlng(self, pixel, mbr, nb_rows, xscale, yscale):
        i, j = pixel
        return (((nb_rows - i) / yscale) + mbr.min_lat), ((j / xscale) + mbr.min_lng)
