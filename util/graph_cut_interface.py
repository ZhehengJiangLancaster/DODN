import cv2
import numpy as np
from scipy.ndimage import generic_filter
from util.graph_cut.GraphMaker import GraphMaker
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,medial_axis
from collections import defaultdict

def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return 255 * ((P[4]==255) and np.sum(P)==510)

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        # self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        # self.weights[(to_node, from_node)] = weight

def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def connect_pairs(source_o,target_o,weights_map):
    graph = Graph()
    for i in range(weights_map.shape[0]):
        for j in range(weights_map.shape[1]):
            source = str(weights_map.shape[1] * i + j)
            source_w = weights_map[i,j]
            ### 8 connection edge
            for n_x in range(-1,2):
                for n_y in range(-1,2):
                    if i+n_x >= 0 and j+n_y>=0 and i+n_x<weights_map.shape[0] and j+n_y<weights_map.shape[1]:
                        target = str(weights_map.shape[1] * (i + n_x) + j + n_y)
                        target_w = weights_map[i+ n_x, j+ n_y]
                        ###exclude self edge
                        if source!=target:
                            graph.add_edge(source,target,target_w-source_w)
    path = dijsktra(graph, source_o, target_o)
    ###map to image
    path_map = np.zeros(weights_map.shape)
    for p in path:
        i = int(int(p) / path_map.shape[1])
        j = int(p) % path_map.shape[1]
        path_map[i,j] = 1

    return path_map

def graph_cut_interface_v3(centerline, distance_map):
    roi_w=32
    roi_h=32
    dis_filter = 30

    ### centerline<1
    foreground_seeds = centerline.nonzero()
    #background_seeds = np.where(centerline == 0)
    result = generic_filter(centerline*255, lineEnds, (3, 3))
    index = np.where(result == 255)
    points_pairs = []
    n_componet_centerline = cv2.connectedComponentsWithStats(centerline.astype('int8'), connectivity=8)
    n_componet_mask = n_componet_centerline[1]
    ###find pair endpoints need to be connected
    for i in range(index[0].shape[0]-1):
        for j in range(i+1,index[0].shape[0] - 1):
            dis = np.sqrt((index[0][i]-index[0][j]) ** 2 + (index[1][i]-index[1][j]) ** 2)
            if dis < dis_filter:
                p1_x, p1_y, p2_x, p2_y = index[0][i],index[1][i],index[0][j],index[1][j]
                ###if they are from the same component
                if n_componet_mask[p1_x,p1_y]!=n_componet_mask[p2_x,p2_y]:
                    points_pairs.append([[p1_x,p1_y],[p2_x, p2_y]])
    ###connect endpoints to the nearest neighbour
    for i in range(index[0].shape[0]):
        roi = centerline[int(index[0][i]-roi_w/2):int(index[0][i]+roi_w/2),
              int(index[1][i]-roi_h/2):int(index[1][i]+roi_h/2)]
        n_componet = cv2.connectedComponentsWithStats(roi.astype('int8'), connectivity=8)
        if n_componet[0]>2:
            minmum_dis = 10000
            for n in(1, n_componet[0]-1):
                if n != n_componet[1][int(roi_w/2),int(roi_w/2)]:
                    neighbour_index = np.where(n_componet[1] == n)
                    for x_index,y_index in zip(*neighbour_index):
                        dis = np.sqrt((x_index-roi_w/2) ** 2 + (y_index-roi_h/2) ** 2)
                        if dis <minmum_dis:
                            minmum_dis = dis
                            neighbour = [int(x_index-roi_w/2+index[0][i]),int(y_index-roi_h/2+index[1][i])]
            if minmum_dis != 10000:
                points_pairs.append([[index[0][i],index[1][i]],neighbour])

    for points_pair in points_pairs:
        x1_roi, x2_roi = min([points_pair[0][0],points_pair[1][0]]), max([points_pair[0][0],points_pair[1][0]])
        y1_roi, y2_roi = min([points_pair[0][1],points_pair[1][1]]), max([points_pair[0][1],points_pair[1][1]])
        weights_map = distance_map[x1_roi:x2_roi+1,y1_roi:y2_roi+1]
        if (points_pair[0][0]-points_pair[1][0])*(points_pair[0][1]-points_pair[1][1])>0:
            source = '0'
            target = str(weights_map.size-1)
        else:
            source = str(weights_map.size-weights_map.shape[1])
            target = str(weights_map.shape[1]-1)
        path_map = connect_pairs(source,target,weights_map)
        path_map = path_map + centerline[x1_roi:x2_roi+1,y1_roi:y2_roi+1]
        path_map = path_map>0
        centerline[x1_roi:x2_roi+1,y1_roi:y2_roi+1] = path_map.astype(int)

    # plt.imshow(skeletonize(graph_maker.segment_overlay>0.1,method='lee'))
    # plt.show()
    return centerline,centerline

def graph_cut_interface_v2(centerline, distance_map):
    roi_w=22
    roi_h=22

    ### centerline<1
    foreground_seeds = centerline.nonzero()
    #background_seeds = np.where(centerline == 0)
    result = generic_filter(centerline*255, lineEnds, (3, 3))
    index = np.where(result==255)
    for i in range(index[0].shape[0]):
        roi = centerline[int(index[0][i]-roi_w/2):int(index[0][i]+roi_w/2),
              int(index[1][i]-roi_h/2):int(index[1][i]+roi_h/2)]
        n_componet = cv2.connectedComponentsWithStats(roi.astype('int8'), connectivity=8)
        diff_endp = np.array(index)-np.repeat(np.expand_dims([index[0][i],index[1][i]],axis=1),
                                                             index[0].shape[0],axis=1)
        dist_endp = np.sqrt(diff_endp[0,:]**2+diff_endp[1,:]**2)
        min_dist_endp = np.min(dist_endp[np.where(dist_endp!=0)])
        min_dist_endp_index = np.where(dist_endp==min_dist_endp)
        min_dist_endp_index = [index[0][min_dist_endp_index],index[1][min_dist_endp_index]]

        ###search endpoint in neighbour
        minimum_roi = np.sqrt((roi_h/2)**2+(roi_w/2)**2)
        new_endp_index = []
        if n_componet[0]>2:
            for n in range(3,n_componet[0]+1):
                sub_index = np.where(n_componet[1]==n)
                diff_roi = np.array(sub_index)-np.repeat(np.expand_dims([roi_w/2,roi_h/2],axis=1),
                                                             sub_index[0].shape[0],axis=1)
                dist_roi = np.sqrt(diff_roi[0, :] ** 2 + diff_roi[1, :] ** 2)
                min_dist_roi = np.min(dist_roi)
                new_endp_index_candi = np.argmin(dist_roi)
                if min_dist_roi < minimum_roi:
                    minimum_roi = min_dist_roi
                    new_endp_index = new_endp_index_candi

        if new_endp_index!=[]:
            if minimum_roi > min_dist_endp:
                dest_point = min_dist_endp_index
            else:
                dest_point =new_endp_index

    kernel = np.ones((21, 21), np.uint8)
    result = cv2.dilate(result, kernel)-result
    mask = centerline + result/255
    background_seeds = np.where(mask <= 0.5)
    # distance_map = cv2.GaussianBlur(distance_map.astype('uint8')*255, (31, 31), 0)
    distance_map = distance_map/np.max(distance_map)
    image = np.stack([distance_map*255,distance_map*255,distance_map*255], axis=-1)
    graph_maker = GraphMaker(image)
    ### cv2 seeds coordinate: x, down arrow ;y, right arrow.  graph and image coordinate: x, right arrow;y, down arrow.
    graph_maker.add_foreground_seeds((foreground_seeds[0],foreground_seeds[1]))
    graph_maker.add_background_seeds((background_seeds[0], background_seeds[1]))
    graph_maker.create_graph()
    # plt.imshow(skeletonize(graph_maker.segment_overlay>0.1,method='lee'))
    # plt.show()
    return mask,skeletonize(graph_maker.segment_overlay>0.1)

def graph_cut_interface(centerline, distance_map):

    ### centerline<1
    foreground_seeds = centerline.nonzero()
    #background_seeds = np.where(centerline == 0)

    result = generic_filter(centerline*255, lineEnds, (3, 3))
    kernel = np.ones((11, 11), np.uint8)
    result = cv2.dilate(result, kernel)-result
    mask = centerline + result/255
    background_seeds = np.where(mask <= 0.5)
    # distance_map = cv2.GaussianBlur(distance_map.astype('uint8')*255, (31, 31), 0)
    distance_map = distance_map/np.max(distance_map)
    image = np.stack([distance_map*255,distance_map*255,distance_map*255], axis=-1)
    graph_maker = GraphMaker(image)
    ### cv2 seeds coordinate: x, down arrow ;y, right arrow.  graph and image coordinate: x, right arrow;y, down arrow.
    graph_maker.add_foreground_seeds((foreground_seeds[0],foreground_seeds[1]))
    graph_maker.add_background_seeds((background_seeds[0], background_seeds[1]))
    graph_maker.create_graph()
    # plt.imshow(skeletonize(graph_maker.segment_overlay>0.1,method='lee'))
    # plt.show()
    return mask,skeletonize(graph_maker.segment_overlay>0.1)