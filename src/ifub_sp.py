import temporal_distances as td
import temporal_graph as tg

import numpy as np


def count_pairs_reachable(graph, landmarks, is_inB, is_inA):
    pairs_number = 0
    num_landmarks = len(landmarks)

    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    # num_landmarks, num_nodes = np.shape(is_inA)
    for j in range(num_nodes):
        reachables_nodes = np.full(num_nodes, False)
        for i in range(num_landmarks):
            if is_inA[i, j]:
                reachables_nodes = np.logical_or(is_inB[i], reachables_nodes)
        count = np.count_nonzero(reachables_nodes)
        pairs_number += count

    # Number of distinct element in A and in B (min(distinct(A), distinct(B))= num BFS classic diameter on landmark
    distinct_A = np.full(num_nodes, False)
    distinct_B = np.full(num_nodes, False)
    for i in range(num_landmarks):
        distinct_A = np.logical_or(is_inA[i], distinct_A)
        distinct_B = np.logical_or(is_inB[i], distinct_B)
    num_distinct_A = np.count_nonzero(distinct_A)
    num_distinct_B = np.count_nonzero(distinct_B)

    return pairs_number, num_distinct_A, num_distinct_B


def eccentricity_reachable(distances, is_inA_B):
    if len(distances) != len(is_inA_B):
        raise Exception("Distances array has an incorrect length!")
    index = -1
    count = 0
    t = np.NINF
    for elem in distances:
        if elem > t and is_inA_B[count]:
            t = elem
            index = count
        count += 1
    return index, t


def sort_remove_unreachable(distances_fw, distances_bw):
    # Sort fw distances in descending order and store associated nodes to each distance in ind_fw array
    ind_fw = np.argsort(-distances_fw)
    dist_fw = distances_fw[ind_fw]
    # Discard unreachable nodes from arrays
    if len(np.where(dist_fw == np.inf)[0]) > 0:
        inf_indices = np.where(dist_fw == np.inf)[0][-1] + 1
        dist_fw = dist_fw[inf_indices:]
        ind_fw = ind_fw[inf_indices:]

    # Sort bw distances in descending order and store associated nodes to each distance in ind_bw array
    ind_bw = np.argsort(-distances_bw)
    dist_bw = distances_bw[ind_bw]
    # Discard unreachable nodes from arrays
    if len(np.where(dist_bw == np.inf)[0]) > 0:
        inf_indices = np.where(dist_bw == np.inf)[0][-1] + 1
        dist_bw = dist_bw[inf_indices:]
        ind_bw = ind_bw[inf_indices:]
    return dist_fw, ind_fw, dist_bw, ind_bw


def who_is_reachable_with_landmarks(graph, landmarks=None):
    """
    :param graph: Graph
    :param landmarks: array of tuples where for each tuple, first elements is landmark nodes and second element is time
    """
    num_visits = 0

    sp_dist = td.ShortestPath()

    g_path_op = sp_dist.gen_opposite(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                     file=graph.get_file_path().rsplit('/', 1)[1])

    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    num_nodes = graph.get_num_nodes()

    # Compute distances forward/backward from/to landmarks nodes
    distances_fw = np.full((len(landmarks), num_nodes), 0, dtype=float)
    distances_bw = np.full((len(landmarks), num_nodes), 0, dtype=float)
    for i in range(len(landmarks)):

        middle_t = landmarks[i][1]
        sp_dist.compute_distances(graph=graph, start_node=int(landmarks[i][0]), min_time=middle_t)
        num_visits += 1
        dist_fw = sp_dist.get_distances()

        sp_dist.compute_distances(graph=g_op, start_node=int(landmarks[i][0]), min_time=-middle_t + 1)
        num_visits += 1
        dist_bw = sp_dist.get_distances()

        distances_fw[i] = dist_fw
        distances_bw[i] = dist_bw

    is_inA = np.full((len(landmarks), num_nodes), False)
    is_inB = np.full((len(landmarks), num_nodes), False)

    distances_sp_fw = []
    ind_sp_fw_node = []
    distances_sp_bw = []
    ind_sp_bw_node = []
    for j in range(len(landmarks)):
        for i in range(num_nodes):
            if distances_fw[j, i] != np.inf:
                is_inB[j, i] = True
            if distances_bw[j, i] != np.inf:
                is_inA[j, i] = True

        dist_sp_fw, ind_sp_fw, dist_sp_bw, ind_sp_bw = sort_remove_unreachable(distances_fw[j], distances_bw[j])
        distances_sp_fw.append(dist_sp_fw)
        ind_sp_fw_node.append(ind_sp_fw)
        distances_sp_bw.append(dist_sp_bw)
        ind_sp_bw_node.append(ind_sp_bw)

    return is_inB, is_inA, distances_sp_fw, ind_sp_fw_node, distances_sp_bw, ind_sp_bw_node, num_visits


def ifub_on_landmarks(graph, landmarks=None):
    """
    :param graph: Graph
    :param landmarks: array of tuples where for each tuple, first element is landmark nodes and second element is time
    """
    if graph.get_latest_node() is not None:
        raise Exception('Shortest path not work with dummy nodes, give me in input weighted graph!')

    graph_name = graph.get_file_path().rsplit('/', 1)[1]
    print("Computing iFUB Diameter of Graph " + graph_name + "...")

    if landmarks is None:
        # Give me most central node
        _, landmarks = graph.get_max_deg_out()

    if np.isscalar(landmarks):
        min_t, max_t = graph.get_time_interval()
        middle_t = round(((max_t - min_t) / 2) + min_t)
        l_tuple = [(landmarks, middle_t)]
        landmarks = np.empty(len(l_tuple), dtype=object)
        landmarks[:] = l_tuple

    num_landmark = len(landmarks)

    is_inB, is_inA, distances_fw, ind_fw_node, distances_bw, ind_bw_node, num_visits = \
        who_is_reachable_with_landmarks(graph=graph, landmarks=landmarks)

    pairs_number, num_distinct_A, num_distinct_B = count_pairs_reachable(graph=graph, landmarks=landmarks,
                                                                         is_inB=is_inB, is_inA=is_inA)
    # print('Pairs reachables with landmarks Graph ' + graph_name + ': ' + str(pairs_number), flush=True)
    # print('Nodes in A ' + graph_name + ': ' + str(num_distinct_A), flush=True)
    # print('Nodes in B ' + graph_name + ': ' + str(num_distinct_B), flush=True)

    # Arrays of indices for locate next node to process relative to each landmark
    indices_fw = np.full(num_landmark, 0)
    indices_bw = np.full(num_landmark, 0)

    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    # Into the arrays to_be_considered, if one item is False, that node has already been considered in another BFS
    to_be_considered_bw = np.full(num_nodes, True)
    to_be_considered_fw = np.full(num_nodes, True)

    sp_dist = td.ShortestPath()

    g_path_op = sp_dist.gen_opposite(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                     file=graph.get_file_path().rsplit('/', 1)[1])

    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    # Get greatest distances fw and bw respectively from and to landmarks
    # Choose the upper bound like the higher upper bound from all landmark's upper bounds
    ub = -1
    idx_landmark = -1
    for i in range(num_landmark):
        if (distances_fw[i][0] + distances_bw[i][0]) > ub:
            ub = distances_fw[i][0] + distances_bw[i][0]
            idx_landmark = i

    lb = 0

    while ub > lb:

        index_node_bw = ind_bw_node[idx_landmark][indices_bw[idx_landmark]]
        sp_dist.compute_distances(graph=graph, start_node=index_node_bw)
        num_visits += 1

        reachables_nodes = is_inB[idx_landmark]
        for i in range(num_landmark):
            if is_inA[i, index_node_bw]:
                reachables_nodes = np.logical_or(reachables_nodes, is_inB[i])

        _, ecc_fw = eccentricity_reachable(sp_dist.get_distances(), is_inA_B=reachables_nodes)
        indices_bw[idx_landmark] += 1
        to_be_considered_bw[index_node_bw] = False

        lb = max(lb, ecc_fw)
        if lb > (ub - 1):
            return lb, num_visits, num_distinct_A, num_distinct_B, pairs_number

        index_node_fw = ind_fw_node[idx_landmark][indices_fw[idx_landmark]]
        sp_dist.compute_distances(graph=g_op, start_node=index_node_fw)
        num_visits += 1

        reachables_nodes = is_inA[idx_landmark]
        for i in range(num_landmark):
            if is_inB[i, index_node_fw]:
                reachables_nodes = np.logical_or(reachables_nodes, is_inA[i])

        _, ecc_bw = eccentricity_reachable(sp_dist.get_distances(), is_inA_B=reachables_nodes)
        indices_fw[idx_landmark] += 1
        to_be_considered_fw[index_node_fw] = False

        lb = max(lb, ecc_bw)
        if lb > (ub - 1):
            return lb, num_visits, num_distinct_A, num_distinct_B, pairs_number

        max_distance = -1
        for i in range(num_landmark):
            while indices_fw[i] < len(distances_fw[i]) and not to_be_considered_fw[ind_fw_node[i][indices_fw[i]]]:
                indices_fw[i] += 1
            while indices_bw[i] < len(distances_bw[i]) and not to_be_considered_bw[ind_bw_node[i][indices_bw[i]]]:
                indices_bw[i] += 1

            if indices_fw[i] < len(distances_fw[i]) and indices_bw[i] < len(distances_bw[i]):
                if (distances_fw[i][indices_fw[i]] + distances_bw[i][indices_bw[i]]) > max_distance:
                    max_distance = distances_fw[i][indices_fw[i]] + distances_bw[i][indices_bw[i]]
                    idx_landmark = i

        if max_distance < 0:  # All nodes have been examined
            return lb, num_visits, num_distinct_A, num_distinct_B, pairs_number

        ub = max_distance

    return lb, num_visits, num_distinct_A, num_distinct_B, pairs_number


if __name__ == '__main__':

    # g = tg.Graph(file_path='graphs/transportation2/belfast_temporal_day.txt.sor', is_directed=True)
    g = tg.Graph(file_path='./graphs/transportation2/kuopio_temporal_day.txt.sor', is_directed=True)
    # Create array of tuples [(landmark_node_1, time_1)...(landmark_node_k, time_k)]
    _, nodes = g.get_max_deg_out(n=9)
    number_landmarks = len(nodes)
    a, b = g.get_time_interval()
    mid_t = round(((b - a) / 2) + a)
    mid_t1 = round(((b - a) / 4) + a)
    # mid_t2 = a
    lmarks = np.empty(number_landmarks * 2, dtype=object)
    k = 0
    # print(nodes)
    for eleme in nodes:
        lmarks[k] = (eleme, mid_t)
        lmarks[k + 1] = (eleme, mid_t1)
        k += 2
    print('Landmarks: ' + str(lmarks))
    print('\n')

    # Call ifub_on_landmarks() on graph and landmarks
    diam, num_visits_ifub, num_dist_A, num_dist_B, pairs_reach = ifub_on_landmarks(graph=g, landmarks=lmarks)
    print('Diameter: ' + str(diam))
    print('Number of visits: ' + str(num_visits_ifub))
