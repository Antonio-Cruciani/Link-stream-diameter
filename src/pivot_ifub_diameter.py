import temporal_distances as td
import temporal_graph as tg

import numpy as np


class PivotIfubDiameter:
    def __init__(self, graph: tg.Graph, pivots, t_alpha=None, t_omega=None):
        """
        :param graph: Temporal Graph (sorted in non-decreasing order with respect to the edge starting times)
        :param pivots: List of tuples (node, time)
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """
        if t_alpha is None:
            t_alpha, _ = graph.get_time_interval()
        if t_omega is None:
            _, t_omega = graph.get_time_interval()
        self._t_alpha, self._t_omega = t_alpha, t_omega

        for p in pivots:
            if t_omega < p[1] < t_alpha:
                raise Exception("Pivot times should be between t_alpha and t_omega")

        self.__graph = graph
        self.__pivots = pivots
        self.__t_alpha = t_alpha
        self.__t_omega = t_omega

        self.__is_inA = None  # Will be a matrix [num_pivots x num_nodes] of boolean
        self.__is_inB = None  # Will be a matrix [num_pivots x num_nodes] of boolean
        self.__reachable_pairs = None
        self.__distinct_inA = None
        self.__distinct_inB = None

        self.__pivot_diameter_st = None
        self.__num_visits_st = None

        self.__pivot_diameter_ft = None
        self.__num_visits_ft = None

        self.__pivot_diameter_eat = None
        self.__num_visits_eat = None

        self.__pivot_diameter_ldt = None
        self.__num_visits_ldt = None

        self.__dist_pivot_fw = []  # List of lists of tuples (distance, nodeIndex)
        self.__dist_pivot_bw = []  # List of lists of tuples (distance, nodeIndex)
        self.__next_node_inB = []  # List of indices for locate next node in B to process relative to each landmark
        self.__next_node_inA = []  # List of indices for locate next node in A to process relative to each landmark
        self.__to_be_considered_inB = []  # List of bool, if one element is False, that node has already been considered
        self.__to_be_considered_inA = []  # List of bool, if one element is False, that node has already been considered

    def get_st(self):
        """
        :returns:
            - ST pivot-diameter
            - Number of visits done
        """
        if self.__pivot_diameter_st is None:
            self.__pivot_diameter_st, self.__num_visits_st = self.__compute_diameter(td.ShortestTime)
        return self.__pivot_diameter_st, self.__num_visits_st

    def get_ft(self):
        """
        :returns:
            - FT pivot-diameter
            - Number of visits done
        """
        if self.__pivot_diameter_st is None:
            self.__pivot_diameter_ft, self.__num_visits_ft = self.__compute_diameter(td.FastestTime)
        return self.__pivot_diameter_ft, self.__num_visits_ft

    def get_eat(self):
        """
        :returns:
            - EAT pivot-diameter
            - Number of visits done
        """
        if self.__pivot_diameter_eat is None:
            self.__pivot_diameter_eat, self.__num_visits_eat = self.__compute_diameter(td.EarliestArrivalTime)
        return self.__pivot_diameter_eat, self.__num_visits_eat

    def get_ldt(self):
        """
        :returns:
            - LDT pivot-diameter
            - Number of visits done
        """
        if self.__pivot_diameter_ldt is None:
            self.__pivot_diameter_ldt, self.__num_visits_ldt = self.__compute_diameter(td.LatestDepartureTime)
        return self.__pivot_diameter_ldt, self.__num_visits_ldt

    def get_reachable_pairs(self):
        """
        :return: Number of pairs of reachable nodes
        """
        if self.__reachable_pairs is None:
            if self.__graph.get_latest_node() is not None:
                self.__compute_sets_a_b(td.EarliestArrivalTime)
            else:
                self.__compute_sets_a_b(td.ShortestTime)
        return self.__reachable_pairs

    def get_distinct_in_a(self):
        """
        :return: Number of distinct nodes in set A
        """
        if self.__distinct_inA is None:
            if self.__graph.get_latest_node() is not None:
                self.__compute_sets_a_b(td.EarliestArrivalTime)
            else:
                self.__compute_sets_a_b(td.ShortestTime)
        return self.__distinct_inA

    def get_distinct_in_b(self):
        """
        :return: Number of distinct nodes in set B
        """
        if self.__distinct_inB is None:
            if self.__graph.get_latest_node() is not None:
                self.__compute_sets_a_b(td.EarliestArrivalTime)
            else:
                self.__compute_sets_a_b(td.ShortestTime)
        return self.__distinct_inB

    def get_is_in_a(self):
        """
        :return: Matrix of boolean [num_pivots x num_nodes] where element [i,j] is True if node j can reach pivot i
        """
        if self.__is_inA is None:
            if self.__graph.get_latest_node() is not None:
                self.__compute_sets_a_b(td.EarliestArrivalTime)
            else:
                self.__compute_sets_a_b(td.ShortestTime)
        return self.__is_inA

    def get_is_in_b(self):
        """
        :return: Matrix of boolean [num_pivots x num_nodes] where element [i,j] is True if pivot i can reach node j
        """
        if self.__is_inB is None:
            if self.__graph.get_latest_node() is not None:
                self.__compute_sets_a_b(td.EarliestArrivalTime)
            else:
                self.__compute_sets_a_b(td.ShortestTime)
        return self.__is_inB

    def __compute_diameter(self, distance_type: type(td.Distance)):
        """
        :param distance_type: Type of distance
        :returns:
            - Pivot-diameter
            - Number of visits done
        """
        distance = distance_type(graph=self.__graph, t_alpha=self.__t_alpha, t_omega=self.__t_omega)
        self.__dist_pivot_fw, self.__dist_pivot_bw, num_visits = self.__compute_sets_a_b(distance_type)

        # List of indices for locate next node to process relative to each landmark
        self.__next_node_inB = np.full(len(self.__pivots), 0)
        self.__next_node_inA = np.full(len(self.__pivots), 0)

        # In the list to_be_considered, if one element is False, that node has already been considered in another visit
        self.__to_be_considered_inB = np.full(self.__graph.get_n(), True)
        self.__to_be_considered_inA = np.full(self.__graph.get_n(), True)

        # Get biggest distances fw and bw respectively from and to landmarks
        # Choose the upper bound like the higher upper bound from all landmark's upper bounds
        ub = -1
        pivot_idx = -1
        for i in range(len(self.__pivots)):
            if distance_type in {td.ShortestTime, td.FastestTime}:
                ub, pivot_idx = self.__select_pivot_st_ft(ub, pivot_idx, i, 0, 0)
            elif distance_type in {td.EarliestArrivalTime}:
                ub, pivot_idx = self.__select_pivot_eat(ub, pivot_idx, i, 0)
            elif distance_type in {td.LatestDepartureTime}:
                ub, pivot_idx = self.__select_pivot_ldt(ub, pivot_idx, i, 0)
            else:
                raise Exception("Distance type is not in ST, LT, EAT, LDT")

        lb = 0
        while ub > lb:
            if distance_type in {td.ShortestTime, td.FastestTime, td.LatestDepartureTime}:
                ecc_fw, num_visits = self.__get_lower_bound_fw(distance, pivot_idx, num_visits)
                lb = max(lb, ecc_fw)
                if lb >= ub:
                    return lb, num_visits

            if distance_type in {td.ShortestTime, td.FastestTime, td.EarliestArrivalTime}:
                ecc_bw, num_visits = self.__get_lower_bound_bw(distance, pivot_idx, num_visits)
                lb = max(lb, ecc_bw)
                if lb >= ub:
                    return lb, num_visits

            ub = -1
            pivot_idx = -1
            for i in range(len(self.__pivots)):
                if distance_type not in {td.LatestDepartureTime}:
                    while self.__next_node_inB[i] < len(self.__dist_pivot_fw[i]) and not \
                            self.__to_be_considered_inB[self.__dist_pivot_fw[i][self.__next_node_inB[i]][1]]:
                        self.__next_node_inB[i] += 1
                if distance_type not in {td.EarliestArrivalTime}:
                    while self.__next_node_inA[i] < len(self.__dist_pivot_bw[i]) and not \
                            self.__to_be_considered_inA[self.__dist_pivot_bw[i][self.__next_node_inA[i]][1]]:
                        self.__next_node_inA[i] += 1

                if self.__next_node_inB[i] < len(self.__dist_pivot_fw[i]) and \
                        self.__next_node_inA[i] < len(self.__dist_pivot_bw[i]):

                    if distance_type in {td.ShortestTime, td.FastestTime}:
                        ub, pivot_idx = self.__select_pivot_st_ft(
                            ub, pivot_idx, i, self.__next_node_inB[i], self.__next_node_inA[i])
                    elif distance_type in {td.EarliestArrivalTime}:
                        ub, pivot_idx = self.__select_pivot_eat(ub, pivot_idx, i, self.__next_node_inB[i])
                    elif distance_type in {td.LatestDepartureTime}:
                        ub, pivot_idx = self.__select_pivot_ldt(ub, pivot_idx, i, self.__next_node_inA[i])
                    else:
                        raise Exception("Distance type is not in ST, LT, EAT, LDT")

            if ub < 0:  # All nodes have been examined
                return lb, num_visits

        return lb, num_visits

    def __compute_sets_a_b(self, distance_type: type(td.Distance)):
        """
        :param distance_type: Type of distance
        :returns:
            - List of forward distances, where each element i corresponding to pivot i is a list of tuple
            (distance, nodeIndex) that is the distance forward from nodeIndex to pivot i
            - List of backward distances where each element i corresponding to pivot i is a list of tuple
            (distance, nodeIndex) that is the distance backward to nodeIndex from pivot i
            -
        """
        dist_fw, dist_bw, num_visits = self.__get_distances_pivots(distance_type)
        dist_pivot_fw, dist_pivot_bw = self.__compute_ina_inb(dist_fw, dist_bw)
        return dist_pivot_fw, dist_pivot_bw, num_visits

    def __get_distances_pivots(self, distance_type):
        """
        Compute distances forward/backward from/to landmarks nodes

        :param distance_type: Type of distance
        :returns:
            - List of distances forward from each pivot
            - List of distances backward to each pivot
        """
        num_visits = 0
        dist_fw = np.full((len(self.__pivots), self.__graph.get_n()), 0, dtype=float)
        dist_bw = np.full((len(self.__pivots), self.__graph.get_n()), 0, dtype=float)

        if distance_type in {td.ShortestTime}:
            for i, p in enumerate(self.__pivots):
                distance_fw = distance_type(graph=self.__graph, t_alpha=p[1], t_omega=self.__t_omega)
                dist_fw[i] = distance_fw.get_fw_distances(source=p[0])
                num_visits += 1

                distance_bw = distance_type(graph=self.__graph, t_alpha=self.__t_alpha, t_omega=p[1])
                dist_bw[i] = distance_bw.get_bw_distances(target=p[0])
                num_visits += 1
        elif distance_type in {td.FastestTime, td.EarliestArrivalTime, td.LatestDepartureTime}:
            for i, p in enumerate(self.__pivots):
                distance_fw = td.EarliestArrivalTime(graph=self.__graph, t_alpha=p[1], t_omega=self.__t_omega)
                dist_fw[i] = distance_fw.get_fw_distances(source=p[0])
                num_visits += 1

                distance_bw = td.LatestDepartureTime(graph=self.__graph, t_alpha=self.__t_alpha, t_omega=p[1])
                dist_bw[i] = distance_bw.get_bw_distances(target=p[0])
                num_visits += 1
        else:
            raise Exception("Distance type is not in ST, LT, EAT, LDT")
        return dist_fw, dist_bw, num_visits

    def __compute_ina_inb(self, dist_fw, dist_bw):
        """
        Compute number of pairs of reachables nodes, number of distinct node in A and B and return two lists of forward
        and backward distances, where each element i corresponding to pivot i is a list of tuple (distance, nodeIndex)
        that is the distance (forward or backward) from (or to) nodeIndex to (or from) pivot i


        :param dist_fw: List where each element i is a list of distances forward from pivot i to each node
        :param dist_bw: List where each element i is a list of distances backward to pivot i from each node
        :returns:
            - List of forward distances, where each element i corresponding to pivot i is a list of tuple
            (distance, nodeIndex) that is the distance forward from nodeIndex to pivot i
            - List of backward distances where each element i corresponding to pivot i is a list of tuple
            (distance, nodeIndex) that is the distance backward to nodeIndex from pivot i
        """
        dist_pivot_fw = []  # list of lists of tuples (distance, nodeIndex)
        dist_pivot_bw = []  # list of lists of tuples (distance, nodeIndex)

        self.__is_inA = np.full((len(self.__pivots), self.__graph.get_n()), False)
        self.__is_inB = np.full((len(self.__pivots), self.__graph.get_n()), False)

        for i in range(len(self.__pivots)):
            for j in range(self.__graph.get_n()):
                if dist_fw[i, j] != np.inf:
                    self.__is_inB[i, j] = True
                if dist_bw[i, j] != np.inf:
                    self.__is_inA[i, j] = True

            d_fw, d_bw = self.__sort_remove_unreachable(dist_fw[i], dist_bw[i])
            dist_pivot_fw.append(d_fw)
            dist_pivot_bw.append(d_bw)
        self.__count_reachable_pairs(self.__is_inA, self.__is_inB)
        self.__distinct_a_b(self.__is_inA, self.__is_inB)
        return dist_pivot_fw, dist_pivot_bw

    @staticmethod
    def __sort_remove_unreachable(dist_fw, dist_bw):
        """
        :param dist_fw: List of distances forward from each pivot
        :param dist_bw: List of distances backward to each pivot
        :returns:
            - List of distances forward from each pivot in descending order and without inf values, each element is a
            tuple (distance, nodeIndex)
            - List of distances backward to each pivot in descending order and without inf values, each element is a
            tuple (distance, nodeIndex)
        """

        # Sort fw distances in descending order and store associated nodes to each distance in ind_fw array
        ind_fw = np.argsort(-dist_fw)
        dist_fw = dist_fw[ind_fw]
        # Discard unreachable nodes from array
        unreachables = np.asarray(dist_fw == np.inf).nonzero()[0]
        if len(unreachables) > 0:
            unreachables_indices = unreachables[-1] + 1
            dist_fw = dist_fw[unreachables_indices:]
            ind_fw = ind_fw[unreachables_indices:]

        # Sort bw distances in descending order and store associated nodes to each distance in ind_bw array
        ind_bw = np.argsort(-dist_bw)
        dist_bw = dist_bw[ind_bw]
        # Discard unreachable nodes from array
        unreachables = np.asarray(dist_bw == np.inf).nonzero()[0]
        if len(unreachables) > 0:
            unreachables_indices = unreachables[-1] + 1
            dist_bw = dist_bw[unreachables_indices:]
            ind_bw = ind_bw[unreachables_indices:]
        return list(zip(dist_fw, ind_fw)), list(zip(dist_bw, ind_bw))

    def __count_reachable_pairs(self, is_inA, is_inB):
        """
        Count pairs of reachable nodes

        :param is_inA: Matrix of size [pivots x num_nodes], where element [i,j] is True if node j can reach pivot i
        :param is_inB: Matrix of size [pivots x num_nodes], where element [i,j] is True if pivot i can reach node j
        """
        count = 0
        for i in range(self.__graph.get_n()):
            reachable_nodes = np.full(self.__graph.get_n(), False)
            for j in range(len(self.__pivots)):
                if is_inA[j, i]:
                    reachable_nodes = np.logical_or(is_inB[j], reachable_nodes)
            count += np.count_nonzero(reachable_nodes)
        self.__reachable_pairs = count

    def __distinct_a_b(self, is_inA, is_inB):
        """
        Count number of distinct elements in A and in B

        :param is_inA: Matrix of size: pivots x num_nodes, where element [i,j] is True if node j can reach pivot i
        :param is_inB: Matrix of size: pivots x num_nodes, where element [i,j] is True if pivot i can reach node j
        """
        distinct_A = np.full(self.__graph.get_n(), False)
        distinct_B = np.full(self.__graph.get_n(), False)
        for i in range(len(self.__pivots)):
            distinct_A = np.logical_or(is_inA[i], distinct_A)
            distinct_B = np.logical_or(is_inB[i], distinct_B)
        self.__distinct_inA = np.count_nonzero(distinct_A)
        self.__distinct_inB = np.count_nonzero(distinct_B)

    def __select_pivot_st_ft(self, ub: int, pivot_idx: int, index_pivot: int, index_node_fw: int, index_node_bw: int):
        """
        :param ub: Current upper bound
        :param pivot_idx: Current pivot index
        :param index_pivot: Pivot index to consider for update the upper bound
        :param index_node_fw: Index of node in B to consider for update the upper bound
        :param index_node_bw: Index of node in A to consider for update the upper bound
        :returns:
            - Upper bound
            - Index of selected pivot
        """
        if (self.__dist_pivot_fw[index_pivot][index_node_fw][0] + self.__dist_pivot_bw[index_pivot][index_node_bw][0]) \
                > ub:
            ub = self.__dist_pivot_fw[index_pivot][index_node_fw][0] + \
                 self.__dist_pivot_bw[index_pivot][index_node_bw][0]
            pivot_idx = index_pivot
        return ub, pivot_idx

    def __select_pivot_eat(self, ub: int, pivot_idx: int, index_pivot: int, index_node: int):
        """
        N.B.: Given that EAT = EAT - t_alpha, for a fair comparison we add the t_alpha of each pivot
        (self.__pivots[i][1]) and subtract global t_alpha

        :param ub: Current upper bound
        :param pivot_idx: Current pivot index
        :param index_pivot: Pivot index to consider for update the upper bound
        :param index_node: Index of node in B to consider for update the upper bound
        :returns:
            - Upper bound
            - Index of selected pivot
        """
        if (self.__dist_pivot_fw[index_pivot][index_node][0] + self.__pivots[index_pivot][1]) - self.__t_alpha > ub:
            ub = self.__dist_pivot_fw[index_pivot][index_node][0] + self.__pivots[index_pivot][1] - self.__t_alpha
            pivot_idx = index_pivot
        return ub, pivot_idx

    def __select_pivot_ldt(self, ub: int, pivot_idx: int, index_pivot: int, index_node: int):
        """
        N.B.: Given that LDT = t_omega - LDT, for a fair comparison we subtract the
        t_omega of each pivot (self.__pivots[i][1]) and we add to -LDT the value of global t_omega

        :param ub: Current upper bound
        :param pivot_idx: Current pivot index
        :param index_pivot: Pivot index to consider for update the upper bound
        :param index_node: Index of node in A to consider for update the upper bound
        :returns:
            - Upper bound
            - Index of selected pivot
        """
        if self.__t_omega - (self.__pivots[index_pivot][1] - self.__dist_pivot_bw[index_pivot][index_node][0]) > ub:
            ub = self.__t_omega - (self.__pivots[index_pivot][1] - self.__dist_pivot_bw[index_pivot][index_node][0])
            pivot_idx = index_pivot
        return ub, pivot_idx

    def __get_lower_bound_fw(self, distance: td.Distance, pivot_idx: int, num_visits: int):
        """
        Compute ssbp from node in A associate to the pivot with the bigger upper bound

        :param distance: Distance object
        :param pivot_idx: Selected Pivot with bigger upper bound
        :param num_visits: Number of visits done
        :returns:
            - Ecentricity forward found
            - Number of visits updated
        """
        node_inA = self.__dist_pivot_bw[pivot_idx][self.__next_node_inA[pivot_idx]][1]
        distances_fw = distance.get_fw_distances(source=node_inA)
        num_visits += 1

        reachable_nodes = self.__is_inB[pivot_idx]
        for i in range(len(self.__pivots)):
            if self.__is_inA[i, node_inA]:
                reachable_nodes = np.logical_or(reachable_nodes, self.__is_inB[i])

        _, ecc = self.__eccentricity_reachable(distances=distances_fw, reachable_nodes=reachable_nodes)
        self.__next_node_inA[pivot_idx] += 1
        self.__to_be_considered_inA[node_inA] = False
        return ecc, num_visits

    def __get_lower_bound_bw(self, distance: td.Distance, pivot_idx: int, num_visits: int):
        """
        Compute sstp to node in B associate to the pivot with the bigger upper bound

        :param distance: Distance object
        :param pivot_idx: Selected Pivot with bigger upper bound
        :param num_visits: Number of visits done
        :returns:
            - Ecentricity backward found
            - Number of visits updated
        """
        node_inB = self.__dist_pivot_fw[pivot_idx][self.__next_node_inB[pivot_idx]][1]
        distances_bw = distance.get_bw_distances(target=node_inB)
        num_visits += 1

        reachable_nodes = self.__is_inA[pivot_idx]
        for i in range(len(self.__pivots)):
            if self.__is_inB[i, node_inB]:
                reachable_nodes = np.logical_or(reachable_nodes, self.__is_inA[i])

        _, ecc_bw = self.__eccentricity_reachable(distances=distances_bw, reachable_nodes=reachable_nodes)
        self.__next_node_inB[pivot_idx] += 1
        self.__to_be_considered_inB[node_inB] = False
        return ecc_bw, num_visits

    @staticmethod
    def __eccentricity_reachable(distances, reachable_nodes):
        """
        :param distances: Distance object
        :param reachable_nodes: List of boolean where each element is True if that node is reachable
        :return:
            - Index of farther node
            - Eccentricity
        """
        if len(distances) != len(reachable_nodes):
            raise Exception("Distances array has an incorrect length!")
        index = -1
        ecc = np.NINF
        for i, d in enumerate(distances):
            if reachable_nodes[i] and d > ecc:  # condition (d != np.inf) is guaranteed by reachable_nodes
                ecc = d
                index = i
        return index, ecc
