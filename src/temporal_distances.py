import backward_bfs as bwbfs
import tg_utils
import temporal_graph as tg

import numpy as np
import logging

from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)


class Distance(ABC):
    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        """
        :param graph: Input Graph (sorted in non-decreasing order with respect to the edge starting times)
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """
        self._name = None
        self._graph = graph
        self._source = None
        self._target = None
        self._eccentricity_fw = None
        self._eccentricity_bw = None
        self._node_farther_fw = None
        self._node_farther_bw = None
        self._reachables_fw = None
        self._reachables_bw = None
        self._dist_fw = None
        self._dist_bw = None
        self._diameter = None
        self._reachable_pairs = None

        if t_alpha is None:
            t_alpha, _ = graph.get_time_interval()
        if t_omega is None:
            _, t_omega = graph.get_time_interval()
        self._t_alpha, self._t_omega = t_alpha, t_omega

    def get_fw_distances(self, source):
        """
        :param source: Source node
        :return: Numpy array of forward distances from the source to each node in the graph
        """
        if self._source != source or self._dist_fw is None:
            self.ssbp(source_node=source)
        return self._dist_fw

    def get_bw_distances(self, target):
        """
        :param target: Target node
        :return: Numpy array of backward distances to the target from each node in the graph
        """
        if self._target != target or self._dist_bw is None:
            self.stbp(target_node=target)
        return self._dist_bw

    def get_node_farther_fw(self, source):
        """
        :param source: Source node
        :return: Index of node farther in all best paths starting from the source
        """
        if self._source != source or self._dist_fw is None:
            self.ssbp(source_node=source)
        return self._node_farther_fw

    def get_node_farther_bw(self, target):
        """
        :param target: Target node
        :return: Index of the furthest node in all best paths arriving to the target
        """
        if self._target != target or self._dist_bw is None:
            self.stbp(target_node=target)
        return self._node_farther_bw

    def get_eccentricity_fw(self, source):
        """
        :param source: Source node
        :return: Forward eccentricity of source node
        """
        if self._source != source or self._dist_fw is None:
            self.ssbp(source_node=source)
        return self._eccentricity_fw

    def get_eccentricity_bw(self, target):
        """
        :param target: Target node
        :return: Backward eccentricity of target node
        """
        if self._target != target or self._dist_bw is None:
            self.stbp(target_node=target)
        return self._eccentricity_bw

    def get_reachables_fw(self, source):
        """
        :param source: Source node
        :return: Number of reachable nodes starting from source node
        """
        if self._source != source or self._dist_fw is None:
            self.ssbp(source_node=source)
        return self._reachables_fw

    def get_reachables_bw(self, target):
        """
        :param target: Target node
        :return: Number of nodes that can reach target node
        """
        if self._target != target or self._dist_bw is None:
            self.stbp(target_node=target)
        return self._reachables_bw

    def get_name(self):
        """
        :return: Distance name
        """
        return self._name

    def get_diameter(self):
        """
        :return: Diameter value
        """
        if self._diameter is None:
            self.compute_diameter()
        return self._diameter

    def get_reachable_pairs(self):
        """
        :return: Number of pairs of reachable nodes in Graph
        """
        if self._reachable_pairs is None:
            self.compute_diameter()
        return self._reachable_pairs

    def compute_diameter(self):
        """
        :returns:
            - Diameter value
            - Starting node in longest path
            - Arrival node in longest path
            - Number of reachable pairs of nodes
        """
        diam = np.NINF
        node_to = -1
        node_from = -1
        reachable_pairs = 0

        num_nodes = self._graph.get_n()

        for node in range(num_nodes):
            self.ssbp(source_node=node)
            reachable_pairs += self.get_reachables_fw(source=node)

            if self._eccentricity_fw > diam:
                diam, node_from, node_to = self._eccentricity_fw, node, self._node_farther_fw

            if (node + 1) % 1000 == 0:
                logging.info('{} visits done...'.format(node+1))

        self._diameter, self._reachable_pairs = diam, reachable_pairs
        return diam, node_from, node_to, reachable_pairs

    @staticmethod
    def _compute_ecc_reach(distances):
        """
        :param distances: Array of distances
        :returns:
            - Index of node at maximum distance
            - Eccentricity of node at maximum distance
            - Number of nodes reachables
        """
        node_idx = -1
        node_count = 0
        reachables = 0
        ecc = np.NINF
        for dist in distances:
            if dist != np.inf:
                reachables += 1
                if dist > ecc:
                    ecc = dist
                    node_idx = node_count
            node_count += 1
        return node_idx, ecc, reachables

    @abstractmethod
    def ssbp(self, source_node: int):
        """
        SINGLE SOURCE BEST PATH

        Compute: Numpy array of distances from source_node to every node v in V within time interval, eccentricity fw
        of source_node, destination node in longest path, number of nodes reachables from source_node
        """
        self._source = source_node

    @abstractmethod
    def stbp(self, target_node: int):
        """
        SINGLE TARGET BEST PATH

        Compute: Numpy array of distances to target_node from every node v in V within time interval, eccentricity bw
        of target_node, starting node in longest path, number of nodes that reaching target_node
        """
        self._target = target_node


class ShortestTime(Distance):
    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        super().__init__(graph, t_alpha, t_omega)
        self._name = 'ST'  # Shortest time

    @staticmethod
    def __compute_st(u: int, v: int, t: int, traversal_time: int, source_node: int, list_of_sorted_lists,
                     distances):
        # print(list_of_sorted_lists)
        if u == source_node:
            if (0, t) not in list_of_sorted_lists[source_node]:
                list_of_sorted_lists[source_node].append((0, t))

        if list_of_sorted_lists[u]:  # check if list is not empty
            for i, elem in enumerate(list_of_sorted_lists[u]):
                if elem[1] <= t:
                    new_d = elem[0] + traversal_time
                    new_arrival_time = t + traversal_time
                    inserted_new = False  # True if new element is inserted
                    if not list_of_sorted_lists[v]:  # Check if list is empty
                        list_of_sorted_lists[v].append((new_d, new_arrival_time))
                        inserted_new = True
                    else:
                        same_arrival = False  # True if there is another elem. with same arrival time into the list
                        for j, e in enumerate(list_of_sorted_lists[v]):
                            if e[1] == new_arrival_time:
                                if e[0] > new_d:
                                    list_of_sorted_lists[v][j] = (new_d, new_arrival_time)
                                    inserted_new = True
                                same_arrival = True
                        if not same_arrival:
                            # check if new path is dominated
                            new_is_dominated = False
                            for pair in list_of_sorted_lists[v]:
                                if (pair[0] < new_d and pair[1] <= new_arrival_time) or (
                                        pair[0] == new_d and pair[1] < new_arrival_time):
                                    new_is_dominated = True
                            if not new_is_dominated:
                                list_of_sorted_lists[v].append((new_d, new_arrival_time))
                                inserted_new = True
                    if inserted_new:
                        # keep only not dominated paths
                        list_of_sorted_lists[v][:] = [e for e in list_of_sorted_lists[v] if not (
                                (e[0] > new_d and e[1] >= new_arrival_time) or
                                (e[0] == new_d and e[1] > new_arrival_time))]
                        # If the minimum duration changes, update it
                        if new_d < distances[v]:
                            distances[v] = new_d
                        # sort list
                        list_of_sorted_lists[v].sort(key=lambda tup: tup[0])
                    break
                else:
                    continue
        return distances

    def __best_path(self, graph: tg.Graph, source_node: int, t_alpha: int, t_omega: int):
        """
        Compute: Numpy array of travel time of best paths from or to source_node depending on input graph
        (if it is a transformed link stream it compute bw stats), within time interval, eccentricity (fw or bw) of
        source_node, destination (or source) node in longest path, number of nodes reachables from (or reaching)
        source_node

        :param graph: Input Graph
        :param source_node: Node to start from
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """

        if graph.get_latest_node() is not None:
            raise Exception('Shortest path does not work with dummy nodes, give me in input a weighted graph!')

        list_of_sorted_lists = []
        for elem in range(graph.get_num_nodes()):
            list_of_sorted_lists.append([])

        dist_fw = np.full((graph.get_num_nodes(),), np.inf)
        dist_fw[source_node] = 0

        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])

            if len(li) == 4:
                traversal_time = int(li[3])
            else:
                traversal_time = 1

            if t >= t_alpha and t + traversal_time <= t_omega:
                dist_fw = self.__compute_st(u=u, v=v, t=t, traversal_time=traversal_time, source_node=source_node,
                                            list_of_sorted_lists=list_of_sorted_lists, distances=dist_fw)
                if not self._graph.get_is_directed():
                    dist_fw = self.__compute_st(u=v, v=u, t=t, traversal_time=traversal_time, source_node=source_node,
                                                list_of_sorted_lists=list_of_sorted_lists, distances=dist_fw)
            elif t >= t_omega:
                break
        node_farther, eccentricity, reachables = self._compute_ecc_reach(distances=dist_fw)
        return dist_fw, node_farther, eccentricity, reachables

    def ssbp(self, source_node: int):
        """
        Compute: Numpy array of travel time of best paths from source_node to every nodes v in V within time
        interval, eccentricity fw of source_node, destination node in longest path, number of nodes reachables from
        source_node

        :param source_node: Node to start from
        :returns:
            - Distances forward
            - Node farther, starting from source_node
            - Eccentricity forward
            - Number of reachables nodes, starting from source_node

        """
        super().ssbp(source_node=source_node)

        self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw = \
            self.__best_path(graph=self._graph, source_node=source_node, t_alpha=self._t_alpha, t_omega=self._t_omega)
        return self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw

    def stbp(self, target_node: int):
        """
        Compute: Numpy array of travel time of best paths to target_node from every nodes v in V within time interval,
        eccentricity bw of target_node, starting node in longest path, number of nodes reaching target_node

        :param target_node: Target node
        :returns:
            - Distances backward
            - Node farther, arriving to target_node
            - Eccentricity forward
            - Number of reachables nodes, arriving to target_node
        """
        super().stbp(target_node=target_node)

        if self._graph.get_latest_node() is not None:
            raise Exception('Shortest path does not work with dummy nodes, give me in input a weighted graph!')

        graph_t_path = tg_utils.transform_graph(graph=self._graph)
        graph_t = tg.Graph(file_path=graph_t_path, is_directed=self._graph.get_is_directed(),
                           latest_node=self._graph.get_latest_node())

        self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw = \
            self.__best_path(graph=graph_t, source_node=target_node, t_alpha=-self._t_omega, t_omega=-self._t_alpha)
        return self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw


class FastestTime(Distance):
    """
    NOTE: This version of FastestTime work only with traversal times = 1 (or when all traversal times are equals)
    """

    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        super().__init__(graph, t_alpha, t_omega)
        self._name = 'FT'  # Fastest time

    @staticmethod
    def __compute_ft(u: int, v: int, t: int, source_node: int, list_of_queues, distances):
        if u == source_node:
            if (t, t) not in list_of_queues[source_node]:
                list_of_queues[source_node].append((t, t))

        if list_of_queues[u]:  # check if list is not empty
            # for elem in reversed(list_of_queues[u]):
            for i, elem in reversed(list(enumerate(list_of_queues[u]))):
                if elem[1] <= t:
                    # list_of_queues[u] = list_of_queues[u][list_of_queues[u].index(elem):]
                    list_of_queues[u] = list_of_queues[u][i:]
                    new_t = elem[0]
                    new_arrival_time = t + 1  # traversal_time = 1
                    if not list_of_queues[v]:  # Check if list is empty
                        list_of_queues[v].append((new_t, new_arrival_time))

                    # If the old path dominates the new (or they are equals), keep only the old
                    elif (list_of_queues[v][-1][0] >= new_t and
                          list_of_queues[v][-1][1] <= new_arrival_time):
                        break

                    # If new path dominates the old, keep only the new
                    elif (list_of_queues[v][-1][0] < new_t and
                          list_of_queues[v][-1][1] >= new_arrival_time) or (list_of_queues[v][-1][0] == new_t and
                                                                            list_of_queues[v][-1][1] >
                                                                            new_arrival_time):
                        list_of_queues[v][-1] = (new_t, new_arrival_time)

                    # Otherwise keep both (newest is added after the oldest)
                    else:
                        list_of_queues[v].append((new_t, new_arrival_time))

                    # If the minimum duration changes, update it
                    if list_of_queues[v][-1][1] - list_of_queues[v][-1][0] < distances[v]:
                        distances[v] = list_of_queues[v][-1][1] - list_of_queues[v][-1][0]

                    break
                else:
                    continue
        return distances

    def __best_path(self, graph: tg.Graph, source_node: int, t_alpha: int, t_omega: int):
        """
        Compute: Numpy array of duration of the best paths from or to source_node depending on input graph (if it is a
        transformed link stream it compute bw stats) , within time interval, eccentricity (fw or bw) of source_node,
        destination (or source) node in longest path, number of nodes reachables from (or reaching) source_node

        :param source_node: Node to start from
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """

        list_of_queues = []
        for elem in range(graph.get_num_nodes()):
            list_of_queues.append([])

        dist_fw = np.full((graph.get_num_nodes(),), np.inf)
        dist_fw[source_node] = 0

        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            if len(li) > 3:
                raise Exception('Line ' + line + ' not have correct number of fields: This FT algorithm work only with '
                                                 'unweighted graph, transform it using dummies nodes before!')
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])

            if t >= t_alpha and t + 1 <= t_omega:
                dist_fw = self.__compute_ft(u=u, v=v, t=t, source_node=source_node, list_of_queues=list_of_queues,
                                            distances=dist_fw)
                if not graph.get_is_directed():
                    dist_fw = self.__compute_ft(u=v, v=u, t=t, source_node=source_node, list_of_queues=list_of_queues,
                                                distances=dist_fw)
            elif t >= t_omega:
                break

        # To handle dummy nodes: discard them from the distances vector
        if graph.get_latest_node() is not None:
            dist_fw = dist_fw[:graph.get_latest_node()]

        node_farther, eccentricity, reachables = self._compute_ecc_reach(distances=dist_fw)
        return dist_fw, node_farther, eccentricity, reachables

    def ssbp(self, source_node: int):
        """
        Compute: Numpy array of duration of best paths from source_node to every nodes v in V within time interval,
        eccentricity fw of source_node, destination node in longest path, number of nodes reachables from
        source_node

        :param source_node: Node to start from
        :returns:
            - Distances forward
            - Node farther, starting from source_node
            - Eccentricity forward
            - Number of reachables nodes, starting from source_node
        """
        super().ssbp(source_node=source_node)
        self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw = \
            self.__best_path(graph=self._graph, source_node=source_node, t_alpha=self._t_alpha, t_omega=self._t_omega)
        return self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw

    def stbp(self, target_node: int):
        """
        Compute: Numpy array of duration of best paths to target_node from every nodes v in V within time interval,
        eccentricity bw of target_node, starting node in longest path, number of nodes reaching target_node

        :param target_node: Target node
        :returns:
            - Distances backward
            - Node farther, arriving to target_node
            - Eccentricity forward
            - Number of reachables nodes, arriving to target_node
        """
        super().stbp(target_node=target_node)
        graph_t_path = tg_utils.transform_graph(graph=self._graph)
        graph_t = tg.Graph(file_path=graph_t_path, is_directed=self._graph.get_is_directed(),
                           latest_node=self._graph.get_latest_node())
        self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw = \
            self.__best_path(graph=graph_t, source_node=target_node, t_alpha=-self._t_omega, t_omega=-self._t_alpha)
        return self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw


class EarliestArrivalTime(Distance):
    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        super().__init__(graph, t_alpha, t_omega)
        self._name = 'EAT'

    def ssbp(self, source_node):
        """
        Compute: Numpy array of the earliest arrival times from source_node to every nodes v in V within time interval,
        eccentricity fw of source_node, destination node in longest path, number of nodes reachables from
        source_node

        :param source_node: Node to start from
        :returns:
            - Distances forward
            - Node farther, starting from source_node
            - Eccentricity forward
            - Number of reachables nodes, starting from source_node
        """
        super().ssbp(source_node=source_node)
        self._dist_fw = np.full((self._graph.get_num_nodes(),), np.inf)
        self._dist_fw[source_node] = self._t_alpha

        for line in self._graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            traversal_time = 1
            if len(li) == 4:
                traversal_time = int(li[3])

            if (t + traversal_time) <= self._t_omega:
                if t >= self._dist_fw[u]:
                    if (t + traversal_time) < self._dist_fw[v]:
                        self._dist_fw[v] = t + traversal_time

                elif not self._graph.get_is_directed() and t >= self._dist_fw[v]:  # If graph is undirected
                    if (t + traversal_time) < self._dist_fw[u]:
                        self._dist_fw[u] = t + traversal_time

            elif t >= self._t_omega:
                break
            else:
                continue
        # To handle dummy nodes: discard them from the distances vector
        if self._graph.get_latest_node() is not None:
            self._dist_fw = self._dist_fw[:self._graph.get_latest_node()]

        self._node_farther_fw, self._eccentricity_fw, self._reachables_fw = \
            self._compute_ecc_reach(distances=self._dist_fw)

        self._dist_fw = self._dist_fw - self._t_alpha
        self._eccentricity_fw = self._eccentricity_fw - self._t_alpha
        return self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw

    def stbp(self, target_node: int):
        """
        Compute: Numpy array of the earliest arrival times to target_node from every nodes v in V within time interval,
        eccentricity bw of target_node, destination node in longest path, number of nodes reaching target_node

        :param target_node: Target node
        :returns:
            - Distances backward
            - Node farther, arriving to target_node
            - Eccentricity forward
            - Number of reachables nodes, arriving to target_node
        """
        super().stbp(target_node=target_node)
        graph_r_path = tg_utils.reverse_edges_sort(graph=self._graph)
        graph_r = tg.Graph(file_path=graph_r_path, is_directed=self._graph.get_is_directed(),
                           latest_node=self._graph.get_latest_node())
        back_bfs = bwbfs.TemporalBackwardBFS(graph=graph_r, target_node=target_node, t_alpha=self._t_alpha,
                                             t_omega=self._t_omega)
        back_bfs.bfs()
        self._dist_bw = back_bfs.get_eat()
        self._eccentricity_bw = back_bfs.get_eccentricity_eat()
        self._node_farther_bw = back_bfs.get_idx_farther_eat()
        self._reachables_bw = back_bfs.get_reachables()

        self._dist_bw = self._dist_bw - self._t_alpha
        self._eccentricity_bw = self._eccentricity_bw - self._t_alpha
        return self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw


class LatestDepartureTime(Distance):
    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        super().__init__(graph, t_alpha, t_omega)
        self._name = 'LDT'

    def ssbp(self, source_node: int):
        """
        Compute: Numpy array of the latest departure times from source_node to every nodes v in V within time interval,
        eccentricity fw of source_node, destination node in longest path, number of nodes reachables from source_node

        :param source_node: Node to start from
        :returns:
            - Distances forward
            - Node farther, starting from source_node
            - Eccentricity forward
            - Number of reachables nodes, starting from source_node
        """
        super().ssbp(source_node=source_node)
        graph_t_path = tg_utils.transform_graph(graph=self._graph)
        graph_t = tg.Graph(file_path=graph_t_path, is_directed=self._graph.get_is_directed(),
                           latest_node=self._graph.get_latest_node())

        graph_tr_path = tg_utils.reverse_edges_sort(graph=graph_t)
        graph_tr = tg.Graph(file_path=graph_tr_path, is_directed=graph_t.get_is_directed(),
                            latest_node=graph_t.get_latest_node())
        back_bfs = bwbfs.TemporalBackwardBFS(graph=graph_tr, target_node=source_node, t_alpha=-self._t_omega,
                                             t_omega=-self._t_alpha)
        back_bfs.bfs()
        self._dist_fw = -(back_bfs.get_eat())
        self._eccentricity_fw = -(back_bfs.get_eccentricity_eat())
        self._node_farther_fw = back_bfs.get_idx_farther_eat()
        self._reachables_fw = back_bfs.get_reachables()

        self._dist_fw = self._t_omega - self._dist_fw
        self._eccentricity_fw = self._t_omega - self._eccentricity_fw
        return self._dist_fw, self._node_farther_fw, self._eccentricity_fw, self._reachables_fw

    def stbp(self, target_node: int):
        """
        Compute: Numpy array of the latest departure times to target_node from every nodes v in V within time interval,
        eccentricity bw of target_node, destination node in longest path, number of nodes reaching target_node

        :param target_node: Target node
        :returns:
            - Distances backward
            - Node farther, arriving to target_node
            - Eccentricity forward
            - Number of reachables nodes, arriving to target_node
        """
        super().stbp(target_node=target_node)
        graph_r_path = tg_utils.reverse_edges_sort(graph=self._graph)
        graph_r = tg.Graph(file_path=graph_r_path, is_directed=self._graph.get_is_directed(),
                           latest_node=self._graph.get_latest_node())

        self._dist_bw = np.full((graph_r.get_num_nodes(),), np.NINF)
        self._dist_bw[target_node] = self._t_omega

        for line in graph_r.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            traversal_time = 1
            if len(li) == 4:
                traversal_time = int(li[3])

            if t >= self._t_alpha:
                if t + traversal_time <= self._dist_bw[v]:
                    if t > self._dist_bw[u]:
                        self._dist_bw[u] = t
                elif not graph_r.get_is_directed() and t + traversal_time <= self._dist_bw[u]:
                    if t > self._dist_bw[v]:
                        self._dist_bw[v] = t
            else:
                break
        # To handle dummy nodes: discard them from the distances vector
        if graph_r.get_latest_node() is not None:
            self._dist_bw = self._dist_bw[:graph_r.get_latest_node()]

        self._dist_bw = self._t_omega - self._dist_bw  # np.NINF become np.inf (consistent with other distances)
        self._node_farther_bw, self._eccentricity_bw, self._reachables_bw = self._compute_ecc_reach(
            distances=self._dist_bw)

        return self._dist_bw, self._node_farther_bw, self._eccentricity_bw, self._reachables_bw
