import numpy as np
# import time
import logging

import tg_utils
import temporal_distances as td
import temporal_graph as tg

logging.basicConfig(level=logging.INFO)


# RUB = Reverse Upper Bound
# Since BW EAT work only with dummy nodes, this algorithm work only with dummy nodes
class RubDiameter:
    def __init__(self, graph: tg.Graph, t_alpha=None, t_omega=None):
        """
        :param graph: Temporal Graph (sorted in non-decreasing order with respect to the edge starting times)
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """
        self.__graph = graph
        self.__diam_eat = None
        self.__diam_ldt = None
        self.__visits_eat = None
        self.__visits_ldt = None
        self.__list_nodes = []

        if t_alpha is None:
            t_alpha, _ = graph.get_time_interval()
        if t_omega is None:
            _, t_omega = graph.get_time_interval()
        self.__t_alpha, self.__t_omega = t_alpha, t_omega

    def get_eat(self):
        """
        Compute EAT diameter and return the diameter value and the number of visits done

        :returns:
            - EAT Diameter
            - Number of visits done
        """
        if self.__diam_eat is None:
            self.__compute_eat()
        return self.__diam_eat, self.__visits_eat

    def get_ldt(self):
        """
        Compute LDT diameter and return the diameter value and the number of visits done

        :returns:
            - LDT Diameter
            - Number of visits done
        """
        if self.__diam_ldt is None:
            self.__compute_ldt()
        return self.__diam_ldt, self.__visits_ldt

    def __compute_eat(self):
        eat = td.EarliestArrivalTime(graph=self.__graph, t_alpha=self.__t_alpha, t_omega=self.__t_omega)
        self.__nodes_order_eat()

        lb = 0
        number_visits = 0
        for pair in self.__list_nodes:
            v = pair[0]
            ub = pair[1] - self.__t_alpha
            ecc = eat.get_eccentricity_bw(target=v)
            number_visits += 1
            # logging.info('Number of visits: {}'.format(number_visits))
            if ecc > lb:
                lb = ecc
            if ub <= lb:
                break
        self.__diam_eat, self.__visits_eat = lb, number_visits

    def __compute_ldt(self):
        ldt = td.LatestDepartureTime(graph=self.__graph, t_alpha=self.__t_alpha, t_omega=self.__t_omega)
        self.__nodes_order_ldt()

        lb = 0
        number_visits = 0
        for pair in self.__list_nodes:
            v = pair[0]
            ub = self.__t_omega + pair[1]
            ecc = ldt.get_eccentricity_fw(source=v)
            number_visits += 1
            # logging.info('Number of visits: {}'.format(number_visits))
            if ecc > lb:
                lb = ecc
            if ub <= lb:
                break
        self.__diam_ldt, self.__visits_ldt = lb, number_visits

    def __nodes_order_eat(self):
        graph_r_path = tg_utils.reverse_edges_sort(graph=self.__graph)
        graph_r = tg.Graph(file_path=graph_r_path, is_directed=self.__graph.get_is_directed(),
                           latest_node=self.__graph.get_latest_node())
        self.__nodes_order(graph=graph_r, t_alpha=self.__t_alpha, t_omega=self.__t_omega)

    def __nodes_order_ldt(self):
        graph_t_path = tg_utils.transform_graph(graph=self.__graph)
        graph_t = tg.Graph(file_path=graph_t_path, is_directed=self.__graph.get_is_directed(),
                           latest_node=self.__graph.get_latest_node())

        graph_tr_path = tg_utils.reverse_edges_sort(graph=graph_t)
        graph_tr = tg.Graph(file_path=graph_tr_path, is_directed=graph_t.get_is_directed(),
                            latest_node=graph_t.get_latest_node())

        self.__nodes_order(graph=graph_tr, t_alpha=-self.__t_omega, t_omega=-self.__t_alpha)

    def __nodes_order(self, graph: tg.Graph, t_alpha: int, t_omega: int):
        """
        :return: A list of tuples (node, time) sorted to be processed
        N.B. This ordering work only if all travel times are equals (as in the case of dummy nodes)
        """
        self.__list_nodes = []
        num_nodes = graph.get_n()
        nodes = np.full(num_nodes, True)

        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            traversal_time = 1
            if len(li) == 4:
                traversal_time = int(li[3])

            if v < num_nodes and nodes[v] and t >= t_alpha and t + traversal_time <= t_omega:
                nodes[v] = False
                self.__list_nodes.append((v, t + traversal_time))

            if not graph.get_is_directed():
                if u < num_nodes and nodes[u] and t >= t_alpha and t + traversal_time <= t_omega:
                    nodes[u] = False
                    self.__list_nodes.append((u, t + traversal_time))
