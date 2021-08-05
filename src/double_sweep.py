import temporal_distances as td
import temporal_graph as tg


class DoubleSweep:
    def __init__(self, graph: tg.Graph, start_node=None, t_alpha=None, t_omega=None):
        """
        :param graph: Temporal Graph (sorted in non-decreasing order with respect to the edge starting times)
        :param start_node: Starting node
        :param t_alpha: Lower bound time interval (if None it is the minimum strarting time in the graph)
        :param t_omega: Upper bound time interval (if None it is the minimum arrival time in the graph)
        """
        self.__graph = graph
        self.__t_alpha = t_alpha
        self.__t_omega = t_omega
        self.__lb_eat = None
        self.__source_eat = None
        self.__destination_eat = None
        self.__lb_ldt = None
        self.__source_ldt = None
        self.__destinantion_ldt = None
        self.__lb_ft = None
        self.__source_ft = None
        self.__destinantion_ft = None
        self.__lb_st = None
        self.__source_st = None
        self.__destinantion_st = None

        if start_node is None:
            _, self.__r = graph.get_max_deg_out(n=1)
            self.__r = self.__r[0]
        else:
            self.__r = start_node

    def get_start_node(self):
        return self.__r

    def get_lb_eat(self):
        """
        Compute 2-Sweep EAT and return the EAT diameter lower bound found, source and destination nodes in the path

        :returns:
            - A lower bound on EAT diameter
            - The source node in the path found
            - The destination node in the path found
        """
        if self.__lb_eat is None:
            self.__lb_eat, self.__source_eat, self.__destination_eat = \
                self.__compute_lb(distance=td.EarliestArrivalTime(graph=self.__graph, t_alpha=self.__t_alpha,
                                                                  t_omega=self.__t_omega))
        return self.__lb_eat, self.__source_eat, self.__destination_eat

    def get_lb_ldt(self):
        """
        Compute 2-Sweep LDT and return the LDT diameter lower bound found, source and destination nodes in the path

        :returns:
            - A lower bound on LDT diameter
            - The source node in the path found
            - The destination node in the path found
        """
        if self.__lb_ldt is None:
            self.__lb_ldt, self.__source_ldt, self.__destinantion_ldt = \
                self.__compute_lb(distance=td.LatestDepartureTime(graph=self.__graph, t_alpha=self.__t_alpha,
                                                                  t_omega=self.__t_omega))
        return self.__lb_ldt, self.__source_ldt, self.__destinantion_ldt

    def get_lb_ft(self):
        """
        Compute 2-Sweep FT and return the FT diameter lower bound found, source and destination nodes in the path

        :returns:
            - A lower bound on FT diameter
            - The source node in the path found
            - The destination node in the path found
        """
        if self.__lb_ft is None:
            self.__lb_ft, self.__source_ft, self.__destinantion_ft = \
                self.__compute_lb(distance=td.FastestTime(graph=self.__graph, t_alpha=self.__t_alpha,
                                                          t_omega=self.__t_omega))
        return self.__lb_ft, self.__source_ft, self.__destinantion_ft

    def get_lb_st(self):
        """
        Compute 2-Sweep ST and return the ST diameter lower bound found, source and destination nodes in the path

        :returns:
            - A lower bound on ST diameter
            - The source node in the path found
            - The destination node in the path found
        """
        if self.__lb_st is None:
            self.__lb_st, self.__source_st, self.__destinantion_st = \
                self.__compute_lb(distance=td.ShortestTime(graph=self.__graph, t_alpha=self.__t_alpha,
                                                           t_omega=self.__t_omega))
        return self.__lb_st, self.__source_st, self.__destinantion_st

    def __compute_lb(self, distance: td.Distance):
        """
        Compute a lower bound on the diameter (corresponding to parameter distance)

        :param distance: Distance object with which to calculate the lower bound
        :returns:
            - Lower bound
            - Source node in the path found
            - Destination node in the path found
        """

        a1 = distance.get_node_farther_fw(source=self.__r)
        lb1 = distance.get_eccentricity_bw(target=a1)
        b1 = distance.get_node_farther_bw(target=a1)

        a2 = distance.get_node_farther_bw(target=self.__r)
        lb2 = distance.get_eccentricity_fw(source=a2)
        b2 = distance.get_node_farther_fw(source=a2)

        if lb1 > lb2:
            return lb1, b1, a1
        else:
            return lb2, a2, b2
