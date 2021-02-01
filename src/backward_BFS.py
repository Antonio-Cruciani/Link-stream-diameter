import numpy as np
from utils import util


class TemporalBackwardBFS:
    def __init__(self, graph, start_node):
        """
        Temporal Backward BFS on graph "graph", it must be in reverse edge stream representation
        N.B. If you use TemporalBackwardBFS to compute fw LDT you must change the sign of distances and eccentricity
        """
        self.__n_nodes = graph.get_num_nodes()
        self.__graph = graph
        # self.__file_path = graph.get_file_path()
        self.__is_directed = graph.get_is_directed()
        self.__s = start_node
        self.__min_time, self.__max_time = graph.get_time_interval()
        self.__node_interval = []
        self.__eat = []
        self.__dur = []
        self.__eccentricity_eat = None
        self.__idx_farther_eat = None
        self.__reachables = 0

    def get_eat(self):
        if len(self.__eat) == 0:
            self.bfs()
        return self.__eat

    def get_dur(self):
        if len(self.__dur) == 0:
            self.bfs()
        return self.__dur

    def get_eccentricity_eat(self):
        if self.__eccentricity_eat is None:
            self.bfs()
        return self.__eccentricity_eat

    def get_idx_farther_eat(self):
        """
        Return the index of node with highest eccentricity
        """
        if self.__idx_farther_eat is None:
            self.bfs()
        return self.__idx_farther_eat

    def get_graph_iter(self):
        return self.__graph.graph_reader()

    def get_reachables(self):
        return self.__reachables

    def bfs(self):
        """
        We assume constant traversal times
        """
        self.__eat = np.full((self.__n_nodes,), np.inf)
        self.__dur = np.full((self.__n_nodes,), np.inf)
        self.__eat[self.__s] = self.__min_time
        self.__dur[self.__s] = 0
        self.__eccentricity_eat = self.__min_time
        self.__idx_farther_eat = self.__s
        # self.node_interval = np.full((self.n_nodes,), Interval(np.inf, np.inf, np.inf, np.inf, np.inf))
        for i in range(self.__n_nodes):
            self.__node_interval.append(Interval(np.inf, np.inf, np.inf, np.inf, np.inf))
        last_t = self.__min_time - 1
        # with open(self.__file_path) as f:
        for line in self.get_graph_iter():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            # if len(li) != 3:
            #    raise Exception('Line ' + line + ' not have correct number of fields')
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            if self.__min_time <= t <= self.__max_time:
                if last_t != t:
                    self.__node_interval[self.__s].set(t + 1, t, t + 1, t, t + 1)
                    last_t = t
                lsu = self.__node_interval[u].min_s()
                lsv = self.__node_interval[v].min_s()
                prev_lsv = self.__node_interval[v].max_s()
                prev_lsu = self.__node_interval[u].max_s()
                if lsu[1] > t and lsv[1] > t:
                    if not self.__is_directed and lsu[0] < lsv[0]:
                        self.__node_interval[v].set_max(lsu[0], lsv[0], t)
                        self.__eat[v] = lsu[0] + 1
                        self.__dur[v] = lsu[0] - t + 1
                    elif lsu[0] > lsv[0]:
                        self.__node_interval[u].set_max(lsv[0], lsu[0], t)
                        self.__eat[u] = lsv[0] + 1
                        self.__dur[u] = lsv[0] - t + 1

                elif lsu[1] > t and lsv[1] == t:
                    if not self.__is_directed and lsu[0] < lsv[0]:
                        self.__node_interval[v].set_min(lsu[0])
                        self.__eat[v] = lsu[0] + 1
                        self.__dur[v] = lsu[0] - t + 1
                    elif lsu[0] > prev_lsv[0]:
                        self.__node_interval[u].set_max(prev_lsv[0], lsu[0], t)
                        self.__eat[u] = prev_lsv[0] + 1
                        self.__dur[u] = prev_lsv[0] - t + 1

                elif lsu[1] == t and lsv[1] > t:
                    if lsv[0] < lsu[0]:
                        self.__node_interval[u].set_min(lsv[0])
                        self.__eat[u] = lsv[0] + 1
                        self.__dur[u] = lsv[0] - t + 1
                    elif not self.__is_directed and lsv[0] > prev_lsu[0]:
                        self.__node_interval[v].set_max(prev_lsu[0], lsv[0], t)
                        self.__eat[v] = prev_lsu[0] + 1
                        self.__dur[v] = prev_lsu[0] - t + 1

                elif lsu[1] == t and lsv[1] == t:
                    if prev_lsv[0] < lsu[0]:
                        self.__node_interval[u].set_min(prev_lsv[0])
                        self.__eat[u] = prev_lsv[0] + 1
                        self.__dur[u] = prev_lsv[0] - t + 1
                    elif not self.__is_directed and prev_lsu[0] < lsv[0]:
                        self.__node_interval[v].set_min(prev_lsu[0])
                        self.__eat[v] = prev_lsu[0] + 1
                        self.__dur[v] = prev_lsu[0] - t + 1

        if self.__graph.get_latest_node() is not None:
            self.__eat = self.__eat[:self.__graph.get_latest_node()]
        self.__idx_farther_eat, self.__eccentricity_eat, self.__reachables = util.get_max_ind(self.__eat, np.NINF)


class Interval:
    def __init__(self, r, l1, s1, l2, s2):
        self.__r = r
        self.__l1 = l1
        self.__s1 = s1
        self.__l2 = l2
        self.__s2 = s2

    def set(self, r, l1, s1, l2, s2):
        self.__r = r
        self.__l1 = l1
        self.__s1 = s1
        self.__l2 = l2
        self.__s2 = s2

    def max_s(self):  # Get previous triple (oldest)
        r = [self.__l1, self.__s1]
        if self.__s2 > self.__s1:
            r[0] = self.__l2
            r[1] = self.__s2
        return r

    def min_s(self):  # Get most recent triple (last inserted)
        r = [self.__l1, self.__s1]
        if self.__s2 < self.__s1:
            r[0] = self.__l2
            r[1] = self.__s2
        return r

    def set_max(self, li, r, t):  # Update oldest triple
        self.__r = r
        if self.__s1 > self.__s2:
            self.__l1 = li
            self.__s1 = t
        else:
            self.__l2 = li
            self.__s2 = t

    def set_min(self, li):  # Extend (update) most recent triple (with smaller t)
        if self.__s1 < self.__s2:
            self.__l1 = li
        else:
            self.__l2 = li
