import numpy as np


class Graph:
    def __init__(self, file_path, is_directed, latest_node=None):
        """
        :param file_path: graph's path
        :param file_path: path of graph's file
        :param is_directed: False indirected graph, True directed graph
        :param latest_node: if it is a value (not None) that value represents latest not dummy node + 1
        """
        self.__file_path = file_path
        self.__txt_list = None
        self.__num_nodes = None
        self.__min_time = None
        self.__max_time = None
        self.__edges_number = None
        self.__is_directed = is_directed
        self.__latest_node = latest_node
        self.__degrees_out = None
        self.__degrees_in = None
        self.__distinct_out_neighbors = None
        self.__distinct_in_neighbors = None
        self.__check_latest_node()

    def get_file_path(self):
        """
        Return the file path of the graph

        :return: Path of the graph file
        """
        return self.__file_path

    def graph_reader(self):
        """
        Iterator on link stream's rows

        :return: Next row of link stream file
        """
        with open(self.__file_path, "r") as f:
            for row in f:
                yield row

        # if self.__txt_list is None:
        #    self.__txt_list = []
        #    with open(self.__file_path, "r") as f:
        #        for row in f:
        #            row = row.rstrip('\n')
        #            self.__txt_list.append(row)
        # for elem in self.__txt_list:
        #    yield elem

    def get_num_nodes(self):
        """
        Total number of nodes in link stream

        :return: Total number of nodes in link stream
        """
        if self.__num_nodes is None:
            self.__graph_proprieties()
        return self.__num_nodes

    def get_time_interval(self):
        """
        Return time interval (t_alpha, t_omega) in which the graph is defined

        :return: Tuple (t_alpha, t_omega)
        """
        if self.__min_time is None or self.__max_time is None:
            self.__graph_proprieties()
        return self.__min_time, self.__max_time

    def get_latest_node(self):
        return self.__latest_node

    def get_n(self):
        """
        :return: Number of nodes of the original Graph
        """
        if self.__latest_node is not None:
            return self.__latest_node
        else:
            return self.get_num_nodes()

    def __check_latest_node(self):
        """
        Check if the latest node not dummy of the link stream is less or equal of the total number of nodes of link
        stream
        """
        if self.__latest_node is not None:
            if self.__latest_node > self.get_num_nodes():
                raise Exception('Parameter latest_node must be less then or equal to the number of nodes in the Graph')

    def get_edges_number(self):
        """
        Return the number of temporal edges in the link stream

        :return: Number of temporal edges in the link stream
        """
        if self.__edges_number is None:
            self.__graph_proprieties()
        return self.__edges_number

    def get_is_directed(self):
        """
        Check if the link stream is directer or undirected

        :return: True if it is directected, False if it is undirected
        """
        return self.__is_directed

    def get_max_deg_out(self, n=1):
        """
        Return a tuple of arrays (degrees[], indices[]) where indices[] is a list of n nodes with highest out-degree in
        decreasing order of out-degree and degrees[] is a list of out-degree for each node in indeces[]

        :param n: Number of nodes to return
        :return: Tuple (degrees[], indices[])
        """
        if self.__degrees_out is None:
            self.__get_degrees()
        sorted_deg_out_index = np.argsort(-self.__degrees_out)
        sorted_deg_out = self.__degrees_out[sorted_deg_out_index]
        return sorted_deg_out[:n], sorted_deg_out_index[:n]

    def get_max_deg_in(self, n=1):
        """
        Return a tuple of arrays (degrees[], indices[]) where indices[] is a list of n nodes with highest in-degree in
        decreasing order of in-degree and degrees[] is a list of in-degree for each node in indeces[]

        :param n: Number of nodes to return
        :return: Tuple (degrees[], indices[])
        """
        if self.__degrees_in is None:
            self.__get_degrees()
        sorted_deg_in_index = np.argsort(-self.__degrees_in)
        sorted_deg_in = self.__degrees_in[sorted_deg_in_index]
        return sorted_deg_in[:n], sorted_deg_in_index[:n]

    def get_max_deg_total(self, n=1):
        """
        Return a tuple of arrays (degrees[], indices[]) where indices[] is a list of n nodes with highest
        (out-degree + in_degree) in decreasing order of that values and degrees[] is a list of (out-degree + in_degree)
        for each node in indeces[]

        :param n: Number of nodes to return
        :return: Tuple (degrees[], indices[])
        """
        if self.__degrees_out is None or self.__degrees_in is None:
            self.__get_degrees()
        sum_degrees = self.__degrees_out + self.__degrees_in
        sorted_degrees_index = np.argsort(-sum_degrees)
        sorted_degrees = sum_degrees[sorted_degrees_index]
        return sorted_degrees[:n], sorted_degrees_index[:n]

    def get_distinct_out_neighbors(self, n=1):
        """
        N.B. WORK ONLY WITH GRAPHS WITHOUT DUMMY NODES

        Return a tuple of arrays (degrees[], indices[]) where indices[] is a list of n nodes with highest number of
        out-neighbors in decreasing order of out-neighbors and degrees[] is a list of the number of out-neighbors for
        each node in indeces[]

        :param n: Number of nodes to return
        :return: Tuple (degrees[], indices[])
        """
        if self.__distinct_out_neighbors is None:
            self.__get_distinct_neighbors()
        sorted_distinct_out_neighbors_index = np.argsort(-self.__distinct_out_neighbors)
        sorted_distinct_out_neighbors = self.__distinct_out_neighbors[sorted_distinct_out_neighbors_index]
        return sorted_distinct_out_neighbors[:n], sorted_distinct_out_neighbors_index[:n]

    def get_distinct_in_neighbors(self, n=1):
        """
        N.B. WORK ONLY WITH GRAPHS WITHOUT DUMMY NODES

        Return a tuple of arrays (degrees[], indices[]) where indices[] is a list of n nodes with highest number of
        in-neighbors in decreasing order of in-neighbors and degrees[] is a list of the number of in-neighbors for
        each node in indeces[]

        :param n: Number of nodes to return
        :return: Tuple (degrees[], indices[])
        """
        if self.__distinct_in_neighbors is None:
            self.__get_distinct_neighbors()
        sorted_distinct_in_neighbors_index = np.argsort(-self.__distinct_in_neighbors)
        sorted_distinct_in_neighbors = self.__distinct_in_neighbors[sorted_distinct_in_neighbors_index]
        return sorted_distinct_in_neighbors[:n], sorted_distinct_in_neighbors_index[:n]

    def get_distinct_total_neighbors(self, n=1):
        """
        :return: The n vertices with max number of distinct neighbors in
        N.B. work only with graphs without dummy nodes
        """
        if self.__distinct_in_neighbors is None or self.__distinct_out_neighbors is None:
            self.__get_distinct_neighbors()
        sum_degrees = self.__distinct_out_neighbors + self.__distinct_in_neighbors
        sorted_degrees_index = np.argsort(-sum_degrees)
        sorted_degrees = sum_degrees[sorted_degrees_index]
        return sorted_degrees[:n], sorted_degrees_index[:n]

    def __graph_proprieties(self):
        """
        Compute some graph properties
        """
        max_node = -1
        min_time = np.inf
        max_time = np.NINF
        num_edges = 0
        for line in self.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            if 3 > len(li) > 4:
                raise Exception('Line ' + line + ' not have correct number of fields')
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            if len(li) == 4:
                traversal_time = int(li[3])
            else:
                traversal_time = 1
            if u > max_node:
                max_node = u
            if v > max_node:
                max_node = v
            if t + traversal_time > max_time:
                max_time = t + traversal_time
            if t < min_time:
                min_time = t
            num_edges += 1
        max_node += 1
        self.__num_nodes = max_node
        self.__min_time = min_time
        self.__max_time = max_time
        self.__edges_number = num_edges

    def __get_degrees(self):
        """
        Compute the degree of each node
        """
        if self.__latest_node is not None:
            self.__degrees_out = np.full((self.__latest_node,), 0)
            self.__degrees_in = np.full((self.__latest_node,), 0)
        else:
            self.__degrees_out = np.full((self.get_num_nodes(),), 0)
            self.__degrees_in = np.full((self.get_num_nodes(),), 0)

        for line in self.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            if 3 > len(li) > 4:
                raise Exception('Line ' + line + ' not have correct number of fields')
            u = int(li[0])
            v = int(li[1])

            if self.__latest_node is not None:
                if u < self.__latest_node:
                    self.__degrees_out[u] += 1
                    if not self.__is_directed:
                        self.__degrees_in[u] += 1
                if v < self.__latest_node:
                    self.__degrees_in[v] += 1
                    if not self.__is_directed:
                        self.__degrees_out[v] += 1
            else:
                self.__degrees_out[u] += 1
                self.__degrees_in[v] += 1
                if not self.__is_directed:
                    self.__degrees_out[v] += 1
                    self.__degrees_in[u] += 1

    def __get_distinct_neighbors(self):
        """
        Compute the number of distinct (in and out) neighbors for each node
        """
        # N.B. work only with graphs without dummy nodes (different dummy nodes can be connected to the same out node)
        out_neighbors = []
        in_neighbors = []
        if self.__latest_node is not None:
            raise Exception("This method compute distinct neighbors only for graphs without dummy nodes!")

        else:
            self.__distinct_out_neighbors = np.full((self.get_num_nodes(),), 0)
            self.__distinct_in_neighbors = np.full((self.get_num_nodes(),), 0)
            for elem in range(self.get_num_nodes()):
                out_neighbors.append(set())
                in_neighbors.append(set())

        for line in self.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            if 3 > len(li) > 4:
                raise Exception('Line ' + line + ' not have correct number of fields')
            u = int(li[0])
            v = int(li[1])

            out_neighbors[u].add(v)
            in_neighbors[v].add(u)
            if not self.__is_directed:
                out_neighbors[v].add(u)
                in_neighbors[u].add(v)

        for i in range(self.get_num_nodes()):
            self.__distinct_out_neighbors[i] = len(out_neighbors[i])
            self.__distinct_in_neighbors[i] = len(in_neighbors[i])
