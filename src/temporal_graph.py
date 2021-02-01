import numpy as np


class Graph:
    def __init__(self, file_path, is_directed, latest_node=None):
        """
        :param file_path: graph's path
        :param file_path: path of graph's file
        :param is_directed: False indirected graph, True directed graph
        :param latest_node: if it is a value (not None), that value represents latest not dummy node + 1
        (i.e. number of not dummies nodes), in this case a transformation ('transform_constant_traversal') has been made
        else it is None
        """
        self.__file_path = file_path
        self.__txt_list = None
        self.__num_nodes = None
        self.__min_time = None
        self.__max_time = None
        self.__edges_number = None
        self.__is_directed = is_directed
        self.__latest_node = latest_node
        self.__chcek_latest_node()
        self.__degrees_out = None
        self.__degrees_in = None

    def get_file_path(self):
        return self.__file_path

    def graph_reader(self):
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
        # if self.__latest_node is not None:
        #    return self.__latest_node
        if self.__num_nodes is None:
            self.__graph_proprieties()
        return self.__num_nodes

    def get_time_interval(self):
        if self.__min_time is None or self.__max_time is None:
            self.__graph_proprieties()
        return self.__min_time, self.__max_time

    def get_latest_node(self):
        return self.__latest_node

    def __chcek_latest_node(self):
        if self.__latest_node is not None:
            if self.__latest_node > self.get_num_nodes():
                raise Exception('Parameter latest_node must be less then or equal to the number of nodes in the Graph')

    def get_edges_number(self):
        if self.__edges_number is None:
            self.__graph_proprieties()
        return self.__edges_number

    def get_is_directed(self):
        return self.__is_directed

    def get_max_deg_out(self, n=1):
        if self.__degrees_out is None:
            self.__get_degrees()
        sorted_deg_out_index = np.argsort(-self.__degrees_out)
        sorted_deg_out = self.__degrees_out[sorted_deg_out_index]
        return sorted_deg_out[:n], sorted_deg_out_index[:n]

    def get_max_deg_in(self, n=1):
        if self.__degrees_in is None:
            self.__get_degrees()
        sorted_deg_in_index = np.argsort(-self.__degrees_in)
        sorted_deg_in = self.__degrees_in[sorted_deg_in_index]
        return sorted_deg_in[:n], sorted_deg_in_index[:n]

    def __graph_proprieties(self):
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


if __name__ == '__main__':
    # g = Graph(file_path='../graphs/Dummy/transportation/grenoble-sa-sorted.txt', is_directed=True, latest_node=1547)
    # print(g.get_max_deg_out(10))

    from utils import util

    # source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/Weighted/'
    # source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/SNAP/'
    # source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/new/'
    # source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/misc/'
    # source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/twitter/sorted_graphs/'
    source_directory = '/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/soc-bitcoin/sorted_graphs/'
    g_list = util.files_for_dimensions(source_directory)

    print('GRAPH NUM_EDGES T_alpha T_omega')
    for g_name in g_list:
        g = Graph(file_path=source_directory + g_name, is_directed=True)
        t_min, t_max = g.get_time_interval()
        print(g_name + " " + str(g.get_edges_number()) + " " + str(t_min) + " " + str(t_max))
