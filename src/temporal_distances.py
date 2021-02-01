import numpy as np

import backward_BFS as Bbfs
from utils import util


class ShortestPath:
    def __init__(self):
        self.__distances = []
        self.__eccentricity = None
        self.__idx_farther = None
        self.__reachables = 0
        self.__min_ecc = np.NINF  # minimum possible eccentricity
        self.__name = 'SP'  # Shortest Path
        self.__opposite_path = None

    def get_distances(self):
        return self.__distances

    def get_idx_farther(self):
        return self.__idx_farther

    def get_eccentricity(self):
        return self.__eccentricity

    def compare_eccentricity(self, ecc, idx):
        if self.__eccentricity > ecc:
            return self.__eccentricity, self.__idx_farther
        else:
            return ecc, idx

    def get_min_ecc(self):
        return self.__min_ecc

    def get_reachables(self):
        return self.__reachables

    def get_name(self):
        return self.__name

    def gen_opposite(self, folder, file):
        self.__opposite_path = util.opposite_graph_ascend_traversal(folder=folder, file=file)
        return self.__opposite_path

    def __compute_sp(self, u, v, t, traversal_time, start_node, list_of_sorted_lists):
        # print(list_of_sorted_lists)
        if u == start_node:
            if (0, t) not in list_of_sorted_lists[start_node]:
                list_of_sorted_lists[start_node].append((0, t))

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
                                (e[0] > new_d and e[1] >= new_arrival_time) or (
                                                              e[0] == new_d and e[1] > new_arrival_time))]

                        # If the minimum duration changes, update it
                        if new_d < self.__distances[v]:
                            self.__distances[v] = new_d
                        # sort list
                        list_of_sorted_lists[v].sort(key=lambda tup: tup[0])
                    break
                else:
                    continue

    def compute_distances(self, graph, start_node, min_time=None, max_time=None):
        """
        :param graph: temporal graph
        :param start_node: node to start from
        :param min_time: lower bound time interval
        :param max_time: upper bound time interval
        :return: numpy array of duration of the fastest path from start_node to every nodes v in V within time interval
        """

        if graph.get_latest_node() is not None:
            raise Exception('Shortest path does not work with dummy nodes, give me in input weighted graph!')

        if min_time is None:
            min_time, _ = graph.get_time_interval()
        if max_time is None:
            _, max_time = graph.get_time_interval()

        list_of_sorted_lists = []
        for elem in range(graph.get_num_nodes()):
            list_of_sorted_lists.append([])

        self.__eccentricity = 0
        self.__idx_farther = start_node
        self.__distances = np.full((graph.get_num_nodes(),), np.inf)
        self.__distances[start_node] = 0

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

            if t >= min_time and t + traversal_time <= max_time:
                self.__compute_sp(u=u, v=v, t=t, traversal_time=traversal_time, start_node=start_node,
                                  list_of_sorted_lists=list_of_sorted_lists)
                if not graph.get_is_directed():
                    self.__compute_sp(u=v, v=u, t=t, traversal_time=traversal_time, start_node=start_node,
                                      list_of_sorted_lists=list_of_sorted_lists)
            elif t >= max_time:
                break

        # To handle dummy nodes: discard them from the distances vector
        # if graph.get_latest_node() is not None:
        #    self.__distances = self.__distances[:graph.get_latest_node()]

        self.__idx_farther, self.__eccentricity, self.__reachables = util.get_max_ind(self.__distances, np.NINF)


class FastestPathOnePass:
    def __init__(self):
        self.__distances = []
        self.__eccentricity = None
        self.__idx_farther = None
        self.__min_ecc = np.NINF  # minimum possible eccentricity
        self.__reachables = 0
        self.__name = 'FP'  # Fastest Path
        self.__opposite_path = None

    def get_distances(self):
        return self.__distances

    def get_idx_farther(self):
        return self.__idx_farther

    def get_eccentricity(self):
        return self.__eccentricity

    def compare_eccentricity(self, ecc, idx):
        if self.__eccentricity > ecc:
            return self.__eccentricity, self.__idx_farther
        else:
            return ecc, idx

    def get_min_ecc(self):
        return self.__min_ecc

    def get_reachables(self):
        return self.__reachables

    def get_name(self):
        return self.__name

    def gen_opposite(self, folder, file):
        self.__opposite_path = util.opposite_graph_ascend(folder=folder, file=file)
        return self.__opposite_path

    def __compute_fp(self, u, v, t, start_node, list_of_queues):
        if u == start_node:
            if (t, t) not in list_of_queues[start_node]:
                list_of_queues[start_node].append((t, t))

        if list_of_queues[u]:  # check if list is not empty
            for i, elem in reversed(list(enumerate(list_of_queues[u]))):
                if elem[1] <= t:
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
                                                                            list_of_queues[v][-1][
                                                                                1] > new_arrival_time):
                        list_of_queues[v][-1] = (new_t, new_arrival_time)

                    # Otherwise keep both (newest is added after the oldest)
                    else:
                        list_of_queues[v].append((new_t, new_arrival_time))

                    # If the minimum duration changes, update it
                    if list_of_queues[v][-1][1] - list_of_queues[v][-1][0] < self.__distances[v]:
                        self.__distances[v] = list_of_queues[v][-1][1] - list_of_queues[v][-1][0]

                    break
                else:
                    continue

    def compute_distances(self, graph, start_node, min_time=None, max_time=None):
        """
        :param graph: temporal graph
        :param start_node: node to start from
        :param min_time: lower bound time interval
        :param max_time: upper bound time interval
        :return: numpy array of duration of the fastest path from start_node to every nodes v in V within time interval
        """

        if min_time is None:
            min_time, _ = graph.get_time_interval()
        if max_time is None:
            _, max_time = graph.get_time_interval()

        list_of_queues = []
        for elem in range(graph.get_num_nodes()):
            list_of_queues.append([])

        self.__eccentricity = 0
        self.__idx_farther = start_node
        self.__distances = np.full((graph.get_num_nodes(),), np.inf)
        self.__distances[start_node] = 0

        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])

            if t >= min_time and t + 1 <= max_time:
                self.__compute_fp(u=u, v=v, t=t, start_node=start_node, list_of_queues=list_of_queues)
                if not graph.get_is_directed():
                    self.__compute_fp(u=v, v=u, t=t, start_node=start_node, list_of_queues=list_of_queues)
            elif t >= max_time:
                break

        # To handle dummy nodes: discard them from the distances vector
        if graph.get_latest_node() is not None:
            self.__distances = self.__distances[:graph.get_latest_node()]

        self.__idx_farther, self.__eccentricity, self.__reachables = util.get_max_ind(self.__distances, np.NINF)


class EarliestArrivalPath:
    def __init__(self):
        self.__distances = []
        self.__eccentricity = None
        self.__idx_farther = None
        self.__min_ecc = np.NINF  # minimum possible eccentricity
        self.__name = 'EAT'

    def get_distances(self):
        return self.__distances

    def get_eccentricity(self):
        return self.__eccentricity

    def get_idx_farther(self):
        return self.__idx_farther

    def compare_eccentricity(self, ecc, idx):
        if self.__eccentricity > ecc:
            return self.__eccentricity, self.__idx_farther
        else:
            return ecc, idx

    def get_min_ecc(self):
        return self.__min_ecc

    def get_name(self):
        return self.__name

    def compute_distances(self, graph, start_node, min_time=None, max_time=None):
        """
        :param graph: temporal graph in its edge stream representation
        :param start_node: node to start from
        :param min_time: lower bound time interval
        :param max_time: upper bound time interval
        :return: numpy array of the earliest arrival times from start_node to every nodes v in V within time interval
        """

        if min_time is None:
            min_time, _ = graph.get_time_interval()
        if max_time is None:
            _, max_time = graph.get_time_interval()

        self.__eccentricity = min_time
        self.__idx_farther = start_node

        self.__distances = np.full((graph.get_num_nodes(),), np.inf)
        self.__distances[start_node] = min_time

        traversal_time = 1
        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            if len(li) == 4:
                traversal_time = int(li[3])

            if (t + traversal_time) <= max_time:
                if t >= self.__distances[u]:
                    if (t + traversal_time) < self.__distances[v]:
                        self.__distances[v] = t + traversal_time
                        if self.__distances[v] > self.__eccentricity and \
                                (graph.get_latest_node() is None or v < graph.get_latest_node()):  # to handle dummy
                            self.__eccentricity = self.__distances[v]
                            self.__idx_farther = v

                elif not graph.get_is_directed() and t >= self.__distances[v]:  # If graph is undirected
                    if (t + traversal_time) < self.__distances[u]:
                        self.__distances[u] = t + traversal_time
                        if self.__distances[u] > self.__eccentricity and \
                                (graph.get_latest_node() is None or u < graph.get_latest_node()):  # to handle dummy
                            # nodes
                            self.__eccentricity = self.__distances[u]
                            self.__idx_farther = u
            elif t >= max_time:
                break
            else:
                continue
        # To handle dummy nodes: discard them from the distances vector
        if graph.get_latest_node() is not None:
            self.__distances = self.__distances[:graph.get_latest_node()]


class LatestDeparturePath:  # ( method compute_distances() compute backward LDT)
    def __init__(self):
        self.__distances = []
        self.__eccentricity = None
        self.__idx_farther = None
        self.__min_ecc = np.inf  # minimum possible eccentricity
        self.__name = 'LDT'

    def get_distances(self):
        return self.__distances

    def get_eccentricity(self):
        return self.__eccentricity

    def get_idx_farther(self):
        return self.__idx_farther

    def compare_eccentricity(self, ecc, idx):
        if self.__eccentricity < ecc:
            return self.__eccentricity, self.__idx_farther
        else:
            return ecc, idx

    def get_min_ecc(self):
        return self.__min_ecc

    def get_name(self):
        return self.__name

    def compute_distances(self, graph, start_node, min_time=None, max_time=None):
        """
        :param graph: temporal graph in REVERSE EDGE STREAM representation
        :param start_node: destination node
        :param min_time: lower bound time interval
        :param max_time: upper bound time interval
        :return: numpy array of the latest departure times from every vertex v in V to dest_node within time interval
        """

        if min_time is None:
            min_time, _ = graph.get_time_interval()
        if max_time is None:
            _, max_time = graph.get_time_interval()

        self.__eccentricity = max_time
        self.__idx_farther = start_node

        self.__distances = np.full((graph.get_num_nodes(),), np.NINF)
        self.__distances[start_node] = max_time

        traversal_time = 1
        for line in graph.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()

            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            if len(li) == 4:
                traversal_time = int(li[3])

            if t >= min_time:
                if t + traversal_time <= self.__distances[v]:
                    if t > self.__distances[u]:
                        self.__distances[u] = t
                        # I can compute eccentricity here (online) because the graph is in reverse edge stream
                        if self.__distances[u] < self.__eccentricity and \
                                (graph.get_latest_node() is None or u < graph.get_latest_node()):
                            self.__eccentricity = self.__distances[u]
                            self.__idx_farther = u

                elif not graph.get_is_directed() and t + traversal_time <= self.__distances[u]:
                    if t > self.__distances[v]:
                        self.__distances[v] = t
                        if self.__distances[v] < self.__eccentricity and \
                                (graph.get_latest_node() is None or v < graph.get_latest_node()):
                            self.__eccentricity = self.__distances[v]
                            self.__idx_farther = v
            else:
                break
        # To handle dummy nodes: discard them from the distances vector
        if graph.get_latest_node() is not None:
            self.__distances = self.__distances[:graph.get_latest_node()]
        # self.__idx_farther, self.__eccentricity = util.get_min_ind(self.__distances, np.inf)


def compute_diameter(graph, distance):
    """
    :param graph: temporal graph
    :param distance: Object distance: FastestPathOnePass, earliestArrivalPath etc...
    :return: diameter value
    """
    ecc = distance.get_min_ecc()
    idx = -1
    reachable_pairs = 0
    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    for node in range(num_nodes):
        distance.compute_distances(graph=graph, start_node=node)

        if distance.get_name() == 'FP' or distance.get_name() == 'SP':
            reachable_pairs += distance.get_reachables()

        ecc, idx = distance.compare_eccentricity(ecc, idx)
        # print(ecc)

        if (node + 1) % 1000 == 0:
            print('1000 visits done...', flush=True)

    if distance.get_name() == 'FP' or distance.get_name() == 'SP':
        print('Reachable Pairs ' + graph.get_file_path().rsplit('/', 1)[1] + ': ' + str(reachable_pairs), flush=True)
    return ecc


if __name__ == '__main__':
    import temporal_graph as tg

    sp_distance = ShortestPath()

    g = tg.Graph(file_path='./graphs/Weighted/kuopio-weighted-sorted.txt', is_directed=True)

    sp_dist = ShortestPath()

    diam = compute_diameter(graph=g, distance=sp_distance)
    print('diam:')
    print(diam)
