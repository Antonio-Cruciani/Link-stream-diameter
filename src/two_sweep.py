import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../utils')

import temporal_distances as td
import temporal_graph as tg
import backward_BFS as Bbfs
from utils import util

from random import randint


def two_sweep_fw_fpsp(graph, fp_sp_distance, r=None):
    """
    :param graph: graph
    :param fp_sp_distance: Object distance fastest or shortest path
    :param r: starting node
    Return
    r: Starting node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """
    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)

    # Create opposite graph to compute backward fastest paths distances
    g_path_op = fp_sp_distance.gen_opposite(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                            file=graph.get_file_path().rsplit('/', 1)[1])
    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    fp_sp_distance.compute_distances(graph=graph, start_node=r)

    i1 = fp_sp_distance.get_idx_farther()

    fp_sp_distance.compute_distances(graph=g_op, start_node=i1)

    d = fp_sp_distance.get_eccentricity()
    i2 = fp_sp_distance.get_idx_farther()

    return r, i2, i1, d


def two_sweep_bw_fpsp(graph, fp_sp_distance, r=None):
    """
    :param graph: graph
    :param fp_sp_distance: Object distance fastest or shortest path
    :param r: starting node
    Return
    r: Start node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)

    # Create opposite graph to compute backward fastest paths distances
    g_path_op = fp_sp_distance.gen_opposite(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                            file=graph.get_file_path().rsplit('/', 1)[1])
    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    fp_sp_distance.compute_distances(graph=g_op, start_node=r)

    i1 = fp_sp_distance.get_idx_farther()

    fp_sp_distance.compute_distances(graph=graph, start_node=i1)

    d = fp_sp_distance.get_eccentricity()
    i2 = fp_sp_distance.get_idx_farther()

    return r, i1, i2, d


def two_sweep_2_fpsp(graph, fp_sp_distance, r=None):
    """
    :param graph: graph
    :param fp_sp_distance: Object distance fastest or shortest path
    :param r: starting node
    Return
    s: Starting node of longest path
    i: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    r, i1, i2, d1 = two_sweep_fw_fpsp(graph, fp_sp_distance, r)
    _, s, i, d = two_sweep_bw_fpsp(graph, fp_sp_distance, r)

    if d1 > d:
        s = i1
        i = i2
        d = d1
    return s, i, d


def two_sweep_forward_eat(graph, r=None):
    """
    Return
    r: Start node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)
    #    print('Random start node: ' + str(r))

    # Create reverse graph to compute backward EAT distances
    g_path_rev = util.reverse_ef(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                 file=graph.get_file_path().rsplit('/', 1)[1])
    g_rev = tg.Graph(file_path=g_path_rev, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    eat_dist = td.EarliestArrivalPath()
    eat_dist.compute_distances(graph=graph, start_node=r)

    i1 = eat_dist.get_idx_farther()

    back_bfs = Bbfs.TemporalBackwardBFS(graph=g_rev, start_node=i1)
    back_bfs.bfs()

    d = back_bfs.get_eccentricity_eat()
    i2 = back_bfs.get_idx_farther_eat()

    return r, i2, i1, d


def two_sweep_backward_eat(graph, r=None):
    """
    Return
    r: Start node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """
    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)
    #    print('Random start node: ' + str(r))

    # Create reverse graph to compute backward EAT distances
    g_path_rev = util.reverse_ef(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                 file=graph.get_file_path().rsplit('/', 1)[1])
    g_rev = tg.Graph(file_path=g_path_rev, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    back_bfs = Bbfs.TemporalBackwardBFS(graph=g_rev, start_node=r)
    back_bfs.bfs()

    i1 = back_bfs.get_idx_farther_eat()

    eat_dist = td.EarliestArrivalPath()
    eat_dist.compute_distances(graph, start_node=i1)

    d = eat_dist.get_eccentricity()
    i2 = eat_dist.get_idx_farther()

    return r, i1, i2, d


def two_sweep_2_eat(graph, r=None):
    """
    Return
    s: Starting node of longest path
    i: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    r, i1, i2, d1 = two_sweep_forward_eat(graph, r)
    _, s, i, d = two_sweep_backward_eat(graph, r)

    if d1 > d:
        s = i1
        i = i2
        d = d1
    return s, i, d


def two_sweep_forward_ldt(graph, r=None):
    """
    Return
    r: Start node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)
    #    print('Random start node: ' + str(r))

    # Create opposite graph to compute forward LDT distances
    g_path_op = util.opposite_graph(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                    file=graph.get_file_path().rsplit('/', 1)[1])
    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    back_bfs = Bbfs.TemporalBackwardBFS(graph=g_op, start_node=r)
    back_bfs.bfs()
    i1 = back_bfs.get_idx_farther_eat()

    # Create reverse graph to compute backward LDT distances
    g_path_rev = util.reverse_ef(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                 file=graph.get_file_path().rsplit('/', 1)[1])
    g_rev = tg.Graph(file_path=g_path_rev, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    ldt_dist = td.LatestDeparturePath()
    ldt_dist.compute_distances(graph=g_rev, start_node=i1)

    d = ldt_dist.get_eccentricity()
    i2 = ldt_dist.get_idx_farther()

    return r, i2, i1, d


def two_sweep_backward_ldt(graph, r=None):
    """
    Return
    r: Start node
    i1: Starting node of longest path
    i2: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """
    if graph.get_latest_node() is not None:
        num_nodes = graph.get_latest_node()
    else:
        num_nodes = graph.get_num_nodes()

    if r is None:
        r = randint(0, num_nodes - 1)
    #    print('Random start node: ' + str(r))

    # Create reverse graph to compute backward LDT distances
    g_path_rev = util.reverse_ef(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                 file=graph.get_file_path().rsplit('/', 1)[1])
    g_rev = tg.Graph(file_path=g_path_rev, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    ldt_dist = td.LatestDeparturePath()
    ldt_dist.compute_distances(graph=g_rev, start_node=r)

    i1 = ldt_dist.get_idx_farther()

    # Create opposite graph to compute forward LDT distances
    g_path_op = util.opposite_graph(folder=graph.get_file_path().rsplit('/', 1)[0] + '/',
                                    file=graph.get_file_path().rsplit('/', 1)[1])
    g_op = tg.Graph(file_path=g_path_op, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())

    back_bfs = Bbfs.TemporalBackwardBFS(graph=g_op, start_node=i1)
    back_bfs.bfs()

    i2 = back_bfs.get_idx_farther_eat()
    d = -(back_bfs.get_eccentricity_eat())

    return r, i1, i2, d


def two_sweep_2_ldt(graph, r=None):
    """
    Return
    s: Starting node of longest path
    i: Arrival node of longest path
    d: distance of farther node find (our diameter estimate)
    """

    r, i1, i2, d1 = two_sweep_forward_ldt(graph, r)
    _, s, i, d = two_sweep_backward_ldt(graph, r)

    if d1 < d:
        s = i1
        i = i2
        d = d1
    return s, i, d


if __name__ == '__main__':
    g = tg.Graph(file_path='/home/marco/Coding/PycharmProjects/Temporal_graphs_diameter/graphs/imdb2/NOTcomputed/'
                           'imdb_all-sorted.txt', is_directed=False)
    degree_node, starting_node = g.get_max_deg_out(n=80)
    print('Starting node: ' + str(starting_node[0]))
    print('Degree node: ' + str(degree_node[0]))
    sp_dist = td.ShortestPath()
    diam_estimation = -1
    a = None
    b = None
    for nod in starting_node:
        a, b, diam_estimation_temp = two_sweep_2_fpsp(graph=g, fp_sp_distance=sp_dist, r=nod)
        print('1 2SW Done...')
        if diam_estimation < diam_estimation_temp:
            diam_estimation = diam_estimation_temp

    print('From: ' + str(a))
    print('To: ' + str(b))
    print('Diameter estimation: ' + str(diam_estimation))
