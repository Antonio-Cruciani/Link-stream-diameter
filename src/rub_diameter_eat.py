import numpy as np
import time

from utils import util
import backward_BFS as Bbfs

import temporal_graph as tg


# rub = Reverse Upper Bound
def nodes_order(g_inv, out_path):
    out_dir = out_path.rsplit('/', 1)[0]
    util.create_dir(out_dir)

    if not util.check_file_exists(out_path):
        if g_inv.get_latest_node() is not None:
            num_nodes = g_inv.get_latest_node()
        else:
            num_nodes = g_inv.get_num_nodes()
        nodes = np.full(num_nodes, True)

        for line in g_inv.graph_reader():
            if not line.strip():  # check if line is blank
                break
            li = line.split()
            u = int(li[0])
            v = int(li[1])
            t = int(li[2])
            if v < num_nodes and nodes[v]:
                nodes[v] = False
                with open(out_path, mode='a') as f:
                    f.write(str(v) + ' ')
                    f.write(str(t) + '\n')
            if not g_inv.get_is_directed():
                if u < num_nodes and nodes[u]:
                    nodes[u] = False
                    with open(out_path, mode='a') as f:
                        f.write(str(u) + ' ')
                        f.write(str(t) + '\n')


def compute_rub_diam(graph_rev, nodes_file):
    number_visits = 0
    l_b, u_p = graph_rev.get_time_interval()
    with open(nodes_file) as f:
        for row in f:
            if not row.strip():  # check if line is blank
                break
            li = row.split()
            v = int(li[0])
            t = int(li[1])
            back_bfs = Bbfs.TemporalBackwardBFS(graph=graph_rev, start_node=v)
            number_visits += 1
            # print('Number of visits: ' + str(number_visits))
            if back_bfs.get_eccentricity_eat() > l_b:
                l_b = back_bfs.get_eccentricity_eat()
            u_p = t + 1
            if u_p <= l_b:
                break
    return l_b, number_visits


def rub_diam_eat(graph):
    folder, g_name = graph.get_file_path().rsplit('/', 1)
    folder += '/'

    path_g_reverse = util.reverse_ef(folder=folder, file=g_name)
    # path_g_reverse = util.reverse(folder=folder, file=g_name)

    dire_g_reverse = path_g_reverse.rsplit('/', 1)[0]
    dire_g_reverse += '/'

    # util.create_dir(dire_g_reverse + 'nodes_reversed/')

    g_rev = tg.Graph(file_path=path_g_reverse, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())
    nodes_file = dire_g_reverse + 'nodes_reversed/nodes_rev_' + g_rev.get_file_path().rsplit('/', 1)[1]

    print('Computing diameter EAT of graph ' + g_name + '...', flush=True)

    start_time = time.time()
    nodes_order(g_inv=g_rev, out_path=nodes_file)
    diam, number_visits = compute_rub_diam(graph_rev=g_rev, nodes_file=nodes_file)
    end_time = time.time()

    total_time = end_time - start_time

    return diam, number_visits, total_time


def rub_diam_ldt(graph):
    folder, g_name = graph.get_file_path().rsplit('/', 1)
    folder += '/'
    path_g_opposite = util.opposite_graph(folder=folder, file=g_name)
    dire_g_opposite = path_g_opposite.rsplit('/', 1)[0]
    dire_g_opposite += '/'

    # util.create_dir(dire_g_opposite + 'nodes_reversed/')

    g_op = tg.Graph(file_path=path_g_opposite, is_directed=graph.get_is_directed(), latest_node=graph.get_latest_node())
    nodes_file = dire_g_opposite + 'nodes_reversed/nodes_rev_' + g_op.get_file_path().rsplit('/', 1)[1]

    print('Computing diameter LDT of graph ' + g_name + '...', flush=True)

    start_time = time.time()
    nodes_order(g_inv=g_op, out_path=nodes_file)
    diam, number_visits = compute_rub_diam(graph_rev=g_op, nodes_file=nodes_file)
    diam = -diam
    end_time = time.time()

    total_time = end_time - start_time

    return diam, number_visits, total_time


if __name__ == '__main__':
    # Show how to use the algorithm
    g = tg.Graph(file_path='graphs/transportation2/belfast_temporal_day.txt.sor', is_directed=True)
    d, n_vis, sec = rub_diam_eat(graph=g)
    print('Diameter: ' + str(d))
    print('Number of visits: ' + str(n_vis))
    print('Time spent: ' + str(sec))

    g = tg.Graph(file_path='graphs/transportation2/belfast_temporal_day.txt.sor', is_directed=True)
    d, n_vis, sec = rub_diam_ldt(graph=g)
    print('Diameter: ' + str(d))
    print('Number of visits: ' + str(n_vis))
    print('Time spent: ' + str(sec))

    """
    source_directory = './graphs/Dummy/transportation/'

    g_list = util.files_for_dimensions(dire=source_directory)
    dummy_nodes = np.loadtxt(source_directory + 'dummy_nodes/dummy_nodes.txt', dtype=str)

    i = 0
    for graph_name in g_list:
        print('Graph ' + graph_name, flush=True)
        print('Dummy node: ' + str(dummy_nodes[i]), flush=True)
        g = tg.Graph(file_path=source_directory + graph_name, is_directed=True, latest_node=int(dummy_nodes[i][1]))

        # d, n_vis, sec = rub_diam_eat(graph=g)
        d, n_vis, sec = rub_diam_ldt(graph=g)
        print('Diameter: ' + str(d), flush=True)
        print('Number of visits: ' + str(n_vis), flush=True)
        print('Time spent: ' + str(sec), flush=True)
        print('\n', flush=True)
        i += 1
    """
