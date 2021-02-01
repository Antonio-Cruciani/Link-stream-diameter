import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg
import two_sweep as ts
import backward_BFS as Bbfs
from utils import util

import random
import math
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute LDT double sweep VS randoms BFS for all link streams whose paths are in file")
    parser.add_argument("file", type=str, help="file path")

    args = parser.parse_args()
    input_path = args.file

    ldt_dist = td.LatestDeparturePath()

    dist_name = 'LDT'
    print('DISTANCE: ' + dist_name + '\n', flush=True)

    with open(input_path, "r") as f:
        for row in f:
            if not row.strip():  # check if line is blank
                continue
            row = row.split()
            if len(row) > 3:
                raise Exception('In Input file, incorrect line: ' + str(row))
            g_path = row[0]
            dummy_node = None
            is_directed = True
            if len(row) > 1:
                dummy_node = int(row[1])
                if dummy_node < 0:
                    dummy_node = None
            if len(row) > 2:
                is_directed = False

            graph_name = g_path.rsplit('/', 1)[1]
            print("\n")
            print(dist_name + ' Graph ' + graph_name, flush=True)
            print(dist_name + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
            print(dist_name + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

            t_min, t_max = g.get_time_interval()

            # Create opposite graph to compute forward LDT distances
            g_path_op = util.opposite_graph(folder=g.get_file_path().rsplit('/', 1)[0] + '/',
                                            file=g.get_file_path().rsplit('/', 1)[1])
            g_op = tg.Graph(file_path=g_path_op, is_directed=g.get_is_directed(),
                            latest_node=g.get_latest_node())

            if g.get_latest_node() is not None:
                num_nodes = g.get_latest_node()
            else:
                num_nodes = g.get_num_nodes()

            a = round(math.log(num_nodes, 2))
            num_2sweep = [1, a, 2 * a, 4 * a]

            print(dist_name + ' Graph ' + graph_name + ' Num of BFS: ' + str(num_2sweep), flush=True)

            if ((4 * a) * 4) >= num_nodes:
                print('GRAPH ' + graph_name + ' SKIPPED', flush=True)
                continue

            start_nodes = random.sample(range(num_nodes), (4 * a) * 4)
            print('2SWEEP RESULTS: ', flush=True)
            # print('i = 1', flush=True)

            _, _, estimation = ts.two_sweep_2_ldt(graph=g, r=start_nodes[0])

            print(dist_name + graph_name + ' StartNode=' + str(start_nodes[0]) + ' i=1 2SWecc: ' +
                  str(t_max - estimation), flush=True)
            print(dist_name + graph_name + ' i=1 MAX2SW: ' + str(t_max - estimation),
                  flush=True)

            print('RND RESULTS:', flush=True)

            rnd_estimation = np.inf

            for h in range(0, 4):
                back_bfs = Bbfs.TemporalBackwardBFS(graph=g_op, start_node=start_nodes[h])
                back_bfs.bfs()
                ecc = -(back_bfs.get_eccentricity_eat())

                print(dist_name + graph_name + ' StartNode=' + str(start_nodes[h]) + ' i=1 RNDecc: ' + str(t_max - ecc),
                      flush=True)

                if ecc < rnd_estimation:
                    rnd_estimation = ecc

            print(dist_name + graph_name + ' i=1 MAXrnd: ' + str(t_max - rnd_estimation), flush=True)

            for h in range(1, len(num_2sweep)):
                print('2SWEEP RESULTS: ', flush=True)
                # print('i = ' + str(num_2sweep[h]), flush=True)
                for j in range(num_2sweep[h - 1], num_2sweep[h]):

                    _, _, ecc = ts.two_sweep_2_ldt(graph=g, r=start_nodes[j])

                    print(dist_name + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                          ' 2SWecc: ' + str(t_max - ecc),
                          flush=True)

                    if ecc < estimation:
                        estimation = ecc

                print(dist_name + graph_name + ' i=' + str(h+1) + ' MAX2SW: ' + str(t_max - estimation),
                      flush=True)

                print('RND RESULTS:', flush=True)
                for j in range((num_2sweep[h - 1]) * 4, (num_2sweep[h]) * 4):

                    back_bfs = Bbfs.TemporalBackwardBFS(graph=g_op, start_node=start_nodes[j])
                    back_bfs.bfs()
                    ecc = -(back_bfs.get_eccentricity_eat())

                    print(dist_name + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                          ' RNDecc: ' + str(t_max - ecc),
                          flush=True)

                    if ecc < rnd_estimation:
                        rnd_estimation = ecc

                print(dist_name + graph_name + ' i=' + str(h+1) + ' MAXrnd: ' + str(t_max - rnd_estimation),
                      flush=True)
