import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg
import two_sweep as ts

import random
import math
import argparse

"""
TWO SPEEP STATS ALL DISTANCES EXCEPT LDT DISTANCE!!
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute FP, SP or EAT double sweep vs randoms BFS for all graps whose paths are in file "
                    "'./TGraphs/TGraphs_list'")
    parser.add_argument("FP_SP_EAT", type=str, help="FP for Fastest Path distance, SP for Shortest Path distance"
                                                    ", EAT for Earliest arrival time distance")
    args = parser.parse_args()
    input_dist = None
    if args.FP_SP_EAT == 'FP':
        input_dist = td.FastestPathOnePass()
    if args.FP_SP_EAT == 'SP':
        input_dist = td.ShortestPath()
    if args.FP_SP_EAT == 'EAT':
        input_dist = td.EarliestArrivalPath()

    distances = [input_dist]

    input_path = './TGraphs/TGraphs_list'

    for dist in distances:

        dist_name = dist.get_name()
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
                print(dist_name + ' Graph ' + graph_name, flush=True)
                print(dist_name + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
                print(dist_name + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

                g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

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
                print('i = 1', flush=True)

                if dist.get_name() == 'EAT':
                    _, _, estimation = ts.two_sweep_2_eat(graph=g, r=start_nodes[0])
                else:
                    _, _, estimation = ts.two_sweep_2_fpsp(graph=g, fp_sp_distance=dist, r=start_nodes[0])

                print(dist_name + graph_name + ' StartNode=' + str(start_nodes[0]) + ' i=1 2SWecc: ' + str(estimation),
                      flush=True)
                print(dist_name + graph_name + ' i=1 MAX2SW: ' + str(estimation),
                      flush=True)

                print('RND RESULTS:', flush=True)

                rnd_estimation = -1

                for h in range(0, 4):

                    dist.compute_distances(graph=g, start_node=start_nodes[h])
                    ecc = dist.get_eccentricity()

                    print(dist_name + graph_name + ' StartNode=' + str(start_nodes[h]) + ' i=1 RNDecc: ' + str(ecc),
                          flush=True)

                    if ecc > rnd_estimation:
                        rnd_estimation = ecc

                print(dist_name + graph_name + ' i=1 MAXrnd: ' + str(rnd_estimation), flush=True)

                for h in range(1, len(num_2sweep)):
                    print('2SWEEP RESULTS: ', flush=True)
                    print('i = ' + str(num_2sweep[h]), flush=True)
                    for j in range(num_2sweep[h - 1], num_2sweep[h]):

                        if dist.get_name() == 'EAT':
                            _, _, ecc = ts.two_sweep_2_eat(graph=g, r=start_nodes[j])
                        else:
                            _, _, ecc = ts.two_sweep_2_fpsp(graph=g, fp_sp_distance=dist, r=start_nodes[j])

                        print(dist_name + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                              ' 2SWecc: ' + str(ecc),
                              flush=True)

                        if ecc > estimation:
                            estimation = ecc

                    print(dist_name + graph_name + ' i=' + str(h+1) + ' MAX2SW: ' + str(estimation),
                          flush=True)

                    print('RND RESULTS:', flush=True)
                    for j in range((num_2sweep[h - 1]) * 4, (num_2sweep[h]) * 4):

                        dist.compute_distances(graph=g, start_node=start_nodes[j])
                        ecc = dist.get_eccentricity()

                        print(dist_name + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                              ' RNDecc: ' + str(ecc),
                              flush=True)

                        if ecc > rnd_estimation:
                            rnd_estimation = ecc

                    print(dist_name + graph_name + ' i=' + str(h+1) + ' MAXrnd: ' + str(rnd_estimation),
                          flush=True)
