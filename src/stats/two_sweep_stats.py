import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg
import two_sweep as ts

import random
import math
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute FT, ST or EAT double sweep VS randoms BFS for all link streams whose paths are in file")
    parser.add_argument("FT_ST_EAT", type=str, help="FT for Fastest Time distance, ST for Shortest Time distance"
                                                    ", EAT for Earliest arrival time distance")
    parser.add_argument("file", type=str, help="file path")

    args = parser.parse_args()
    if args.FT_ST_EAT == 'FT':
        input_dist = td.FastestPathOnePass()
    elif args.FT_ST_EAT == 'ST':
        input_dist = td.ShortestPath()
    elif args.FT_ST_EAT == 'EAT':
        input_dist = td.EarliestArrivalPath()
    else:
        raise Exception('Error on argument FT_ST_EAT. Use FT for FT distance, ST for ST distance, EAT for'
                        ' EAT distance')

    distances = [input_dist]

    input_path = args.file

    for dist in distances:

        print('DISTANCE: ' + args.FT_ST_EAT + '\n', flush=True)

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
                print(args.FT_ST_EAT + ' Graph ' + graph_name, flush=True)
                print(args.FT_ST_EAT + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
                print(args.FT_ST_EAT + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

                g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

                t_min, t_max = g.get_time_interval()

                if dist.get_name() == 'EAT':
                    resize = t_min
                else:
                    resize = 0

                if g.get_latest_node() is not None:
                    num_nodes = g.get_latest_node()
                else:
                    num_nodes = g.get_num_nodes()

                a = round(math.log(num_nodes, 2))
                num_2sweep = [1, a, 2 * a, 4 * a]

                print(args.FT_ST_EAT + ' Graph ' + graph_name + ' Num of BFS: ' + str(num_2sweep), flush=True)

                if ((4 * a) * 4) >= num_nodes:
                    print('GRAPH ' + graph_name + ' SKIPPED', flush=True)
                    continue

                start_nodes = random.sample(range(num_nodes), (4 * a) * 4)
                print('2SWEEP RESULTS: ', flush=True)
                # print('i = 1', flush=True)

                if dist.get_name() == 'EAT':
                    _, _, estimation = ts.two_sweep_2_eat(graph=g, r=start_nodes[0])
                else:
                    _, _, estimation = ts.two_sweep_2_fpsp(graph=g, fp_sp_distance=dist, r=start_nodes[0])

                print(args.FT_ST_EAT + graph_name + ' StartNode=' + str(start_nodes[0]) + ' i=1 2SWecc: ' +
                      str(estimation - resize), flush=True)
                print(args.FT_ST_EAT + graph_name + ' i=1 MAX2SW: ' + str(estimation - resize),
                      flush=True)

                print('RND RESULTS:', flush=True)

                rnd_estimation = -1

                for h in range(0, 4):

                    dist.compute_distances(graph=g, start_node=start_nodes[h])
                    ecc = dist.get_eccentricity()

                    print(args.FT_ST_EAT + graph_name + ' StartNode=' + str(start_nodes[h]) + ' i=1 RNDecc: ' +
                          str(ecc - resize), flush=True)

                    if ecc > rnd_estimation:
                        rnd_estimation = ecc

                print(args.FT_ST_EAT + graph_name + ' i=1 MAXrnd: ' + str(rnd_estimation - resize), flush=True)

                for h in range(1, len(num_2sweep)):
                    print('2SWEEP RESULTS: ', flush=True)
                    # print('i = ' + str(num_2sweep[h]), flush=True)
                    for j in range(num_2sweep[h - 1], num_2sweep[h]):

                        if dist.get_name() == 'EAT':
                            _, _, ecc = ts.two_sweep_2_eat(graph=g, r=start_nodes[j])
                        else:
                            _, _, ecc = ts.two_sweep_2_fpsp(graph=g, fp_sp_distance=dist, r=start_nodes[j])

                        print(args.FT_ST_EAT + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                              ' 2SWecc: ' + str(ecc - resize),
                              flush=True)

                        if ecc > estimation:
                            estimation = ecc

                    print(args.FT_ST_EAT + graph_name + ' i=' + str(h+1) + ' MAX2SW: ' + str(estimation - resize),
                          flush=True)

                    print('RND RESULTS:', flush=True)
                    for j in range((num_2sweep[h - 1]) * 4, (num_2sweep[h]) * 4):

                        dist.compute_distances(graph=g, start_node=start_nodes[j])
                        ecc = dist.get_eccentricity()

                        print(args.FT_ST_EAT + graph_name + ' StartNode=' + str(start_nodes[j]) + ' i=' + str(h+1) +
                              ' RNDecc: ' + str(ecc - resize),
                              flush=True)

                        if ecc > rnd_estimation:
                            rnd_estimation = ecc

                    print(args.FT_ST_EAT + graph_name + ' i=' + str(h+1) + ' MAXrnd: ' + str(rnd_estimation - resize),
                          flush=True)
