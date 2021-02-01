import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg

import time
import argparse

if __name__ == '__main__':

    # ldt_distance = td.LatestDeparturePath()
    """
    NB: LDT NEED REVERSED GRAPH!!
    """
    # eat_distance = td.EarliestArrivalPath()

    parser = argparse.ArgumentParser(
        description="Compute FP or SP diameter for all graps whose paths are in file "
                    "'./TGraphs/TGraphs_list'")
    parser.add_argument("FPSP", type=str, help="FP for Fastest Path diameter, SP for Shortest Path diameter")
    args = parser.parse_args()
    dist = None
    if args.FPSP == 'FP':
        dist = td.FastestPathOnePass()
    if args.FPSP == 'SP':
        dist = td.ShortestPath()

    distances = [dist]

    input_path = './TGraphs/TGraphs_list'

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

            for dist in distances:
                dist_name = dist.get_name()
                print('DISTANCE: ' + dist_name)

                graph_name = g_path.rsplit('/', 1)[1]
                print(dist_name + ' Graph ' + graph_name, flush=True)
                print(dist_name + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
                print(dist_name + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

                g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
                start_time = time.time()
                diameter = td.compute_diameter(graph=g, distance=dist)
                end_time = time.time()
                diam_time = end_time - start_time
                print(dist_name + graph_name + ' DIAMETER: ' + str(diameter), flush=True)
                # print(dist_name + graph_name + ' TIME: ' + str(diam_time), flush=True)
                print('\n', flush=True)
