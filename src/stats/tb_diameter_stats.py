import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg

import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute FT or ST diameter for all link streams whose paths are in file")
    parser.add_argument("FT_ST", type=str, help="FT for Fastest Time diameter, ST for Shortest Time diameter")
    parser.add_argument("file", type=str, help="file path")
    args = parser.parse_args()

    if args.FT_ST == 'FT':
        dist = td.FastestPathOnePass()
    elif args.FT_ST == 'ST':
        dist = td.ShortestPath()
    else:
        raise Exception('Error on argument FT_ST. Use FT for FT diameter or ST for ST diameter')

    distances = [dist]

    input_path = args.file

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

                print('DISTANCE: ' + args.FT_ST)

                graph_name = g_path.rsplit('/', 1)[1]
                print(args.FT_ST + ' Graph ' + graph_name, flush=True)
                print(args.FT_ST + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
                print(args.FT_ST + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

                g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
                start_time = time.time()
                diameter = td.compute_diameter(graph=g, distance=dist)
                end_time = time.time()
                diam_time = end_time - start_time
                print(args.FT_ST + graph_name + ' DIAMETER: ' + str(diameter), flush=True)
                # print(dist_name + graph_name + ' TIME: ' + str(diam_time), flush=True)
                print('\n', flush=True)
