import sys

sys.path.insert(1, '../')

import temporal_graph as tg
import temporal_distances as td

# import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute EAT or LDT or FT or ST diameter for all link streams whose paths are in file")
    parser.add_argument("EAT_LDT_FT_ST", type=str, help="EAT for Earliest Arrival Time Diameter, "
                                                        "LDT for Latest Departure Time Diameter, "
                                                        "FT for Fastest Time Diameter, "
                                                        "ST for Shortest Time Diameter")
    parser.add_argument("file", type=str, help="File path")
    args = parser.parse_args()

    if args.EAT_LDT_FT_ST == 'EAT':
        dist = td.EarliestArrivalTime
    elif args.EAT_LDT_FT_ST == 'LDT':
        dist = td.LatestDepartureTime
    elif args.EAT_LDT_FT_ST == 'FT':
        dist = td.FastestTime
    elif args.EAT_LDT_FT_ST == 'ST':
        dist = td.ShortestTime
    else:
        raise Exception('Error on argument EAT_LDT_FT_ST. Use EAT for EAT-Diameter, LDT for LDT-Diameter, '
                        'FT for FT-Diameter or ST for ST-Diameter')
    input_path = args.file

    with open(input_path, "r") as f:
        for row in f:
            if not row.strip():  # check if line is blank
                continue
            row = row.split()
            if len(row) > 3:
                raise Exception('In input file, incorrect line: {}'.format(row))
            g_path = row[0]
            dummy_node = None
            is_directed = True
            if len(row) > 1 and int(row[1]) >= 0:
                dummy_node = int(row[1])
            if len(row) > 2:
                is_directed = False

            graph_name = g_path.rsplit('/', 1)[1]
            print('DISTANCE: ' + args.EAT_LDT_FT_ST, flush=True)
            print("{} TEXTBOOK-DIAMETER GRAPH={} DUMMY_NODE={} IS_DIRECTED={}"
                  .format(args.EAT_LDT_FT_ST, graph_name, dummy_node, is_directed), flush=True)

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
            distance = dist(graph=g)
            # start_time = time.time()
            diameter = distance.get_diameter()
            pairs_reach = distance.get_reachable_pairs()
            # end_time = time.time()
            # diam_time = end_time - start_time
            print("{} GRAPH={} PAIRS_REACHABLES={}".format(args.EAT_LDT_FT_ST, graph_name, pairs_reach), flush=True)
            print("{} GRAPH={} DIAMETER={}".format(args.EAT_LDT_FT_ST, graph_name, diameter), flush=True)
            # print(dist_name + graph_name + ' TIME: ' + str(diam_time), flush=True)
            print('\n', flush=True)
