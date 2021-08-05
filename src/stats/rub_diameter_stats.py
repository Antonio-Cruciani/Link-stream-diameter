import sys

sys.path.insert(1, '../')

import temporal_graph as tg
import rub_diameter_eat as rub

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute EAT or LDT diameter for all link streams whose paths are in file")
    parser.add_argument("EAT_LDT", type=str, help="EAT for Earliest arrival time diameter, "
                                                  "LDT for Latest departure time diameter")
    parser.add_argument("file", type=str, help="File path")
    args = parser.parse_args()

    if args.EAT_LDT not in {"EAT", "LDT"}:
        raise Exception('Incorrect input argument, use EAT for EAT-Diameter or LDT for LDT-Diameter')

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

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
            graph_name = g_path.rsplit('/', 1)[1]
            print('DISTANCE: ' + args.EAT_LDT, flush=True)
            print("{} GRAPH={} DUMMY_NODE={} IS_DIRECTED={}"
                  .format(args.EAT_LDT, graph_name, dummy_node, is_directed), flush=True)

            if args.EAT_LDT == 'EAT':
                print('DIAMETER EAT ON GRAPH: ' + graph_name + '...', flush=True)
                rub_diam = rub.RubDiameter(graph=g)
                d, n_vis = rub_diam.get_eat()
                print('DIAMETER_EAT={}'.format(d), flush=True)
                print('NUMBER_OF_VISITS={}'.format(n_vis), flush=True)
                print('NUMBER_OF_NODES={}'.format(g.get_n()), flush=True)

            elif args.EAT_LDT == 'LDT':
                print('DIAMETER LDT ON GRAPH: ' + graph_name + '...', flush=True)
                rub_diam = rub.RubDiameter(graph=g)
                d, n_vis = rub_diam.get_ldt()
                print('DIAMETER_LDT={}'.format(d), flush=True)
                print('NUMBER_OF_VISITS={}'.format(n_vis), flush=True)
                print('NUMBER_OF_NODES={}'.format(g.get_n()), flush=True)

            print('\n', flush=True)
