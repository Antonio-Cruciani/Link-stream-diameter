import sys

sys.path.insert(1, '../')

import temporal_graph as tg
import rub_diameter_eat as rub

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute EAT or LDT diameter for all graps whose paths are in file "
                    "'./TGraphs/TGraphs_list'")
    parser.add_argument("EAT_LDT", type=str, help="EAT for Earliest arrival time diameter, "
                                                  "LDT for Latest departure time diameter")
    args = parser.parse_args()

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

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

            graph_name = g_path.rsplit('/', 1)[1]
            print('Graph ' + graph_name, flush=True)
            print('Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
            print('Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

            if g.get_latest_node() is not None:
                num_nodes = g.get_latest_node()
            else:
                num_nodes = g.get_num_nodes()

            if args.EAT_LDT == 'EAT':
                print('DIAMETER EAT ON GRAPH: ' + g.get_file_path().rsplit('/', 1)[1] + '...', flush=True)
                d, n_vis, sec = rub.rub_diam_eat(graph=g)
                print('Diameter EAT: ' + str(d), flush=True)
                print('Number of visits: ' + str(n_vis), flush=True)
                print('Number of nodes: ' + str(num_nodes), flush=True)
                # print('Time spent: ' + str(sec), flush=True)

            elif args.EAT_LDT == 'LDT':
                print('DIAMETER LDT ON GRAPH: ' + g.get_file_path().rsplit('/', 1)[1] + '...', flush=True)
                d, n_vis, sec = rub.rub_diam_ldt(graph=g)
                print('Diameter LDT: ' + str(d), flush=True)
                print('Number of visits: ' + str(n_vis), flush=True)
                print('Number of nodes: ' + str(num_nodes), flush=True)
                # print('Time spent: ' + str(sec), flush=True)

            else:
                raise Exception('Incorrect input argument, use EAT for EAT diameter or LDT for LDT diameter')

            print('\n', flush=True)
