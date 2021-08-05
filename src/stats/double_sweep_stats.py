import sys

sys.path.insert(1, '../')

import temporal_distances as td
import temporal_graph as tg
import double_sweep as d_sw

import random
import math
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute EAT or LDT or FT or ST double sweep VS randoms BFS for all link streams whose paths "
                    "are in file")
    parser.add_argument("EAT_LDT_FT_ST", type=str, help="EAT for Earliest Arrival Time Lower Bound, "
                                                        "LDT for Latest Departure Time Lower Bound, "
                                                        "FT for Fastest Time Lower Bound, "
                                                        "ST for Shortest Time Lower Bound")
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
        raise Exception('Error on argument EAT_LDT_FT_ST. Use EAT for EAT-Lower Bound, LDT for LDT-Lower Bound, '
                        'FT for FT-Lower Bound or ST for ST-Lower Bound')
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

            print('\nDISTANCE: ' + args.EAT_LDT_FT_ST + '\n', flush=True)
            graph_name = g_path.rsplit('/', 1)[1]
            print(args.EAT_LDT_FT_ST + ' Graph ' + graph_name, flush=True)
            print(args.EAT_LDT_FT_ST + ' Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
            print(args.EAT_LDT_FT_ST + ' Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

            a = round(math.log(g.get_n(), 2))
            num_2sweep = [1, a, 2*a, 4*a]
            num_visits = ["4", "4*log(n)", "8*log(n)", "16*log(n)"]

            print(args.EAT_LDT_FT_ST + ' Graph ' + graph_name + ' Num of BFS: ' + str(num_2sweep), flush=True)

            if ((4 * a) * 4) >= g.get_n():
                print('GRAPH ' + graph_name + ' SKIPPED: It is too small, 4*log(n) > n', flush=True)
                continue

            start_nodes = random.sample(range(g.get_n()), (4 * a) * 4)

            print('2SWEEP RESULTS: ', flush=True)

            double_sw = d_sw.DoubleSweep(graph=g, start_node=start_nodes[0])
            if args.EAT_LDT_FT_ST == 'EAT':
                lb, _, _ = double_sw.get_lb_eat()
            elif args.EAT_LDT_FT_ST == 'LDT':
                lb, _, _ = double_sw.get_lb_ldt()
            elif args.EAT_LDT_FT_ST == 'FT':
                lb, _, _ = double_sw.get_lb_ft()
            elif args.EAT_LDT_FT_ST == 'ST':
                lb, _, _ = double_sw.get_lb_st()

            print("{} {} StartNode={} NumberVisits={} 2sw_lb={}".format(args.EAT_LDT_FT_ST, graph_name, start_nodes[0],
                                                                        num_visits[0], lb), flush=True)
            print("{} {} NumberVisits={} MAX_2sw={}".format(args.EAT_LDT_FT_ST, graph_name, num_visits[0], lb),
                  flush=True)

            print('RND RESULTS:', flush=True)
            rnd_estimation = -1
            for h in range(0, 4):

                distance = dist(graph=g)
                ecc = distance.get_eccentricity_fw(source=start_nodes[h])

                print("{} {} StartNode={} NumberVisits={} Rnd_ecc={}".format(
                    args.EAT_LDT_FT_ST, graph_name, start_nodes[0], num_visits[0], ecc), flush=True)

                if ecc > rnd_estimation:
                    rnd_estimation = ecc

            print("{} {} NumberVisits={} MAX_rnd={}".format(args.EAT_LDT_FT_ST, graph_name, num_visits[0],
                                                            rnd_estimation), flush=True)

            for h in range(1, len(num_2sweep)):
                print('2SWEEP RESULTS: ', flush=True)
                for j in range(num_2sweep[h - 1], num_2sweep[h]):

                    double_sw = d_sw.DoubleSweep(graph=g, start_node=start_nodes[j])
                    if args.EAT_LDT_FT_ST == 'EAT':
                        new_lb, _, _ = double_sw.get_lb_eat()
                    elif args.EAT_LDT_FT_ST == 'LDT':
                        new_lb, _, _ = double_sw.get_lb_ldt()
                    elif args.EAT_LDT_FT_ST == 'FT':
                        new_lb, _, _ = double_sw.get_lb_ft()
                    elif args.EAT_LDT_FT_ST == 'ST':
                        new_lb, _, _ = double_sw.get_lb_st()

                    print("{} {} StartNode={} NumberVisits={} 2sw_lb={}"
                          .format(args.EAT_LDT_FT_ST, graph_name, start_nodes[j], num_visits[h], new_lb), flush=True)

                    if new_lb > lb:
                        lb = new_lb

                print("{} {} NumberVisits={} MAX_2sw={}".format(args.EAT_LDT_FT_ST, graph_name, num_visits[h], lb),
                      flush=True)

                print('RND RESULTS:', flush=True)
                for j in range((num_2sweep[h - 1]) * 4, (num_2sweep[h]) * 4):

                    distance = dist(graph=g)
                    ecc = distance.get_eccentricity_fw(source=start_nodes[j])

                    print("{} {} StartNode={} NumberVisits={} Rnd_ecc={}".format(
                        args.EAT_LDT_FT_ST, graph_name, start_nodes[j], num_visits[h], ecc), flush=True)

                    if ecc > rnd_estimation:
                        rnd_estimation = ecc

                print("{} {} NumberVisits={} MAX_rnd={}".format(args.EAT_LDT_FT_ST, graph_name, num_visits[h],
                                                                rnd_estimation), flush=True)
