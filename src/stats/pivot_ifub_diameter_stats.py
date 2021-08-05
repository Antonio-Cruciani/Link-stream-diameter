import sys

sys.path.insert(1, '../')

import temporal_graph as tg
import pivot_ifub_diameter as ifub_pd

import math
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Compute EAT or LDT or FT or ST pivot diameter for all link streams whose paths are in file")
    parser.add_argument("EAT_LDT_FT_ST", type=str, help="EAT for Earliest Arrival Time Pivot-Diameter, "
                                                        "LDT for Latest Departure Time Pivot-Diameter, "
                                                        "FT for Fastest Time Pivot-Diameter, "
                                                        "ST for Shortest Time Pivot-Diameter")
    parser.add_argument("NumberHubs", type=str, help="Number of hubs, use 'log' for log(n) hubs")
    parser.add_argument("NumberTimes", type=int, help="Number of equally spaced times")
    parser.add_argument("file", type=str, help="File path")
    args = parser.parse_args()

    if args.EAT_LDT_FT_ST not in {"EAT", "LDT", "FT", "ST"}:
        raise Exception('Error on argument EAT_LDT_FT_ST. Use EAT for EAT-Pivot-Diameter, LDT for LDT-Pivot-Diameter, '
                        'FT for FT-Pivot-Diameter or ST for ST-Pivot-Diameter')

    num_h = args.NumberHubs
    t = args.NumberTimes
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
            print('\nDISTANCE: ' + args.EAT_LDT_FT_ST, flush=True)
            print("{} iFUB PIVOT-DIAMETER GRAPH={} DUMMY_NODE={} IS_DIRECTED={}"
                  .format(args.EAT_LDT_FT_ST, graph_name, dummy_node, is_directed), flush=True)

            g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
            t_alpha, t_omega = g.get_time_interval()

            if num_h == 'log':
                h = round(math.log(g.get_n(), 2))
            else:
                h = int(num_h)

            pivots = []
            # Give me h hubs
            _, hub = g.get_max_deg_out(n=h)
            print('HUB={}'.format(hub), flush=True)
            tw = [t_alpha + (i / (t + 1) * (t_omega - t_alpha)) for i in range(1, t + 1)]
            for node in hub:
                for ti in tw:
                    pivots.append((node, ti))
            print('PIVOTS:\n{}'.format(pivots), flush=True)

            ifub_piv_diam = ifub_pd.PivotIfubDiameter(graph=g, pivots=pivots)

            if args.EAT_LDT_FT_ST == 'EAT':
                diam, n_visits = ifub_piv_diam.get_eat()
            elif args.EAT_LDT_FT_ST == 'LDT':
                diam, n_visits = ifub_piv_diam.get_ldt()
            elif args.EAT_LDT_FT_ST == 'FT':
                diam, n_visits = ifub_piv_diam.get_ft()
            elif args.EAT_LDT_FT_ST == 'ST':
                diam, n_visits = ifub_piv_diam.get_st()

            print("{} NumTimes={} NumHubs={} Graph={} DIAMETER={}".format(args.EAT_LDT_FT_ST, t, num_h, graph_name,
                                                                          diam), flush=True)

            print("{} NumTimes={} NumHubs={} Graph={} NUMBER_OF_VISITS={}".format(args.EAT_LDT_FT_ST, t, num_h,
                                                                                  graph_name, n_visits), flush=True)

            print("{} NumTimes={} NumHubs={} Graph={} PAIRS_REACHABLES_THROUGH_PIVOTS={}"
                  .format(args.EAT_LDT_FT_ST, t, num_h, graph_name, ifub_piv_diam.get_reachable_pairs()), flush=True)

            print("{} NumTimes={} NumHubs={} Graph={} DISTINCT_ELEMENTS_IN_A={}"
                  .format(args.EAT_LDT_FT_ST, t, num_h, graph_name, ifub_piv_diam.get_distinct_in_a()), flush=True)

            print("{} NumTimes={} NumHubs={} Graph={} DISTINCT_ELEMENTS_IN_B={}"
                  .format(args.EAT_LDT_FT_ST, t, num_h, graph_name, ifub_piv_diam.get_distinct_in_b()), flush=True)

            print("{} NumTimes={} NumHubs={} Graph={} MIN(DISTINCT_A, DISTINCT_B)={}"
                  .format(args.EAT_LDT_FT_ST, t, num_h, graph_name,
                          min(ifub_piv_diam.get_distinct_in_a(), ifub_piv_diam.get_distinct_in_b())), flush=True)

            print('\n', flush=True)
