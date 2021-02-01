
import sys

sys.path.insert(1, '../')

import temporal_graph as tg
import ifub_fp

import math
import argparse

if __name__ == '__main__':

    input_path = './TGraphs/TGraphs_list'

    parser = argparse.ArgumentParser(
        description="Compute iFUB Fastest Path for all graps whose paths are in file"
                    " './TGraphs/TGraphs_list'")
    parser.add_argument("NumberHubs", type=str, help="Number of hubs, log for log hubs")
    parser.add_argument("NumberTimes", type=int, help="Number of times")
    args = parser.parse_args()
    dist = None
    h = args.NumberHubs
    t = args.NumberTimes

    num_hub = [h]
    num_times = [t]

    for k, n_h in enumerate(num_hub):
        for idx, i in enumerate(num_times):

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
                    print('\n')
                    print('FP iFUB Graph ' + graph_name, flush=True)
                    print('FP iFUB Graph ' + graph_name + ' Dummy node: ' + str(dummy_node), flush=True)
                    print('FP iFUB Graph ' + graph_name + ' is_directed: ' + str(is_directed), flush=True)

                    g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)

                    if g.get_latest_node() is not None:
                        num_nodes = g.get_latest_node()
                    else:
                        num_nodes = g.get_num_nodes()

                    t_min, t_max = g.get_time_interval()

                    if n_h == 'log':
                        log_num_nodes = round(math.log(num_nodes, 2))
                        hubs = log_num_nodes
                    else:
                        hubs = int(n_h)

                    #############################
                    print('\nSTRATEGY 3', flush=True)
                    landmarks = []
                    # Give me n_h hub
                    _, hub = g.get_max_deg_out(n=hubs)
                    print('HUB' + str(hub), flush=True)
                    tw = [t_min + (j / (i + 1) * (t_max - t_min)) for j in range(1, i+1)]
                    for h in hub:
                        for t in tw:
                            landmarks.append((h, t))
                    print('---', flush=True)
                    print(landmarks, flush=True)

                    diam, num_visits_ifub, num_distinct_A, num_distinct_B, pairs_reach = ifub_fp.ifub_on_landmarks(
                        graph=g, landmarks=landmarks)
        
                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' DIAMETER: ' + str(diam), flush=True)
                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' NUMBER OF VISITS: ' + str(num_visits_ifub), flush=True)

                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' Pairs reachables with landmarks: ' + str(pairs_reach), flush=True)
                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' DISTINCT elements in A: ' + str(num_distinct_A), flush=True)
                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' DISTINCT elements in B: ' + str(num_distinct_B), flush=True)
                    print('S3 FP NumTimes=' + str(i) + ' NumHubs=' + str(num_hub[k]) + ' ' + graph_name +
                          ' Min(Distinct A, Distinct B), maxBFS classic: ' +
                          str(min(num_distinct_A, num_distinct_B)), flush=True)
        
                    print('\n', flush=True)
