import sys

sys.path.insert(1, '../')
import argparse



import temporal_graph as tg
import temporal_distances as td
import backward_bfs as bwbfs
import tg_utils
from pivot_ifub_diameter import PivotIfubDiameter as ifub

class Greedy:
    def __init__(self, inputPath,k):
        self.input_path = inputPath
        self.k = k

    def greedy_pivot(self):
        with open(self.input_path, "r") as f:
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
                #graph_name = g_path.rsplit('/', 1)[1]
                print("Computing pivots")
                g = tg.Graph(file_path=self.input_path, is_directed=is_directed, latest_node=dummy_node)
                t_min,t_max = g.get_time_interval()
                nv = g.get_num_nodes()
                print("NODE NUMBER ",nv, "T_min = ",t_min,"  T_max = ",t_max)
                pivots = {}
                for node in range(0,nv):
                    if node in g.get_temporal_adj_list().keys():
                        for t in g.get_temporal_adj_list()[node].keys():
                            pivots[(node,t)] = {
                                "BW":[],
                                "FW" :[]
                            }
                print("PIVOT NUMBER ",len(pivots))
                for pivot in pivots.keys():
                    print("COMPUTING PIVOT = ",pivot)
                    a = ifub(g,[pivot])
                    a.get_reachable_pairs()
                    matrix, FW, BW = a.get_is_in_b()
                    pivots[pivot]["BW"] = BW
                    pivots[pivot]["FW"] = FW
                print(pivots)


input ='/home/antoniocruciani/PycharmProjects/Link-stream-diameter/src/stats/graphs/SNAP/sx-mathoverflow-a2q.txt.sor'
k = 3
alg = Greedy(input,k)
alg.greedy_pivot()