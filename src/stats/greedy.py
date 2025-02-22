from ast import Raise
import sys
import pickle
import time
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

    def generate_temporal_pivot_candidates(self):
        # Opening the input file containing the addresses of the temporal graphs
        with open(self.input_path, "r") as f:
            # Read the file row by row
            for row in f:
                # check if line is blank and skip it in affirmative case
                if not row.strip():  
                    continue

                # Split the parameters in the row
                row = row.split()

                # Check that the row contains the right ammount of attributes
                if len(row) > 3:
                    raise Exception('In input file, incorrect line: {}'.format(row))

                # Read the filename of a temporal graph and its properties
                g_path = row[0]
                dummy_node = None
                is_directed = True
                if len(row) > 1 and int(row[1]) >= 0:
                    dummy_node = int(row[1])
                if len(row) > 2:
                    is_directed = False
                
                # Creating the temporal graph
                #graph_name = g_path.rsplit('/', 1)[1]
                print("Computing pivots")
                g = tg.Graph(file_path=g_path, is_directed=is_directed, latest_node=dummy_node)
                
                # Get the max and minimun time steps in the graph and the number of nodes
                t_min,t_max = g.get_time_interval()
                nv = g.get_num_nodes()
                print("NODE NUMBER ",nv, "T_min = ",t_min,"  T_max = ",t_max)

                # Create all temporal pivot candidates
                pivots = {}
                for node in range(0,nv):
                    if node in g.get_temporal_adj_list().keys():
                        for t in g.get_temporal_adj_list()[node].keys():
                            pivots[(node,t)] = {
                                "BW":[],
                                "FW" :[]
                            }

                print("PIVOT NUMBER ",len(pivots))

                # Get the pairs reachable throught each temporal pivot candidate
                i = 0
                for pivot in pivots.keys():
                    #print("COMPUTING PIVOT = ",pivot)
                    a = ifub(g,[pivot])
                    a.get_reachable_pairs()
                    matrix, FW, BW = a.get_is_in_b()
                    pivots[pivot]["BW"] = BW
                    pivots[pivot]["FW"] = FW
                    if (i % 100 == 0 and i != 0):
                        print("PROCESSED ", i, " PIVOTS ")
                    i += 1
                #print(pivots)
                print("Computed result for each pivot")

                outfile = open(g_path + '_pivot_candidates.pickle','wb')
                pickle.dump(pivots, outfile)
                outfile.close()


    def greedy_pivot(self):
         # Opening the input file containing the addresses of the temporal graphs
        with open(self.input_path, "r") as f:
            # Read the file row by row
            for row in f:
                # check if line is blank and skip it in affirmative case
                if not row.strip():  
                    continue

                # Split the parameters in the row
                row = row.split()

                # Check that the row contains the right ammount of attributes
                if len(row) > 3:
                    raise Exception('In input file, incorrect line: {}'.format(row))

                # Read the filename of a temporal graph and its properties
                g_path = row[0]

                # Read the temporal pivot candidates for that graph
                infile = open(g_path + '_pivot_candidates.pickle','rb')
                p_candidates = pickle.load(infile)
                infile.close()

                if self.k < 1:
                    argparse.ArgumentError('K have to be at least 1')

                # Get as a first pivot the temporal node that reaches more pairs
                best_pivot = None
                best_pivot_pairs = 0
                best_pivot_BW = []
                best_pivot_FW = []

                for pivot_candidate in p_candidates:
                    pivot_candidate_pairs = len(pivot_candidate['BW']) * len(pivot_candidate['FW'])
                    if pivot_candidate_pairs > best_pivot_pairs:
                        best_pivot = pivot_candidate
                        best_pivot_pairs = pivot_candidate_pairs
                        best_pivot_BW = pivot_candidate['BW']
                        best_pivot_FW = pivot_candidate['FW']

                del(p_candidates[best_pivot])

                # Select the rest of the pivots
                pivot_set = [best_pivot]
                pivot_set_BW = set(best_pivot_BW)
                pivot_set_FW = set(best_pivot_FW)

                for i in range(1, self.k):
                    best_pivot = None
                    best_pivot_pairs = 0
                    best_pivot_BW = set()
                    best_pivot_FW = set()

                    # Select as new pivot that one that increases the most the reach of the pivot set
                    for pivot_candidate in p_candidates:
                        pivot_candidate_BW = pivot_set_BW.union(pivot_candidate['BW'])
                        pivot_candidate_FW = pivot_set_FW.union(pivot_candidate['FW'])

                        pivot_candidate_pairs = len(pivot_candidate_BW) * len(pivot_candidate_FW)

                        if pivot_candidate_pairs >= best_pivot_pairs:
                            best_pivot = pivot_candidate
                            best_pivot_pairs = pivot_candidate_pairs
                            best_pivot_BW = pivot_candidate_BW
                            best_pivot_FW = pivot_candidate_FW

                    pivot_set.append(best_pivot)
                    pivot_set_pairs = best_pivot_pairs
                    pivot_set_BW = best_pivot_BW
                    pivot_set_FW = best_pivot_FW

                    del(p_candidates(best_pivot))

                
        return pivot_set




input ='/home/antoniocruciani/PycharmProjects/Link-stream-diameter/fileB'
k = 3
alg = Greedy(input,k)
alg.generate_temporal_pivot_candidates()