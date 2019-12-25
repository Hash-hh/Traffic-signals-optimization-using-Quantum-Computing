import networkx as nx
import numpy as np
import dwave_qbsolv as QBSolv
import time
import statistics
from itertools import combinations
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from matplotlib import pyplot as plt
import pandas as pd


def I(i, j, k):
    """This function converts double index to single index"""
    return k*(i-1) + (j-1)


def make_and_solve(number_of_nodes, step=5, runs=5, token=''):

    # nu = pd.read_csv('class1.csv', dtype=object)  # Dataframe to export values in csv format


    G = nx.Graph()

    nodes = range(1, number_of_nodes + 1)

    edges = list(combinations(nodes, 2))
    # print("edges: ", edges)

    G.add_nodes_from(nodes)

    time_list = []
    edge_no_list = []
    response_list = [[] for i in range(runs)]
    time_list_i = [[] for i in range(runs)]

    for i in range(0, len(edges), step):
        # print("i: ", i)
        # print("slice: ", edges[i:i + step])
        time_taken_list = []

        for run in range(0,runs):

            G.add_edges_from(edges[i:i + step])


            n = number_of_nodes  # n = number of nodes
            k = n  # k = no. of colors
            # n = G.number_of_nodes()  # n = number of nodes
            # print("no. of nodes: ", n)
            N = n * k  # N = dimension of Q matrix

            Q = [[0 for x in range(N)] for y in range(N)]

            # Step 1: add -1 in the diagonal
            for i in range(len(Q)):
                Q[i][i] = -1

            # Step 2: add 2 in blocks
            for s in range(0, N, k):

                for i in range(k):
                    for j in range(i + 1, k):
                        Q[s + i][s + j] = 2

            # Step 3: add 1 with respect to the edges
            edges_i = G.edges()

            for edge in edges_i:
                i, j = edge

                # putting three 1s for every edge
                Q[I(i, 1, k)][I(j, 1, k)] = 1
                Q[I(i, 2, k)][I(j, 2, k)] = 1
                Q[I(i, 3, k)][I(j, 3, k)] = 1

            # print_Q()

            # print("Q is: " + str(len(Q[0])) + " x " + str(len(Q[1])) + " dim ")

            c = np.savetxt('Q_004.csv', Q, delimiter=', ')

            # converting the Q matrix to a dictionary by only saving the non-zero values
            Q_dict = {(row, col): val
                      for row, data in enumerate(Q)
                      for col, val in enumerate(data)
                      if val != 0}

            # token = "DEV-08ef0bb0211afbf99b0563a2428ab5f95f200634"

            endpoint = 'https://cloud.dwavesys.com/sapi'

            # response = QBSolv().sample_qubo(Q_dict)

            if token == '':

                start = time.clock()
                response = QBSolv.QBSolv().sample_qubo(Q_dict)  # Classical
                end = time.clock()

            else:

                start = time.clock()
                # sampler = EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint))
                # response = QBSolv.QBSolv().sample_qubo(Q_dict, solver=sampler)  # QPU

                response = EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint)).sample_qubo(Q_dict,
                                                                                                        num_reads=1)

                print("response power by ARCH uhuhu: ", response)

                end = time.clock()

            response_list[run].append(response)
            time_taken_list.append(end - start)
            time_list_i[run].append(end - start)

            # create lists for easy viewing
            big_list = []
            # for i in response:
                # big_list.append(list(i.values()))
                # print(list(i.values()))
                # print()

        # print("number of edges: ", len(G.edges))
        edge_no_list.append(len(G.edges))
        mean_time = statistics.mean(time_taken_list)
        # print("Time: ", mean_time)
        # print()

        time_list.append(mean_time)
        # print("Hist_list: ", hist_list)

    # plt.hist(hist_list, bins=range(len(hist_list)+1))
    # plt.plot(range(len(time_list)), time_list)

    df = pd.DataFrame({'No. of edges': edge_no_list,
                       'Time': time_list,
                       'Time1': time_list_i[0],
                       'Time2': time_list_i[1],
                       'response1': response_list[0],
                       'response2': response_list[1]})



    df.to_csv('Output/GC' + str(number_of_nodes) + '.csv', index=False)
    df.to_pickle('Output/GC' + str(number_of_nodes) + '.pkl')


    # plt.show()



# just have to call this function multiple times

# make_and_solve(7, step=5, runs=2)



