import numpy as np
# import numpy.core._dtype_ctypes
# from grid_graph import create_graph, random_density_list, node_index
from grid_graph import two_nodes_relation
# import copy
from Q_proper import Q
from graph import graph, ordered_dict_generator, nearby_nodes_from_ordered_dict
# import networkx as nx
from qbsolv import QBSolve_classical_solution, solution_slicer, QBSolve_quantum_solution
# from modes_to_signals import signal_state
from modes_to_signals import sel_rules_to_mode_list, modes_to_signals
# import time
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
from simulation_under_hood import traffic_init, traffic_sim, cost_list_generator, cost_lists_init, f_lists_init
# from traffic_flow_simulator import cost_list_updater
from visualization import xy_from_pos_list, x_y_from_xy, pos_shift, graph_topo, color_gen
# from scipy.misc import imread
# from matplotlib.pyplot import imread
from Graph_coloring import make_and_solve
import pandas as pd
from dwave.cloud import Client

import sys


# [up, down, right, left] for each node. up for example represents those who want to go up (naturally)
# cost_lists = [[5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30]]
# f_list = [[0.7, 0.2], [0.7, 0.2], [0.5, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2], [0.7, 0.2],
#           [0.7, 0.2], [0.7, 0.2]]
# cost_lists = [[5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30], [5, 20, 25, 30]]


def modes_from_Q(G, times_list, token, graph_ordered_dict, cost_lists, lembda, runtime, run, first_time):

    Q_ = Q(G, graph_ordered_dict, cost_lists, lembda, runtime, run, first_time)

    if token == "":   # Solve classically if no token given, otherwise send to QC using the entered token.
        sol = QBSolve_classical_solution(Q_, times_list)
    else:
        sol = QBSolve_quantum_solution(Q_, times_list, token)

    sliced_sol = solution_slicer(sol)
    print("Sliced Sol: ", sliced_sol)
    error = 0
    for i in sliced_sol:
        counter = 0
        for j in i:
            if j == 1:
                counter += 1
            if counter > 1:
                # print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n"
                #       "ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n"
                #       "ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n"
                #       "ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n"
                #       "ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n")
                error += 1
    mode_list = sel_rules_to_mode_list(sliced_sol)

    return mode_list, error



def modes_from_cycle(G, last_signal_list):

    no_of_nodes = len(list(G.nodes()))

    if last_signal_list[0] == 0:
        return [1]*no_of_nodes

    if last_signal_list[0] == 1:
        return [2]*no_of_nodes

    if last_signal_list[0] == 2:
        return [3]*no_of_nodes

    if last_signal_list[0] == 3:
        return [4]*no_of_nodes

    if last_signal_list[0] == 4:
        return [5]*no_of_nodes

    if last_signal_list[0] == 5:
        return [0]*no_of_nodes





def metric_q3(last_signal_list_, temp_list, first_time = False):
    """This function returns a parameter that quantifies how well Q3 is working - namely, how many cars pass
    through signals unhindered"""

    # if last_signal_ == True and current_signal_ == True:
    #         return len(temp_list)

    for item in last_signal_list_:
        if not item:
            return 0

    return len(temp_list)



def metric_q3_bad(G, graph_ordered_dict_, mode, cost_list, b):
    """This function returns a parameter that quantifies how well Q3 is working - namely, how many cars pass
    through signals unhindered"""

    badness = 0
    up, down, right, left = nearby_nodes_from_ordered_dict(graph_ordered_dict_, b)

    if mode == 0:
        if right != 'False':
            badness += G[b][right]['speed'] * 0.7 * cost_list[2]
        if down != 'False':
            badness += G[b][down]['speed'] * 0.3 * cost_list[2]

        if left != 'False':
            badness += G[b][left]['speed'] * 0.7 * cost_list[3]
        if up != 'False':
            badness += G[b][up]['speed'] * 0.3 * cost_list[3]

    if mode == 1:
        if down != 'False':
            badness += G[b][down]['speed'] * 0.7 * cost_list[1]
            badness += G[b][down]['speed'] * 0.3 * cost_list[2]
        if left != 'False':
            badness += G[b][left]['speed'] * 0.3 * cost_list[1]
            badness += G[b][left]['speed'] * 0.7 * cost_list[3]

        if right != 'False':
            badness += G[b][right]['speed'] * 0.7 * cost_list[2]

        if up != 'False':
            badness += G[b][up]['speed'] * 0.3 * cost_list[3]

    if mode == 2:
        if up != 'False':
            badness += G[b][up]['speed'] * 0.7 * cost_list[0]
            badness += G[b][up]['speed'] * 0.3 * cost_list[3]
        if right != 'False':
            badness += G[b][right]['speed'] * 0.3 * cost_list[0]
            badness += G[b][right]['speed'] * 0.7 * cost_list[2]
        if down != 'False':
            badness += G[b][down]['speed'] * 0.3 * cost_list[2]
        if left != 'False':
            badness += G[b][left]['speed'] * 0.7 * cost_list[3]


    if mode == 3:
        if up != 'False':
            badness += G[b][up]['speed'] * 0.7 * cost_list[0]
        if right != 'False':
            badness += G[b][right]['speed'] * 0.3 * cost_list[0]
        if down != 'False':
            badness += G[b][down]['speed'] * 0.7 * cost_list[1]
        if left != 'False':
            badness += G[b][left]['speed'] * 0.3 * cost_list[1]

    if mode == 4:
        if up != 'False':
            badness += G[b][up]['speed'] * 0.7 * cost_list[0]
            badness += G[b][up]['speed'] * 0.3 * cost_list[3]
        if right != 'False':
            badness += G[b][right]['speed'] * 0.3 * cost_list[0]
        if down != 'False':
            badness += G[b][down]['speed'] * 0.7 * cost_list[1]
        if left != 'False':
            badness += G[b][left]['speed'] * 0.3 * cost_list[1]
            badness += G[b][left]['speed'] * 0.7 * cost_list[3]


    if mode == 5:
        if up != 'False':
            badness += G[b][up]['speed'] * 0.7 * cost_list[0]
        if right != 'False':
            badness += G[b][right]['speed'] * 0.3 * cost_list[0]
            badness += G[b][right]['speed'] * 0.7 * cost_list[2]
        if down != 'False':
            badness += G[b][down]['speed'] * 0.7 * cost_list[1]
            badness += G[b][down]['speed'] * 0.3 * cost_list[2]
        if left != 'False':
            badness += G[b][left]['speed'] * 0.3 * cost_list[1]


    return badness



def metric_q1(cost_lists_):
    """This function returns a parameter that quantifies how well Q1 is working -- namely, how many cars at any given
    time are stopped / are positioned bumper to bumper"""

    num = 0

    for item in cost_lists_:
        for j in item:
            num += j

    return num





def Main(token):

    """The mother function that, when called by the GUI, runs the entire program"""
    """UNCOMMENT THE FOLLOWING LINE FOR THE FINAL PROGRAM: (KEEP COMMENTED FOR TESTING/DEBUGGING)"""
    # sys.stdout = open("Output/OUTPUT.txt", "w")


    nu = pd.read_csv('skeli.csv', dtype=object)  # Dataframe to export values in csv format

    dim = 6
    lembda = 60
    # lembda = 0.00000000000000000000000000000000000000000001

    times_list = []

    graph_ordered_dict = ordered_dict_generator(dim)
    print("graph_ordered_dict", graph_ordered_dict)

    G = graph(graph_ordered_dict)

    # cost_lists = cost_lists_init(G.number_of_nodes(), 5, 10)
    cost_lists = cost_lists_init(G.number_of_nodes(), repeat_list=[30,30,30,30])

    f_lists = f_lists_init(G.number_of_nodes())

    # print("graph nodes: ", list(G.edges()))

    edges = list(G.edges())
    nodes = list(G.nodes())

    color_list = color_gen(len(edges))
    # print("color list: ", color_list)
    # print("color list: ", edges)

    traffic_init(graph_ordered_dict, G, cost_lists, color_list)

    # for i in range(len(edges)):
    #     print("pos_list", len(G[edges[i][0]][edges[i][1]]['pos_list']))
    #
    # print("\n\n\n\n\n")

    runtime_list = []

    goodness = 0
    badness = 0

    # diff = 0

    # start_ = time.clock()

    run = 0

    error = 0

    mode_list_ = modes_from_Q(G, times_list, token, graph_ordered_dict, cost_lists, lembda, 1, run, first_time=True)
    mode_list = mode_list_[0]
    error += mode_list_[1]


    # mode_list = [0] * len(list(G.nodes))

    signals = modes_to_signals(mode_list)

    # end_ = time.clock()

    # first_run_time = end_ - start_

    # runtime_list.append(first_run_time)

    mode_list = list(np.zeros(36))
    # mode_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # mode_list = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # mode_list = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    # mode_list = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    # mode_list = [4, 4, 4, 4, 4, 4, 4, 4, 4]
    # mode_list = [5, 5, 5, 5, 5, 5, 5, 5, 5]

    print("MODES:\n\n", mode_list)

    graph_topo(G, dim)

    # start = time.clock()

    stopped_list = []
    stopped = metric_q1(cost_lists)
    stopped_list.append(stopped)

    # last_signal = False


    buff = 4   # q3 related

    # last_signal_list = [False for i in range(buff)]
    last_signal_list = []
    signal_list = []

    # print("LEN EDGES: ", len(edges))

    while False:
    # while run < 150:

        # print("Runtime List: ", runtime_list)
        # avg_runtime = np.mean(runtime_list)
        avg_stopped = np.mean(stopped_list)  # quantifies how much q1 is dominant

        # start_for_q = time.clock()

        # if diff >= 2:
        if run % 5 == 0 and run != 0:
        # if run != 0:
            # correct_time_start = time.clock()
            # print("cost lists: ", cost_lists)

            mode_list_ = modes_from_Q(G, times_list, token, graph_ordered_dict, cost_lists, lembda, 1, run,
                                     first_time=False)  # 1 is the avg runtime
            mode_list = mode_list_[0]


            error += mode_list_[1]


            # mode_list = modes_from_cycle(G, mode_list)





            signals = modes_to_signals(mode_list)
            # print("signals: ", signals)

            print("\n\nMODES:", mode_list)
            nu['Modes'][run] = mode_list

        signal_list.append(signals)




            # correct_time_end = time.clock()

            # correction_time = correct_time_end - correct_time_start
            # start = time.clock()

            # start_for_q = time.clock()

        # print("MODE LIST: ", mode_list)
        # print("COST LISTS: ", cost_lists)
        # print(edges)

        # big_pos_list = []
        # xy = []

        # big_x = ()
        # big_y = ()

        # big_color = ()

        for i in range(len(edges)):

            a = edges[i][0]
            b = edges[i][1]

            # print("a: ", a)
            # print("b: ", b)
            # print("\n")

            pos_list_stick = G[a][b]['pos_list_stick']
            # print("pos list stick: ", pos_list_stick)

            relation = two_nodes_relation(nodes, a, b, graph_ordered_dict)
            # print("relation: ", relation)
            # print("signals[a]: ", signals[a])

            # print("signal_list: ", signal_list)
            # print("signal_list[-1]: ", signal_list[-1])
            # print("signal_list[-1][0]: ", signal_list[-1][0])
            # print("signal_list: ", signal_list)


            mode = mode_list[b]
            cost_list = cost_lists[b]


            if run >= buff:

                for i in range(1, buff+1):


                    if relation == 'up':
                        current_signal = signal_list[-i][a][0]

                    if relation == 'down':
                        current_signal = signal_list[-i][a][1]

                    if relation == 'right':
                        current_signal = signal_list[-i][a][2]

                    if relation == 'left':
                        current_signal = signal_list[-i][a][3]


                    last_signal_list.append(current_signal)

            # print("last_signal_list: ", last_signal_list)

            # print("current_signal: ", current_signal)
            # print("last_signal: ", last_signal)
            # print("\n")

            goodness += metric_q3(last_signal_list, pos_list_stick)
            badness += metric_q3_bad(G, graph_ordered_dict, mode, cost_list, b)



            last_signal_list = []

            # last_signal = current_signal
            # last_signal_list(current_signal)

            # pos_list_ = G[a][b]['pos_list']

            # print("G[", a, "][", b, "]['pos_list_temp]: ", G[a][b]['pos_list_temp'])

            # print("pos_list in list: ", pos_list)



            # color_list_ = tuple(G[a][b]['color_list'])




            # print("color list of ", edges[i][0], edges[i][1], color_list_)


            # xy = xy_from_pos_list(G, a, b, 1000, pos_list_, dim, dim)


            # print("len xy: ", len(xy))



            # a = edges[i][0]
            # b = edges[i][1]

            # horizontal = G[a][b]['horizontal']
            # forward = G[a][b]['forward']

            # shifted_xy = pos_shift(xy, horizontal, forward)

            # shifted_xy = xy

            # x, y = x_y_from_xy(shifted_xy)
            # print("len x: ", len(x))
            # x, y = x_y_from_xy(xy)

            # big_x = big_x + x
            # big_y = big_y + y

            # big_color = big_color + color_list_

        # print("len of big color: ", len(big_color))
        # print("no. of cars: ", len(big_x))

        traffic_sim(graph_ordered_dict, G, f_lists, mode_list, 1)
        cost_lists = cost_list_generator(G, graph_ordered_dict)

        # img = imread("media/graph_BG.jpg")

        # plt.scatter(big_x, big_y, s=5, color=big_color)
        # plt.imshow(img, zorder=0, extent=[1000, 4000, 1500, 4500])

        # plt.xlim((1000, 4000))
        # plt.ylim((1500, 4500))

        # plt.show()




        # plt.pause(0.000005)
        # plt.clf()
        # plt.close()

        # end = time.clock()

        # end_for_q = time.clock()

        # runtime = end_for_q - start_for_q
        # runtime_list.append(runtime)

        stopped = metric_q1(cost_lists)
        stopped_list.append(stopped)

        # diff = end - start
        # print("diff: ", diff)

        print("\nStoppedness: ", avg_stopped)
        print("Goodness: ", goodness)
        print("Badness: ", badness)


        nu['Stoppedness'][run] = avg_stopped
        nu['Goodness'][run] = goodness
        nu['Badness'][run] = badness

        run += 1

        counter = 0

        print("\nNumber of errors related to lambda: ", error)

        for i in range(len(edges)):

            a = edges[i][0]
            b = edges[i][1]

            counter += len(G[a][b]['pos_list'])

            # if G[a][b]['pos_list'] == []:
            #     counter += 1

        print("\nTimes List:  ", times_list)
        Average_QBSolv_time = sum(times_list) / len(times_list)
        print("Average QBSolv time: ", Average_QBSolv_time)

        nu['Times List'][run] = times_list
        nu['Average QBSolv time'][run] = Average_QBSolv_time


        if run == 300:
            print("Number of cars remaining on the map: ", counter)
            nu['Remaining Cars'] = counter
            return 0

            # if counter == len(edges):
            #     print("TOTAL RUNS: ", run)
            #     sys.exit(0)

    nu.to_csv('Output/results.csv', index=False)
    nu.to_pickle('Output/results.pkl')



    # print("PROBLEM 1 DONEEEE")
    #
    #
    # #### Graph Coloring part ####
    #
    # make_and_solve(5, step=10, runs=2, token=token)
    # print("GC 1 DONEEE")
    # make_and_solve(20, step=20, runs=2, token=token)
    # print("GC 2 DONEEE")
    # make_and_solve(25, step=30, runs=2, token=token)
    # print("GC 3 DONEEE")
    # make_and_solve(30, step=40, runs=2, token=token)
    # print("GC 4 DONEEE")
    # make_and_solve(35, step=60, runs=2, token=token)
    # print("GC 5 DONEEE")
    #
    #
    # print("GC DONEEEE")


# token = 'DEV-4fb0a5c1dd3c45a0d1e7a4b5dc1cdf14fcd6105b'
# # token = ''
#
# Main(token)
