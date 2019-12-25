import dwave_qbsolv as QBSolv
import time
from dwave.system.samplers import DWaveSampler
# from dwave_qbsolv import QBSolv
from dwave.system.composites import EmbeddingComposite

from dwave.cloud import Client

# from Q_proper import Q
# from itertools import repeat


def Q_dict(Q):
    """This function changes the Q from matrix form to Dict form usable by QBSolv"""

    # Q = -1 * Q
    


    keys = []
    QDist_list = []

    for i in range(len(Q[0])):
        for j in range(len(Q[0])):
            if Q[i][j] != 0:
                keys.append((i, j))
                QDist_list.append(Q[i][j])

    Qdict = {keys[i]: QDist_list[i] for i in range(len(keys))}

    return Qdict


def QBSolve_quantum_solution(Q, times_list, token, print_energy=False):
    """This function use QC to get solution dictionary"""

    Qdict = Q_dict(Q)

    # print("Q_DICT: ", Qdict)

    endpoint = 'https://cloud.dwavesys.com/sapi'

    start = time.clock()

    # sampler = EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint))
    # sampler = DWaveSampler(token=token, endpoint=endpoint)

    # response = sampler.sample_qubo(Qdict, num_reads=5000)



    response = EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint)).sample_qubo(Qdict, num_reads=1, chain_strength=2)

    # client = Client(endpoint='https://cloud.dwavesys.com/sapi', token='secret')

    # with Client.from_config(token=token, endpoint=endpoint) as client:
    #
    #
    #     solver = client.get_solver()
    #
    #     # print("SOLVER: ", solver)
    #
    #     # solver = 'DW_2000Q_2_1'
    #
        # computation = EmbeddingComposite(solver.sample_qubo(Qdict, num_reads=1))
        # computation = solver.sample_qubo(Qdict, num_reads=1)
        # computation.wait()
    #
    #     print("COMP TIME", computation.samples[0])






    # print("response powered by ARCH:", response)

    # response = QBSolv.QBSolv().sample_qubo(Qdict, solver=sampler)

    end = time.clock()

    if print_energy:
        print("energies=" + str(list(response.data_vectors['energy'])))

    time_taken = end - start

    times_list.append(time_taken)

    print("Time taken by quantum QBSolv: ", time_taken)

    qb_solution = list(response.samples())

    qb_solution_list = list(qb_solution[0].values())

    return qb_solution_list


def QBSolve_classical_solution(Q, times_list, print_energy=False):
    """This function use classical QBSolve to get solution dictionary"""

    Qdict = Q_dict(Q)

    start = time.clock()

    response = QBSolv.QBSolv().sample_qubo(Qdict)

    qb_solution = list(response.samples())

    if print_energy:
        print("energies=" + str(list(response.data_vectors['energy'])))

    end = time.clock()

    time_taken = end - start

    print("Time taken by classical QBSolv: ", time_taken)

    times_list.append(time_taken)

    qb_solution_list = list(qb_solution[0].values())

    return qb_solution_list


def solution_slicer(qb_solution_list):

    sliced_sol = [qb_solution_list[x:x + 6] for x in range(0, len(qb_solution_list), 6)]

    # print("Sliced solution: ", sliced_sol)

    return sliced_sol
