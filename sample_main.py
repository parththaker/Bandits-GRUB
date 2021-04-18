import matplotlib.pyplot as plt
import numpy as np
import toml
import graph_algo
import support_func

with_reset = False
check_last_cluster = True
cluster_size = 10

# TODO : np.matrix is gonna be deprecated soon. Need to switch to np.array. Need to check for cross-compatibility.
setup = '1vw_med_graph_SBM_static'
data = toml.load('sample_config.toml')
Degree = np.matrix(data[setup]['Degree'])
Adj = np.matrix(data[setup]['Adj'])
node_means = np.array(data[setup]['node_means'])
nodes = data[setup]['nodes']


def run_algo(GB, printer):
    """
    Run one of the algo picked from graph_algo.py

    Parameters
    ----------
    GB : One of the algorithms defined in graph_algo.py
    printer : Name tag for the algorithm (only required for printing)

    Returns
    -------
    Time_tracker_GB : Array with time marker when associated with arm elimination
    Lcluster_GB : Time marker when the best cluster was remaining
    Lnode_GB : Time marker when the last arm was remaining
    """

    i = 0
    t = 0
    Time_tracker_GB = []
    Lcluster_GB = []
    Lnode_GB = []

    flip = 1
    remainder = int(nodes)
    Time_tracker_GB.append(0)
    while (1):
        i += 1
        t += support_func.round_function(i)
        GB.play_round(support_func.round_function(i))

        if remainder != len(GB.remaining_nodes):
            for j in range(remainder - len(GB.remaining_nodes)):
                Time_tracker_GB.append(t)
            remainder = len(GB.remaining_nodes)

        if len(GB.remaining_nodes) <= cluster_size and flip:
            Lcluster_GB.append(t)
            flip = 0

        if len(GB.remaining_nodes) == 1:
            Lnode_GB.append(t)
            break

    print("Node indices remaining : ", GB.remaining_nodes)
    print("Total time taken by algo ", printer, " : ", t)
    return Time_tracker_GB, Lnode_GB, Lcluster_GB


if __name__ == "__main__":
    """
    Sample code where we run 3 different algorithms from graph_algo.py. 
    Testing is performed by one run of each algorithm on a sample graph bandit problem from sample_config.toml
    """

    GB = graph_algo.GraphBanditEliminationAlgo(Degree, Adj, node_means)
    GB_2 = graph_algo.GraphBanditEliminationAlgoImpSampling(Degree, Adj, node_means, eps=0.0)
    Base = graph_algo.GraphBanditBaseLine(Degree, Adj, node_means)

    Time_tracker_GB, _, _ = run_algo(GB, printer="GB")
    Time_tracker_GB_2, _, _ = run_algo(GB_2, printer="GB_2")
    Time_tracker_Base, _, _ = run_algo(Base, printer="Base")

    plt.plot(Time_tracker_GB_2, 100*np.ones(GB_2.dim) - range(len(Time_tracker_GB_2)), label='Proposed algo', linewidth=2.0)
    plt.plot(Time_tracker_GB, 100*np.ones(GB.dim) - range(len(Time_tracker_GB)), label='Valko et.al', linewidth=2.0)
    plt.plot(Time_tracker_Base, 100*np.ones(Base.dim)-range(len(Time_tracker_Base)), label='Cyclic algo', linewidth=2.0)
    plt.title("No. of remaining arms vs time steps")
    plt.xlabel("Time steps")
    plt.ylabel("No. of remaining arms")
    plt.grid()
    plt.legend()
    plt.show()
