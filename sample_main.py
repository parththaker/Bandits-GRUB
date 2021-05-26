import matplotlib.pyplot as plt
import numpy as np
import toml
import graph_algo
import graph_generator
import support_func


load_sample = False
read_config = True


def load_parameters():
    """

    Function to call graph_generator.py to provide graph data with the config setup from one of the following:
    1. Loads the sample graph setup from 'sample_setup.toml'
    2. Reading the setup from 'config.toml' and then generate graph.
    3. Generating graph from the default setup hardcodede in the function.

    Returns
    -------
    data : dictionary with graph related data (Degree, adjacency, means, total nodes).
    """
    # FIXME : load_sample = True is not compatible. Need to fix errors.
    if load_sample:

        setup = '1vw_med_graph_SBM_static'
        toml_data = toml.load('sample_setup.toml')

        # TODO : Not implemented 'node_per_cluster' functionality in the .toml setup yet.
        node_per_cluster = 10
        clusters = 10

        data = toml_data[setup]

    elif read_config:

        module = toml.load('config.toml')
        config_data = module['system']
        node_per_cluster = config_data['node_per_cluster']
        clusters = config_data['clusters']
        p = config_data['p']
        q = config_data['q']
        graph_type = config_data['graph']

        cluster_means = node_per_cluster * clusters * np.ones(clusters) - node_per_cluster * np.arange(0, 1,
                                                                                                       1.0 / clusters)

        data = graph_generator.call_generator(node_per_cluster, clusters, p, cluster_means, graph_type, q=q)

    # Keeptng the following option as a default parameter setup
    else:

        node_per_cluster = 10
        clusters = 10
        p = 0.9
        q = 0.001

        cluster_means = node_per_cluster * clusters * np.ones(clusters) - node_per_cluster * np.arange(0, 1,
                                                                                                       1.0 / clusters)

        data = graph_generator.call_generator(node_per_cluster, clusters, p, cluster_means, 'SBM', q=q)

    return data, node_per_cluster, clusters


def run_algo(GB, printer, cluster_size):
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
    Testing is performed by one run of each algorithm on a sample graph bandit problem from sample_setup.toml
    """

    """
    Phase 1 : Setup graph data
    -------------------------
    
    This is implemented either by reading from .toml config file or generate directly from graph_generator.py
    """

    data, node_per_cluster, clusters = load_parameters()

    """
    Graph types supported: Changes regarding the same can be made in config.toml
    I. SBM
    II. N-cluster graph 
        Possible graph types for intra cluster graphs so far:
            1. Tree
            2. BA
            3. Line
            4. Complete
            5. ER
            6. Star
            7. Wheel
    """

    # TODO : np.matrix is gonna be deprecated soon. Need to switch to np.array. Need to check for cross-compatibility.

    Degree = np.matrix(data['Degree'])
    Adj = np.matrix(data['Adj'])
    node_means = np.array(data['node_means'])
    nodes = data['nodes']

    # Optimization routine to find mean distribution with epsilon error bound.
    new_means = support_func.find_means(Degree-Adj, 2.0, node_means)
    node_means = new_means

    # Set the max mean to be 1.2 times the max mean value of all other nodes. The factor can be changed
    a = toml.load('config.toml')
    factor = a['system']['best_mean_factor']
    node_means[0] = max(new_means[1:])*factor

    """
    Phase 2 : Run competing algorithms
    ----------------------------------
    
    Run the choice of algorithms defined in graph_algo.py
    """

    GB = graph_algo.MaxVarianceArmAlgo(Degree, Adj, node_means)
    GB_2 = graph_algo.MaxDiffVarAlgo(Degree, Adj, node_means, eps=0.0)
    GB_det = graph_algo.OneStepMinDetAlgo(Degree, Adj, node_means, eps=0.0)
    GB_sum = graph_algo.OneStepMinSumAlgo(Degree, Adj, node_means, eps=0.0)
    Base = graph_algo.NoGraphAlgo(Degree, Adj, node_means)

    Time_tracker_GB, _, _ = run_algo(GB, printer="GB", cluster_size=node_per_cluster)
    Time_tracker_GB_2, _, _ = run_algo(GB_2, printer="GB_2", cluster_size=node_per_cluster)
    Time_tracker_GB_det, _, _ = run_algo(GB_det, printer="GB_det", cluster_size=node_per_cluster)
    Time_tracker_GB_sum, _, _ = run_algo(GB_sum, printer="GB_sum", cluster_size=node_per_cluster)
    Time_tracker_Base, _, _ = run_algo(Base, printer="Base", cluster_size=node_per_cluster)

    """
    Phase 3 : Performance review
    ----------------------------
    
    Plot the different performance plots. Here we plot 'number of nodes still in consideration' vs 'time' for
    different competing algorithms. 
    """
    plt.plot(Time_tracker_GB_2, node_per_cluster*clusters*np.ones(GB_2.dim) - range(len(Time_tracker_GB_2)), color='blue', marker='o', markersize=4, label='Proposed algo', linewidth=2.0)
    plt.plot(Time_tracker_GB_det, node_per_cluster*clusters*np.ones(GB_det.dim) - range(len(Time_tracker_GB_det)), color='black', marker='.', markersize=4, label='Det algo', linewidth=2.0)
    plt.plot(Time_tracker_GB_sum, node_per_cluster*clusters*np.ones(GB_sum.dim) - range(len(Time_tracker_GB_sum)), color='magenta', marker='^', markersize=4, label='Sum algo', linewidth=2.0)
    plt.plot(Time_tracker_GB, node_per_cluster*clusters*np.ones(GB.dim) - range(len(Time_tracker_GB)), color='orange', marker='', markersize=4, label='Valko et.al', linewidth=3.0)
    plt.plot(Time_tracker_Base, node_per_cluster*clusters*np.ones(Base.dim)-range(len(Time_tracker_Base)), color='green', marker='', markersize=4, label='Cyclic algo', linewidth=2.0)
    plt.title("No. of remaining arms vs time steps")
    plt.xlabel("Time steps")
    plt.ylabel("No. of remaining arms")
    plt.grid()
    plt.legend()
    plt.show()
