"""
Graph Generating Library

All the functions for generating different type of graph based models for Best-Arm Identification
Current supported models:
    1. Erdos Renyi graph
    2. Stochastic block model
    3. Clustered graph with structure in each cluster as follows:
        - Tree
        - Star
        - Erdos-Renyi
        - Complete
        - Barabasi-Albert
        - Line
        - Wheel

"""

import numpy as np
import csv
import random
import networkx
import matplotlib.pyplot as plt
import networkx as nx


def add_neighbours(cluster_index, Adj, check, factor=0.5):
    """
    Creates a list of nodes which are densely connected between each other.
    
    Parameters
    ----------
    cluster_index : Node index of nodes belonging to a densely connected cluster
    Adj : Adjacency matrix of the graph
    check : Temp array

    """
    if len(check)==0:
        return cluster_index

    dim = len(Adj)
    for i in range(dim):
        switch = 0
        for j in check:
            if Adj[i, j]==1:
                switch +=1
        # print(Adj[i, j], switch)
        if switch >= len(check)*factor:
            # print("why is it going here?")
            # print(switch, len(check), check)
            cluster_index.append(i)
    return cluster_index


def clean_num_list(array):
    """

    Patch function to make array consistent with usage

    Parameters
    ----------
    array : 1-D array

    """
    new_array = []
    for i in array:
        try:
            new_array.append(i[0])
        except:
            new_array.append(i)

    return new_array


def better_subsample(Adj, num_nodes):
    """
    Subroutine to subsample open-source graphs to perform experiments on a smaller scale.
    This function prioritises finding connected clusters of nodes.

    Parameters
    ----------
    Adj : Adjacency matrix of the graph
    num_nodes : Number of nodes in the subsampled graph

    """
    dim = len(Adj)
    common_nodes = []

    new_node_index = np.random.randint(0, num_nodes)
    new_node_list = [new_node_index]
    new_node_list = add_neighbours(new_node_list, Adj, [new_node_index])
    print(new_node_list, len(new_node_list))

    for i in range(len(new_node_list)):
        common_nodes = add_neighbours(common_nodes, Adj, new_node_list[:i])

    print(len(common_nodes))

    for k in new_node_list:
        if k not in common_nodes:
            common_nodes.append(k)

    # print(len(common_nodes), "Before")
    common_nodes = clean_num_list(common_nodes)
    # print(len(common_nodes), "After")

    while len(common_nodes) < num_nodes:
        new_node_index = random.sample(common_nodes, 1)
        new_node_list = [new_node_index]
        new_node_list = add_neighbours(new_node_list, Adj, [new_node_index])
        # print(len(new_node_list))

        print(len(common_nodes), len(common_nodes)< num_nodes)

        for i in range(len(new_node_list)):
            common_nodes = add_neighbours(common_nodes, Adj, new_node_list[:i])
        # print(len(common_nodes), "After")

        for k in new_node_list:
            if k not in common_nodes:
                common_nodes.append(k)
        common_nodes = clean_num_list(common_nodes)

    print(len(common_nodes))

    new_node_indices = common_nodes
    new_Adj = np.zeros((len(new_node_indices), len(new_node_indices)))
    for index_i, i in enumerate(new_node_indices):
        for index_j, j in enumerate(new_node_indices):
            new_Adj[index_i, index_j] = Adj[i, j]
    new_Deg = np.zeros((len(new_node_indices), len(new_node_indices)))
    for i in range(len(new_node_indices)):
        new_Deg[i,i] = sum([new_Adj[i, j] for j in range(len(new_node_indices))])
    return new_Deg, new_Adj


def subsample(Adj, num_nodes):
    """
    Subroutine to subsample open-source graphs to perform experiments on a smaller scale

    Parameters
    ----------
    Adj : Adjacency matrix of the graph
    num_nodes : Number of nodes in the subsampled graph

    """
    total_nodes = range(len(Adj))
    new_node_indices = random.sample(total_nodes, num_nodes)
    new_Adj = np.zeros((len(new_node_indices), len(new_node_indices)))
    for index_i, i in enumerate(new_node_indices):
        for index_j, j in enumerate(new_node_indices):
            new_Adj[index_i, index_j] = Adj[i, j]
    new_Deg = np.zeros((len(new_node_indices), len(new_node_indices)))
    for i in range(len(new_node_indices)):
        new_Deg[i,i] = sum([new_Adj[i, j] for j in range(len(new_node_indices))])
    return new_Deg, new_Adj


def show_graph_with_labels(adjacency_matrix):
    """
    Plots the graph structure based on adjacency matrix data.

    Parameters
    ----------
    adjacency_matrix : np.array() object encoding the edges of the graph.

    """
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=1)
    plt.show()


def create_dict(num, k):
    """

    Parameters
    ----------
    num : number of clusters
    k : number of nodes

    Returns
    -------
    A : dict

    """
    A = {}
    for i in range(k):
        A[i] = i + k*num
    return A


def generate_means(k, num, cluster_means):
    """
    Create array with same mean value for nodes in the same cluster.

    Parameters
    ----------
    k : number of nodes per cluster
    num : number of clusters
    cluster_means : size(num) - array with value of means per cluster

    Returns
    -------
    means : size(k*num) - array with value of mean per node.
    """

    # TODO : Currently only allows for the setting with equal means for each cluster.
    #        Need to implement other modules (noisy, etc.)

    means = np.zeros(k*num)

    if len(cluster_means)!=num:
        print("No of cluster means is not equal to number of clusters. Returning 0 mean vector.")
        return means

    for i in range(num):
        for j in range(k):
            means[j+i*k] = cluster_means[i]

    return means


def modify_means(means, index, new_mean):
    """
    Insert a new value to the existing array

    Parameters
    ----------
    means : array-like object
    index : location to insert the value
    new_mean : new value to insert

    Returns
    -------
    array-like object with size len(means) + 1

    """
    return np.insert(means, index, new_mean)


def modify_matrix(mat, index, new_value):
    """
    Insert row-column of new value to the existing matrix

    Parameters
    ----------
    mat : matrix-like object
    index : location to insert a new row and column
    new_value : new value to insert

    Returns
    -------
    matrix-like object with shape (len(mat) + 1, len(mat)+ 1)

    """
    temp_mat = np.insert(mat, index, new_value, axis=0)
    return np.insert(temp_mat, index, new_value, axis=1)


def one_off_setup(adj, degree, means, one_off_mean, index=0):
    """
    Wrapper function for adding an isolated node to the current graph

    Parameters
    ----------
    adj : Adjacency matrix
    degree : Degree matrix
    means : Mean vector
    one_off_mean : Mean value of isolated node
    index : location of the isolated node

    Returns
    -------
    new_adj : New adjacency matrix
    new_degree : New degree matrix
    new_means : New mean vector

    """
    new_adj = modify_matrix(adj, index, 0)
    new_degree = modify_matrix(degree, index, 0)
    new_means = modify_means(means, index, one_off_mean)

    return new_adj, new_degree, new_means


def fill_dict(new_dict, mean, adj, degree):
    """
    Create dictionary entry to store in .toml

    Parameters
    ----------
    new_dict : Dictionary object
    mean : Mean vector
    adj : Adjacency matrix
    degree : Degree matrix

    Returns
    -------
    new_dict : Filled up dictionary entry

    """
    new_dict["nodes"] = len(mean)
    new_dict["node_means"] = mean
    new_dict["Adj"] = adj
    new_dict["Degree"] = degree
    return new_dict


def lastfm():
    # dim = 7624  #LastFM
    dim = 37700  #Github Social
    Adj = np.zeros((dim, dim))
    Deg = np.zeros((dim, dim))

    # with open('/home/pkthaker/PycharmProjects/GraphBandits/lasftm_asia/lastfm_asia_edges.csv') as edgefile:
    with open('/home/pkthaker/PycharmProjects/GraphBandits/github_social/musae_git_edges.csv') as edgefile:
        reader = csv.reader(edgefile)
        for row in reader:

            try:
                i = int(row[0])
                j = int(row[1])
                Adj[i, j] = 1
                Adj[j, i] = 1
                Deg[i, i] += 1
                Deg[j, j] += 1
            except:
                print(row)
                pass

    # exit()
    return Deg, Adj


def sbm(k, num, p, q):
    """
    Stochastic-Block-Model graph generator

    Parameters
    ----------
    k : number of nodes per cluster
    num : number of clusters
    p : probability of intra cluster connection
    q : probability of inter cluster connection

    Returns
    -------
    Degree : Degree matrix
    Adj : Adjacency matrix

    """

    # TODO : Need to write code for different-sized clusters.

    sizes = k*np.ones(num, dtype=np.int8)
    probs = (p-q)*np.diag(np.ones(num))+q*np.ones((num, num))

    g = nx.stochastic_block_model(sizes, probs, seed=42)

    nodes = k*num
    Adj = np.zeros((nodes, nodes))
    Degree = np.zeros((nodes, nodes))

    for i, j in g.edges():
        Adj[i, j] = 1.0
        Adj[j, i] = 1.0
        Degree[i, i] += 1.0
        Degree[j, j] += 1.0

    return Degree, Adj


def n_cluster_graph(k, num, g_type, p=0.1, m=2):
    """
    Clustered graph

    Parameters
    ----------
    k : number of nodes per cluster
    num : number of clusters
    g_type : Type of graph structure for individual clusters
    p(optional) : probability of intra cluster connection

    Returns
    -------
    Degree : Degree matrix
    Adj : Adjacency matrix

    """
    g = []
    n = k * num
    for i in range(num):
        mapping = create_dict(len(g), k)
        a = select_graph_generator(k, g_type, p, m)
        b = networkx.relabel_nodes(a, mapping)
        g.append(b)

    h = networkx.Graph()
    for i in range(num):
        h.add_nodes_from(g[i])
        h.add_edges_from(g[i].edges())

    Adj = np.zeros((n, n))
    Degree = np.zeros((n, n))

    for i, j in h.edges():
        Adj[i, j] = 1.0
        Adj[j, i] = 1.0
        Degree[i, i] += 1.0
        Degree[j, j] += 1.0

    for i in range(n):
        for j in range(n):
            Adj[i, j] = np.format_float_positional(Adj[i, j], trim='k')
            Degree[i, j] = np.format_float_positional(Degree[i, j], trim='k')

    return Degree, Adj


def select_graph_generator(k, g_type='complete', p=1.0, m=2):
    """
    Generate graph object based on choice of graph structure

    Parameters
    ----------
    k : number of nodes in the cluster
    g_type : Type of graph for the cluster
    p : probability of connection (required for Erdos-Renyi graph)

    Returns
    -------
    a : networkx object associated with the generated graph on k nodes

    """
    if g_type == 'complete':
        a = networkx.generators.random_graphs.erdos_renyi_graph(k, 1.0)
    elif g_type == 'ER':
        a = networkx.generators.random_graphs.erdos_renyi_graph(k, p)
    elif g_type == 'star':
        a = networkx.generators.star_graph(k - 1)
    elif g_type == 'wheel':
        a = networkx.generators.wheel_graph(k)
    elif g_type == 'line':
        a = networkx.generators.path_graph(k)
    elif g_type == 'tree':
        a = networkx.generators.random_powerlaw_tree(k, gamma=2.0, tries=10000)
    elif g_type == 'BA':
        a = networkx.generators.barabasi_albert_graph(k, m=m)
    else:
        print("Mentioned graph not in list")
        raise ValueError
    return a


def call_generator(node_per_cluster, clusters, p, cluster_means, graph_type, q=0.0, m=2, isolate=True):
    """
    Wrapper function to call for getting graph related matrices

    Parameters
    ----------
    node_per_cluster : number of nodes per cluster
    clusters : number of cluster
    p : probability of intra cluster connection
    cluster_means : size(clusters) - array with value of means per cluster

    Returns
    -------
    new_dict : dictionary with graph related data.

    """
    if graph_type == 'SBM':
        deg, adj = sbm(node_per_cluster, clusters, p, q)
        mean_vector = generate_means(node_per_cluster, clusters, cluster_means)

    elif graph_type == 'LastFM':
        deg, adj = lastfm()
        deg, adj = better_subsample(adj, 200)
        mean_vector = 100*np.ones(len(deg))
        show_graph_with_labels(adj)
        # print(sorted(np.linalg.eigvals(deg-adj)))
        # exit()
    else:
        deg, adj = n_cluster_graph(node_per_cluster, clusters, graph_type, p, m)
        mean_vector = generate_means(node_per_cluster, clusters, cluster_means)


    if isolate:
        adj, deg, mean_vector = one_off_setup(adj, deg, mean_vector, max(mean_vector)*1.10)
        adj[0,1] = 1
        adj[1, 0] = 1
        deg[0,0] = 1
        deg[1,1] += 1

    # show_graph_with_labels(adj)
    # exit()

    new_dict = {}
    new_dict = fill_dict(new_dict, mean_vector, adj, deg)
    return new_dict
