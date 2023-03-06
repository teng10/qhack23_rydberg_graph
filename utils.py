import numpy as np
import matplotlib.pyplot as plt  # required for plotting

from functools import partial

from collections import Counter

from scipy.spatial.distance import jensenshannon

import networkx as nx
import pennylane as qml

def get_counts_summed_probabilities(state_counts, shots, averaged_nodes=None):
    """Aggregate probabilities of the summed states from AHS shot results
    
    Args:
       state_counts: dictionary of state counts from simulations

    Returns:
        np.ndarray: probability of each summed state
    """
    # state_counts = get_counts(result) # get state counts from simulator result # old version
    n_nodes = len(list(state_counts.keys())[0]) # number of nodes in graph
    n_prob = np.zeros(n_nodes+1) # initialize array of probabilities for possible number of spins up
    total_counts = {}
    total_shots = sum(state_counts.values())
    for k, v in state_counts.items(): # sum counts of states with same number of spins up `1`
        new_k = k.count('1')
        total_counts[new_k] = total_counts.get(new_k, 0) + v
    for k, v in total_counts.items(): # convert counts to probabilities
        n_prob[k] = v / total_shots
    assert np.isclose(np.sum(n_prob), 1.0), f"total shots are {total_shots}, n_total is {np.sum(n_prob)}" # ensure probabilities sum to 1    
    if averaged_nodes is not None:
        n_expectationB = sum(n_prob[averaged_nodes]) # expectation value of up spins in B
        averaged_nodesC = list(set(range(n_nodes)) - set(averaged_nodes)) # nodes not in B
        n_expectationC = sum(n_prob[averaged_nodesC]) 
        return n_expectationB, n_expectationC
    return n_prob

def get_expectations_nodes(state_counts, shots, averaged_nodes=None):
    """Compute expectations of up spins in averaged_nodes and up spins in the complement of averaged_nodes
    
    Args:
       result (braket.tasks.analog_hamiltonian_simulation_quantum_task_result.AnalogHamiltonianSimulationQuantumTaskResult)

    Returns:
        np.ndarray: (2,) array for expectation value of up spins in averaged_nodes and 
        up spins in the complement of averaged_nodes
    """
    def _get_counts_node(state_counts, node):
        """Helper function: Aggregate state counts from AHS shot results for a single node"""
        counts_node = [0, 0]
        for k, v in state_counts.items(): # sum spins up `1` and down `0` for node
            new_k = int(k[node])
            counts_node[new_k] = counts_node[new_k] + v
        return np.array(counts_node)

    # state_counts = get_counts(result) # get state counts from simulator result # old version
    n_nodes = len(list(state_counts.keys())[0]) # number of nodes in graph
    n_expectation_lists = []
    total_shots = sum(state_counts.values())
    for node in averaged_nodes:
        counts_node = _get_counts_node(state_counts, node)
        n_expectation_lists.append(counts_node[1] / total_shots)
    averaged_nodesC = list(set(range(n_nodes)) - set(averaged_nodes)) # nodes not in B      
    n_expectation_listsC = []   
    for node in averaged_nodesC:
        counts_node = _get_counts_node(state_counts, node)
        n_expectation_listsC.append(counts_node[1] / total_shots)
    return np.mean(n_expectation_lists), np.mean(n_expectation_listsC)

def compute_kernel(counts_graph, kernel_fn):
    """Compute the kernel matrix for the given results for all the graphs and kernel function.
    
    Args:
        counts_graph (list): The list of counts for each graph
        kernel_fn (function): The kernel function to use

    Returns:
        np.ndarray: The kernel matrix
    """
    n_graphs = len(counts_graph)
    kernel_matrix = np.zeros((n_graphs, n_graphs))
    for i in range(n_graphs):
        for j in range(i+1, n_graphs):
            kernel_matrix[i, j] = kernel_fn(counts_graph[i], counts_graph[j])
            kernel_matrix[j, i] = kernel_matrix[i, j]
    return kernel_matrix


def outcome_counts(counts_graph1, counts_graph2, shots, my_nodesB=None):
    """If no nodes function, specified, compute the kernel function for a probability defined by 
    the counts of the summed states.
    If nodes are specified, also return outcome counts for the specified nodes.
    
    Args:
        counts_graph1 (dict): The counts for graph 1
        counts_graph2 (dict): The counts for graph 2

    Returns:
        float: The kernel function value
    """
    n_prob1 = get_counts_summed_probabilities(counts_graph1, shots=shots)
    n_prob2 = get_counts_summed_probabilities(counts_graph2, shots=shots)
    js_divergence = jensenshannon(n_prob1, n_prob2) # Jensen-Shannon divergence
    if my_nodesB is not None:
        n_prob1, n_expectationB1, n_expectationC1 = get_counts_summed_probabilities(counts_graph, 
                                                                                  shots, my_nodesB)
        n_prob2, n_expectationB2, n_expectationC2 = get_counts_summed_probabilities(counts_graph,
                                                                                    shots, my_nodesB)
        # js_divergence = jensenshannon(n_prob1, n_prob2) # Jensen-Shannon divergence
        return np.array([n_expectationB1, n_expectationC1, n_expectationB2, n_expectationC2])
    # n_prob_list.append(n_prob)
    return np.exp(-js_divergence)



def compute_outcomes(counts_graph, outcome_fn):
    """
    Compute the outcomes for the given results for all the graphs and outcome function.
    """
    n_graphs = len(counts_graph)
    outcomes = []
    for i in range(n_graphs):
        outcomes.append(outcome_fn(counts_graph[i]))
    return np.stack(outcomes)



#@title Helper functions for plotting
def plot_graphs(axs, idices, graphs):
    n_plots = len(idices)
    for i, idx in enumerate(idices):
        ax = axs[i]
        G = graphs[idx]
        nx.draw(G, with_labels=True, ax=ax, pos=nx.spring_layout(G))
        ax.set_title("Graph {}".format(idx))

def plot_adj_mats(axs, adj_mats, indices):
    for i, idx in enumerate(indices):
        ax = axs[i]
        B = adj_mats[idx]
        ax.imshow(B)
        ax.set_title("Graph {}".format(idx))


#@title Helper functions for pennylane demo
def build_hamiltonian(G, omega, delta):
    """
    Builds the approximated Hamiltonian for the given graph.
    Args:
        G (nx.Graph): Graph to build the Hamiltonian for.
        omega (float): Frequency of the oscillator.
        delta (float): Detuning of the oscillator.
    Returns:
        qml.Hamiltonian: Hamiltonian for the given graph.
    """    
    obs = []
    coeffs = []

    V_r = np.sqrt(omega**2 + delta**2)  # rad / sec # Rydberg interaction strength converted from 

    for node in G.nodes(): # construct onsite terms
        coeffs.extend([omega / 2., - delta / 2.])
        obs.extend([qml.PauliX(node), qml.PauliZ(node)])
    for edge in G.edges():
        coeffs.extend([V_r / 4., V_r / 2., V_r / 2.]) 
        obs.extend([qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]),
                            qml.PauliZ(edge[0]),
                            qml.PauliZ(edge[1])])
    return qml.Hamiltonian(coeffs, obs)        

# Helper function for computing summed over counts from probabilities
def _convert_to_counts(probs, n_wires):
    """Converts a list of probabilities to a dictionary of normalized counts histogram."""
    counts = {}
    for i, prob in enumerate(probs):
        str_basis = np.binary_repr(i, width=n_wires)
        counts[str_basis] = prob
    dic_out = {x:y for x,y in counts.items() if y!=0.}
    return dic_out

def outcome_probs(probs1, probs2, n_wires, my_nodesB=None):
    """If no nodes function, specified, compute the kernel function for a probability defined by 
    the counts of the summed states.
    If nodes are specified, also return outcome counts for the specified nodes.
    
    Args:
        counts_graph1 (dict): The counts for graph 1
        counts_graph2 (dict): The counts for graph 2

    Returns:
        float: The kernel function value
    """
    def _convert_to_counts(probs, n_wires):
        """Converts a list of probabilities to a dictionary of normalized counts histogram."""
        counts = {}
        for i, prob in enumerate(probs):
            str_basis = np.binary_repr(i, width=n_wires)
            counts[str_basis] = prob
        dic_out = {x:y for x,y in counts.items() if y!=0.}
        return dic_out  
    n_prob1 = _convert_to_counts(probs1)
    n_prob2 = _convert_to_counts(probs2)
    js_divergence = jensenshannon(n_prob1, n_prob2) # Jensen-Shannon divergence
    # if my_nodesB is not None:
    #     n_prob1, n_expectationB1, n_expectationC1 = _convert_to_counts(probs1, 
    #                                                                               shots, my_nodesB)
    #     n_prob2, n_expectationB2, n_expectationC2 = _convert_to_counts(probs2,
    #                                                                                 shots, my_nodesB)
    #     # js_divergence = jensenshannon(n_prob1, n_prob2) # Jensen-Shannon divergence
    #     return np.array([n_expectationB1, n_expectationC1, n_expectationB2, n_expectationC2])
    # n_prob_list.append(n_prob)
    return np.exp(-js_divergence)