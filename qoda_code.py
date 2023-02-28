import qoda
from qoda import spin

def final_circuit_ansatz(fixed_params, var_par, graph_node_list, graph_edges_list, graph_node_feat, repeatations = 1):
    v = len(graph_node_list)
    kernel = qoda.make_kernel()
    qubits = kernel.qalloc(v)

    for k in range(0, repeatations):
        for node in range(v):
            for i in range(var_par.shape[0]):
                if i % 2 == 0:
                    kernel.rx(parameter = fixed_params[i]*graph_node_feat[node][i], target = qubits[node])
                if i % 2 == 1:
                    kernel.rx(parameter = fixed_params[i]*graph_node_feat[node][i], target = qubits[node])
        
        for i in range(v):
            kernel.rz(parameter = var_par[3*v], target = qubits[i])
            kernel.ry(parameter = var_par[3*v+1], target = qubits[i])
            kernel.rz(parameter = var_par[3*v+2], target = qubits[i])
        
        for edge in graph_edges_list:
            kernel.z(target = qubits[edge[0]])
            kernel.z(target = qubits[edge[1]])

        for q in range(v + 1):
            kernel.cx(control = qubits[q], target = qubits[(q+1)%(v+1)])
    kernel.mz(qubits)
    return kernel

# fixed_params = 
# var_par =  
# graph_node_list =
# graph_edges_list =
# graph_node_feat =

# @ Steffano Need help here especially what is node and what does final_cost returns
def final_cost(subgraphs, params_init, var_param):
    subgraph=subgraphs[node]["subgraph"]
    sub_node, sub_edge= subgraphs[node]["subgraph"]
    sub_feat=subgraphs[node]["sub_features"]
    kernel = final_circuit_ansatz(params_init, var_param, sub_node, sub_edge, sub_feat, repeatations = 2)

    return #add something here