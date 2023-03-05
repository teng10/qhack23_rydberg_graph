import qoda
from qoda import spin
import qiskit
import time
import pennylane as qml



def final_circuit_ansatz(fixed_params, var_par, graph_node_list, graph_edges_list, graph_node_feat, repeatations = 1):
    v = len(graph_node_list)
    kernel = qoda.make_kernel()
    qubits = kernel.qalloc(v)

    for k in range(0, repeatations):
        for node in range(v):
            for i in range(params.shape[0]):
                if i % 2 == 0:
                    kernel.rx(parameter = param_fixed[i]*graph_node_feat[node][i], target = qubits[node])
                if i % 2 == 1:
                    kernel.rx(parameter = param_fixed[i]*graph_node_feat[node][i], target = qubits[node])
        
        for i in range(v):
            kernel.rz(parameter = var_par[3*qubit], target = qubits[i])
            kernel.ry(parameter = var_par[3*qubit+1], target = qubits[i])
            kernel.rz(parameter = var_par[3*qubit+2], target = qubits[i])
        
        for edge in graph_edges_list:
            kernel.z(target = qubits[edge[0]])
            kernel.z(target = qubits[edge[1]])

        for q in range(v + 1):
            kernel.cx(control = qubits[q], target = qubits[(qubit+1)%(V+1)])
    kernel.mz(qubits)
    return kernel

# fixed_params = 
# var_par =  
# graph_node_list =
# graph_edges_list =
# graph_node_feat =
qbs = 29
def final_circuit_ansatz_2():
    v = qbs
    repeatations = 1
    kernel = qoda.make_kernel()
    qubits = kernel.qalloc(v)

    for k in range(0, repeatations):
        for node in range(v):
            for i in range(3):
                if i % 2 == 0:
                    kernel.x(target = qubits[node])
                if i % 2 == 1:
                    kernel.x(target = qubits[node])
        
        for i in range(v):
            kernel.z(target = qubits[i])
            kernel.y(target = qubits[i])
            kernel.z(target = qubits[i])
        
        for edge in range(v-1):
            kernel.z(target = qubits[edge])
            kernel.z(target = qubits[edge + 1])

        for q in range(v + 1):
            kernel.cx(control = qubits[q], target = qubits[(q+1)%(v+1)])
    kernel.mz(qubits)
    return kernel

def final_cost(subgraphs, params_init, var_param):
    subgraph=subgraphs[node]["subgraph"]
    sub_node, sub_edge= subgraphs[node]["subgraph"]
    sub_feat=subgraphs[node]["sub_features"]
    kernel = final_circuit_ansatz(params_init, var_param, sub_node, sub_edge, sub_feat, repeatations = 2)
    kernel.sample()

    return 1

edge_list = [[1, 1]]
start = time.time()
for i in range(0, 1000):
    kernel = final_circuit_ansatz_2()
    #counts = qoda.sample(final_circuit_ansatz_2())
end = time.time()

#print(end - start)

def circuitfinalansatz():
    ####Preparation of the circuit according to the encoding of the node features
    ###Inputs: param_fixed are the initialization parameters to encode the information in the nodes
    ###""    : var_par are the variational parameters for the classification problem
    V=qbs
    for node in range(V): 
        for i in range(3):
            if i % 2 == 0:
                qml.RX(0.0, wires=[node])
            if i % 2 == 1:
                qml.RX(0.0, wires=[node])
    
    for qubit in range(V):
        #print(qubit)
        qml.RZ(0.0, wires=qubit)
        qml.RY(0.0, wires=qubit)
        qml.RZ(0.0, wires=qubit)
    
    for edge in range(V-1):
        #print(edge,type(edge))
        qml.IsingZZ(phi=0.0,wires=[edge,edge+1])
        #print("edge couple",edge[0],edge[1])
    
    for qubit in range(V+1): 
        qml.CNOT(wires=[qubit, (qubit+1)%(V+1)])
    return qml.probs(wires=range(V))
dev=qml.device('default.qubit', wires=29)

#@qml.qnode(dev, interface="tf")

for k in range(8, 30):
    qbs = k
    #print(k)
    start = time.time()
    for i in range(0, 1000):
        kernel = circuitfinalansatz()
        #counts = qoda.sample(final_circuit_ansatz_2())
    end = time.time()
    print(end - start)
    '''start = time.time()
    for i in range(0, 1000):
        kernel = final_circuit_ansatz_2()
        #counts = qoda.sample(final_circuit_ansatz_2())
    end = time.time()

    print(end - start)'''
    

    