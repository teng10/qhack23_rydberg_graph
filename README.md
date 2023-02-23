# qhack23_project
==================
 - Team name: QuantuMother
 - In this Readme.md for QHack 2023 `project name`, we are including notebooks or pdfs in this repository. This repository will contain the entire project.

## Completion Criteria:

1. We want to implement the quantum feature map (QEK) in reference [1], where they demonstrate a competitive result in classifying graph-structured datasets compared to graph neural networks (GNN). 
2. We can implement quantum GNN for the same classification tasks.
3. We aim to improve the accuracy of QGNN by use of CAFQA [4] to get best initialization points for QGNN. Since, CAFQA works on classical simulator we wish to demonstrate the symbiosis of classical computing and Quantum computing leading to improvement of QC performance.

## Getting started

**Here are some steps::**
1. Set up AWS Bracket, because it will be convenient to use this. 
 - see [tutorial notebook](tutorialAWS.ipynb)
2. Finding dataset in reference [1]. #teng10: we will also need a very small toy dataset (~12 nodes?), rather than the full dataset there. We can also generate our own to begin with, probably the easiest way. 
    - add description of the dataset 
    - describe what are the impacts of successful classifications
3. One thing still worth looking into is how does the training work? 

    - In the paper, there are a few steps:
        - choose hamiltonian parameter evolution time $t$ (so that all the parameters in the hamiltonian $\Omega(t)$ and $\delta(t)$ will be fixed)
        - determine best initialization points on classical computer using CAFQA
        - use neural atom QPU to simulate the evolution and hence the kernel
        - postprocess with SVM and proceed with classification
    - can we use a variant of the training procedure?
    - see Figure 3: ![diagram](training_diagram.png)

Useful demo:
 - Bracket [tutorial notebook](https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html#braket-get-started-analyzing-simulator-results) and [other notebook examples](https://github.com/aws/amazon-braket-examples/tree/main/examples/analog_hamiltonian_simulation) and [blog post on optimization](https://aws.amazon.com/blogs/quantum-computing/optimization-with-rydberg-atom-based-quantum-processor/)
 - Pennylane https://pennylane.ai/qml/demos/tutorial_qgrnn.html
 - Pennylane with Bracket https://docs.aws.amazon.com/braket/latest/developerguide/hybrid.html
 - NVIDIA Bracket acceleration [link1](https://aws.amazon.com/blogs/quantum-computing/accelerate-your-simulations-of-hybrid-quantum-algorithms-on-amazon-braket-with-nvidia-cuquantum-and-pennylane/) and [link2](https://github.com/aws/amazon-braket-examples/blob/main/examples/hybrid_jobs/5_Parallelize_training_for_QML/Parallelize_training_for_QML.ipynb)

 
# References

[1] [main reference](https://arxiv.org/pdf/2211.16337.pdf)

[2] [Maria Schuld's kernel method for graph problem, could use some inspirations for dataset/benchmark](https://arxiv.org/pdf/1905.12646.pdf)

[3] [related to quantum enhanced GNN](https://arxiv.org/pdf/2210.10610.pdf)

[4] [CAFQA] https://arxiv.org/abs/2202.12924

## Some related references

[Optimization for combinatorial problem](https://arxiv.org/abs/2202.09372)

[Creating toric code topological state](https://arxiv.org/pdf/2112.03923.pdf)



