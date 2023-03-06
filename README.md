# qhack23_project
==================
 - Team name: QuantuMother
 - In this Readme.md for QHack 2023 `Quantum kernel methods for graph-structured data
`, we are including notebooks or pdfs in this repository. This repository will contain the entire project.

[Presentation Google slides](https://docs.google.com/presentation/d/1Wuv7EtU-KJm0gStl8GUhptq0lXEqrKhG2jeLSjxC7D4/edit#slide=id.p)

## Summary:

1. We implemented the quantum feature map (QEK) in reference [1], in three different approaches:
    1. AWS Braket local simulater. We use Braket's module `analog_hamiltonian_simulation` for simulating the Rydberg hamiltonian.  See [demo notebook Braket](demos/demo_two_graphs_braket.ipynb) in `demos/` folder.
    2. QuEra Aquila QPU. See [train notebook Aquila](TrainingNotebooks/train_aquila.ipynb) in `TrainingNotebooks/` folder.
    3. PennyLane QML simulator. We also trotterize the analog hamiltonian to digital quantum circuits. The difference here is that we are only preserving geometrically local interaction terms on a graph (see [slide 9](https://docs.google.com/presentation/d/1Wuv7EtU-KJm0gStl8GUhptq0lXEqrKhG2jeLSjxC7D4/edit#slide=id.g21368949250_0_144) in presentation). The resulting kernels are qualitatively different because of missing long-range interactions. See [train notebook pennylane](TrainingNotebooks/train_pennylane.ipynb) in `TrainingNotebooks/` folder.
    - We also tried to parrallelize the simulations with `SV1` device from AWS. However, we did not find a competitive speed-up compared to the bare PennyLane simulation. See [notebook pennylane sv1](TrainingNotebooks/train_pennylane_sv1.ipynb). 
2. We implemented decompositional quantum graph neural network (D-QGNN) for the same classification tasks. This algorithm decomposes a graph into smaller sub-graphs to reduce the circuit size. See [D-QGNN notebook]((D)QGNN.ipynb).
3. We aim to improve the accuracy of QGNN classification by use of CAFQA [4] to get best initialization points for the QGNN ansatz. Since, CAFQA works on classical simulator we wish to demonstrate the symbiosis of classical computing and Quantum computing leading to improvement of QC performance. However, we find that our classical-quantum simulations using QODA has worse performance than PennyLane. 
See [qoda code](qoda_code.py) and [slide 16](https://docs.google.com/presentation/d/1Wuv7EtU-KJm0gStl8GUhptq0lXEqrKhG2jeLSjxC7D4/edit#slide=id.g2135e9cb0b3_3_11). 

## Results

1. We (qualitatively) reproduced results in Ref [1], using a toy model of two simple graphs using Braket simulator. We see QEK's capability of learning an optimal evolution time to achieve maximum distinguishability between two graphs. ([Slide 6](https://docs.google.com/presentation/d/1Wuv7EtU-KJm0gStl8GUhptq0lXEqrKhG2jeLSjxC7D4/edit#slide=id.g2135e9cb0b3_0_5) )
2. We implemented a truncated and trotterized version of QEK using PennyLane. We find that with only such short range interactions, no optimal time is present, and therefore we suggest the long-range interactions on Aquila neural atom QPU are important. ([slide 9](https://docs.google.com/presentation/d/1Wuv7EtU-KJm0gStl8GUhptq0lXEqrKhG2jeLSjxC7D4/edit#slide=id.g21368949250_0_144))
3. To demonstrate QEK's capability beyond the toy model, we ran the same algorithm on two graphs from PTC-FM dataset, and we found optimal evolution time. **TODO**: We want to finish training and analyzing the full PTC-FM dataset, which is in progress. 

Note: the results pickle files are in [results folder](results_QEK). 

## Workflow

1. Candidate dataset: [PTC-FM](https://paperswithcode.com/dataset/ptc)
- We picked the PTC-FM dataset, containing graph labels based on carcinogenicity on rats. Being able to compute a binary classification in such compounds is a relevant task either for drug discovery purposes and for health safety concerns. A procedure that works on this dataset should be easily extendible to similar dataset such as MUTAG. 
- See [Notebook for loading dataset](Datasets/Datasets.ipynb) and [Datasets folder](Datasets)

<!-- -The ogbl-ddi dataset is a homogeneous, unweighted, undirected graph, representing the drug-drug interaction network. Each node represents an FDA-approved or experimental drug. Edges represent interactions between drugs and can be interpreted as a phenomenon where the joint effect of taking the two drugs together is considerably different from the expected effect in which drugs act independently of each other.
Plus: Finding dataset in reference [1], and benchmark our results with the reference paper. We will also generate our own dataset to begin with, probably the most straigthforward way to set up the initial parameters and get an idea of where there is room for optimization.  -->
    
2. Training algorithm for QEK: 
    - We follow ref [1] (which is for running on QPUs), and modify accordingly for approaches 1.1 and 1.3. 
        - choose hamiltonian parameter evolution time $t$ (so that all the parameters in the hamiltonian $\Omega(t)$ and $\delta(t)$ will be fixed)
        - determine best initialization points on classical computer using CAFQA
        - use neural atom QPU to simulate the evolution and hence the kernel
        - [todo] postprocess with SVM and proceed with classification
    - see Figure 3: ![diagram](training_diagram.png)

3. Training algorithm for QGNN:
    - [todo] to be added.
    - Reason to use CAFQA - VQA's and many QML methods are sensitive to points where they are intialized. If they are initialized at proper points they may reach converegence faster and have a lower risk of getting stuck at local minimas. CAFQA converts the ansatz of VQA and QML methods into clifford circuits which are efficiently simulable on classical computers and obtains the intialization points. Once, we find the best intialization points using CAFQA, VQA's have been shown to reach better accuracies with faster converegcne rates. We envision similar acceleration for our QGNN approach.
Useful demo:
 - Bracket [tutorial notebook](https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html#braket-get-started-analyzing-simulator-results) and [other notebook examples](https://github.com/aws/amazon-braket-examples/tree/main/examples/analog_hamiltonian_simulation) and [blog post on optimization](https://aws.amazon.com/blogs/quantum-computing/optimization-with-rydberg-atom-based-quantum-processor/), [braket doc](https://amazon-braket-sdk-python.readthedocs.io/en/latest/), [pennylane braket plugin](https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/)
 - Pennylane https://pennylane.ai/qml/demos/tutorial_qgrnn.html
 - Pennylane with Bracket https://docs.aws.amazon.com/braket/latest/developerguide/hybrid.html
 - NVIDIA Bracket acceleration [link1](https://aws.amazon.com/blogs/quantum-computing/accelerate-your-simulations-of-hybrid-quantum-algorithms-on-amazon-braket-with-nvidia-cuquantum-and-pennylane/) and [link2](https://github.com/aws/amazon-braket-examples/blob/main/examples/hybrid_jobs/5_Parallelize_training_for_QML/Parallelize_training_for_QML.ipynb)
 - [Blog post on hamiltonian construction](https://pennylane.ai/blog/2021/05/how-to-construct-and-load-hamiltonians-in-pennylane/)
 
# Acknowledgements
- We acknowledge power-ups from AWS, NVIDIA, through which most of the simulations shown were performed. We also thank power-up from IBM. 
- We thank Nihir Chadderwala and Eric Kessler from AWS for their help and debugging assistance.
- We thank the QHack team for a fun and exciting experience. 

# References

[1] [Main reference for QEK with neural atoms](https://arxiv.org/pdf/2211.16337.pdf)

[2] [Another kernel method for graph problem based on  Gaussian Boson Sampler---> benchmark](https://arxiv.org/pdf/1905.12646.pdf)

[3] [Quantum enhanced GNN](https://arxiv.org/pdf/2210.10610.pdf)

[4] [CAFQA](https://arxiv.org/abs/2202.12924)

## Some related references

[Optimization for combinatorial problem](https://arxiv.org/abs/2202.09372)

[Creating toric code topological state](https://arxiv.org/pdf/2112.03923.pdf)



