**Paper Title**: Optimal control of traffic signals using quantum annealing

**Paper Link**: https://link.springer.com/article/10.1007/s11128-020-02815-1

**Abstract**: Quadratic unconstrained binary optimization (QUBO) is the mathematical formalism for phrasing and solving a class of optimization problems that are combinatorial in nature. Due to their natural equivalence with the two-dimensional Ising model for ferromagnetism in statistical mechanics, problems from the QUBO class can be solved on quantum annealing hardware. In this paper, we report a QUBO formatting of the problem of optimal control of time-dependent traffic signals on an artificial grid-structured road network so as to ease the flow of traffic, and the use of D-Wave Systems’ quantum annealer to solve it. Since current-generation D-Wave annealers have a limited number of qubits and limited inter-qubit connectivity, we adopt a hybrid (classical/quantum) approach to this problem. As traffic flow is a continuous and evolving phenomenon, we address this time-dependent problem by adopting a workflow to generate and solve multiple problem instances periodically.

Run this program on D-Wave by inserting your D-Wave LEAP API token in line 319 of main.py

If no token is entered, optimization calls will be handled locally.

