AiQC Q-Day MiniHackathon — Variational Quantum Optimization Challenge

Abstraction Bucket: WEIGHTED MAX-CUT

Use case        : Data-Center Server Partitioning

PROBLEM

We have 5 servers (S1..S5). Each pair of servers that talks to each
other has a "data traffic weight" (GB/s) on the edge between them.
We want to split the servers into TWO groups (e.g. two racks / two
availability zones) such that the total cross-group traffic is
MAXIMIZED. Why maximize instead of minimize? In this toy scenario,
high cross-link traffic represents load we want spread across two
independent network spines for redundancy/throughput -- a "weighted
max-cut" framing of the load-balancing problem (this is the standard
benchmark form of weighted Max-Cut, chosen for clarity).

Why Quantum ?

Finding the exact global maximum for a weighted Max-Cut problem is classically NP-hard. The brute-force solution scales exponentially as $\mathcal{O}(2^n)$, where $n$ is the number of servers. While a 5-server grid has only 32 possibilities, scaling this to a real-world data center with 100+ servers requires exploring $2^{100}$ combinations—an astronomical scale that clogs classical supercomputers.

Variational Quantum Algorithms (VQAs) map the discrete problem onto a continuous landscape of quantum parameters. By leveraging Quantum Superposition and Entanglement, algorithms like QAOA and VQE can evaluate entire solution spaces simultaneously, providing a robust near-term (NISQ-era) framework to tackle combinatorial optimization without hitting classical exponential scaling bottlenecks

Project Report : [click here] (./report_AQC_Hackathon.docx)

