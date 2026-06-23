"""
=====================================================================
AiQC Q-Day MiniHackathon — Variational Quantum Optimization Challenge
Abstraction Bucket: WEIGHTED MAX-CUT
Use case        : Data-Center Server Partitioning
=====================================================================

PROBLEM
-------
We have 5 servers (S1..S5). Each pair of servers that talks to each
other has a "data traffic weight" (GB/s) on the edge between them.
We want to split the servers into TWO groups (e.g. two racks / two
availability zones) such that the total cross-group traffic is
MAXIMIZED. Why maximize instead of minimize? In this toy scenario,
high cross-link traffic represents load we want spread across two
independent network spines for redundancy/throughput -- a "weighted
max-cut" framing of the load-balancing problem (this is the standard
benchmark form of weighted Max-Cut, chosen for clarity).

-----------------------------------------------------------------
1) QUBO FORMULATION
-----------------------------------------------------------------
Let x_i in {0,1} indicate which group server i is placed in.
For an edge (i, j) with weight w_ij, the edge is CUT (servers in
different groups) exactly when x_i != x_j. We can write this with
the identity:

        cut(i,j) = x_i + x_j - 2*x_i*x_j        (1 if cut, 0 if not)

So the objective to MAXIMIZE is:

        E(x) = sum_{(i,j) in Edges} w_ij * (x_i + x_j - 2*x_i*x_j)

Expanding, this is already in QUBO form  E(x) = sum_i Q_ii x_i
+ sum_{i<j} Q_ij x_i x_j  with:

        Q_ii  = sum_{j: (i,j) in E} w_ij     (linear / diagonal terms)
        Q_ij  = -2 * w_ij                    (quadratic coupling terms)

Since the problem is UNCONSTRAINED (any split of servers into two
groups is valid -- there's no "budget" or cardinality constraint),
no penalty terms are required. This is what makes Max-Cut one of the
cleanest QUBO-native problems.

-----------------------------------------------------------------
2) ISING MAPPING
-----------------------------------------------------------------
Quantum hardware naturally evaluates Pauli-Z operators (eigenvalues
+1 / -1), not binary 0/1 variables. We substitute:

        x_i = (1 - Z_i) / 2        where Z_i in {+1, -1}

Substituting into cut(i,j) = x_i + x_j - 2*x_i*x_j and simplifying,
the +1/-1 cross terms collapse into a beautifully simple form:

        cut(i,j)  =  (1 - Z_i*Z_j) / 2

So the cost Hamiltonian we actually diagonalize on the quantum
computer is:

    H_C = sum_{(i,j) in E} w_ij * (1 - Z_i Z_j) / 2

The constant term (sum w_ij / 2) just shifts the energy and doesn't
affect *where* the optimum is, so for variational optimization we
only need to minimize the operative piece:

    H_C' = -sum_{(i,j) in E} w_ij * Z_i Z_j      (minimizing H_C'
                                                    maximizes the cut)

This is exactly the operator built in code below (see
`weighted_edges` -> `qp.Z(w1) @ qp.Z(w2)` for QAOA, and the
`SparsePauliOp` ZZ terms for VQE/Qiskit).

-----------------------------------------------------------------
3) QAOA MECHANICS (PennyLane)
-----------------------------------------------------------------
QAOA alternates two problem-aware unitaries, p times ("p layers"):
  - Cost unitary   U_C(gamma) = exp(-i*gamma*H_C')   encodes the
    graph structure directly: a CNOT-RZ-CNOT motif per edge applies
    a phase proportional to w_ij*gamma whenever two connected qubits
    disagree.
  - Mixer unitary  U_B(beta)  = exp(-i*beta*sum_i X_i)  is a simple
    bank of RX rotations that lets amplitude flow between bitstrings,
    so the optimizer can explore the solution space.
Both layers are interleaved starting from an equal superposition
(Hadamard on every wire), and (gamma, beta) are tuned classically to
minimize <H_C'>.

-----------------------------------------------------------------
4) VQE MECHANICS (Qiskit) -- THE COMPARATIVE PIECE
-----------------------------------------------------------------
VQE solves the *exact same* Hamiltonian H_C' but swaps QAOA's
graph-aware circuit for a generic, "hardware-efficient" Ansatz
(here: RealAmplitudes, all real-amplitude RY rotations + linear
CNOT entanglers) that has NO knowledge of which edges exist in the
graph. The classical optimizer (COBYLA) only ever sees expectation
values <H_C'> from the Estimator primitive and tunes the Ansatz's
rotation angles blindly.

KEY ARCHITECTURAL DIFFERENCE (this is what the slide deck should
emphasize):
  - QAOA's circuit is DERIVED from the problem (cost unitary is
    literally built from the graph's edges/weights) -> fewer, more
    targeted parameters, but a fixed/limited circuit family.
  - VQE's circuit is PROBLEM-AGNOSTIC (same Ansatz would be used for
    any 5-qubit Hamiltonian) -> more general / more expressive in
    principle, but has to "discover" the problem structure purely
    through optimization, with no built-in physical intuition about
    the graph.
=====================================================================
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution

import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw

# Fix the random number generator so we get the same results every run
np.random.seed(42)

# ---------------------------------------------------------------
# DATASET (toy, as required: 5 nodes / 6 weighted edges)
# ---------------------------------------------------------------
n_wires = 5

graph = rx.PyGraph()
graph.add_nodes_from(range(1, n_wires + 1))  # node PAYLOADS are 1..5 (S1..S5)
# Define the connections between servers and the data traffic weight between them
weighted_edges = [
    (0, 1, 3),
    (1, 2, 2),
    (0, 3, 1),
    (2, 4, 1),
    (3, 4, 2),
    (1, 4, 3),
]

graph.add_edges_from(weighted_edges)
pos = rx.spring_layout(graph, seed=2)

# NOTE: rustworkx node *indices* are 0..4, but each node's *payload* is
# already 1..5 (set above), so labels should use the payload directly
# (f"S{x}") rather than the index (f"S{x+1}", which mislabels S1 as "S2").
def server_label(payload):
    return f"S{payload}"


# =================================================================
# QUANTUM CIRCUITS — QAOA (PennyLane)
# =================================================================

# Cost Hamiltonian H_C' = -sum w_ij Z_i Z_j  (see derivation above)
def U_B(beta):
    """Mixer layer: rotates every qubit to spread amplitude across bitstrings."""
    for wire in range(n_wires):
        qp.RX(2 * beta, wires=wire)


def U_C(gamma):
    """Cost layer: encodes the weighted graph directly into a phase per edge."""
    for w1, w2, weight in weighted_edges:
        qp.CNOT(wires=[w1, w2])
        qp.RZ(weight * gamma, wires=w2)
        qp.CNOT(wires=[w1, w2])


def bitstring_to_int(bit_string_sample):
    """Convert a binary outcome (e.g. [0,1,0,1,0]) into an integer."""
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])


dev = qp.device("lightning.qubit", wires=n_wires)


@qp.set_shots(20)  # 20 samples per optimization step
@qp.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    # Step 1: equal superposition over all 2^5 server-partitions
    for wire in range(n_wires):
        qp.Hadamard(wires=wire)

    # Step 2: alternate cost & mixer layers, p times
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)

    if return_samples:
        return qp.sample()

    # H_C (un-shifted form, includes the "+w_ij" so larger = better cut)
    C = qp.sum(*(weight * (qp.Z(w1) @ qp.Z(w2)) for w1, w2, weight in weighted_edges))
    return qp.expval(C)


def objective(params):
    """Negative cut weight (PennyLane optimizers minimize, so we negate to maximize the cut)."""
    total_weight = sum(w for _, _, w in weighted_edges)
    return -0.5 * (total_weight - circuit(*params))


qaoa_history = []


def qaoa_maxcut(n_layers=2):
    print(f"\nRunning QAOA with p={n_layers:d} layers")

    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)
    opt = qp.AdagradOptimizer(stepsize=0.1)

    params = init_params.copy()
    steps = 500

    for i in range(steps):
        params = opt.step(objective, params)
        current_energy = circuit(*params)
        qaoa_history.append(current_energy)

        if (i + 1) % 50 == 0:
            print(f"Objective after step {i + 1:3d}: {-objective(params): .7f}")

    # Final measurement: sample the optimized circuit to find the most likely partition
    bitstrings = qp.set_shots(circuit, shots=100)(*params, return_samples=True)
    sampled_ints = [bitstring_to_int(string) for string in bitstrings]

    counts = np.bincount(np.array(sampled_ints), minlength=2 ** n_wires)
    most_freq_bit_string = np.argmax(counts)

    print(f"Optimized Angles:\ngamma: {params[0]}\nbeta:  {params[1]}")
    print(f"The best server group partition is: {most_freq_bit_string:05b}")

    return -objective(params), sampled_ints, most_freq_bit_string


qaoa_best_cut, int_samples2, qaoa_best_int = qaoa_maxcut(n_layers=2)


# =================================================================
# QUANTUM CIRCUITS — VQE (Qiskit) — comparative algorithm
# =================================================================
print("\n" + "=" * 40)
print("Starting VQE Part using Qiskit 2.0+")
print("=" * 40)

from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize

# Build the same Ising cost Hamiltonian H_C' = -sum w_ij Z_i Z_j
# (sign convention: SparsePauliOp coefficients carry +w_ij, and VQE
#  minimizes <H>, which is equivalent to maximizing the cut -- same
#  physics as the QAOA `objective` above, just expressed via Qiskit's
#  operator algebra instead of PennyLane's.)
pauli_list = []
for w1, w2, weight in weighted_edges:
    pauli_str = ["I"] * n_wires
    pauli_str[w1] = "Z"
    pauli_str[w2] = "Z"
    pauli_list.append(("".join(reversed(pauli_str)), weight))

qiskit_hamiltonian = SparsePauliOp.from_list(pauli_list)

# Problem-agnostic hardware-efficient Ansatz (no graph knowledge baked in)
vqe_ansatz = real_amplitudes(num_qubits=n_wires, entanglement="linear", reps=1)
vqe_ansatz = vqe_ansatz.decompose()

estimator = AerEstimator()
vqe_history = []


def vqe_objective_qiskit(params):
    job = estimator.run([(vqe_ansatz, qiskit_hamiltonian, params)])
    result = job.result()[0]
    return float(result.data.evs)


num_params = vqe_ansatz.num_parameters
init_vqe_params = 0.01 * np.random.rand(num_params)

print(f"Optimizing VQE with {num_params} parameters...")


def callback_fn(xk):
    current_energy = vqe_objective_qiskit(xk)
    vqe_history.append(current_energy)


res = minimize(
    vqe_objective_qiskit,
    init_vqe_params,
    method="COBYLA",
    callback=callback_fn,
    options={"maxiter": 380},
)

print("\nOptimization Finished!")

optimized_vqe_circuit = vqe_ansatz.assign_parameters(res.x)
optimized_vqe_circuit.measure_all()

sampler = AerSampler()
sampler_job = sampler.run([optimized_vqe_circuit], shots=100)
vqe_counts = sampler_job.result()[0].data.meas.get_counts()

best_vqe_bitstring = max(vqe_counts, key=vqe_counts.get)
print(f"VQE Best server group partition is: {best_vqe_bitstring}")

# Cut weight achieved by the VQE solution (computed classically, for the title)
def cut_weight(bitstring_msb_first):
    # bitstring is Qiskit-ordered (qubit 4 ... qubit 0); convert to our x_i convention
    bits = [int(b) for b in bitstring_msb_first[::-1]]  # bits[i] = value of server i
    return sum(w for (i, j, w) in weighted_edges if bits[i] != bits[j])


qaoa_bitstring = f"{qaoa_best_int:05b}"
qaoa_cut = sum(
    w for (i, j, w) in weighted_edges if qaoa_bitstring[::-1][i] != qaoa_bitstring[::-1][j]
)
vqe_cut = cut_weight(best_vqe_bitstring)
optimal_cut = sum(w for _, _, w in weighted_edges) / 2 + max(
    0, sum(w for _, _, w in weighted_edges) / 2
)
# (Brute-force the true optimum for an honest reference line on the convergence plots)
best_brute_cut = 0
optimal_int = 0
for k in range(2 ** n_wires):
    bits = [(k >> b) & 1 for b in range(n_wires)]
    c = sum(w for (i, j, w) in weighted_edges if bits[i] != bits[j])
    if c > best_brute_cut:
        best_brute_cut = c
        optimal_int = k
optimal_energy = -best_brute_cut  # H_C' minimum = -(max cut weight)

print(f"\nBrute-force optimal cut weight: {best_brute_cut} GB/s")
print(f"QAOA achieved cut weight:        {qaoa_cut} GB/s")
print(f"VQE achieved cut weight:         {vqe_cut} GB/s")


# =====================================================================
# FIGURE 1 — "BEFORE vs AFTER" (required hackathon deliverable):
# unpartitioned network next to both quantum solutions, side by side.
# =====================================================================
plt.style.use("dark_background")


def draw_server_graph(ax, node_color, **extra_kwargs):
    """
    Thin wrapper around rustworkx's mpl_draw with two fixes for this
    dark-themed dashboard:
      1) mpl_draw() unconditionally calls fig.set_facecolor("w") every
         time it runs, which silently overwrites the dark_background
         style. We restore the dark facecolor on both the figure and
         this axes right after drawing.
      2) mpl_draw()'s default edge_color is black, which is invisible
         against a black background, so we explicitly pass a visible
         light-gray edge color (overridable via extra_kwargs).
    """
    edge_color = extra_kwargs.pop("edge_color", "#9aa0a6")
    font_color = extra_kwargs.pop("font_color", "white")
    mpl_draw(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        labels=server_label,
        node_color=node_color,
        edge_color=edge_color,
        font_color=font_color,
        **extra_kwargs,
    )
    ax.get_figure().set_facecolor("black")
    ax.set_facecolor("black")


fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(
    "Before vs After: Quantum-Optimized Server Partitioning\n"
    "Weighted Max-Cut • 5 Servers • 6 Data-Exchange Links",
    fontsize=15,
    fontweight="bold",
)

# --- BEFORE: unpartitioned network ---
axs1[0].set_title("BEFORE\nServer Network (unpartitioned)", fontsize=12, pad=12)
colors_init = ["#6c757d"] * n_wires
draw_server_graph(
    axs1[0],
    colors_init,
    node_size=900,
    width=[w for _, _, w in weighted_edges],
)

# --- AFTER: QAOA solution ---
axs1[1].set_title(
    f"AFTER (QAOA)\n|{qaoa_bitstring}⟩  cut = {qaoa_cut} GB/s", fontsize=12, pad=12
)
colors_qaoa = ["#00f5ff" if c == "1" else "#ff5a5a" for c in qaoa_bitstring]
draw_server_graph(
    axs1[1],
    colors_qaoa,
    node_size=900,
    width=[w for _, _, w in weighted_edges],
)

# --- AFTER: VQE solution ---
axs1[2].set_title(
    f"AFTER (VQE)\n|{best_vqe_bitstring}⟩  cut = {vqe_cut} GB/s", fontsize=12, pad=12
)
colors_vqe = ["#00f5ff" if c == "1" else "#ff5a5a" for c in best_vqe_bitstring]
draw_server_graph(
    axs1[2],
    colors_vqe,
    node_size=900,
    width=[w for _, _, w in weighted_edges],
)

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig("before_after.png", dpi=150, facecolor=fig1.get_facecolor())
print("\nSaved: before_after.png")


# =====================================================================
# FIGURE 2 — Full 2x4 comparative dashboard (QAOA vs VQE)
# =====================================================================
fig, axs = plt.subplots(2, 4, figsize=(26, 14))
fig.suptitle(
    "QAOA (PennyLane)  vs  VQE (Qiskit) — Data Center Server Partitioning\n"
    "Weighted Max-Cut • 5 Servers • 6 Data-Exchange Links",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# --- Row 1, Col 1: Server Network (unpartitioned) ---
axs[0, 0].set_title("Server Network\n(unpartitioned)", fontsize=12, pad=15)
draw_server_graph(axs[0, 0], colors_init, node_size=700)

# --- Row 1, Col 2: QAOA Partition Graph Solution ---
axs[0, 1].set_title(
    f"QAOA Solution  |{qaoa_bitstring}⟩\n(PennyLane)  cut={qaoa_cut} GB/s",
    fontsize=12,
    pad=15,
)
draw_server_graph(axs[0, 1], colors_qaoa, node_size=700)

# --- Row 1, Col 3: VQE Partition Graph Solution ---
axs[0, 2].set_title(
    f"VQE Solution  |{best_vqe_bitstring}⟩\n(Qiskit)  cut={vqe_cut} GB/s",
    fontsize=12,
    pad=15,
)
draw_server_graph(axs[0, 2], colors_vqe, node_size=700)

# --- Row 1, Col 4: unused (kept blank to balance the 2x4 grid) ---
axs[0, 3].axis("off")

# --- Row 2, Col 1: QAOA Optimization Curve ---
axs[1, 0].set_title("QAOA Convergence", fontsize=12)
axs[1, 0].plot(qaoa_history, color="#00f5ff", linewidth=2)
axs[1, 0].axhline(
    y=optimal_energy, color="#39ff14", linestyle="--", label=f"Optimal {optimal_energy:.3f}"
)
axs[1, 0].set_xlabel("Iteration")
axs[1, 0].set_ylabel("⟨H⟩")
axs[1, 0].legend()

# --- Row 2, Col 2: VQE Optimization Curve ---
axs[1, 1].set_title("VQE Convergence", fontsize=12)
axs[1, 1].step(range(len(vqe_history)), vqe_history, color="#ff9f43", linewidth=2)
axs[1, 1].axhline(
    y=optimal_energy, color="#39ff14", linestyle="--", label=f"Optimal {optimal_energy:.3f}"
)
axs[1, 1].set_xlabel("Iteration")
axs[1, 1].set_ylabel("⟨H⟩")
axs[1, 1].legend()

# --- Row 2, Col 3 & 4: Frequency Histograms (PennyLane-tutorial style) ---
# Each panel shows ALL 32 possible bitstrings on the x-axis, with the raw
# sample frequency (count out of 100 shots) on the y-axis. The optimal
# bitstring is highlighted in green so the solution "pops" visually,
# exactly like the official PennyLane QAOA MaxCut demo dashboards.

qaoa_freqs = np.bincount(int_samples2, minlength=2 ** n_wires)
vqe_freqs = np.zeros(2 ** n_wires, dtype=int)
for bitstr, count in vqe_counts.items():
    rev_bitstr = bitstr[::-1]
    vqe_freqs[int(rev_bitstr, 2)] = count

# True optimal bitstring (brute-forced earlier as `optimal_int`)
all_bitstrings = [f"{k:0{n_wires}b}" for k in range(2 ** n_wires)]
x_all = np.arange(2 ** n_wires)


def plot_freq_histogram(ax, freqs, title, bar_color):
    bar_colors = [
        "#39ff14" if k == optimal_int else bar_color for k in range(2 ** n_wires)
    ]
    ax.bar(x_all, freqs, color=bar_colors)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("bitstrings")
    ax.set_ylabel("freq.")
    ax.set_xticks(x_all)
    ax.set_xticklabels(all_bitstrings, rotation=90, fontsize=7)


plot_freq_histogram(axs[1, 2], qaoa_freqs, "QAOA Samples\n(optimal in green)", "#00f5ff")
plot_freq_histogram(axs[1, 3], vqe_freqs, "VQE Samples\n(optimal in green)", "#ff9f43")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("dashboard.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved: dashboard.png")

plt.show()






    
