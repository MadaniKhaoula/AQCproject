"""

Install:
    !pip install pennylane qiskit qiskit-aer scipy matplotlib numpy networkx
"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.optimize import minimize

# ── PennyLane (QAOA) ──────────────────────────────────────────────────────────
import pennylane as qml
from pennylane import qaoa

# ── Qiskit (VQE) ──────────────────────────────────────────────────────────────
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import RealAmplitudes

print("=" * 70)
print("  QAOA (PennyLane) + VQE (Qiskit) — Server Cluster Partitioning")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────────────────────


n_qubits = 5

weighted_edges = [
    (0, 1, 3),    
    (1, 2, 2),   
    (0, 3, 1),   
    (2, 4, 1),   
    (3, 4, 2),   
    (1, 4, 3),    
]

edges   = [(i, j) for i, j, _ in weighted_edges]
weights = {(i, j): w for i, j, w in weighted_edges}

# positions for visualisation
positions = {0:(0,0), 1:(2,0), 2:(4,0), 3:(1,2), 4:(3,2)}
labels    = {i: f"S{i+1}" for i in range(n_qubits)}

total_weight = sum(w for _, _, w in weighted_edges)

print("\n[STEP 1] Problem Definition")
print(f"  Servers  : {n_qubits}  (S1 – S5)")
print(f"  Edges    : {len(weighted_edges)}  data-exchange pairs")
print(f"  Max possible cut weight : {total_weight} GB/s")
for i, j, w in weighted_edges:
    print(f"    S{i+1}—S{j+1}  {w} GB/s")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — QUBO FORMULATION
# ─────────────────────────────────────────────────────────────────────────────
"""
Weighted Max-Cut QUBO:

  Maximise:  C(x) = Σ w(i,j) · (xi + xj - 2·xi·xj)

  The term (xi + xj - 2·xi·xj) = 1 when xi ≠ xj  (edge is cut)
                                = 0 when xi = xj   (edge not cut)

  Convert to minimisation (standard QUBO form):
  E(x) = −Σ w(i,j) · (xi + xj - 2·xi·xj)

  Expand and rewrite using xi² = xi  (binary property):
  E(x) = −Σ w(i,j)·xi - Σ w(i,j)·xj + 2·Σ w(i,j)·xi·xj

  In matrix form  E(x) = xᵀQx :
    Qᵢᵢ = −Σⱼ w(i,j)     (diagonal: negative weighted degree)
    Qᵢⱼ =  2·w(i,j)       (off-diagonal: positive coupling)
"""
#the matrix Q
Q = np.zeros((n_qubits, n_qubits))
for i, j, w in weighted_edges:
    Q[i][i] -= w
    Q[j][j] -= w
    Q[i][j] += 2 * w   # upper triangle only (i < j guaranteed)

print("\n[STEP 2] QUBO Matrix  Q  (E(x) = xᵀQx)")
header = "       " + "".join(f"  S{k+1} " for k in range(n_qubits))
print(header)
for i in range(n_qubits):
    row = f"  S{i+1}  " + "".join(f"{Q[i][k]:+5.0f} " for k in range(n_qubits))
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ISING HAMILTONIAN  (shared by QAOA and VQE)
# ─────────────────────────────────────────────────────────────────────────────
"""
Substitute  xi = (1 − Zᵢ) / 2  into E(x), drop constants:

  Diagonal (linear Zᵢ terms):
    hᵢ = ½ · Qᵢᵢ + ¼ · Σⱼ≠ᵢ Qᵢⱼ

  Off-diagonal (ZᵢZⱼ coupling):
    Jᵢⱼ = Qᵢⱼ / 4  =  w(i,j) / 2

Quantum Hamiltonian:
  Ĥ = Σᵢ hᵢ Zᵢ  +  Σ_{(i,j)} Jᵢⱼ ZᵢZⱼ

Ground state of Ĥ  =  optimal partition of servers.
"""

J_coupling = {(i, j): w / 2.0 for i, j, w in weighted_edges}

h_field = np.zeros(n_qubits)
for i in range(n_qubits):
    h_field[i] = 0.5 * Q[i][i]
    for j in range(n_qubits):
        if j != i:
            h_field[i] += 0.25 * Q[i][j] if i < j else 0.25 * Q[j][i]

print("\n[STEP 3] Ising Hamiltonian Coefficients")
print("  Linear fields  hᵢ  (Zᵢ terms):")
for q in range(n_qubits):
    print(f"    h[S{q+1}] = {h_field[q]:+.4f}")
print("  Couplings  Jᵢⱼ  (ZᵢZⱼ terms):")
for (i, j), Jval in J_coupling.items():
    print(f"    J[S{i+1},S{j+1}] = {Jval:+.4f}   (from {weights[(i,j)]} GB/s edge)")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — brute-force classical solution (feasible for 5 qubits = 32 states)
# ─────────────────────────────────────────────────────────────────────────────

def cut_weight(bitstring: str) -> float:
    
    x = [int(b) for b in bitstring]
    return sum(w * abs(x[i] - x[j]) for i, j, w in weighted_edges)

def ising_energy(bitstring: str) -> float:
    """
   '1' → Cluster A → z = −1
   '0' → Cluster B → z = +1
    """
    z = [1 - 2 * int(b) for b in bitstring]
    e  = sum(h_field[q] * z[q] for q in range(n_qubits))
    e += sum(J_coupling[(i,j)] * z[i] * z[j] for i, j, _ in weighted_edges)
    return e

def brute_force():
    best_e, best_s = float("inf"), ""
    for bits in range(2 ** n_qubits):
        s = format(bits, f"0{n_qubits}b")
        e = ising_energy(s)
        if e < best_e:
            best_e, best_s = e, s
    return best_s, best_e

bf_string, bf_energy = brute_force()
bf_cut    = cut_weight(bf_string)
bf_clusterA = [f"S{i+1}" for i, b in enumerate(bf_string) if b == "1"]
bf_clusterB = [f"S{i+1}" for i, b in enumerate(bf_string) if b == "0"]

print("\n[BRUTE FORCE REFERENCE]")
print(f"  Best bitstring : |{bf_string}⟩")
print(f"  Cluster A (1)  : {bf_clusterA}")
print(f"  Cluster B (0)  : {bf_clusterB}")
print(f"  Cut weight     : {bf_cut:.0f} GB/s  out of {total_weight} GB/s total")
print(f"  Ising energy   : {bf_energy:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  PART A — QAOA  via PennyLane
# ═════════════════════════════════════════════════════════════════════════════


print("\n" + "=" * 70)
print("  PART A — QAOA  (PennyLane)")
print("=" * 70)

# ── A1. Build PennyLane Hamiltonians ─────────────────────────────────────────

cost_coeffs, cost_ops = [], []

for q in range(n_qubits):                        # linear Zᵢ terms
    cost_coeffs.append(float(h_field[q]))
    cost_ops.append(qml.PauliZ(q))

for (i, j), Jval in J_coupling.items():          # ZᵢZⱼ coupling terms
    cost_coeffs.append(float(Jval))
    cost_ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

cost_hamiltonian  = qml.Hamiltonian(cost_coeffs, cost_ops)

mixer_hamiltonian = qml.Hamiltonian(
    [-1.0] * n_qubits,
    [qml.PauliX(q) for q in range(n_qubits)]
)

print("\n[A1] Cost Hamiltonian built  —  Weighted Max-Cut on server graph")
print(f"     Terms: {len(cost_coeffs)}  ({n_qubits} linear + {len(J_coupling)} ZZ couplings)")

# ── A2. QAOA quantum circuit ──────────────────────────────────────────────────



dev_qaoa = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev_qaoa)
def qaoa_circuit(params):
    """
    params shape: (2, p)
      params[0] = [γ₁,…,γₚ]   cost angles
      params[1] = [β₁,…,βₚ]   mixer angles
    """
    gammas, betas = params[0], params[1]

    # 1. Equal superposition |+⟩^⊗5
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    # 2. p alternating cost + mixer layers
    for layer in range(p):
        qaoa.cost_layer(gammas[layer], cost_hamiltonian)   # encodes server graph
        qaoa.mixer_layer(betas[layer], mixer_hamiltonian)  # explores partitions

    return qml.expval(cost_hamiltonian)

print(f"[A2] QAOA circuit: {p} layers, {2*p} parameters total")

# ── A3. Classical optimisation ────────────────────────────────────────────────

print("[A3] Optimising with COBYLA …")

qaoa_energy_history = []

def qaoa_cost(flat_params):
    e = float(qaoa_circuit(flat_params.reshape(2, p)))
    qaoa_energy_history.append(e)
    return e

rng = np.random.default_rng(42)
init_qaoa = rng.uniform(0, np.pi, size=2 * p)

res_qaoa = minimize(qaoa_cost, init_qaoa, method="COBYLA",
                    options={"maxiter": 500, "rhobeg": 0.5})

best_params_qaoa = res_qaoa.x.reshape(2, p)
print(f"  Optimised γ = {best_params_qaoa[0].round(4)}")
print(f"  Optimised β = {best_params_qaoa[1].round(4)}")
print(f"  Minimum ⟨Ĥ⟩  = {res_qaoa.fun:.4f}")

# ── A4. Read statevector & extract best partition ─────────────────────────────

@qml.qnode(qml.device("default.qubit", wires=n_qubits))
def qaoa_statevector(params):
    gammas, betas = params[0], params[1]
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for layer in range(p):
        qaoa.cost_layer(gammas[layer], cost_hamiltonian)
        qaoa.mixer_layer(betas[layer], mixer_hamiltonian)
    return qml.state()

sv_qaoa        = np.array(qaoa_statevector(best_params_qaoa))
probs_qaoa     = np.abs(sv_qaoa) ** 2
qaoa_probs_dict = {format(idx, f"0{n_qubits}b"): float(pr)
                   for idx, pr in enumerate(probs_qaoa)}

qaoa_answer   = max(qaoa_probs_dict, key=qaoa_probs_dict.get)
qaoa_clusterA = [f"S{i+1}" for i, b in enumerate(qaoa_answer) if b == "1"]
qaoa_clusterB = [f"S{i+1}" for i, b in enumerate(qaoa_answer) if b == "0"]

print("\n[A4] QAOA Result")
print(f"  Top bitstring  : |{qaoa_answer}⟩  (prob = {qaoa_probs_dict[qaoa_answer]:.4f})")
print(f"  Cluster A (1)  : {qaoa_clusterA}")
print(f"  Cluster B (0)  : {qaoa_clusterB}")
print(f"  Cut weight     : {cut_weight(qaoa_answer):.0f} GB/s")
print(f"  Ising energy   : {ising_energy(qaoa_answer):.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  PART B — VQE  via Qiskit
# ═════════════════════════════════════════════════════════════════════════════


print("\n" + "=" * 70)
print("  PART B — VQE  (Qiskit)")
print("=" * 70)

# ── B1. Build Hamiltonian as Qiskit SparsePauliOp ────────────────────────────
"""
Qiskit uses Pauli strings with RIGHT-TO-LEFT qubit ordering.
'IZZII' = Z on qubit 2 and Z on qubit 3 (counting from right = qubit 0).
"""

def pauli_z(qubit, n):
    s = ["I"] * n
    s[n - 1 - qubit] = "Z"
    return "".join(s)

def pauli_zz(qi, qj, n):
    s = ["I"] * n
    s[n - 1 - qi] = "Z"
    s[n - 1 - qj] = "Z"
    return "".join(s)

pauli_list = (
    [(pauli_z(q, n_qubits), float(h_field[q])) for q in range(n_qubits)] +
    [(pauli_zz(i, j, n_qubits), float(Jval))   for (i,j), Jval in J_coupling.items()]
)

hamiltonian_qiskit = SparsePauliOp.from_list(pauli_list)

print("\n[B1] Qiskit Hamiltonian (SparsePauliOp):")
for term, coeff in pauli_list:
    print(f"  {coeff:+.4f} · {term}")

# ── B2. Build the Ansatz ──────────────────────────────────────────────────────


reps_vqe = 2
ansatz   = RealAmplitudes(num_qubits=n_qubits, reps=reps_vqe)

print(f"\n[B2] Ansatz: RealAmplitudes | reps={reps_vqe} | "
      f"parameters={ansatz.num_parameters}")

# ── B3. Energy evaluation using exact statevector ────────────────────────────

vqe_energy_history = []

def vqe_energy(params):
    """Bind params → statevector → ⟨ψ|Ĥ|ψ⟩."""
    bound = ansatz.assign_parameters(params)
    sv    = Statevector(bound)
    e     = sv.expectation_value(hamiltonian_qiskit).real
    vqe_energy_history.append(float(e))
    return float(e)

# ── B4. Classical optimisation ────────────────────────────────────────────────

print("[B3] Optimising with L-BFGS-B …")

init_vqe = rng.uniform(0, np.pi, size=ansatz.num_parameters)

res_vqe = minimize(vqe_energy, init_vqe, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-9})

print(f"  Minimum ⟨Ĥ⟩  = {res_vqe.fun:.4f}")

# ── B5. Read statevector & extract best partition ─────────────────────────────

bound_final  = ansatz.assign_parameters(res_vqe.x)
sv_final     = Statevector(bound_final)
probs_vqe    = sv_final.probabilities()

vqe_probs_dict = {format(idx, f"0{n_qubits}b"): float(pr)
                  for idx, pr in enumerate(probs_vqe)}

vqe_answer   = max(vqe_probs_dict, key=vqe_probs_dict.get)
vqe_clusterA = [f"S{i+1}" for i, b in enumerate(vqe_answer) if b == "1"]
vqe_clusterB = [f"S{i+1}" for i, b in enumerate(vqe_answer) if b == "0"]

print("\n[B4] VQE Result")
print(f"  Top bitstring  : |{vqe_answer}⟩  (prob = {vqe_probs_dict[vqe_answer]:.4f})")
print(f"  Cluster A (1)  : {vqe_clusterA}")
print(f"  Cluster B (0)  : {vqe_clusterB}")
print(f"  Cut weight     : {cut_weight(vqe_answer):.0f} GB/s")
print(f"  Ising energy   : {ising_energy(vqe_answer):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)
print(f"  {'Method':<22} {'Bitstring':<12} {'Cut (GB/s)':<12} {'Energy':>8}")
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*8}")
print(f"  {'Brute Force':<22} |{bf_string}⟩    "
      f"{bf_cut:<12.0f} {bf_energy:>8.4f}")
print(f"  {'QAOA (PennyLane)':<22} |{qaoa_answer}⟩    "
      f"{cut_weight(qaoa_answer):<12.0f} {ising_energy(qaoa_answer):>8.4f}")
print(f"  {'VQE  (Qiskit)':<22} |{vqe_answer}⟩    "
      f"{cut_weight(vqe_answer):<12.0f} {ising_energy(vqe_answer):>8.4f}")

print(f"\n  QAOA matched optimal? "
      f"{'YES ✓' if ising_energy(qaoa_answer) <= bf_energy+1e-6 else 'NO — sub-optimal'}")
print(f"  VQE  matched optimal? "
      f"{'YES ✓' if ising_energy(vqe_answer)  <= bf_energy+1e-6 else 'NO — sub-optimal'}")

print(f"\n  Architectural difference:")
print(f"    QAOA : circuit tailored to the server graph  ({2*p} params)")
print(f"    VQE  : generic RealAmplitudes ansatz         ({ansatz.num_parameters} params)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[PLOT] Generating figure …")

# ── Colour palette ────────────────────────────────────────────────────────────
DARK  = "#0d1117"; PANEL = "#161b22"; GRID  = "#21262d"
WHITE = "#e6edf3"; MUTED = "#8b949e"
CYAN  = "#00e5ff"; AMBER = "#ffab40"; GREEN = "#69ff47"
RED   = "#ff6b6b"

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── helper: draw server graph ─────────────────────────────────────────────────
def draw_server_graph(ax, bits, title, clusterA_color, clusterB_color):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=10, pad=8, fontweight="bold")

    for i, j, w in weighted_edges:
        xi, yi = positions[i]; xj, yj = positions[j]
        x_i = int(bits[i]); x_j = int(bits[j])
        is_cut = (x_i != x_j)
        lw    = 1.0 + w * 0.8          # thicker = more traffic
        color = GREEN if is_cut else MUTED
        alpha = 0.95 if is_cut else 0.35
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw,
                zorder=1, alpha=alpha)
        # weight label on edge
        mx, my = (xi+xj)/2, (yi+yj)/2
        ax.text(mx, my+0.12, f"{w}GB/s", ha="center", va="bottom",
                fontsize=6.5, color=GREEN if is_cut else MUTED, zorder=5)

    for q in range(n_qubits):
        x, y = positions[q]
        in_A = bits[q] == "1"
        color = clusterA_color if in_A else clusterB_color
        ax.scatter(x, y, s=700, c=color, edgecolors=WHITE,
                   linewidths=1.8, zorder=3)
        ax.text(x, y, f"S{q+1}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=DARK, zorder=4)
        cluster_lbl = "A" if in_A else "B"
        ax.text(x, y-0.28, cluster_lbl, ha="center", va="top",
                fontsize=7, color=color, zorder=4)

    ax.set_xlim(-0.8, 4.8); ax.set_ylim(-0.8, 2.8); ax.axis("off")

# ── Row 0: graph panels ───────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
draw_server_graph(ax0, "00000",
                  "Server Network\n(unpartitioned)", MUTED, MUTED)

ax1 = fig.add_subplot(gs[0, 1])
draw_server_graph(ax1, qaoa_answer,
                  f"QAOA Solution  |{qaoa_answer}⟩\n"
                  f"(PennyLane)  cut={cut_weight(qaoa_answer):.0f} GB/s",
                  CYAN, RED)

ax2 = fig.add_subplot(gs[0, 2])
draw_server_graph(ax2, vqe_answer,
                  f"VQE Solution   |{vqe_answer}⟩\n"
                  f"(Qiskit)  cut={cut_weight(vqe_answer):.0f} GB/s",
                  AMBER, RED)

# ── helper: style axes ────────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Iteration", color=MUTED, fontsize=8)
    ax.set_ylabel("⟨Ĥ⟩", color=MUTED, fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

# ── Row 1 left: QAOA convergence ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(qaoa_energy_history, color=CYAN, lw=1.4, alpha=0.9)
ax3.axhline(bf_energy, color=GREEN, lw=1.2, ls="--",
            label=f"Optimal {bf_energy:.3f}")
style_ax(ax3, "QAOA Convergence")
ax3.legend(fontsize=8, facecolor=GRID, labelcolor=WHITE, edgecolor=MUTED)

# ── Row 1 middle: VQE convergence ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(vqe_energy_history, color=AMBER, lw=1.4, alpha=0.9)
ax4.axhline(bf_energy, color=GREEN, lw=1.2, ls="--",
            label=f"Optimal {bf_energy:.3f}")
style_ax(ax4, "VQE Convergence")
ax4.legend(fontsize=8, facecolor=GRID, labelcolor=WHITE, edgecolor=MUTED)

# ── Row 1 right: probability histogram ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)

top_n     = 8
qaoa_top  = sorted(qaoa_probs_dict.items(), key=lambda x: -x[1])[:top_n]
vqe_top   = sorted(vqe_probs_dict.items(),  key=lambda x: -x[1])[:top_n]
all_labels = list(dict.fromkeys([k for k,_ in qaoa_top] +
                                 [k for k,_ in vqe_top]))[:top_n]

x_pos = np.arange(len(all_labels)); bar_w = 0.38
ax5.bar(x_pos - bar_w/2,
        [qaoa_probs_dict.get(l, 0) for l in all_labels],
        bar_w, color=CYAN,  alpha=0.85, label="QAOA")
ax5.bar(x_pos + bar_w/2,
        [vqe_probs_dict.get(l, 0)  for l in all_labels],
        bar_w, color=AMBER, alpha=0.85, label="VQE")

if bf_string in all_labels:
    ax5.axvline(all_labels.index(bf_string), color=GREEN, lw=1.3,
                ls="--", alpha=0.7, label=f"Optimal |{bf_string}⟩")

ax5.set_xticks(x_pos)
ax5.set_xticklabels(all_labels, rotation=45, ha="right",
                    fontsize=7, color=MUTED)
ax5.set_title("Probability Distribution\n(Top bitstrings)",
              color=WHITE, fontsize=10, fontweight="bold")
ax5.set_ylabel("Probability", color=MUTED, fontsize=8)
ax5.tick_params(colors=MUTED)
ax5.legend(fontsize=8, facecolor=GRID, labelcolor=WHITE, edgecolor=MUTED)
for sp in ax5.spines.values():
    sp.set_edgecolor(GRID)

fig.suptitle(
    "QAOA (PennyLane)  vs  VQE (Qiskit) — Data Center Server Partitioning\n"
    "Weighted Max-Cut · 5 Servers · 6 Data-Exchange Links",
    color=WHITE, fontsize=13, fontweight="bold", y=0.98
)

out_img = "server_partition_results.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"  Figure saved → {out_img}")
plt.close()

print("\n✅  All done!\n")