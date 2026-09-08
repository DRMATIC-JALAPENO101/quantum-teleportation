"""
Quantum Teleportation Simulation — Starter Code
Team project: Quantum Networking (Undergraduate)
Uses: Qiskit 1.x
Install: pip install qiskit qiskit-aer matplotlib
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# PRESET STATES  (convenience constants)
# ─────────────────────────────────────────────

PRESETS = {
    "|0⟩": (0.0,           0.0),
    "|1⟩": (np.pi,         0.0),
    "|+⟩": (np.pi / 2,     0.0),
    "|−⟩": (np.pi / 2,     np.pi),
    "|i⟩": (np.pi / 2,     np.pi / 2),
}


# ─────────────────────────────────────────────
# 1. PREPARE THE UNKNOWN STATE TO TELEPORT
# ─────────────────────────────────────────────

def prepare_unknown_state(theta=np.pi / 3, phi=np.pi / 4):
    """
    Creates a single-qubit state |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    This is the state Alice wants to teleport to Bob.

    Use PRESETS dict for well-known states, e.g.:
        theta, phi = PRESETS["|+⟩"]
    """
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


# ─────────────────────────────────────────────
# 2. BUILD THE TELEPORTATION CIRCUIT
# ─────────────────────────────────────────────

def build_teleportation_circuit(theta=np.pi / 3, phi=np.pi / 4):
    """
    Full 3-qubit teleportation circuit:
      q[0] = qubit A — Alice's unknown state |ψ⟩
      q[1] = qubit B — Alice's half of the Bell pair
      q[2] = qubit C — Bob's half of the Bell pair

    Steps:
      1. Prepare |ψ⟩ on qubit A
      2. Generate Bell pair on B and C (entanglement)
      3. Alice does Bell measurement on A+B (2 classical bits)
      4. Bob applies correction gates on C based on measurement
    """
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)

    # Step 1: Prepare Alice's unknown state on qubit A
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])
    qc.barrier(label="prepare |ψ⟩")

    # Step 2: Create Bell pair between B and C
    qc.h(q[1])           # Hadamard on B → superposition
    qc.cx(q[1], q[2])    # CNOT → entanglement
    qc.barrier(label="bell pair")

    # Step 3: Alice's Bell measurement on A and B
    qc.cx(q[0], q[1])    # CNOT: A controls B
    qc.h(q[0])           # Hadamard on A
    qc.barrier(label="alice measures")
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # Step 4: Bob applies corrections on C based on classical bits
    x_body = QuantumCircuit(1)
    x_body.x(0)
    qc.if_else((c[1], 1), x_body, None, [q[2]], [])

    z_body = QuantumCircuit(1)
    z_body.z(0)
    qc.if_else((c[0], 1), z_body, None, [q[2]], [])

    return qc


# ─────────────────────────────────────────────
# 3. NOISE MODEL (simulates real hardware)
# ─────────────────────────────────────────────

def build_noise_model(error_rate=0.05):
    """
    Adds depolarizing noise — simulates decoherence.
    error_rate: 0 = perfect, 0.1 = ~10% error per gate
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1),     ['h', 'ry', 'rz', 'x', 'z'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate * 2, 2), ['cx'])
    return noise_model


# ─────────────────────────────────────────────
# 4. COMPUTE FIDELITY (how well did teleportation work?)
# ─────────────────────────────────────────────

def estimate_fidelity_analytic(noise_rate, num_hops=1):
    """
    Analytic fidelity estimate using the depolarizing channel model.

    For each hop the circuit applies ~7 single-qubit gates and 1 CNOT.
    Depolarizing channel contracts the Bloch vector by (1 - 2p/3) per
    single-qubit gate and (1 - 4p_2/5) per two-qubit gate (p_2 = 2*rate).

    Returns:
        float: estimated fidelity in [0, 1]
    """
    if noise_rate <= 0:
        return 1.0
    f_single = (1 - (2 / 3) * noise_rate) ** 7      # 7 single-qubit gates per hop
    f_two    = (1 - (4 / 5) * (noise_rate * 2)) ** 1 # 1 CNOT per hop
    per_hop  = f_single * f_two
    return max(0.0, per_hop ** num_hops)


def compute_fidelity(theta, phi, noise_rate=0.0):
    """
    Simulates teleportation and returns:
      - counts       : dict of measurement outcomes from 1024 shots
      - fidelity     : float, state fidelity between ideal and noisy output
      - ideal_state  : Statevector of what Bob should receive

    Uses density-matrix simulation when noise_rate > 0 so the fidelity
    calculation is exact (not a rough linear estimate).
    """
    # Ideal target state
    ideal_qc = QuantumCircuit(1)
    ideal_qc.ry(theta, 0)
    ideal_qc.rz(phi, 0)
    ideal_state = Statevector(ideal_qc)

    qc = build_teleportation_circuit(theta, phi)

    if noise_rate > 0:
        # Density-matrix path: save Bob's qubit state after all corrections
        qc_dm = build_teleportation_circuit(theta, phi)
        qc_dm.save_density_matrix(qubits=[2], label='bob_dm')
        sim_dm = AerSimulator(method='density_matrix')
        result_dm = sim_dm.run(qc_dm, noise_model=build_noise_model(noise_rate), shots=1).result()
        bob_dm   = result_dm.data()['bob_dm']
        fidelity = float(state_fidelity(DensityMatrix(ideal_state), bob_dm))

        # Also get shot counts for the histogram
        sim_sv = AerSimulator(method='statevector')
        counts = sim_sv.run(qc, noise_model=build_noise_model(noise_rate), shots=1024).result().get_counts()
    else:
        sim = AerSimulator(method='statevector')
        counts   = sim.run(qc, shots=1024).result().get_counts()
        fidelity = 1.0   # ideal teleportation is perfect

    return counts, fidelity, ideal_state


# ─────────────────────────────────────────────
# 5. MULTI-HOP TELEPORTATION (Alice → Relay → Bob)
# ─────────────────────────────────────────────

def multi_hop_teleportation(theta=np.pi / 3, phi=np.pi / 4):
    """
    Simulates a multi-hop quantum network, e.g.:
      Node 1 (Alice) → Node 2 (Relay) → Node 3 (Bob)

    Each hop is an independent teleportation with its own noise rate.
    Cumulative fidelity uses the analytic depolarizing model (not a
    linear shortcut), so results are physically meaningful.

    Returns:
        float: cumulative fidelity after all hops
    """
    print("=== Multi-hop Quantum Teleportation ===")
    print(f"Original state: θ={theta:.2f} rad,  φ={phi:.2f} rad\n")

    hops = [
        ("Alice → Relay", 0.02),
        ("Relay → Bob",   0.03),
    ]

    cumulative_fidelity = 1.0

    for i, (label, noise_rate) in enumerate(hops):
        counts, hop_fidelity, _ = compute_fidelity(theta, phi, noise_rate=noise_rate)
        analytic_fid            = estimate_fidelity_analytic(noise_rate, num_hops=1)
        cumulative_fidelity    *= analytic_fid

        print(f"Hop {i+1}: {label}")
        print(f"  Noise rate         : {noise_rate:.1%}")
        print(f"  Simulated fidelity : {hop_fidelity:.4f}")
        print(f"  Analytic estimate  : {analytic_fid:.4f}")
        print(f"  Cumulative fidelity: {cumulative_fidelity:.4f}\n")

    print(f"Final fidelity after {len(hops)} hops: {cumulative_fidelity:.2%}")
    return cumulative_fidelity


# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────

def visualize_circuit(theta=np.pi / 3, phi=np.pi / 4):
    """Draw and save the circuit diagram."""
    qc  = build_teleportation_circuit(theta, phi)
    fig = qc.draw(output='mpl', style='iqp', fold=60)
    plt.title("Quantum Teleportation Circuit", fontsize=14)
    plt.tight_layout()
    plt.savefig("teleportation_circuit.png", dpi=150, bbox_inches='tight')
    print("Circuit saved to teleportation_circuit.png")
    return fig


def visualize_bloch_sphere(theta=np.pi / 3, phi=np.pi / 4, title_suffix=""):
    """Show Bloch sphere of the given state."""
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    sv  = Statevector(qc)
    fig = plot_bloch_multivector(sv)
    fig.suptitle(f"Alice's Unknown Qubit State |ψ⟩{title_suffix}", fontsize=13)
    fname = "bloch_sphere.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Bloch sphere saved to {fname}")
    return fig


def visualize_bob_before_after(theta=np.pi / 3, phi=np.pi / 4, noise_rate=0.0):
    """
    Side-by-side Bloch spheres showing Bob's qubit state
    BEFORE and AFTER correction for all 4 Bell measurement outcomes.

    Bob's state before correction depends on which outcome Alice got:
      00 → already correct  (no gates needed)
      01 → Z-flipped phase  (Z corrects it)
      10 → X-flipped amplitudes (X corrects it)
      11 → X then Z needed

    After correction all 4 converge to Alice's original |ψ⟩.
    """
    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)

    before_states = {
        '00': Statevector(np.array([alpha,  beta ])),
        '01': Statevector(np.array([alpha, -beta ])),
        '10': Statevector(np.array([beta,   alpha])),
        '11': Statevector(np.array([-beta,  alpha])),
    }
    correction_label = {
        '00': 'None',
        '01': 'Z gate',
        '10': 'X gate',
        '11': 'X + Z',
    }

    # Ideal "after" state
    ideal_qc = QuantumCircuit(1)
    ideal_qc.ry(theta, 0)
    ideal_qc.rz(phi, 0)
    ideal_sv = Statevector(ideal_qc)

    # Noisy "after" density matrix (if noise_rate > 0)
    noisy_after_bvec = None
    if noise_rate > 0:
        qc_dm = build_teleportation_circuit(theta, phi)
        qc_dm.save_density_matrix(qubits=[2], label='bob_dm')
        sim_dm = AerSimulator(method='density_matrix')
        dm = sim_dm.run(qc_dm, noise_model=build_noise_model(noise_rate), shots=1).result().data()['bob_dm']
        rho = np.array(dm.data)
        noisy_after_bvec = np.array([
            2 * np.real(rho[0, 1]),
            2 * np.imag(rho[1, 0]),
            np.real(rho[0, 0] - rho[1, 1]),
        ])

    outcomes = ['00', '01', '10', '11']
    fig, axes = plt.subplots(4, 2, figsize=(8, 18),
                              subplot_kw=dict(projection='3d'))
    fig.suptitle("Bob's Qubit: Before vs After Correction", fontsize=14, y=1.01)

    from qiskit.visualization.bloch import Bloch

    def sv_to_bloch(sv):
        """Convert a Statevector to a [x, y, z] Bloch vector."""
        a, b = sv.data[0], sv.data[1]
        return [
            2 * np.real(np.conj(a) * b),
            2 * np.imag(np.conj(a) * b),
            float(np.real(np.conj(a) * a - np.conj(b) * b)),
        ]

    ideal_bvec = sv_to_bloch(ideal_sv)

    # Compute noisy after-fidelity once (reused for every row label)
    if noise_rate > 0 and noisy_after_bvec is not None:
        after_fid   = float(state_fidelity(DensityMatrix(ideal_sv), dm))
        after_label_suffix = f"\n(noisy, rate={noise_rate:.1%})"
    else:
        after_fid          = 1.0
        after_label_suffix = ""

    for row, outcome in enumerate(outcomes):
        sv_before  = before_states[outcome]
        bvec_before = sv_to_bloch(sv_before)
        dot         = abs(np.dot(sv_before.data.conj(), ideal_sv.data)) ** 2

        # ── Before ──
        b_before = Bloch(axes=axes[row, 0])
        b_before.vector_color = ['#f48fb1']
        b_before.add_vectors(bvec_before)
        b_before.render()
        axes[row, 0].set_title(
            f"Outcome {outcome} — BEFORE\n"
            f"Correction: {correction_label[outcome]}\n"
            f"Fidelity with Alice: {dot:.2%}",
            fontsize=9,
        )

        # ── After ──
        b_after = Bloch(axes=axes[row, 1])
        b_after.vector_color = ['#81c784']
        after_bvec = noisy_after_bvec if (noise_rate > 0 and noisy_after_bvec is not None) else ideal_bvec
        b_after.add_vectors(after_bvec)
        b_after.render()
        axes[row, 1].set_title(
            f"Outcome {outcome} — AFTER correction\n"
            f"Fidelity with Alice: {after_fid:.2%}{after_label_suffix}",
            fontsize=9,
        )

    plt.tight_layout()
    fname = f"bob_before_after{'_noisy' if noise_rate > 0 else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Bob before/after saved to {fname}")
    return fig


def plot_noise_vs_fidelity(num_hops=1):
    """
    Show how noise degrades teleportation fidelity using the analytic
    depolarizing model (replaces the old linear approximation).
    """
    noise_rates    = np.linspace(0, 0.15, 60)
    fidelities     = [estimate_fidelity_analytic(n, num_hops) for n in noise_rates]
    fidelities_old = [max(0, 1 - n * 8) for n in noise_rates]   # kept for comparison

    plt.figure(figsize=(7, 4))
    plt.plot(noise_rates * 100, fidelities,     'o-', color='#1D9E75', linewidth=2,
             markersize=3, label=f'Analytic model ({num_hops} hop{"s" if num_hops>1 else ""})')
    plt.plot(noise_rates * 100, fidelities_old, '--', color='#aaa',    linewidth=1.2,
             label='Old linear estimate (less accurate)')
    plt.axhline(y=0.9, color='#E85D24', linestyle='--', alpha=0.6, label='90% fidelity threshold')
    plt.fill_between(noise_rates * 100, fidelities, alpha=0.15, color='#1D9E75')
    plt.xlabel("Depolarizing Error Rate (%)")
    plt.ylabel("Teleportation Fidelity")
    plt.title("Noise vs. Fidelity in Quantum Teleportation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_fidelity.png", dpi=150, bbox_inches='tight')
    print("Noise plot saved to noise_fidelity.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Quantum Teleportation Demo")
    print("=" * 40)

    # ── Choose a state (swap for any PRESETS entry) ──
    theta, phi = np.pi / 3, np.pi / 4
    # theta, phi = PRESETS["|+⟩"]   # uncomment to try a preset

    # [1] Circuit diagram
    print("\n[1] Drawing circuit...")
    visualize_circuit(theta, phi)

    # [2] Alice's Bloch sphere
    print("\n[2] Drawing Alice's Bloch sphere...")
    visualize_bloch_sphere(theta, phi)

    # [3] Ideal (noiseless) simulation
    print("\n[3] Running ideal (noiseless) simulation...")
    counts_ideal, fid_ideal, _ = compute_fidelity(theta, phi, noise_rate=0)
    print(f"  Measurement outcomes : {counts_ideal}")
    print(f"  State fidelity       : {fid_ideal:.4f}")

    # [4] Noisy simulation
    print("\n[4] Running noisy simulation (5% error)...")
    counts_noisy, fid_noisy, _ = compute_fidelity(theta, phi, noise_rate=0.05)
    print(f"  Measurement outcomes : {counts_noisy}")
    print(f"  State fidelity       : {fid_noisy:.4f}")
    print(f"  Analytic estimate    : {estimate_fidelity_analytic(0.05):.4f}")

    # [5] Multi-hop network
    print("\n[5] Simulating multi-hop network...")
    multi_hop_teleportation(theta, phi)

    # [6] Bob's before/after Bloch spheres (noiseless)
    print("\n[6] Visualising Bob's qubit before and after correction (ideal)...")
    visualize_bob_before_after(theta, phi, noise_rate=0.0)

    # [7] Bob's before/after Bloch spheres (noisy)
    print("\n[7] Visualising Bob's qubit before and after correction (noisy, 5%)...")
    visualize_bob_before_after(theta, phi, noise_rate=0.05)

    # [8] Noise vs fidelity analysis
    print("\n[8] Plotting noise vs fidelity...")
    plot_noise_vs_fidelity(num_hops=1)

    print("\nDone! Check the saved PNG files for visuals.")
