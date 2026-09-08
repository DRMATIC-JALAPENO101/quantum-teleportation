"""
Quantum Teleportation — Streamlit Web Demo
Run with: streamlit run app.py
Install:  pip install streamlit qiskit qiskit-aer matplotlib numpy
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
from qiskit.visualization import plot_bloch_multivector

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Quantum Teleportation Demo",
    page_icon="⚛",
    layout="wide",
)

st.markdown("""
<style>
  /* Subtle quantum-themed background */
  section[data-testid="stSidebar"] { background: #0d1b2a; }
  section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: #1e4d8c; }
  
  .state-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #e0eaf4;
    height: 100%;
  }
  .state-card h4 {
    margin: 0 0 4px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64b5f6;
  }
  .state-card .label {
    font-size: 11px;
    color: #7a9bbf;
    margin-bottom: 8px;
  }
  
  .correction-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .badge-before { background: #3d1f1f; color: #f48fb1; border: 1px solid #7b3f4e; }
  .badge-after  { background: #1b3a2a; color: #81c784; border: 1px solid #2e7d52; }
  
  .arrow-col {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    color: #64b5f6;
    height: 100%;
    padding-top: 60px;
  }
  
  .fidelity-box {
    text-align:center;
    padding:16px;
    border-radius:12px;
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
  }
</style>
""", unsafe_allow_html=True)

st.title("⚛ Quantum Teleportation Network")
st.caption("An undergraduate project simulation")

# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────

st.sidebar.header("Qubit State Controls")
st.sidebar.markdown("Define Alice's unknown qubit state |ψ⟩ using Bloch sphere angles.")

theta = st.sidebar.slider("θ (polar angle)", 0.0, float(np.pi), float(np.pi / 3), step=0.05,
                           help="Controls the |0⟩ vs |1⟩ mix. θ=0 → |0⟩, θ=π → |1⟩")
phi = st.sidebar.slider("φ (azimuthal angle)", 0.0, float(2 * np.pi), float(np.pi / 4), step=0.05,
                         help="Controls the phase of the superposition")

st.sidebar.divider()
st.sidebar.header("Noise Model")
noise_on = st.sidebar.toggle("Enable decoherence", value=False)
noise_rate = st.sidebar.slider("Error rate per gate", 0.001, 0.15, 0.05, step=0.005,
                                 disabled=not noise_on,
                                 help="Depolarizing noise probability per gate operation")

st.sidebar.divider()
st.sidebar.header("Network Hops")
num_hops = st.sidebar.selectbox("Number of relay nodes", [1, 2, 3], index=0,
                                   help="More hops = more fidelity loss")

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", width="stretch")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def build_teleportation_circuit(theta, phi):
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)

    # Prepare |ψ⟩
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])
    qc.barrier(label="|ψ⟩")

    # Bell pair
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier(label="entangle")

    # Alice measures
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="measure")
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # Bob corrects
    x_body = QuantumCircuit(1)
    x_body.x(0)
    qc.if_else((c[1], 1), x_body, None, [q[2]], [])

    z_body = QuantumCircuit(1)
    z_body.z(0)
    qc.if_else((c[0], 1), z_body, None, [q[2]], [])

    return qc


def build_noise_model(rate):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 1), ['h', 'ry', 'rz', 'x', 'z'])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate * 2, 2), ['cx'])
    return nm


def simulate(theta, phi, noise_on, noise_rate):
    qc = build_teleportation_circuit(theta, phi)
    sim = AerSimulator(method='statevector')
    kwargs = {}
    if noise_on:
        kwargs['noise_model'] = build_noise_model(noise_rate)
    job = sim.run(qc, shots=1024, **kwargs)
    return job.result().get_counts()


def estimate_fidelity(noise_on, noise_rate, num_hops):
    if not noise_on:
        return 1.0
    per_hop = max(0, 1 - noise_rate * 8)
    return per_hop ** num_hops


def get_state_label(theta, phi):
    t = round(theta / np.pi, 2)
    p = round(phi / np.pi, 2)
    return f"|ψ⟩ = cos({t}π/2)|0⟩ + e^(i·{p}π) sin({t}π/2)|1⟩"


def get_target_statevector(theta, phi):
    """Alice's original state."""
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return Statevector(qc)


def get_bob_state_before_correction(theta, phi, measurement_outcome: str):
    """
    Simulate what Bob's qubit looks like BEFORE correction for a given
    2-bit measurement outcome (e.g. '00', '01', '10', '11').
    The Bell measurement collapses Bob's qubit to one of 4 states.
    outcome[0] = Alice's q[1] bit, outcome[1] = Alice's q[0] bit (Qiskit bit order)
    """
    # In Qiskit counts, bit string is c[1]c[0] (MSB left), so:
    #   c[0] corresponds to q[0] (Z-correction bit)
    #   c[1] corresponds to q[1] (X-correction bit)
    bit_c0 = int(measurement_outcome[1])  # Z bit
    bit_c1 = int(measurement_outcome[0])  # X bit

    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)

    # Map from Bell basis: Bob's state before correction
    #  00 ->  alpha|0⟩ + beta|1⟩       (already correct)
    #  01 ->  alpha|0⟩ - beta|1⟩       (Z error)
    #  10 ->  beta|0⟩  + alpha|1⟩      (X error)
    #  11 -> -beta|0⟩  + alpha|1⟩      (XZ error — but X first then Z)
    if   bit_c1 == 0 and bit_c0 == 0:
        coeffs = np.array([alpha,  beta ])
    elif bit_c1 == 0 and bit_c0 == 1:
        coeffs = np.array([alpha, -beta ])
    elif bit_c1 == 1 and bit_c0 == 0:
        coeffs = np.array([beta,   alpha])
    else:  # 11
        coeffs = np.array([-beta,  alpha])

    return Statevector(coeffs / np.linalg.norm(coeffs))


def get_bob_state_after_correction(theta, phi, noise_on, noise_rate):
    """
    Bob's state after correction.
    Ideal: same as Alice's. With noise: apply density matrix simulation.
    """
    if not noise_on:
        return get_target_statevector(theta, phi)
    # With noise we simulate the full circuit with save_density_matrix
    qc = build_teleportation_circuit(theta, phi)
    qc.save_density_matrix(qubits=[2], label='bob_dm')
    sim = AerSimulator(method='density_matrix')
    noise_model = build_noise_model(noise_rate)
    result = sim.run(qc, noise_model=noise_model, shots=1).result()
    dm = result.data()['bob_dm']
    return dm


def compute_bloch_vector_from_dm(dm):
    """Extract Bloch vector [x,y,z] from a 2x2 density matrix."""
    rho = np.array(dm.data)
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])


def plot_bloch_from_statevector(sv, title="", title_color="#e0eaf4"):
    fig = plot_bloch_multivector(sv)
    fig.set_size_inches(3.0, 3.0)
    if title:
        fig.suptitle(title, fontsize=10, color=title_color, y=0.98)
    fig.patch.set_facecolor('#0d1b2a')
    return fig


def plot_bloch_from_bloch_vector(bvec, title="", title_color="#e0eaf4"):
    """Plot a Bloch sphere for an arbitrary Bloch vector (may be mixed state)."""
    from qiskit.visualization.bloch import Bloch
    b = Bloch()
    b.vector_color = ['#f48fb1']
    b.add_vectors(bvec)
    b.render()
    fig = b.fig
    fig.set_size_inches(3.0, 3.0)
    if title:
        fig.suptitle(title, fontsize=10, color=title_color, y=0.98)
    fig.patch.set_facecolor('#0d1b2a')
    return fig


# ─────────────────────────────────────────────
# MAIN LAYOUT — TOP ROW: Alice | Circuit | Network
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns([1.0, 2.2, 1.0])

with col1:
    st.subheader("Alice's Qubit State")
    st.markdown(f"`{get_state_label(theta, phi)}`")

    sv_alice = get_target_statevector(theta, phi)
    fig_bloch = plot_bloch_from_statevector(sv_alice)
    st.pyplot(fig_bloch, width="stretch")
    plt.close()

    st.caption("This is the unknown state |ψ⟩ Alice wants to teleport.")

with col2:
    st.subheader("Teleportation Circuit")
    qc = build_teleportation_circuit(theta, phi)
    fig_circ = qc.draw(output='mpl', style='iqp', fold=60)
    fig_circ.set_size_inches(9, 3.5)
    st.pyplot(fig_circ, width="stretch")
    plt.close()

    st.markdown("""
<div style='background:linear-gradient(135deg,#0d1b2a,#112240);border:1px solid #1e3a5f;
            border-radius:10px;padding:14px 18px;font-size:13px;color:#c8d8e8;line-height:1.75;margin-top:8px'>
  <b style='color:#64b5f6;font-size:13px'>How the circuit works — step by step</b><br><br>
  <b style='color:#90caf9'>① Prepare |ψ⟩</b> &nbsp;—&nbsp;
    Alice encodes her unknown qubit using an <code>Ry(θ)</code> rotation followed by <code>Rz(φ)</code>.
    This sets the qubit anywhere on the Bloch sphere.<br><br>
  <b style='color:#90caf9'>② Create entanglement</b> &nbsp;—&nbsp;
    A Hadamard gate (<code>H</code>) puts qubit B into superposition, then a
    <code>CNOT</code> entangles B with C, forming a shared Bell pair between Alice and Bob.<br><br>
  <b style='color:#90caf9'>③ Alice's Bell measurement</b> &nbsp;—&nbsp;
    A <code>CNOT</code> from A→B followed by <code>H</code> on A rotates the system into
    the Bell basis. Alice then measures both A and B, collapsing the system and yielding
    2 classical bits (00 / 01 / 10 / 11).<br><br>
  <b style='color:#90caf9'>④ Bob corrects</b> &nbsp;—&nbsp;
    Alice sends her 2 bits to Bob over a classical channel.
    Bob applies <code>X</code> if bit₁ = 1, then <code>Z</code> if bit₀ = 1.
    After correction, his qubit C is identical to Alice's original |ψ⟩.
</div>
""", unsafe_allow_html=True)

with col3:
    st.subheader("Network & Fidelity")

    fidelity = estimate_fidelity(noise_on, noise_rate, num_hops)
    fid_pct = round(fidelity * 100, 1)

    color = "#1D9E75" if fid_pct >= 90 else "#BA7517" if fid_pct >= 70 else "#A32D2D"
    st.markdown(
        f"""<div class='fidelity-box'>
        <div style='font-size:13px;color:#7a9bbf;margin-bottom:4px'>Estimated Fidelity</div>
        <div style='font-size:40px;font-weight:600;color:{color}'>{fid_pct}%</div>
        <div style='font-size:12px;color:#557799;margin-top:4px'>after {num_hops} hop{"s" if num_hops > 1 else ""}</div>
        </div>""",
        unsafe_allow_html=True
    )

    st.divider()

    nodes = ["Alice"] + [f"Relay {i}" for i in range(1, num_hops)] + ["Bob"]
    network_str = " → ".join(nodes)
    st.markdown(f"**Network path:** `{network_str}`")

    for i in range(num_hops):
        hop_fid = max(0, 1 - (noise_rate * 8 if noise_on else 0))
        st.progress(hop_fid, text=f"Hop {i+1}: {nodes[i]} → {nodes[i+1]} | fidelity {hop_fid:.0%}")

    if noise_on and fid_pct < 90:
        st.warning(f"⚠ Fidelity below 90% — quantum error correction may be needed.")
    elif not noise_on:
        st.success("✓ Ideal channel — perfect teleportation (fidelity 100%)")
    else:
        st.success(f"✓ Good fidelity despite noise — teleportation successful")


# ─────────────────────────────────────────────
# BOB'S STATE — BEFORE & AFTER CORRECTION
# ─────────────────────────────────────────────

st.divider()
st.subheader("🔭 Bob's Qubit: Before vs After Correction")
st.caption(
    "After Alice's Bell measurement, Bob's qubit collapses to one of 4 possible states depending on "
    "the 2 classical bits Alice sends him. The correction gates (X and/or Z) then rotate his qubit back "
    "to match Alice's original state exactly."
)

# Pick representative outcomes to display (all 4 possible Bell outcomes)
outcomes_to_show = ['00', '01', '10', '11']
outcome_labels = {
    '00': 'Alice sends 00 — no correction needed',
    '01': 'Alice sends 01 — Z gate applied',
    '10': 'Alice sends 10 — X gate applied',
    '11': 'Alice sends 11 — X then Z gate applied',
}
correction_desc = {
    '00': 'None',
    '01': 'Z gate',
    '10': 'X gate',
    '11': 'X + Z gates',
}

sv_alice = get_target_statevector(theta, phi)
bob_after_dm = get_bob_state_after_correction(theta, phi, noise_on, noise_rate)

# Header row
h0, h1, h2, h3 = st.columns([0.5, 1, 0.15, 1])
with h1:
    st.markdown("<div style='text-align:center'><span class='correction-badge badge-before'>⚡ BEFORE CORRECTION</span></div>", unsafe_allow_html=True)
with h3:
    st.markdown("<div style='text-align:center'><span class='correction-badge badge-after'>✓ AFTER CORRECTION</span></div>", unsafe_allow_html=True)

for outcome in outcomes_to_show:
    sv_before = get_bob_state_before_correction(theta, phi, outcome)

    row_label, before_col, arrow_col, after_col = st.columns([0.5, 1, 0.15, 1])

    with row_label:
        st.markdown(
            f"""<div style='height:100%;display:flex;flex-direction:column;justify-content:center;
                           padding-top:40px;font-size:12px;color:#7a9bbf;line-height:1.6'>
              <div style='font-family:monospace;font-size:15px;color:#64b5f6;font-weight:700'>
                Outcome {outcome}
              </div>
              <div>{outcome_labels[outcome]}</div>
              <div style='margin-top:6px;color:#4a7a9b'>
                <b style='color:#c8d8e8'>Correction:</b> {correction_desc[outcome]}
              </div>
            </div>""",
            unsafe_allow_html=True
        )

    with before_col:
        fig_before = plot_bloch_from_statevector(sv_before, title_color="#f48fb1")
        # Overlay a subtle red tint in the title to signal "uncorrected"
        for ax in fig_before.axes:
            ax.set_title("")
        st.pyplot(fig_before, width="stretch")
        plt.close()

        # Compute angle difference from Alice's state
        bloch_before = np.real(sv_before.data)
        dot = abs(np.dot(sv_before.data.conj(), sv_alice.data)) ** 2
        st.markdown(
            f"<div style='text-align:center;font-size:11px;color:#f48fb1'>Fidelity with Alice: <b>{dot:.2%}</b></div>",
            unsafe_allow_html=True
        )

    with arrow_col:
        st.markdown(
            "<div style='text-align:center;font-size:22px;padding-top:60px;color:#64b5f6'>→</div>",
            unsafe_allow_html=True
        )

    with after_col:
        if not noise_on:
            # Perfect correction — show Alice's exact state
            fig_after = plot_bloch_from_statevector(sv_alice, title_color="#81c784")
            for ax in fig_after.axes:
                ax.set_title("")
            st.pyplot(fig_after, width="stretch")
            plt.close()
            st.markdown(
                "<div style='text-align:center;font-size:11px;color:#81c784'>Fidelity with Alice: <b>100.00%</b></div>",
                unsafe_allow_html=True
            )
        else:
            # Noisy — show mixed/degraded state from density matrix
            bvec = compute_bloch_vector_from_dm(bob_after_dm)
            fig_after = plot_bloch_from_bloch_vector(bvec, title_color="#81c784")
            st.pyplot(fig_after, width="stretch")
            plt.close()
            # Fidelity: Tr(rho_target * rho_noisy)
            rho_target = DensityMatrix(sv_alice)
            fid_after = state_fidelity(rho_target, bob_after_dm)
            st.markdown(
                f"<div style='text-align:center;font-size:11px;color:#81c784'>Fidelity with Alice: <b>{fid_after:.2%}</b></div>",
                unsafe_allow_html=True
            )

    st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0'>", unsafe_allow_html=True)

st.caption(
    "**Key insight:** Regardless of the measurement outcome (00/01/10/11), Bob's corrected state always "
    "matches Alice's original |ψ⟩. This is the essence of quantum teleportation — the classical bits "
    "tell Bob *which* correction to apply, not *what* the state is."
)


# ─────────────────────────────────────────────
# SIMULATION RESULTS (runs when button pressed)
# ─────────────────────────────────────────────

if run_btn:
    st.divider()
    st.subheader("Simulation Results")

    with st.spinner("Running quantum simulation..."):
        counts = simulate(theta, phi, noise_on, noise_rate)

    rcol1, rcol2 = st.columns(2)

    with rcol1:
        st.markdown("**Measurement outcomes (1024 shots)**")
        st.caption("These are Alice's 2 classical bits sent to Bob. Distribution shows noise effects.")

        fig_counts, ax = plt.subplots(figsize=(5, 3))
        outcomes = sorted(counts.keys())
        values = [counts[k] for k in outcomes]
        bars = ax.bar(outcomes, values, color=['#1D9E75' if noise_on else '#185FA5'] * len(outcomes), alpha=0.8, edgecolor='none')
        ax.set_xlabel("Measurement outcome (bit 0, bit 1)", fontsize=11)
        ax.set_ylabel("Count (of 1024 shots)", fontsize=11)
        ax.set_title("Bell Measurement Distribution")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(val), ha='center', fontsize=10)
        st.pyplot(fig_counts, width="stretch")
        plt.close()

    with rcol2:
        st.markdown("**Noise vs. Fidelity curve**")
        st.caption("How fidelity degrades as noise increases — key insight for real quantum networks.")

        noise_range = np.linspace(0, 0.15, 50)
        fidelity_curve = [max(0, 1 - n * 8) for n in noise_range]

        fig_nf, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(noise_range * 100, fidelity_curve, color='#185FA5', linewidth=2)
        ax2.fill_between(noise_range * 100, fidelity_curve, alpha=0.12, color='#185FA5')
        ax2.axhline(0.9, color='#E85D24', linestyle='--', alpha=0.7, linewidth=1.2, label='90% threshold')
        if noise_on:
            ax2.axvline(noise_rate * 100, color='#1D9E75', linestyle=':', linewidth=1.5, label=f'Current rate ({noise_rate*100:.1f}%)')
        ax2.set_xlabel("Error Rate (%)")
        ax2.set_ylabel("Fidelity")
        ax2.set_title("Decoherence Impact on Fidelity")
        ax2.legend(fontsize=9)
        st.pyplot(fig_nf, width="stretch")
        plt.close()

    # Summary table
    st.divider()
    st.subheader("Protocol Summary")
    summary_data = {
        "Parameter": ["Input state θ", "Input state φ", "Noise enabled", "Error rate", "Hops", "Estimated fidelity", "Classical bits sent"],
        "Value": [f"{theta:.3f} rad", f"{phi:.3f} rad", "Yes" if noise_on else "No",
                  f"{noise_rate:.1%}" if noise_on else "0%", str(num_hops),
                  f"{fid_pct}%", "2 bits per teleportation"]
    }
    st.table(summary_data)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Qiskit & Streamlit | Quantum Networking Project")
