"""
Quantum Teleportation — Streamlit Web Demo
Run with: streamlit run app.py
Install:  pip install streamlit qiskit qiskit-aer matplotlib numpy
"""

from pathlib import Path

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

  .fidelity-box {
    text-align:center; padding:16px; border-radius:12px;
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
  }
  .info-callout {
    background: #0e2236;
    border-left: 3px solid #64b5f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #c8d8e8;
    line-height: 1.6;
    margin: 12px 0;
  }
  .info-callout b { color: #64b5f6; }
  .preset-label {
    font-size: 11px; color: #7a9bbf; margin-bottom: 4px; text-transform: uppercase;
    letter-spacing: 0.06em;
  }
</style>
""", unsafe_allow_html=True)

st.title("⚛ Quantum Teleportation Network")
st.caption("An undergraduate project simulation")

_anim_path = Path(__file__).with_name("quantum_flow_animation.html")
with st.expander("Data-flow animation — encode, decode, and eavesdropper", expanded=False):
    st.caption(
        "This is `quantum_flow_animation.html`. Turn on **Eavesdropper peeks** to watch Eve "
        "measure a qubit in transit: Bob still runs the 2-bit decode sequence, but it fails "
        "because the state already collapsed."
    )
    if _anim_path.exists():
        st.components.v1.html(_anim_path.read_text(encoding="utf-8"), height=820, scrolling=True)
    else:
        st.error("quantum_flow_animation.html was not found next to app.py.")

# ─────────────────────────────────────────────
# SESSION STATE — persist simulation results
# ─────────────────────────────────────────────

if "sim_counts" not in st.session_state:
    st.session_state.sim_counts = None
if "sim_params" not in st.session_state:
    st.session_state.sim_params = None

# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────

st.sidebar.header("Qubit State Controls")
st.sidebar.markdown("Define Alice's unknown qubit state |ψ⟩ using Bloch sphere angles.")

# Preset state buttons
PRESETS = {
    "|0⟩":  (0.0,        0.0),
    "|1⟩":  (float(np.pi), 0.0),
    "|+⟩":  (float(np.pi/2), 0.0),
    "|−⟩":  (float(np.pi/2), float(np.pi)),
    "|i⟩":  (float(np.pi/2), float(np.pi/2)),
}

st.sidebar.markdown('<div class="preset-label">Quick presets</div>', unsafe_allow_html=True)
preset_cols = st.sidebar.columns(len(PRESETS))
for i, (label, (t, p)) in enumerate(PRESETS.items()):
    if preset_cols[i].button(label, width="stretch", key=f"preset_{label}"):
        st.session_state["theta_val"] = t
        st.session_state["phi_val"]   = p

theta = st.sidebar.slider(
    "θ (polar angle)", 0.0, float(np.pi),
    value=st.session_state.get("theta_val", float(np.pi / 3)),
    step=0.05, help="Controls the |0⟩ vs |1⟩ mix. θ=0 → |0⟩, θ=π → |1⟩",
    key="theta_slider",
)
phi = st.sidebar.slider(
    "φ (azimuthal angle)", 0.0, float(2 * np.pi),
    value=st.session_state.get("phi_val", float(np.pi / 4)),
    step=0.05, help="Controls the phase of the superposition",
    key="phi_slider",
)
# sync session state back if sliders moved manually
st.session_state["theta_val"] = theta
st.session_state["phi_val"]   = phi

st.sidebar.divider()
st.sidebar.header("Noise Model")
noise_on = st.sidebar.toggle("Enable decoherence", value=False)
noise_rate = st.sidebar.slider(
    "Error rate per gate", 0.001, 0.15, 0.05, step=0.005,
    disabled=not noise_on,
    help="Depolarizing noise probability per gate operation",
)

st.sidebar.divider()
st.sidebar.header("Network Hops")
num_hops = st.sidebar.selectbox(
    "Number of relay nodes", [1, 2, 3], index=0,
    help="More hops = more fidelity loss",
)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", width="stretch")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS  (cached for performance)
# ─────────────────────────────────────────────

def build_teleportation_circuit(theta, phi):
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])
    qc.barrier(label="|ψ⟩")
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier(label="entangle")
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="measure")
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    x_body = QuantumCircuit(1); x_body.x(0)
    qc.if_else((c[1], 1), x_body, None, [q[2]], [])
    z_body = QuantumCircuit(1); z_body.z(0)
    qc.if_else((c[0], 1), z_body, None, [q[2]], [])
    return qc


def build_noise_model(rate):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 1), ['h', 'ry', 'rz', 'x', 'z'])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate * 2, 2), ['cx'])
    return nm


@st.cache_data(show_spinner=False)
def simulate(theta, phi, noise_on, noise_rate):
    qc = build_teleportation_circuit(theta, phi)
    sim = AerSimulator(method='statevector')
    kwargs = {}
    if noise_on:
        kwargs['noise_model'] = build_noise_model(noise_rate)
    return sim.run(qc, shots=1024, **kwargs).result().get_counts()


def estimate_fidelity_analytic(noise_on, noise_rate, num_hops):
    """
    Analytic fidelity for depolarizing noise through the teleportation circuit.
    Each hop applies ~8 single-qubit gates + 1 two-qubit gate.
    Depolarizing channel: F = 1 - (2/3)*p for single-qubit,
                              1 - (4/5)*p for two-qubit (approximate).
    Combined per-hop fidelity, then raised to num_hops power.
    """
    if not noise_on:
        return 1.0
    # 7 single-qubit gates, 1 CNOT per hop (simplified model)
    f1 = (1 - (2/3) * noise_rate) ** 7
    f2 = (1 - (4/5) * (noise_rate * 2)) ** 1
    per_hop = f1 * f2
    return max(0.0, per_hop ** num_hops)


@st.cache_data(show_spinner=False)
def get_target_statevector(theta, phi):
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return Statevector(qc)


@st.cache_data(show_spinner=False)
def get_bob_state_before_correction(theta, phi, measurement_outcome: str):
    bit_c0 = int(measurement_outcome[1])
    bit_c1 = int(measurement_outcome[0])
    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)
    if   bit_c1 == 0 and bit_c0 == 0: coeffs = np.array([alpha,  beta ])
    elif bit_c1 == 0 and bit_c0 == 1: coeffs = np.array([alpha, -beta ])
    elif bit_c1 == 1 and bit_c0 == 0: coeffs = np.array([beta,   alpha])
    else:                              coeffs = np.array([-beta,  alpha])
    return Statevector(coeffs / np.linalg.norm(coeffs))


@st.cache_data(show_spinner=False)
def get_bob_state_after_correction(theta, phi, noise_on, noise_rate):
    if not noise_on:
        return get_target_statevector(theta, phi)
    qc = build_teleportation_circuit(theta, phi)
    qc.save_density_matrix(qubits=[2], label='bob_dm')
    sim = AerSimulator(method='density_matrix')
    result = sim.run(qc, noise_model=build_noise_model(noise_rate), shots=1).result()
    return result.data()['bob_dm']


def compute_bloch_vector_from_dm(dm):
    rho = np.array(dm.data)
    return np.array([2*np.real(rho[0,1]), 2*np.imag(rho[1,0]), np.real(rho[0,0]-rho[1,1])])


def get_state_label(theta, phi):
    t = round(theta / np.pi, 2)
    p = round(phi / np.pi, 2)
    return f"|ψ⟩ = cos({t}π/2)|0⟩ + e^(i·{p}π) sin({t}π/2)|1⟩"


def plot_bloch_from_statevector(sv, title_color="#e0eaf4"):
    fig = plot_bloch_multivector(sv)
    fig.set_size_inches(3.0, 3.0)
    fig.patch.set_facecolor('#0d1b2a')
    return fig


def plot_bloch_from_bloch_vector(bvec, title_color="#e0eaf4"):
    from qiskit.visualization.bloch import Bloch
    b = Bloch()
    b.vector_color = ['#f48fb1']
    b.add_vectors(bvec)
    b.render()
    fig = b.fig
    fig.set_size_inches(3.0, 3.0)
    fig.patch.set_facecolor('#0d1b2a')
    return fig


# ─────────────────────────────────────────────
# MAIN LAYOUT — Alice | Circuit | Network
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
    Alice encodes her unknown qubit using <code>Ry(θ)</code> then <code>Rz(φ)</code>,
    placing it anywhere on the Bloch sphere.<br><br>
  <b style='color:#90caf9'>② Create entanglement</b> &nbsp;—&nbsp;
    <code>H</code> puts qubit B into superposition; a <code>CNOT</code> entangles B with C,
    creating a shared Bell pair between Alice and Bob.<br><br>
  <b style='color:#90caf9'>③ Alice's Bell measurement</b> &nbsp;—&nbsp;
    <code>CNOT</code> from A→B then <code>H</code> on A rotates into the Bell basis.
    Alice measures A and B, yielding 2 classical bits (00 / 01 / 10 / 11).<br><br>
  <b style='color:#90caf9'>④ Bob corrects</b> &nbsp;—&nbsp;
    Alice sends 2 bits over a classical channel. Bob applies <code>X</code> if bit₁=1,
    then <code>Z</code> if bit₀=1 — recovering |ψ⟩ exactly.
</div>
""", unsafe_allow_html=True)

with col3:
    st.subheader("Network & Fidelity")
    fidelity = estimate_fidelity_analytic(noise_on, noise_rate, num_hops)
    fid_pct  = round(fidelity * 100, 1)
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
    st.markdown(f"**Network path:** `{' → '.join(nodes)}`")
    for i in range(num_hops):
        hop_fid = max(0, 1 - (noise_rate * 8 if noise_on else 0))
        st.progress(hop_fid, text=f"Hop {i+1}: {nodes[i]} → {nodes[i+1]} | {hop_fid:.0%}")
    if noise_on and fid_pct < 90:
        st.warning("⚠ Fidelity below 90% — quantum error correction may be needed.")
    elif not noise_on:
        st.success("✓ Ideal channel — perfect teleportation (fidelity 100%)")
    else:
        st.success("✓ Good fidelity despite noise — teleportation successful")


# ─────────────────────────────────────────────
# WHAT IS NOT TELEPORTED — misconception callout
# ─────────────────────────────────────────────

st.markdown("""
<div class='info-callout'>
  <b>⚠ Common misconception: nothing physical travels from Alice to Bob.</b><br>
  Quantum teleportation moves the <em>information describing</em> |ψ⟩ — not any particle.
  Alice's original qubit is <em>destroyed</em> by her measurement (no-cloning theorem),
  and Bob reconstructs an identical state using entanglement + 2 classical bits.
  The classical channel limits transfer to at most the speed of light — no FTL communication.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROTOCOL SUMMARY TABLE — always visible
# ─────────────────────────────────────────────

st.divider()
st.subheader("Protocol Summary")
summary_data = {
    "Parameter": ["Input state θ", "Input state φ", "Noise enabled", "Error rate",
                  "Hops", "Estimated fidelity (analytic)", "Classical bits sent"],
    "Value": [f"{theta:.3f} rad", f"{phi:.3f} rad",
              "Yes" if noise_on else "No",
              f"{noise_rate:.1%}" if noise_on else "0%",
              str(num_hops), f"{fid_pct}%", "2 bits per teleportation"]
}
st.table(summary_data)


# ─────────────────────────────────────────────
# BOB'S STATE — BEFORE & AFTER CORRECTION
# ─────────────────────────────────────────────

st.divider()
st.subheader("🔭 Bob's Qubit: Before vs After Correction")
st.caption(
    "After Alice's Bell measurement, Bob's qubit collapses to one of 4 states depending on "
    "the 2 classical bits. The correction gates rotate his qubit back to match Alice's original state."
)

outcomes_to_show = ['00', '01', '10', '11']
outcome_labels = {
    '00': 'No correction needed',
    '01': 'Z gate applied',
    '10': 'X gate applied',
    '11': 'X then Z gate applied',
}
correction_desc = {'00': 'None', '01': 'Z gate', '10': 'X gate', '11': 'X + Z gates'}

sv_alice     = get_target_statevector(theta, phi)
bob_after_dm = get_bob_state_after_correction(theta, phi, noise_on, noise_rate)

# Compute fidelity after correction once
if noise_on:
    rho_target = DensityMatrix(sv_alice)
    fid_after  = state_fidelity(rho_target, bob_after_dm)
    fid_after_str = f"{fid_after:.2%}"
else:
    fid_after_str = "100.00%"

# Header row
h0, h1, h2, h3, h4 = st.columns([0.55, 1, 0.12, 1, 0.4])
with h1:
    st.markdown("<div style='text-align:center'><span class='correction-badge badge-before'>⚡ BEFORE CORRECTION</span></div>", unsafe_allow_html=True)
with h3:
    st.markdown("<div style='text-align:center'><span class='correction-badge badge-after'>✓ AFTER CORRECTION</span></div>", unsafe_allow_html=True)
with h4:
    st.markdown("<div style='text-align:center;font-size:11px;color:#7a9bbf;padding-top:4px'>All 4 outcomes converge<br>to the same corrected state</div>", unsafe_allow_html=True)

for outcome in outcomes_to_show:
    sv_before = get_bob_state_before_correction(theta, phi, outcome)
    dot       = abs(np.dot(sv_before.data.conj(), sv_alice.data)) ** 2

    row_label, before_col, arrow_col, after_col, note_col = st.columns([0.55, 1, 0.12, 1, 0.4])

    with row_label:
        st.markdown(
            f"""<div style='padding-top:36px;font-size:12px;color:#7a9bbf;line-height:1.7'>
              <div style='font-family:monospace;font-size:16px;color:#64b5f6;font-weight:700'>Outcome {outcome}</div>
              <div>{outcome_labels[outcome]}</div>
              <div style='margin-top:4px'><b style='color:#c8d8e8'>Gates:</b> {correction_desc[outcome]}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with before_col:
        fig_before = plot_bloch_from_statevector(sv_before, title_color="#f48fb1")
        for ax in fig_before.axes: ax.set_title("")
        st.pyplot(fig_before, width="stretch")
        plt.close()
        st.markdown(
            f"<div style='text-align:center;font-size:11px;color:#f48fb1'>Fidelity with Alice: <b>{dot:.2%}</b></div>",
            unsafe_allow_html=True
        )

    with arrow_col:
        st.markdown("<div style='text-align:center;font-size:22px;padding-top:56px;color:#64b5f6'>→</div>", unsafe_allow_html=True)

    with after_col:
        # Show the single corrected state (same for all outcomes)
        if not noise_on:
            fig_after = plot_bloch_from_statevector(sv_alice, title_color="#81c784")
        else:
            bvec      = compute_bloch_vector_from_dm(bob_after_dm)
            fig_after = plot_bloch_from_bloch_vector(bvec, title_color="#81c784")
        for ax in fig_after.axes: ax.set_title("")
        st.pyplot(fig_after, width="stretch")
        plt.close()
        st.markdown(
            f"<div style='text-align:center;font-size:11px;color:#81c784'>Fidelity with Alice: <b>{fid_after_str}</b></div>",
            unsafe_allow_html=True
        )

    with note_col:
        if outcome == '00':
            note = "Bob already holds Alice's state — no action needed."
        elif outcome == '01':
            note = "A Z gate flips the phase of |1⟩, restoring the correct sign."
        elif outcome == '10':
            note = "An X gate (bit-flip) swaps the amplitudes of |0⟩ and |1⟩."
        else:
            note = "X swaps amplitudes first, then Z fixes the phase — full recovery."
        st.markdown(
            f"<div style='padding-top:38px;font-size:11px;color:#7a9bbf;line-height:1.6'>{note}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0'>", unsafe_allow_html=True)

st.caption(
    "**Key insight:** Regardless of the measurement outcome, Bob's corrected state always matches "
    "Alice's |ψ⟩. The classical bits tell Bob *which* correction to apply — not *what* the state is."
)


# ─────────────────────────────────────────────
# SIMULATION RESULTS — persisted via session state
# ─────────────────────────────────────────────

if run_btn:
    with st.spinner("Running quantum simulation…"):
        counts = simulate(theta, phi, noise_on, noise_rate)
    st.session_state.sim_counts = counts
    st.session_state.sim_params = dict(theta=theta, phi=phi, noise_on=noise_on,
                                       noise_rate=noise_rate, num_hops=num_hops, fid_pct=fid_pct)

if st.session_state.sim_counts is not None:
    p = st.session_state.sim_params
    counts = st.session_state.sim_counts

    st.divider()
    st.subheader("Simulation Results")
    if p["theta"] != theta or p["phi"] != phi or p["noise_on"] != noise_on:
        st.info("ℹ Parameters changed since last run — press ▶ Run Simulation to refresh.")

    rcol1, rcol2 = st.columns(2)

    with rcol1:
        st.markdown("**Measurement outcomes (1024 shots)**")
        st.caption("Alice's 2 classical bits sent to Bob. Even distribution confirms correct teleportation.")
        fig_counts, ax = plt.subplots(figsize=(5, 3))
        outcomes_k = sorted(counts.keys())
        values     = [counts[k] for k in outcomes_k]
        bar_color  = '#1D9E75' if p["noise_on"] else '#185FA5'
        bars = ax.bar(outcomes_k, values, color=bar_color, alpha=0.8, edgecolor='none')
        ax.set_xlabel("Measurement outcome", fontsize=11)
        ax.set_ylabel("Count (of 1024 shots)", fontsize=11)
        ax.set_title("Bell Measurement Distribution")
        ax.axhline(256, color='#aaa', linestyle='--', linewidth=0.8, label='Expected (256)')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+8, str(val), ha='center', fontsize=10)
        st.pyplot(fig_counts, width="stretch")
        plt.close()

    with rcol2:
        st.markdown("**Noise vs. Fidelity curve (analytic)**")
        st.caption("Derived from the depolarizing channel model — more accurate than a linear approximation.")
        noise_range    = np.linspace(0.001, 0.15, 100)
        fidelity_curve = [estimate_fidelity_analytic(True, n, p["num_hops"]) for n in noise_range]
        fig_nf, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(noise_range * 100, fidelity_curve, color='#185FA5', linewidth=2)
        ax2.fill_between(noise_range * 100, fidelity_curve, alpha=0.12, color='#185FA5')
        ax2.axhline(0.9, color='#E85D24', linestyle='--', alpha=0.7, linewidth=1.2, label='90% threshold')
        if p["noise_on"]:
            ax2.axvline(p["noise_rate"]*100, color='#1D9E75', linestyle=':', linewidth=1.5,
                        label=f'Current ({p["noise_rate"]*100:.1f}%) → {p["fid_pct"]}%')
        ax2.set_xlabel("Error Rate (%)")
        ax2.set_ylabel("Fidelity")
        ax2.set_title(f"Decoherence Impact ({p['num_hops']} hop{'s' if p['num_hops']>1 else ''})")
        ax2.legend(fontsize=9)
        st.pyplot(fig_nf, width="stretch")
        plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Qiskit & Streamlit | Quantum Networking Project")
