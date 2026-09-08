# Quantum Teleportation Network — Presentation Outline
**Team of 10 | Undergraduate Project**

---

## Slide 1 — Title Slide
**Quantum Teleportation: From Theory to Network Simulation**
- Team names & roles
- Department / course name
- Date

---

## Slide 2 — What is Quantum Teleportation? (1 min)
**Hook:** "We can send the complete description of a quantum state — without it ever physically travelling."

Key points to cover:
- Not science fiction — proven experimentally since 1997 (Zeilinger et al.)
- Not faster-than-light — always needs a classical channel too
- Transmits quantum *state*, not matter or information beyond classical limits
- Foundation of the quantum internet

---

## Slide 3 — Why Does This Matter? (1 min)
**Real-world relevance:**
- Quantum Key Distribution (QKD) — unbreakable encryption
- Distributed quantum computing — linking quantum processors
- Quantum sensing networks — ultra-precise measurements
- Long-distance quantum communication

---

## Slide 4 — The Ingredients (2 min)
**Three things needed for teleportation:**

1. **Entanglement** — A shared Bell pair between Alice and Bob
   - Show Bell state equation: |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
   - Visual: two correlated coins that always land opposite

2. **Quantum measurement** — Alice's Bell measurement on her qubit + the unknown state
   - Collapses the entangled pair
   - Produces 2 classical bits

3. **Classical communication** — Sending those 2 bits to Bob
   - This is the *only* signal — so no FTL
   - Bob uses bits to apply a correction gate

---

## Slide 5 — The Protocol Step by Step (3 min)
**Walk through the 4 steps with diagrams:**

```
Step 1: Generate Bell pair (B, C) — shared between Alice and Bob
Step 2: Alice performs Bell measurement on (A, B)
         → collapses state, produces 2 classical bits
Step 3: Alice sends bits over classical channel (phone, internet, etc.)
Step 4: Bob applies correction gate to C
         → C now holds the exact state |ψ⟩
```

**Key insight to emphasize:** "Alice's qubit A is destroyed in measurement. But |ψ⟩ is perfectly reconstructed at Bob's C. This respects the no-cloning theorem."

---

## Slide 6 — Live Circuit Demo (5 min) ← Qiskit demo here
**[Switch to terminal / live code]**

Run `quantum_teleportation.py` and show:
- Circuit diagram output
- Bloch sphere of Alice's state
- Measurement outcome distribution
- "Now watch what happens when we add noise..."

Talking points:
- Explain each barrier in the circuit
- Point out the classical register (the 2 bits)
- Show the conditional gates (if_else) — Bob's correction

---

## Slide 7 — Interactive Demo (5 min) ← Streamlit app here
**[Switch to browser: `streamlit run app.py`]**

Walk through the UI:
1. Drag θ and φ sliders — watch the Bloch sphere update
2. Toggle noise OFF → fidelity = 100%
3. Toggle noise ON → fidelity drops
4. Increase error rate → show the fidelity curve
5. Add more hops → cumulative fidelity decreases
6. Run simulation → show measurement histogram

**Key talking point:** "Each extra hop multiplies the fidelity loss. A 3-hop network with 5% per-gate noise drops to ~50% fidelity — this is why quantum repeaters are a major research problem."

---

## Slide 8 — Multi-Hop Network & Real Challenges (2 min)
**From lab to network:**

| Challenge | Why it matters |
|---|---|
| Decoherence | Quantum states decay — ~100ms in superconducting qubits |
| Quantum memory | Need to store qubits while waiting for classical signals |
| Entanglement swapping | Extending entanglement across relay nodes |
| Error correction | Surface codes, stabilizer codes — expensive overhead |
| Distance | China's Micius satellite: 1,200 km QKD in 2020 |

---

## Slide 9 — Our Simulation Architecture (1 min)
**What we built:**

```
quantum_teleportation.py   ← Core Qiskit simulation engine
app.py                     ← Streamlit interactive web frontend
```

- Noise model: depolarizing error on all gates
- Multi-hop: sequential teleportation with cumulative fidelity tracking
- Visualizations: Bloch sphere, circuit diagram, noise-fidelity curve
- Tech stack: Qiskit 1.x, Qiskit-Aer, Streamlit, Matplotlib

---

## Slide 10 — Results & Takeaways (1 min)
**What our simulation shows:**

- Ideal teleportation: 100% fidelity, any qubit state correctly transmitted
- With 5% gate error: fidelity drops to ~60% over 3 hops
- Even 1% error rate causes measurable degradation over networks
- Classical communication is the bottleneck — not the quantum channel

**What we learned:** Quantum error correction is not optional for real networks — it's the central engineering challenge.

---

## Slide 11 — References & Further Reading
- Nielsen & Chuang, *Quantum Computation and Quantum Information* (2000)
- Bouwmeester et al., "Experimental quantum teleportation", *Nature* (1997)
- IBM Quantum / Qiskit documentation: qiskit.org
- Wehner, Elkouss, Hanson, "Quantum internet: A vision for the road ahead", *Science* (2018)
- China's satellite QKD: Liao et al., *Nature* (2017)

---

## Slide 12 — Q&A
**Anticipated questions to prepare for:**

- *"Does teleportation break the no-cloning theorem?"*
  → No — Alice's original is destroyed in measurement. No copy is ever made.

- *"Why can't we use this for FTL communication?"*
  → Bob can't extract any information from C until he gets Alice's 2 bits classically.

- *"How is this different from classical encryption?"*
  → The quantum state itself is transmitted — not just a key. The security comes from measurement collapse, not computational hardness.

- *"What hardware runs actual quantum teleportation?"*
  → Superconducting qubits (IBM, Google), trapped ions (IonQ, Quantinuum), photonic systems (for long-distance).

---

## Presentation Tips

- **Total time:** ~20–25 minutes including demo
- **Speaker assignments:** Assign one speaker per section; all 10 members should speak
- **Demo prep:** Test `streamlit run app.py` on presentation laptop the day before
- **Backup:** Export Bloch sphere + circuit as PNG images in case live demo fails
- **Slide design:** Use dark background with quantum-themed colors (teal, purple) — avoid generic blue gradients

---

*End of outline*
