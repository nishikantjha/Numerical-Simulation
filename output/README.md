## Summary of Numerical Simulation: Nonlinear Wave Equation

### 1. Numerical Approach

I implemented the **Symmetric Split-Step Fourier Method (SSFM)** to solve the **Nonlinear Schrödinger Equation**. This method was selected for its superior efficiency *(O(N log N))* and stability compared to finite difference methods:

- **Operator Splitting:**  
  The problem is split into a linear dispersive step (solved in the Fourier domain) and a nonlinear step (solved in the time domain).

- **Symmetry (Strang Splitting):**  
  I used the sequence  
  `Linear(dt/2) → Nonlinear(dt) → Linear(dt/2)`  
  to ensure 2nd-order accuracy in time *(O(dt²))*.

- **Vectorization:**  
  The solver is fully vectorized using **NumPy**, achieving simulation times of **< 0.1 seconds** for 1000 steps on a 1024-point grid.

---

### 2. Parameter Choices

- **Grid Size (N = 1024):**  
  A power of 2 was chosen to optimize FFT performance. This provides sufficient resolution for the Gaussian pulse without aliasing.

- **Domain (T = 40):**  
  The window is large enough that the pulse tails decay to machine zero before the boundaries, preventing periodic boundary artifacts.

- **Time Step (dt = 0.01):**  
  Sufficiently small to minimize the commutation error between the linear and nonlinear operators.

---

### 3. Results of Qualitative Tests (All 3 cases chosen had PHYSICAL difference)

#### Case 1: Remove the Nonlinear Term

- **Modification:**  
  `enable_nonlinearity = False`

- **Qualitative Expectation:**  
  Without the nonlinear term, the equation becomes a linear dispersive equation (similar to how a wave packet spreads out in quantum mechanics). The pulse should broaden (spread out) over time, and its peak height should decrease to conserve total energy.

- **Observation (from `evolution_no_nonlinearity.png`):**  
  The heatmap shows the bright central spot widening as time increases.  
  The *Pulse Profile Comparison* confirms that the final orange pulse is significantly wider and shorter than the initial blue pulse.  
  The output logs from the command line confirm the simulation completed successfully with **Final Power: 1.7725**, identical to the initial power.

- **Conclusion:**  
  The difference is **Physical**. This is the natural effect of dispersion, where different frequency components travel at different speeds, causing the wave packet to spread.

---

#### Case 2: Remove the Dispersive Term

- **Modification:**  
  `enable_dispersion = False`

- **Qualitative Expectation:**  
  Without dispersion, there is no mechanism for the pulse to move or spread spatially.  
  The nonlinearity only affects the **phase** of the complex field, not its magnitude.  
  Therefore, the intensity profile `|u(τ)|²` should remain exactly frozen in place.

- **Observation (from `evolution_no_dispersion.png`):**  
  The heatmap is a perfectly straight vertical line.  
  The *Pulse Profile Comparison* shows the orange line (Final) sitting exactly on top of the blue line (Initial).

- **Conclusion:**  
  The difference is **Physical**. The system is undergoing *Self-Phase Modulation*, where the wave rotates in the complex plane but does not change shape.

---

#### Case 3: Add a Linear Loss Term

- **Modification:**  
  `linear_loss = 0.05`

- **Qualitative Expectation:**  
  This term represents energy dissipation (like friction).  
  The pulse shape might evolve similarly to the baseline case, but the overall brightness (intensity) should fade away exponentially.

- **Observation (from `evolution_with_loss.png`):**  
  The heatmap starts bright but fades into darkness at the bottom.  
  The final pulse profile is significantly smaller in amplitude compared to the baseline simulation.  
  The output logs show a massive drop in power from **1.7725** to **0.6520**.

- **Conclusion:**  
  The difference is **Physical**. Energy is being removed from the system.  
  The loss term is correctly dissipating energy.

---

### 4. Numerical Issues & Diagnostic Checks

- **Conservation:**  
  The diagnostic check confirmed that **Power (P)** was conserved with **0.000000% deviation** in all conservative test cases.

- **Deviation in Loss Case:**  
  The observed ~63% deviation in the loss case is expected and confirms that the code correctly implements non-conservative physics.

- **Boundary Handling:**  
  No numerical reflections were observed at the boundaries, confirming that the window size **T = 40** was sufficient for the chosen pulse width.