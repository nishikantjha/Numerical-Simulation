import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from dataclasses import dataclass
from typing import List
import time

# 1. Setup Logging
# We use handlers to send logs to both the terminal and a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("simulation.log", mode='w'), # 'w' overwrites the file each run
        logging.StreamHandler()                          # Prints to console
    ]
)
logger = logging.getLogger(__name__)

# 2. Configuration Management (FAIR Principles: Explicit Parameters)
@dataclass
class SimulationConfig:
    """
    Configuration dataclass to ensure reproducibility and easy parameter tuning.
    """
    # Grid parameters
    N: int = 1024           # Number of grid points (Power of 2 for FFT efficiency)
    T_window: float = 40.0  # Domain length [-T/2, T/2]
    
    # Time stepping
    dt: float = 0.01        # Time step size
    steps: int = 1000       # Number of time steps
    
    # Physics toggles (for Qualitative Tests)
    enable_dispersion: bool = True
    enable_nonlinearity: bool = True
    linear_loss: float = 0.0  # Coefficient for linear loss term
    
    # Initial Condition parameters
    pulse_amplitude: float = 1.0
    pulse_width: float = 1.0

    @property
    def tau_grid(self) -> np.ndarray:
        """Generates the spatial grid (tau)."""
        return np.linspace(-self.T_window/2, self.T_window/2, self.N, endpoint=False)

    @property
    def omega_grid(self) -> np.ndarray:
        """Generates the frequency grid (omega) for FFT."""
        # standard FFT frequency ordering: [0, 1, ..., N/2-1, -N/2, ..., -1] * (2pi/T)
        return np.fft.fftfreq(self.N, d=self.T_window/self.N) * 2 * np.pi


class NLSESolver:
    """
    A numerical solver for the Nonlinear Schrödinger Equation using the
    Symmetric Split-Step Fourier Method.
    
    Equation: du/dt = i * d^2u/dtau^2 + i * |u|^2 * u
    """
    
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.tau = config.tau_grid
        self.omega = config.omega_grid
        
        # State variable (Complex field)
        self.u: np.ndarray = np.zeros(self.cfg.N, dtype=np.complex128)
        
        # Pre-compute Linear Dispersion Operator (Optimization)
        # Operator in Fourier space: D = -i * omega^2
        # Half-step propagator: exp(D * dt/2) = exp(-i * omega^2 * dt/2)
        if self.cfg.enable_dispersion:
            # Note: The equation is i * d^2u/dtau^2. FT(d^2/dtau^2) -> -omega^2.
            # So linear term is -i * omega^2.
            self.dispersion_op = np.exp(1j * (-self.omega**2) * (self.cfg.dt / 2))
        else:
            self.dispersion_op = np.ones_like(self.omega, dtype=np.complex128)

        # History for visualization
        self.history_u: List[np.ndarray] = []
        self.history_t: List[float] = []
        self.power_log: List[float] = []

    def set_initial_condition(self):
        """Sets a Gaussian pulse initial condition."""
        # Gaussian: A * exp(-tau^2 / (2*width^2))
        self.u = self.cfg.pulse_amplitude * np.exp(-(self.tau**2) / (2 * self.cfg.pulse_width**2))
        logger.info("Initial condition set: Gaussian Pulse")

    def _linear_step(self):
        """Performs half-step linear dispersion in Fourier domain."""
        u_hat = np.fft.fft(self.u)
        u_hat *= self.dispersion_op
        self.u = np.fft.ifft(u_hat)

    def _nonlinear_step(self):
        """Performs full-step nonlinearity in Real domain."""
        if self.cfg.enable_nonlinearity:
            # Operator: N = i * |u|^2
            # Propagator: exp(i * |u|^2 * dt)
            nonlinear_phase = np.exp(1j * (np.abs(self.u)**2) * self.cfg.dt)
            self.u *= nonlinear_phase
            
        # Handle Linear Loss if present (Task 3.5)
        if self.cfg.linear_loss > 0:
            # Decay: exp(-loss * dt)
            self.u *= np.exp(-self.cfg.linear_loss * self.cfg.dt)

    def calculate_power(self) -> float:
        """
        Calculates the conserved quantity P = integral(|u|^2 dtau).
        Uses Trapezoidal rule for integration.
        """
        intensity = np.abs(self.u)**2
        dtau = self.cfg.T_window / self.cfg.N
        return np.sum(intensity) * dtau

    def evolve(self):
        """Main evolution loop using Symmetric Split-Step."""
        start_time = time.time()
        
        # Save initial state
        self.history_u.append(self.u.copy())
        self.history_t.append(0.0)
        self.power_log.append(self.calculate_power())

        for step in range(1, self.cfg.steps + 1):
            # Symmetric Strang Splitting: Linear(dt/2) -> Nonlinear(dt) -> Linear(dt/2)
            
            # 1. First Linear Half-Step
            self._linear_step()
            
            # 2. Nonlinear Full-Step
            self._nonlinear_step()
            
            # 3. Second Linear Half-Step
            self._linear_step()
            
            # Logging and Storage
            if step % (self.cfg.steps // 20) == 0:  # Store 20 snapshots
                self.history_u.append(self.u.copy())
                self.history_t.append(step * self.cfg.dt)
                
            # Diagnostic Check (Power Conservation)
            current_power = self.calculate_power()
            self.power_log.append(current_power)

        elapsed = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed:.4f}s. Final Power: {self.power_log[-1]:.4f}")

    def plot_results(self, title_suffix: str = ""):
        """Generates visualization of the evolution."""
        history_arr = np.array(self.history_u)
        intensity_map = np.abs(history_arr)**2
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Space-Time Heatmap
        # Using extent to map array indices to physical units
        extent = [-self.cfg.T_window/2, self.cfg.T_window/2, self.history_t[-1], 0]
        im = ax[0].imshow(intensity_map, aspect='auto', extent=extent, cmap='inferno')
        ax[0].set_title(f"Field Intensity Evolution {title_suffix}")
        ax[0].set_xlabel("Tau (Time Delay)")
        ax[0].set_ylabel("Time (Evolution)")
        plt.colorbar(im, ax=ax[0], label="|u|^2")

        # Plot 2: Initial vs Final Pulse
        ax[1].plot(self.tau, np.abs(self.history_u[0])**2, label="Initial (t=0)", linestyle='--')
        ax[1].plot(self.tau, np.abs(self.history_u[-1])**2, label=f"Final (t={self.history_t[-1]:.1f})")
        ax[1].set_title("Pulse Profile Comparison")
        ax[1].set_xlabel("Tau")
        ax[1].set_ylabel("Intensity")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"output/evolution_{title_suffix.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        logger.info(f"Plot saved to {filename}")
        plt.close() # Close to free memory

    def check_conservation(self):
        """Logs the deviation in power conservation."""
        p_initial = self.power_log[0]
        p_final = self.power_log[-1]
        deviation = abs(p_final - p_initial) / p_initial * 100
        logger.info(f"Power Conservation Deviation: {deviation:.6f}%")


def run_qualitative_tests():
    """
    Runs the scenarios requested in Task 3.
    """
    import os
    if not os.path.exists('output'):
        os.makedirs('output')

    # 1. Standard Run (Baseline)
    logger.info("--- Running Baseline Simulation ---")
    cfg_base = SimulationConfig()
    solver = NLSESolver(cfg_base)
    solver.set_initial_condition()
    solver.evolve()
    solver.plot_results("Baseline")
    solver.check_conservation()

    # 2. Remove Nonlinearity (Task 3.1)
    logger.info("--- Test: No Nonlinearity ---")
    cfg_linear = SimulationConfig(enable_nonlinearity=False)
    solver = NLSESolver(cfg_linear)
    solver.set_initial_condition()
    solver.evolve()
    solver.plot_results("No Nonlinearity")
    solver.check_conservation()
    # Expectation: Pulse spreads due to dispersion, amplitude decreases, no shape change.

    # 3. Remove Dispersion (Task 3.2)
    logger.info("--- Test: No Dispersion ---")
    cfg_nodisp = SimulationConfig(enable_dispersion=False)
    solver = NLSESolver(cfg_nodisp)
    solver.set_initial_condition()
    solver.evolve()
    solver.plot_results("No Dispersion")
    solver.check_conservation()
    # Expectation: Pulse shape |u|^2 remains constant (frozen), only phase changes.

    # 4. Add Linear Loss (Task 3.5)
    logger.info("--- Test: Linear Loss ---")
    cfg_loss = SimulationConfig(linear_loss=0.05)
    solver = NLSESolver(cfg_loss)
    solver.set_initial_condition()
    solver.evolve()
    solver.plot_results("With Loss")
    solver.check_conservation() # Should NOT be conserved

if __name__ == "__main__":
    # CLI for flexibility
    parser = argparse.ArgumentParser(description="NLSE Solver for RSE Interview")
    parser.add_argument("--test", action="store_true", help="Run qualitative test suite")
    args = parser.parse_args()

    if args.test:
        run_qualitative_tests()
    else:
        # Default run
        run_qualitative_tests()