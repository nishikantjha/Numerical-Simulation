import pytest
import numpy as np
from solution import SimulationConfig, NLSESolver

def test_config_initialization():
    """Test that configuration defaults are reasonable."""
    cfg = SimulationConfig()
    assert cfg.N == 1024
    assert cfg.dt == 0.01
    assert len(cfg.tau_grid) == 1024

def test_power_conservation_baseline():
    """
    Physics Test: In a standard NLSE system, Power must be conserved.
    We allow a small numerical error tolerance.
    """
    cfg = SimulationConfig(steps=100)
    solver = NLSESolver(cfg)
    solver.set_initial_condition()
    
    initial_power = solver.calculate_power()
    solver.evolve()
    final_power = solver.calculate_power()
    
    # Check if deviation is less than 0.1%
    deviation = abs(final_power - initial_power) / initial_power
    assert deviation < 0.001, f"Power conservation violated! Deviation: {deviation}"

def test_no_dispersion_intensity():
    """
    Physics Test: Without dispersion, the intensity profile |u|^2 should not change,
    only the phase changes.
    """
    cfg = SimulationConfig(enable_dispersion=False, steps=50)
    solver = NLSESolver(cfg)
    solver.set_initial_condition()
    
    initial_intensity = np.abs(solver.u)**2
    solver.evolve()
    final_intensity = np.abs(solver.u)**2
    
    # Check that intensity profile is identical
    np.testing.assert_allclose(initial_intensity, final_intensity, atol=1e-5)

def test_loss_decay():
    """
    Physics Test: With loss, power should decrease.
    """
    cfg = SimulationConfig(linear_loss=0.1, steps=50)
    solver = NLSESolver(cfg)
    solver.set_initial_condition()
    
    initial_power = solver.calculate_power()
    solver.evolve()
    final_power = solver.calculate_power()
    
    assert final_power < initial_power, "Power did not decay despite loss term."