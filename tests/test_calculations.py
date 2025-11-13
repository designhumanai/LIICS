#!/usr/bin/env python3
"""
Unit Tests for LIICS Framework Calculations
============================================
Validates numerical accuracy of all core calculations:
- Ψ_LLM computation
- D_max predictions
- Statistical measures (mean, σ, 95% CI)
- Sensitivity analysis
- Decomposition consistency

Based on: "An Empirical Invariant for Transformer Scaling: 
          Towards an Information Incompleteness Hypothesis"
Author: Viktor N. Savitskiy (ORCID: 0000-0003-1356-7260)
Version: 2.1 (November 2025)
License: MIT

Usage:
    pytest tests/test_calculations.py -v
    python -m pytest tests/test_calculations.py --cov
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import calculation functions
try:
    from compute_psi_canonical import (
        compute_psi, compute_k, compute_c, 
        compute_statistics, MODELS, PSI_CANONICAL
    )
    from sensitivity_analysis import calculate_psi_grid
    from predict_future_models import (
        predict_dmax, predict_with_uncertainty,
        compute_architectural_efficiency,
        VALIDATED_MODELS
    )
except ImportError as e:
    pytest.skip(f"Cannot import modules: {e}", allow_module_level=True)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def canonical_params():
    """Standard normalization parameters."""
    return {
        "H": 2.0,
        "V": 1e6
    }

@pytest.fixture
def chinchilla_params():
    """Chinchilla model parameters (ground truth)."""
    return {
        "N": 70e9,
        "D": 1.40e12,
        "L": 80,
        "E": 8192
    }

@pytest.fixture
def palm_params():
    """PaLM model parameters."""
    return {
        "N": 540e9,
        "D": 0.78e12,
        "L": 118,
        "E": 18432
    }

@pytest.fixture
def llama_params():
    """LLaMA-65B model parameters (corrected)."""
    return {
        "N": 65.2e9,
        "D": 1.40e12,  # CORRECTED from 1.0T
        "L": 80,
        "E": 8192
    }

# ============================================================================
# TEST: PSI COMPUTATION
# ============================================================================

class TestPsiComputation:
    """Test Ψ_LLM calculation accuracy."""
    
    def test_chinchilla_psi(self, chinchilla_params, canonical_params):
        """Chinchilla Ψ_LLM should be 1.34×10⁻¹¹."""
        psi = compute_psi(
            chinchilla_params["N"],
            chinchilla_params["D"],
            chinchilla_params["L"],
            chinchilla_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        expected = 1.34e-11
        assert np.isclose(psi, expected, rtol=0.01), \
            f"Chinchilla Ψ = {psi:.4e}, expected {expected:.4e}"
    
    def test_palm_psi(self, palm_params, canonical_params):
        """PaLM Ψ_LLM should be 1.03×10⁻¹¹."""
        psi = compute_psi(
            palm_params["N"],
            palm_params["D"],
            palm_params["L"],
            palm_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        expected = 1.03e-11
        assert np.isclose(psi, expected, rtol=0.01), \
            f"PaLM Ψ = {psi:.4e}, expected {expected:.4e}"
    
    def test_llama_psi_corrected(self, llama_params, canonical_params):
        """LLaMA-65B Ψ_LLM should be 1.44×10⁻¹¹ (with D=1.4T correction)."""
        psi = compute_psi(
            llama_params["N"],
            llama_params["D"],
            llama_params["L"],
            llama_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        expected = 1.44e-11
        assert np.isclose(psi, expected, rtol=0.01), \
            f"LLaMA-65B Ψ = {psi:.4e}, expected {expected:.4e}"
    
    def test_psi_positive(self, chinchilla_params, canonical_params):
        """Ψ_LLM must be positive."""
        psi = compute_psi(
            chinchilla_params["N"],
            chinchilla_params["D"],
            chinchilla_params["L"],
            chinchilla_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        assert psi > 0, "Ψ_LLM must be positive"
    
    def test_psi_dimensionless(self, chinchilla_params, canonical_params):
        """Ψ_LLM should be dimensionless (order of magnitude check)."""
        psi = compute_psi(
            chinchilla_params["N"],
            chinchilla_params["D"],
            chinchilla_params["L"],
            chinchilla_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        # For compute-optimal models, Ψ should be O(10⁻¹¹)
        assert 1e-12 < psi < 1e-10, \
            f"Ψ_LLM = {psi:.4e} outside expected range [1e-12, 1e-10]"

# ============================================================================
# TEST: DECOMPOSITION CONSISTENCY
# ============================================================================

class TestDecomposition:
    """Test Ψ_LLM = k·L·E decomposition."""
    
    def test_decomposition_chinchilla(self, chinchilla_params, canonical_params):
        """Verify Ψ = k·L·E for Chinchilla."""
        psi = compute_psi(
            chinchilla_params["N"],
            chinchilla_params["D"],
            chinchilla_params["L"],
            chinchilla_params["E"],
            canonical_params["H"],
            canonical_params["V"]
        )
        k = compute_k(
            chinchilla_params["N"],
            chinchilla_params["D"],
            canonical_params["H"],
            canonical_params["V"]
        )
        psi_reconstructed = k * chinchilla_params["L"] * chinchilla_params["E"]
        
        assert np.isclose(psi, psi_reconstructed, rtol=1e-10), \
            f"Decomposition failed: Ψ={psi:.6e} ≠ k·L·E={psi_reconstructed:.6e}"
    
    def test_decomposition_all_models(self, canonical_params):
        """Verify decomposition for all models."""
        for name, params in MODELS.items():
            psi = compute_psi(
                params["N"], params["D"], params["L"], params["E"],
                canonical_params["H"], canonical_params["V"]
            )
            k = compute_k(
                params["N"], params["D"],
                canonical_params["H"], canonical_params["V"]
            )
            psi_reconstructed = k * params["L"] * params["E"]
            
            assert np.isclose(psi, psi_reconstructed, rtol=1e-10), \
                f"{name}: Ψ={psi:.6e} ≠ k·L·E={psi_reconstructed:.6e}"
    
    def test_c_coefficient_range(self, chinchilla_params):
        """c coefficient should be in range [10, 15] for standard Transformers."""
        c = compute_c(
            chinchilla_params["N"],
            chinchilla_params["L"],
            chinchilla_params["E"]
        )
        assert 10 <= c <= 15, \
            f"c = {c:.2f} outside standard range [10, 15]"

# ============================================================================
# TEST: STATISTICAL MEASURES
# ============================================================================

class TestStatistics:
    """Test statistical calculations (mean, σ, 95% CI)."""
    
    def test_mean_computation(self):
        """Mean of compute-optimal models should be 1.27×10⁻¹¹."""
        psi_values = [1.34e-11, 1.03e-11, 1.44e-11]  # Chinchilla, PaLM, LLaMA
        mean = np.mean(psi_values)
        expected = 1.27e-11
        
        assert np.isclose(mean, expected, rtol=0.01), \
            f"Mean Ψ = {mean:.4e}, expected {expected:.4e}"
    
    def test_std_computation(self):
        """Sample std should be 0.21×10⁻¹¹."""
        psi_values = [1.34e-11, 1.03e-11, 1.44e-11]
        std = np.std(psi_values, ddof=1)  # Sample std
        expected = 0.21e-11
        
        assert np.isclose(std, expected, rtol=0.05), \
            f"Sample σ = {std:.4e}, expected {expected:.4e}"
    
    def test_cv_computation(self):
        """Coefficient of variation should be 16.5%."""
        psi_values = [1.34e-11, 1.03e-11, 1.44e-11]
        mean = np.mean(psi_values)
        std = np.std(psi_values, ddof=1)
        cv = (std / mean) * 100
        expected = 16.5
        
        assert np.isclose(cv, expected, rtol=0.05), \
            f"CV = {cv:.1f}%, expected {expected:.1f}%"
    
    def test_95ci_bounds(self):
        """95% CI should be [0.75, 1.79]×10⁻¹¹."""
        psi_values = [1.34e-11, 1.03e-11, 1.44e-11]
        stats = compute_statistics(psi_values)
        
        expected_lower = 0.75e-11
        expected_upper = 1.79e-11
        
        assert np.isclose(stats["ci_lower"], expected_lower, rtol=0.05), \
            f"CI lower = {stats['ci_lower']:.4e}, expected {expected_lower:.4e}"
        assert np.isclose(stats["ci_upper"], expected_upper, rtol=0.05), \
            f"CI upper = {stats['ci_upper']:.4e}, expected {expected_upper:.4e}"
    
    def test_canonical_value(self):
        """Canonical Ψ_LLM should match computed mean."""
        psi_values = [1.34e-11, 1.03e-11, 1.44e-11]
        mean = np.mean(psi_values)
        
        assert np.isclose(mean, PSI_CANONICAL, rtol=0.01), \
            f"Canonical Ψ = {PSI_CANONICAL:.4e}, computed mean = {mean:.4e}"

# ============================================================================
# TEST: SENSITIVITY ANALYSIS
# ============================================================================

class TestSensitivity:
    """Test sensitivity analysis calculations."""
    
    def test_linear_scaling_H(self):
        """Ψ should scale linearly with H."""
        models = {"test": {"N": 70e9, "D": 1.4e12, "L": 80, "E": 8192}}
        
        H_values = [1.8, 2.0, 2.2]
        V = 1e6
        
        psi_values = []
        for H in H_values:
            psi = compute_psi(
                models["test"]["N"],
                models["test"]["D"],
                models["test"]["L"],
                models["test"]["E"],
                H, V
            )
            psi_values.append(psi)
        
        # Check linearity: Ψ(H2) / Ψ(H1) = H2 / H1
        ratio_psi = psi_values[1] / psi_values[0]  # 2.0 / 1.8
        ratio_H = H_values[1] / H_values[0]
        
        assert np.isclose(ratio_psi, ratio_H, rtol=0.01), \
            f"Non-linear H scaling: Ψ ratio = {ratio_psi:.4f}, H ratio = {ratio_H:.4f}"
    
    def test_linear_scaling_V(self):
        """Ψ should scale linearly with V."""
        models = {"test": {"N": 70e9, "D": 1.4e12, "L": 80, "E": 8192}}
        
        H = 2.0
        V_values = [0.7e6, 1.0e6, 1.3e6]
        
        psi_values = []
        for V in V_values:
            psi = compute_psi(
                models["test"]["N"],
                models["test"]["D"],
                models["test"]["L"],
                models["test"]["E"],
                H, V
            )
            psi_values.append(psi)
        
        ratio_psi = psi_values[1] / psi_values[0]
        ratio_V = V_values[1] / V_values[0]
        
        assert np.isclose(ratio_psi, ratio_V, rtol=0.01), \
            f"Non-linear V scaling: Ψ ratio = {ratio_psi:.4f}, V ratio = {ratio_V:.4f}"
    
    def test_cv_invariance(self):
        """CV should remain constant across H×V variations."""
        from sensitivity_analysis import MODELS as SENS_MODELS
        
        H_values = np.array([1.8, 2.0, 2.2])
        V_values = np.array([0.7e6, 1.0e6, 1.3e6])
        
        df = calculate_psi_grid(SENS_MODELS, H_values, V_values)
        
        cv_values = df["cv_percent"].values
        cv_mean = np.mean(cv_values)
        cv_std = np.std(cv_values)
        
        # CV should be constant (16.5% ± 0.1%)
        assert np.isclose(cv_mean, 16.5, atol=0.5), \
            f"Mean CV = {cv_mean:.2f}%, expected 16.5%"
        assert cv_std < 0.1, \
            f"CV std = {cv_std:.4f} too high (should be ~0)"

# ============================================================================
# TEST: D_MAX PREDICTIONS
# ============================================================================

class TestPredictions:
    """Test D_max prediction accuracy."""
    
    def test_chinchilla_prediction(self):
        """Chinchilla prediction should match actual (0% error)."""
        model = VALIDATED_MODELS["chinchilla"]
        d_pred = predict_dmax(
            model["N"], model["L"], model["E"]
        )
        d_actual = model["D_actual"]
        
        error_rel = abs(d_pred - d_actual) / d_actual * 100
        
        assert error_rel < 1.0, \
            f"Chinchilla error = {error_rel:.1f}%, expected <1%"
    
    def test_palm_prediction(self):
        """PaLM prediction should have ~8% error."""
        model = VALIDATED_MODELS["palm"]
        d_pred = predict_dmax(
            model["N"], model["L"], model["E"]
        )
        d_actual = model["D_actual"]
        
        error_rel = abs(d_pred - d_actual) / d_actual * 100
        
        assert 5 <= error_rel <= 10, \
            f"PaLM error = {error_rel:.1f}%, expected 5-10%"
    
    def test_llama_prediction(self):
        """LLaMA prediction should have ~13% error."""
        model = VALIDATED_MODELS["llama_65b"]
        d_pred = predict_dmax(
            model["N"], model["L"], model["E"]
        )
        d_actual = model["D_actual"]
        
        error_rel = abs(d_pred - d_actual) / d_actual * 100
        
        assert 10 <= error_rel <= 15, \
            f"LLaMA error = {error_rel:.1f}%, expected 10-15%"
    
    def test_mae_accuracy(self):
        """Mean Absolute Error should be ~10.8%."""
        errors = []
        for key in ["chinchilla", "palm", "llama_65b"]:
            model = VALIDATED_MODELS[key]
            d_pred = predict_dmax(
                model["N"], model["L"], model["E"]
            )
            d_actual = model["D_actual"]
            error_rel = abs(d_pred - d_actual) / d_actual * 100
            errors.append(error_rel)
        
        mae = np.mean(errors)
        expected_mae = 10.8
        
        assert np.isclose(mae, expected_mae, rtol=0.15), \
            f"MAE = {mae:.1f}%, expected {expected_mae:.1f}%"
    
    def test_uncertainty_bounds(self):
        """Predictions with uncertainty should contain actual values."""
        for key in ["chinchilla", "palm", "llama_65b"]:
            model = VALIDATED_MODELS[key]
            d_pred, d_lower, d_upper = predict_with_uncertainty(
                model["N"], model["L"], model["E"]
            )
            d_actual = model["D_actual"]
            
            assert d_lower <= d_actual <= d_upper, \
                f"{model['name']}: Actual {d_actual/1e12:.2f}T not in CI " \
                f"[{d_lower/1e12:.2f}, {d_upper/1e12:.2f}]T"

# ============================================================================
# TEST: EDGE CASES AND ROBUSTNESS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_zero_parameters(self):
        """Should raise error or return inf for N=0."""
        with pytest.raises((ZeroDivisionError, ValueError)):
            compute_psi(N=0, D=1e12, L=80, E=8192, H=2.0, V=1e6)
    
    def test_zero_data(self):
        """Should raise error or return inf for D=0."""
        with pytest.raises((ZeroDivisionError, ValueError)):
            compute_psi(N=70e9, D=0, L=80, E=8192, H=2.0, V=1e6)
    
    def test_large_model(self):
        """Should handle very large models (10T parameters)."""
        psi = compute_psi(
            N=10e12, D=10e12, L=500, E=32768,
            H=2.0, V=1e6
        )
        assert psi > 0 and np.isfinite(psi), \
            "Failed to handle 10T parameter model"
    
    def test_small_model(self):
        """Should handle small models (1B parameters)."""
        psi = compute_psi(
            N=1e9, D=10e9, L=12, E=768,
            H=2.0, V=1e6
        )
        assert psi > 0 and np.isfinite(psi), \
            "Failed to handle 1B parameter model"
    
    def test_negative_inputs(self):
        """Should reject negative inputs."""
        with pytest.raises((ValueError, AssertionError)):
            compute_psi(N=-70e9, D=1.4e12, L=80, E=8192, H=2.0, V=1e6)

# ============================================================================
# TEST: NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test numerical precision and stability."""
    
    def test_float_precision(self):
        """Results should be stable under float32/float64."""
        params = {"N": 70e9, "D": 1.4e12, "L": 80, "E": 8192, "H": 2.0, "V": 1e6}
        
        psi_64 = compute_psi(**params)
        
        # Convert to float32 and back
        params_32 = {k: np.float32(v) for k, v in params.items()}
        psi_32 = float(compute_psi(**params_32))
        
        rel_error = abs(psi_64 - psi_32) / psi_64
        
        assert rel_error < 1e-6, \
            f"Float precision instability: {rel_error:.2e} relative error"
    
    def test_repeated_calculation(self):
        """Repeated calculations should give identical results."""
        params = {"N": 70e9, "D": 1.4e12, "L": 80, "E": 8192, "H": 2.0, "V": 1e6}
        
        results = [compute_psi(**params) for _ in range(100)]
        
        assert len(set(results)) == 1, \
            "Non-deterministic calculation detected"
    
    def test_commutative_operations(self):
        """Order of operations should not affect result."""
        N, D, L, E, H, V = 70e9, 1.4e12, 80, 8192, 2.0, 1e6
        
        # Method 1: Direct formula
        psi1 = (L * E * H * V) / (N * D)
        
        # Method 2: Reordered
        psi2 = (H * V * L * E) / (D * N)
        
        assert np.isclose(psi1, psi2, rtol=1e-15), \
            "Non-commutative numerical error"

# ============================================================================
# TEST: INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests across modules."""
    
    def test_end_to_end_validation(self):
        """Full workflow: compute Ψ → predict D → validate."""
        # Step 1: Compute Ψ for Chinchilla
        psi = compute_psi(70e9, 1.4e12, 80, 8192, 2.0, 1e6)
        
        # Step 2: Use canonical Ψ to predict D
        d_pred = predict_dmax(70e9, 80, 8192)
        
        # Step 3: Validate against actual
        d_actual = 1.4e12
        error = abs(d_pred - d_actual) / d_actual * 100
        
        assert error < 5, \
            f"End-to-end error {error:.1f}% too high"
    
    def test_consistency_across_modules(self):
        """Same model should give consistent results across modules."""
        from compute_psi_canonical import MODELS as PSI_MODELS
        from predict_future_models import VALIDATED_MODELS as PRED_MODELS
        
        # Check Chinchilla consistency
        assert PSI_MODELS["Chinchilla"]["N"] == PRED_MODELS["chinchilla"]["N"]
        assert PSI_MODELS["Chinchilla"]["D"] == PRED_MODELS["chinchilla"]["D_actual"]

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

# ============================================================================
# MAIN (for standalone execution)
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
