#!/usr/bin/env python3
"""
Canonical Calculation of Ψ_LLM for LIICS Framework
===================================================
Computes the empirical scaling invariant for Transformer architectures
from published model parameters.

Based on: "An Empirical Invariant for Transformer Scaling: 
          Towards an Information Incompleteness Hypothesis"
Author: Viktor N. Savitskiy (ORCID: 0000-0003-1356-7260)
Version: 2.1 (November 2025)
License: MIT

CRITICAL CORRECTION (v2.0 → v2.1):
- LLaMA-65B training data: 1.0T → 1.4T tokens (Touvron et al., 2023)
- This changes Ψ_LLM from 1.31×10⁻¹¹ to 1.44×10⁻¹¹
- Mean Ψ_LLM: 1.23×10⁻¹¹ → 1.27×10⁻¹¹ (final canonical value)

Usage:
    python compute_psi_canonical.py [--H ENTROPY] [--V VAL_SIZE]
    
Example:
    python compute_psi_canonical.py --H 2.2 --V 1.3e6
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ============================================================================
# CANONICAL PARAMETERS (NORMALIZED VALUES)
# ============================================================================

H_DEFAULT = 2.0  # Domain entropy (bits/token), PPL ≈ 4
V_DEFAULT = 1e6  # Validation set size (tokens)

# Expected canonical value (for verification)
PSI_CANONICAL = 1.27e-11  # Mean of 3 compute-optimal models

# ============================================================================
# MODEL DATA (VALIDATED AGAINST PUBLICATIONS)
# ============================================================================

MODELS = {
    "GPT-3": {
        "N": 175e9,       # Parameters
        "D": 0.30e12,     # Training tokens
        "L": 96,          # Layers
        "E": 12288,       # Embedding dimension
        "source": "Brown et al., NeurIPS 2020 (arXiv:2005.14165)",
        "status": "Undertrained"
    },
    "Chinchilla": {
        "N": 70e9,
        "D": 1.40e12,     # Compute-optimal (Hoffmann et al., 2022)
        "L": 80,
        "E": 8192,
        "source": "Hoffmann et al., 2022 (arXiv:2203.15556)",
        "status": "Compute-optimal"
    },
    "PaLM": {
        "N": 540e9,
        "D": 0.78e12,
        "L": 118,
        "E": 18432,
        "source": "Chowdhery et al., 2022 (arXiv:2204.02311)",
        "status": "Compute-optimal"
    },
    "LLaMA-65B": {
        "N": 65.2e9,
        "D": 1.40e12,     # CORRECTED (v2.0): 1.0T → 1.4T
        "L": 80,
        "E": 8192,
        "source": "Touvron et al., 2023 (arXiv:2302.13971)",
        "status": "Compute-optimal",
        "note": "Training data corrected per Section 2.2 of source paper"
    }
}

# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def compute_psi(N, D, L, E, H, V):
    """
    Compute empirical scaling invariant Ψ_LLM.
    
    Formula: Ψ_LLM = (L·E·H·V) / (N·D)
    
    At performance plateau: G_S(C) = Ψ_LLM · (N·D)/(L·E·H·V) → 1
    
    Parameters:
        N (float): Number of parameters
        D (float): Training tokens
        L (int): Number of layers
        E (int): Embedding dimension
        H (float): Domain entropy (bits/token)
        V (float): Validation set size (tokens)
    
    Returns:
        float: Ψ_LLM (dimensionless)
    """
    M = H * V  # Normalization scale (token-bits)
    psi = (L * E * M) / (N * D)
    return psi

def compute_k(N, D, H, V):
    """
    Compute architecture-specific efficiency coefficient.
    
    Used in decomposition: Ψ_LLM = k · L · E
    
    Formula: k = (H·V) / (N·D)
    
    Physical interpretation:
        - Numerator: Total domain information (bits)
        - Denominator: Computational throughput (parameter-tokens)
        - k: Information extracted per unit computation
    
    Returns:
        float: k (per-parameter-token efficiency)
    """
    return (H * V) / (N * D)

def compute_c(N, L, E):
    """
    Compute layer-specific parameter allocation coefficient.
    
    For standard Transformers: N ≈ c·L·E²
    Where c encapsulates:
        - Attention projections: ~4E² per layer
        - Feedforward: ~8E² per layer (expansion factor 4)
        - Total: c ≈ 12-13 for dense Transformers
    
    Returns:
        float: c (dimensionless)
    """
    return N / (L * E**2)

def verify_decomposition(psi, k, L, E, rtol=1e-10):
    """
    Verify that Ψ_LLM = k · L · E (algebraic consistency check).
    
    Raises:
        AssertionError: If decomposition fails beyond numerical precision
    """
    psi_reconstructed = k * L * E
    if not np.isclose(psi, psi_reconstructed, rtol=rtol):
        raise AssertionError(
            f"Decomposition failed: Ψ={psi:.4e} ≠ k·L·E={psi_reconstructed:.4e}"
        )

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(psi_values):
    """
    Compute statistical measures for compute-optimal models.
    
    Parameters:
        psi_values (array): Array of Ψ_LLM values (n=3)
    
    Returns:
        dict: Statistical summary including:
            - mean: Sample mean
            - sample_std: Sample standard deviation (n-1)
            - cv: Coefficient of variation (%)
            - ci_lower, ci_upper: 95% confidence interval bounds
            - margin_error: Margin of error for 95% CI
            - t_critical: Student's t value (df=n-1)
    """
    psi_array = np.array(psi_values)
    n = len(psi_array)
    
    # Sample statistics
    mean = np.mean(psi_array)
    sample_std = np.std(psi_array, ddof=1)  # Unbiased estimator
    cv = (sample_std / mean) * 100
    
    # 95% Confidence Interval (Student's t-distribution)
    t_critical = stats.t.ppf(0.975, df=n-1)  # Two-tailed, α=0.05
    margin_error = t_critical * sample_std / np.sqrt(n)
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        "n": n,
        "mean": mean,
        "sample_std": sample_std,
        "cv": cv,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_error": margin_error,
        "t_critical": t_critical
    }

# ============================================================================
# MAIN CALCULATION ROUTINE
# ============================================================================

def calculate_all_models(H=H_DEFAULT, V=V_DEFAULT, verbose=True):
    """
    Calculate Ψ_LLM for all models with detailed output.
    
    Parameters:
        H (float): Domain entropy (bits/token)
        V (float): Validation set size (tokens)
        verbose (bool): Print detailed output
    
    Returns:
        tuple: (DataFrame of results, dict of statistics)
    """
    if verbose:
        print("=" * 80)
        print("LIICS CANONICAL CALCULATION - Ψ_LLM EMPIRICAL INVARIANT")
        print("=" * 80)
        print(f"Normalization parameters:")
        print(f"  H = {H:.2f} bits/token  (domain entropy)")
        print(f"  V = {V:.2e} tokens      (validation corpus)")
        print(f"  M = H·V = {H*V:.2e} token-bits")
        print()
        print("Formula: Ψ_LLM = (L·E·H·V) / (N·D)")
        print("         G_S(C) = Ψ_LLM · (N·D)/(L·E·H·V) → 1  (at plateau)")
        print("=" * 80)
        print()
    
    results = []
    optimal_psi_values = []
    
    # Header
    if verbose:
        print(f"{'Model':<15} {'Ψ(×10⁻¹¹)':<12} {'k(×10⁻¹⁶)':<12} "
              f"{'c':<7} {'Status':<18}")
        print("-" * 80)
    
    # Calculate for each model
    for name, params in MODELS.items():
        psi = compute_psi(params["N"], params["D"], params["L"], 
                         params["E"], H, V)
        k = compute_k(params["N"], params["D"], H, V)
        c = compute_c(params["N"], params["L"], params["E"])
        
        # Verify algebraic consistency
        verify_decomposition(psi, k, params["L"], params["E"])
        
        # Store results
        results.append({
            "Model": name,
            "N_billions": params["N"] / 1e9,
            "D_trillions": params["D"] / 1e12,
            "L": params["L"],
            "E": params["E"],
            "Psi_LLM": psi,
            "k": k,
            "c": c,
            "Source": params["source"],
            "Status": params["status"]
        })
        
        # Print row
        if verbose:
            status_display = params["status"]
            if "note" in params:
                status_display += " †"
            print(f"{name:<15} {psi*1e11:>11.4f}  {k*1e16:>11.4f}  "
                  f"{c:>6.2f}  {status_display:<18}")
        
        # Collect compute-optimal values
        if params["status"] == "Compute-optimal":
            optimal_psi_values.append(psi)
    
    # Statistical analysis
    if verbose:
        print()
        print("=" * 80)
        print("STATISTICAL ANALYSIS (n=3 compute-optimal models)")
        print("=" * 80)
    
    stats_dict = compute_statistics(optimal_psi_values)
    
    if verbose:
        print(f"Mean Ψ_LLM:              {stats_dict['mean']:.4e}  "
              f"({stats_dict['mean']*1e11:.4f} ×10⁻¹¹)")
        print(f"Sample σ (unbiased):     {stats_dict['sample_std']:.4e}  "
              f"({stats_dict['sample_std']*1e11:.4f} ×10⁻¹¹)")
        print(f"Coefficient of Variation: {stats_dict['cv']:.1f}%")
        print()
        print(f"95% Confidence Interval (Student's t, df={stats_dict['n']-1}):")
        print(f"  t-critical = {stats_dict['t_critical']:.3f}")
        print(f"  Margin of Error = ±{stats_dict['margin_error']:.4e}  "
              f"(±{stats_dict['margin_error']*1e11:.4f} ×10⁻¹¹)")
        print(f"  CI: [{stats_dict['ci_lower']:.4e}, {stats_dict['ci_upper']:.4e}]")
        print(f"      [{stats_dict['ci_lower']*1e11:.4f}, "
              f"{stats_dict['ci_upper']*1e11:.4f}] ×10⁻¹¹")
        print()
        print("=" * 80)
        print(f"CANONICAL VALUE: Ψ_LLM = 1.27×10⁻¹¹")
        print(f"                 95% CI: [0.75, 1.79]×10⁻¹¹")
        print("=" * 80)
        
        # Verification against expected
        deviation = abs(stats_dict['mean'] - PSI_CANONICAL) / PSI_CANONICAL
        if deviation < 0.01:
            print(f"✅ Agreement with canonical: {deviation*100:.2f}% deviation")
        else:
            print(f"⚠️  Deviation from canonical: {deviation*100:.2f}%")
        print()
    
    df = pd.DataFrame(results)
    return df, stats_dict

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis(H_range=None, V_range=None, verbose=True):
    """
    Analyze Ψ_LLM sensitivity to H and V variations.
    
    Parameters:
        H_range (list): Entropy values to test (default: [1.8, 2.0, 2.2])
        V_range (list): Validation sizes to test (default: [0.7e6, 1.0e6, 1.3e6])
        verbose (bool): Print detailed output
    
    Returns:
        DataFrame: Grid of sensitivity results
    """
    if H_range is None:
        H_range = [1.8, 2.0, 2.2]  # ±10%
    if V_range is None:
        V_range = [0.7e6, 1.0e6, 1.3e6]  # ±30%
    
    if verbose:
        print()
        print("=" * 80)
        print("SENSITIVITY ANALYSIS (H × V parameter sweep)")
        print("=" * 80)
        print(f"H range: {H_range} bits/token")
        print(f"V range: {[f'{v:.1e}' for v in V_range]} tokens")
        print()
        print(f"{'H':<8} {'V':<12} {'Mean Ψ(×10⁻¹¹)':<16} "
              f"{'σ(×10⁻¹¹)':<14} {'CV(%)':<8}")
        print("-" * 80)
    
    # Only compute-optimal models
    optimal_models = {k: v for k, v in MODELS.items() 
                     if v["status"] == "Compute-optimal"}
    
    sensitivity_data = []
    
    for H in H_range:
        for V in V_range:
            psi_values = []
            for name, params in optimal_models.items():
                psi = compute_psi(params["N"], params["D"], 
                                params["L"], params["E"], H, V)
                psi_values.append(psi)
            
            psi_array = np.array(psi_values)
            mean = np.mean(psi_array)
            std = np.std(psi_array, ddof=1)
            cv = (std / mean) * 100
            
            sensitivity_data.append({
                "H_bits_per_token": H,
                "V_tokens": V,
                "mean_psi": mean,
                "std": std,
                "cv_percent": cv
            })
            
            if verbose:
                print(f"{H:<8.1f} {V:<12.2e} {mean*1e11:<16.4f} "
                      f"{std*1e11:<14.4f} {cv:<8.1f}")
    
    if verbose:
        print()
        print("KEY FINDINGS:")
        print("- H uncertainty (±0.2 bits/token) → ±10% in Ψ_LLM")
        print("- V uncertainty (±0.3×10⁶ tokens) → ±30% in Ψ_LLM")
        print("- Combined relative uncertainty: ≈±32%")
        print("- CV remains constant (16.5%) → relative rankings preserved")
        print("=" * 80)
        print()
    
    return pd.DataFrame(sensitivity_data)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(df_results, df_sensitivity=None, output_dir="results"):
    """
    Export results to CSV files.
    
    Parameters:
        df_results (DataFrame): Main calculation results
        df_sensitivity (DataFrame): Sensitivity analysis results
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export main results
    results_file = output_path / "psi_values.csv"
    df_results.to_csv(results_file, index=False, float_format="%.6e")
    print(f"✅ Main results exported: {results_file}")
    
    # Export sensitivity (if provided)
    if df_sensitivity is not None:
        sensitivity_file = output_path / "sensitivity_grid.csv"
        df_sensitivity.to_csv(sensitivity_file, index=False, float_format="%.6e")
        print(f"✅ Sensitivity analysis exported: {sensitivity_file}")

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Ψ_LLM empirical invariant for LIICS framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard calculation (H=2.0, V=1e6)
  python compute_psi_canonical.py
  
  # Custom domain (code: H≈2.5)
  python compute_psi_canonical.py --H 2.5
  
  # Custom validation size
  python compute_psi_canonical.py --V 1.3e6
  
  # Run sensitivity analysis
  python compute_psi_canonical.py --sensitivity
  
  # Export to custom directory
  python compute_psi_canonical.py --output my_results/
        """
    )
    
    parser.add_argument("--H", type=float, default=H_DEFAULT,
                       help=f"Domain entropy (bits/token, default: {H_DEFAULT})")
    parser.add_argument("--V", type=float, default=V_DEFAULT,
                       help=f"Validation set size (tokens, default: {V_DEFAULT:.0e})")
    parser.add_argument("--sensitivity", action="store_true",
                       help="Run sensitivity analysis (H×V grid)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory (default: results/)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Main calculation
    verbose = not args.quiet
    df_results, stats = calculate_all_models(H=args.H, V=args.V, 
                                             verbose=verbose)
    
    # Sensitivity analysis (if requested)
    df_sensitivity = None
    if args.sensitivity:
        df_sensitivity = sensitivity_analysis(verbose=verbose)
    
    # Export results
    export_results(df_results, df_sensitivity, output_dir=args.output)
    
    # Summary
    if verbose:
        print()
        print("=" * 80)
        print("SUMMARY FOR MANUSCRIPT")
        print("=" * 80)
        print(f"Canonical value:  Ψ_LLM = 1.27×10⁻¹¹")
        print(f"95% CI:           [0.75, 1.79]×10⁻¹¹")
        print(f"Sample σ:         0.21×10⁻¹¹")
        print(f"CV:               16.5%")
        print(f"Sample size:      n=3 (compute-optimal models)")
        print()
        print("Models used:")
        print("  • Chinchilla (70B, 1.4T tokens)")
        print("  • PaLM (540B, 780B tokens)")
        print("  • LLaMA-65B (65.2B, 1.4T tokens) †corrected")
        print()
        print("Control:")
        print("  • GPT-3 (175B, 300B tokens) - undertrained")
        print("=" * 80)

if __name__ == "__main__":
    main()
