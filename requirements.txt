#!/usr/bin/env python3
"""
Canonical Recalculation of Ψ_LLM with Sensitivity Analysis
Based on corrected LLaMA-65B data (D = 1.4T tokens)
Author: Viktor N. Savitskiy
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# CANONICAL PARAMETERS (Source of Truth)
# ============================================================================

# Domain normalization (conservative estimates)
H_BASE = 2.0  # bits/token (perplexity ≈ 4)
V_BASE = 1e6  # validation tokens

# Model data (from master_table.csv)
MODELS = {
    "GPT-3": {
        "N": 175e9, "D": 0.30e12, "L": 96, "E": 12288,
        "status": "Undertrained"
    },
    "Chinchilla": {
        "N": 70e9, "D": 1.40e12, "L": 80, "E": 8192,
        "status": "Compute-optimal"
    },
    "PaLM": {
        "N": 540e9, "D": 0.78e12, "L": 118, "E": 18432,
        "status": "Compute-optimal"
    },
    "LLaMA-65B": {
        "N": 65.2e9, "D": 1.40e12, "L": 80, "E": 8192,  # CORRECTED
        "status": "Compute-optimal"
    }
}

# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def compute_psi(N, D, L, E, H=H_BASE, V=V_BASE):
    """
    Compute Ψ_LLM = (L·E·H·V) / (N·D)
    
    Returns:
        Ψ_LLM in scientific notation
    """
    M = H * V  # Normalization scale (token-bits)
    psi = (L * E * M) / (N * D)
    return psi

def compute_k(N, D, H=H_BASE, V=V_BASE):
    """
    Compute architecture-specific constant k = (H·V) / (N·D)
    
    Used in decomposition: Ψ = k·L·E
    """
    return (H * V) / (N * D)

def compute_c_coefficient(N, L, E):
    """
    Compute layer-specific parameter allocation coefficient
    N ≈ c·L·E²  =>  c = N / (L·E²)
    
    For standard Transformers: c ≈ 12-13
    """
    return N / (L * E**2)

# ============================================================================
# MAIN CALCULATION
# ============================================================================

def calculate_all_psi():
    """Calculate Ψ for all models and compute statistics"""
    
    print("="*70)
    print("CANONICAL PSI RECALCULATION")
    print("="*70)
    print(f"Base parameters: H = {H_BASE} bits/token, V = {V_BASE:.0e} tokens")
    print(f"Normalization: M = H·V = {H_BASE * V_BASE:.2e} token-bits")
    print("="*70)
    print()
    
    results = []
    optimal_psi_values = []
    
    print(f"{'Model':<15} {'Ψ (×10⁻¹¹)':<12} {'k (×10⁻¹⁴)':<12} {'c':<8} {'Status'}")
    print("-"*70)
    
    for name, params in MODELS.items():
        psi = compute_psi(params["N"], params["D"], params["L"], params["E"])
        k = compute_k(params["N"], params["D"])
        c = compute_c_coefficient(params["N"], params["L"], params["E"])
        
        # Verify decomposition: Ψ = k·L·E
        psi_verify = k * params["L"] * params["E"]
        assert np.isclose(psi, psi_verify, rtol=1e-10), "Decomposition failed"
        
        results.append({
            "Model": name,
            "N": params["N"],
            "D": params["D"],
            "L": params["L"],
            "E": params["E"],
            "Ψ": psi,
            "k": k,
            "c": c,
            "Status": params["status"]
        })
        
        print(f"{name:<15} {psi*1e11:>11.4f}  {k*1e14:>11.4f}  {c:>7.2f}  {params['status']}")
        
        if params["status"] == "Compute-optimal":
            optimal_psi_values.append(psi)
    
    print()
    
    # Statistics for compute-optimal models
    optimal_psi_values = np.array(optimal_psi_values)
    mean_psi = np.mean(optimal_psi_values)
    sample_std = np.std(optimal_psi_values, ddof=1)  # Sample std (n-1)
    pop_std = np.std(optimal_psi_values, ddof=0)     # Population std
    
    # 95% Confidence Interval (Student's t, df=2)
    n = len(optimal_psi_values)
    t_critical = stats.t.ppf(0.975, df=n-1)  # Two-tailed, 97.5th percentile
    margin_error = t_critical * sample_std / np.sqrt(n)
    ci_lower = mean_psi - margin_error
    ci_upper = mean_psi + margin_error
    
    print("="*70)
    print("STATISTICS (Compute-Optimal Models Only: n=3)")
    print("="*70)
    print(f"Mean Ψ_LLM:          {mean_psi:.4e} ({mean_psi*1e11:.4f} ×10⁻¹¹)")
    print(f"Sample σ (n-1):      {sample_std:.4e} ({sample_std*1e11:.4f} ×10⁻¹¹)")
    print(f"Population σ (n):    {pop_std:.4e} ({pop_std*1e11:.4f} ×10⁻¹¹)")
    print(f"Coeff. of Variation: {(sample_std/mean_psi)*100:.1f}%")
    print()
    print(f"95% CI (t={t_critical:.3f}, df={n-1}):")
    print(f"  [{ci_lower:.4e}, {ci_upper:.4e}]")
    print(f"  [{ci_lower*1e11:.4f}, {ci_upper*1e11:.4f}] ×10⁻¹¹")
    print(f"Margin of Error:     ±{margin_error:.4e} (±{margin_error*1e11:.4f} ×10⁻¹¹)")
    print("="*70)
    print()
    
    # Recommended canonical value (rounded to 2 significant figures)
    canonical_psi = np.round(mean_psi, 13)  # Keep precision, display rounded
    print(f"RECOMMENDED CANONICAL: Ψ_LLM = {canonical_psi:.2e}")
    print(f"                       (Display as: 1.27×10⁻¹¹)")
    print("="*70)
    print()
    
    return pd.DataFrame(results), {
        "mean": mean_psi,
        "sample_std": sample_std,
        "pop_std": pop_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_error": margin_error,
        "canonical": canonical_psi
    }

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis():
    """
    Analyze how Ψ_LLM varies with H and V uncertainties
    """
    
    print()
    print("="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)
    print()
    
    # Parameter ranges
    H_range = [1.8, 2.0, 2.2]  # ±10% around base
    V_range = [0.7e6, 1.0e6, 1.3e6]  # ±30% around base
    
    # Compute-optimal models only
    optimal_models = {k: v for k, v in MODELS.items() 
                     if v["status"] == "Compute-optimal"}
    
    print(f"{'H (bits/token)':<18} {'V (tokens)':<15} {'Mean Ψ (×10⁻¹¹)':<18} {'σ (×10⁻¹¹)':<15} {'CV %'}")
    print("-"*80)
    
    sensitivity_data = []
    
    for H in H_range:
        for V in V_range:
            psi_values = []
            for name, params in optimal_models.items():
                psi = compute_psi(params["N"], params["D"], 
                                params["L"], params["E"], H=H, V=V)
                psi_values.append(psi)
            
            psi_values = np.array(psi_values)
            mean = np.mean(psi_values)
            std = np.std(psi_values, ddof=1)
            cv = (std/mean)*100
            
            sensitivity_data.append({
                "H": H,
                "V": V,
                "mean_psi": mean,
                "std": std,
                "cv": cv
            })
            
            print(f"{H:<18.1f} {V:<15.2e} {mean*1e11:<18.4f} {std*1e11:<15.4f} {cv:<.1f}")
    
    print()
    print("Interpretation:")
    print("- H uncertainty (±0.2) propagates as ±10% in Ψ")
    print("- V uncertainty (±0.3×10⁶) propagates as ±30% in Ψ")
    print("- Combined uncertainty: approximately ±32% relative error")
    print("="*70)
    print()
    
    return pd.DataFrame(sensitivity_data)

# ============================================================================
# COMPARISON WITH OLD VALUES
# ============================================================================

def compare_with_old():
    """Compare new calculations with previous manuscript values"""
    
    print()
    print("="*70)
    print("COMPARISON WITH PREVIOUS MANUSCRIPT")
    print("="*70)
    print()
    
    old_psi_main = 1.23e-11  # From main manuscript Table 1
    old_psi_supp = 1.10e-11  # From Supplementary
    
    # Recalculate with OLD LLaMA data (D = 1.0T)
    old_llama_psi = compute_psi(
        N=65.2e9, D=1.00e12,  # OLD D value
        L=80, E=8192
    )
    
    old_mean = np.mean([
        compute_psi(70e9, 1.40e12, 80, 8192),   # Chinchilla
        compute_psi(540e9, 0.78e12, 118, 18432), # PaLM
        old_llama_psi  # LLaMA with OLD D
    ])
    
    # New calculation
    new_llama_psi = compute_psi(65.2e9, 1.40e12, 80, 8192)
    new_mean = np.mean([
        compute_psi(70e9, 1.40e12, 80, 8192),
        compute_psi(540e9, 0.78e12, 118, 18432),
        new_llama_psi
    ])
    
    print(f"OLD manuscript (main):      Ψ = {old_psi_main:.2e} (1.23×10⁻¹¹)")
    print(f"OLD manuscript (supp):      Ψ = {old_psi_supp:.2e} (1.10×10⁻¹¹)")
    print()
    print(f"OLD calculation (D_LLaMA=1.0T):")
    print(f"  LLaMA-65B Ψ:              {old_llama_psi:.4e} ({old_llama_psi*1e11:.4f}×10⁻¹¹)")
    print(f"  Mean (3 models):          {old_mean:.4e} ({old_mean*1e11:.4f}×10⁻¹¹)")
    print()
    print(f"NEW calculation (D_LLaMA=1.4T, CORRECTED):")
    print(f"  LLaMA-65B Ψ:              {new_llama_psi:.4e} ({new_llama_psi*1e11:.4f}×10⁻¹¹)")
    print(f"  Mean (3 models):          {new_mean:.4e} ({new_mean*1e11:.4f}×10⁻¹¹)")
    print()
    print(f"CHANGE: {((new_mean - old_mean)/old_mean)*100:+.1f}% increase in mean Ψ")
    print(f"REASON: LLaMA-65B data correction (D: 1.0T → 1.4T)")
    print("="*70)
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Main calculation
    df_results, stats = calculate_all_psi()
    
    # Sensitivity analysis
    df_sensitivity = sensitivity_analysis()
    
    # Comparison with old values
    compare_with_old()
    
    # Export results
    df_results.to_csv("results_psi_canonical.csv", index=False)
    df_sensitivity.to_csv("results_sensitivity.csv", index=False)
    
    print("\n✅ Results exported to:")
    print("   - results_psi_canonical.csv")
    print("   - results_sensitivity.csv")
    print()
    print("📊 Recommended for manuscript:")
    print(f"   Ψ_LLM = 1.27×10⁻¹¹  (mean of 3 compute-optimal models)")
    print(f"   95% CI: [{stats['ci_lower']*1e11:.2f}, {stats['ci_upper']*1e11:.2f}]×10⁻¹¹")
    print(f"   Sample σ = {stats['sample_std']*1e11:.2f}×10⁻¹¹")
    print()
