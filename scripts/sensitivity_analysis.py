#!/usr/bin/env python3
"""
Sensitivity Analysis for LIICS Framework
=========================================
Analyzes how the empirical scaling invariant Ψ_LLM varies with 
normalization parameters H (domain entropy) and V (validation size).

Based on: "An Empirical Invariant for Transformer Scaling: 
          Towards an Information Incompleteness Hypothesis"
Author: Viktor N. Savitskiy (ORCID: 0000-0003-1356-7260)
Version: 2.1 (November 2025)
License: MIT

Reproduces: Supplementary Material Table 3 (Sensitivity Grid)

Key Findings:
- H uncertainty (±10%) propagates linearly to Ψ_LLM (±10%)
- V uncertainty (±30%) propagates linearly to Ψ_LLM (±30%)
- Combined relative uncertainty: ≈±32%
- Coefficient of Variation (16.5%) remains constant → rankings preserved

Usage:
    python sensitivity_analysis.py [--H_min H_MIN] [--H_max H_MAX] 
                                   [--V_min V_MIN] [--V_max V_MAX]
                                   [--grid_points N] [--plot]
    
Example:
    python sensitivity_analysis.py --grid_points 5 --plot
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib not available - plotting disabled")
    print("   Install with: pip install matplotlib")

# ============================================================================
# MODEL DATA (COMPUTE-OPTIMAL ONLY)
# ============================================================================

MODELS = {
    "Chinchilla": {
        "N": 70e9,
        "D": 1.40e12,
        "L": 80,
        "E": 8192
    },
    "PaLM": {
        "N": 540e9,
        "D": 0.78e12,
        "L": 118,
        "E": 18432
    },
    "LLaMA-65B": {
        "N": 65.2e9,
        "D": 1.40e12,  # Corrected value
        "L": 80,
        "E": 8192
    }
}

# ============================================================================
# DEFAULT PARAMETER RANGES
# ============================================================================

# Domain entropy range (bits/token)
H_MIN_DEFAULT = 1.8  # General text: lower bound
H_MAX_DEFAULT = 2.2  # Code/math: upper bound
H_BASE = 2.0         # Canonical value

# Validation set size range (tokens)
V_MIN_DEFAULT = 0.7e6   # Smaller benchmarks
V_MAX_DEFAULT = 1.3e6   # Larger corpora
V_BASE = 1.0e6          # Canonical value

# Grid resolution
GRID_POINTS_DEFAULT = 3  # 3×3 grid for manuscript table

# ============================================================================
# CORE CALCULATION
# ============================================================================

def compute_psi(N, D, L, E, H, V):
    """
    Compute Ψ_LLM = (L·E·H·V) / (N·D)
    
    Returns:
        float: Ψ_LLM (dimensionless)
    """
    return (L * E * H * V) / (N * D)

def calculate_psi_grid(models, H_values, V_values):
    """
    Calculate Ψ_LLM across H×V parameter grid.
    
    Parameters:
        models (dict): Model parameters
        H_values (array): Entropy values to test
        V_values (array): Validation sizes to test
    
    Returns:
        DataFrame: Grid with columns [H, V, mean_psi, std, cv, min, max]
    """
    results = []
    
    for H, V in product(H_values, V_values):
        psi_values = []
        
        for name, params in models.items():
            psi = compute_psi(
                params["N"], params["D"],
                params["L"], params["E"],
                H, V
            )
            psi_values.append(psi)
        
        psi_array = np.array(psi_values)
        
        # Statistics
        mean = np.mean(psi_array)
        std = np.std(psi_array, ddof=1)  # Sample std
        cv = (std / mean) * 100
        psi_min = np.min(psi_array)
        psi_max = np.max(psi_array)
        
        results.append({
            "H_bits_per_token": H,
            "V_tokens": V,
            "mean_psi": mean,
            "std": std,
            "cv_percent": cv,
            "min_psi": psi_min,
            "max_psi": psi_max
        })
    
    return pd.DataFrame(results)

# ============================================================================
# UNCERTAINTY PROPAGATION ANALYSIS
# ============================================================================

def analyze_uncertainty_propagation(df):
    """
    Analyze how H and V uncertainties propagate to Ψ_LLM.
    
    Parameters:
        df (DataFrame): Sensitivity grid results
    
    Returns:
        dict: Uncertainty propagation statistics
    """
    # Find baseline (H=2.0, V=1.0e6)
    baseline = df[(df["H_bits_per_token"] == H_BASE) & 
                  (df["V_tokens"] == V_BASE)].iloc[0]
    
    psi_base = baseline["mean_psi"]
    
    # H variation (V constant at base)
    h_rows = df[df["V_tokens"] == V_BASE]
    h_variation = (h_rows["mean_psi"].std() / psi_base) * 100
    
    # V variation (H constant at base)
    v_rows = df[df["H_bits_per_token"] == H_BASE]
    v_variation = (v_rows["mean_psi"].std() / psi_base) * 100
    
    # Combined uncertainty (quadrature)
    combined = np.sqrt(h_variation**2 + v_variation**2)
    
    return {
        "baseline_psi": psi_base,
        "h_variation_percent": h_variation,
        "v_variation_percent": v_variation,
        "combined_uncertainty_percent": combined,
        "h_range": (h_rows["H_bits_per_token"].min(), 
                    h_rows["H_bits_per_token"].max()),
        "v_range": (v_rows["V_tokens"].min(), 
                    v_rows["V_tokens"].max())
    }

# ============================================================================
# VISUALIZATION (OPTIONAL)
# ============================================================================

def plot_sensitivity_heatmap(df, output_path="results/sensitivity_heatmap.png"):
    """
    Create heatmap visualization of Ψ_LLM across H×V grid.
    
    Parameters:
        df (DataFrame): Sensitivity grid results
        output_path (str): Output file path
    """
    if not PLOTTING_AVAILABLE:
        print("⚠️  Plotting skipped (matplotlib not available)")
        return
    
    # Pivot table for heatmap
    pivot = df.pivot(
        index="V_tokens",
        columns="H_bits_per_token",
        values="mean_psi"
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap 1: Absolute values
    im1 = ax1.imshow(pivot.values * 1e11, aspect='auto', cmap='viridis',
                     origin='lower')
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels([f"{h:.1f}" for h in pivot.columns])
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([f"{v:.1e}" for v in pivot.index])
    ax1.set_xlabel("H (bits/token)", fontsize=12)
    ax1.set_ylabel("V (tokens)", fontsize=12)
    ax1.set_title("Mean Ψ_LLM (×10⁻¹¹)", fontsize=14, fontweight='bold')
    
    # Add values as text
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax1.text(j, i, f"{pivot.values[i, j]*1e11:.2f}",
                           ha="center", va="center", color="white", fontsize=10)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Ψ_LLM (×10⁻¹¹)", fontsize=11)
    
    # Heatmap 2: Coefficient of Variation
    pivot_cv = df.pivot(
        index="V_tokens",
        columns="H_bits_per_token",
        values="cv_percent"
    )
    
    im2 = ax2.imshow(pivot_cv.values, aspect='auto', cmap='RdYlGn_r',
                     origin='lower', vmin=15, vmax=18)
    ax2.set_xticks(range(len(pivot_cv.columns)))
    ax2.set_xticklabels([f"{h:.1f}" for h in pivot_cv.columns])
    ax2.set_yticks(range(len(pivot_cv.index)))
    ax2.set_yticklabels([f"{v:.1e}" for v in pivot_cv.index])
    ax2.set_xlabel("H (bits/token)", fontsize=12)
    ax2.set_ylabel("V (tokens)", fontsize=12)
    ax2.set_title("Coefficient of Variation (%)", fontsize=14, fontweight='bold')
    
    # Add values
    for i in range(len(pivot_cv.index)):
        for j in range(len(pivot_cv.columns)):
            text = ax2.text(j, i, f"{pivot_cv.values[i, j]:.1f}",
                           ha="center", va="center", color="black", fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("CV (%)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap saved: {output_path}")

def plot_uncertainty_bars(df, output_path="results/uncertainty_bars.png"):
    """
    Create bar plot showing uncertainty contributions.
    
    Parameters:
        df (DataFrame): Sensitivity grid results
        output_path (str): Output file path
    """
    if not PLOTTING_AVAILABLE:
        return
    
    uncertainty = analyze_uncertainty_propagation(df)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['H uncertainty\n(±0.2 bits/token)',
                  'V uncertainty\n(±0.3×10⁶ tokens)',
                  'Combined\n(quadrature)']
    values = [
        uncertainty["h_variation_percent"],
        uncertainty["v_variation_percent"],
        uncertainty["combined_uncertainty_percent"]
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Relative Uncertainty (%)", fontsize=12)
    ax.set_title("Uncertainty Propagation in Ψ_LLM", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Uncertainty plot saved: {output_path}")

# ============================================================================
# FORMATTED OUTPUT
# ============================================================================

def print_sensitivity_table(df, title="SENSITIVITY ANALYSIS"):
    """
    Print formatted sensitivity table (reproduce manuscript Table 3).
    
    Parameters:
        df (DataFrame): Sensitivity grid results
        title (str): Table title
    """
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    print()
    print(f"{'H':<10} {'V':<15} {'Mean Ψ(×10⁻¹¹)':<18} "
          f"{'σ(×10⁻¹¹)':<15} {'CV(%)':<10} {'Range(×10⁻¹¹)'}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        h = row["H_bits_per_token"]
        v = row["V_tokens"]
        mean = row["mean_psi"] * 1e11
        std = row["std"] * 1e11
        cv = row["cv_percent"]
        psi_min = row["min_psi"] * 1e11
        psi_max = row["max_psi"] * 1e11
        
        # Highlight baseline
        marker = " *" if (h == H_BASE and v == V_BASE) else "  "
        
        print(f"{h:<10.1f} {v:<15.2e} {mean:<18.4f} "
              f"{std:<15.4f} {cv:<10.1f} [{psi_min:.2f}, {psi_max:.2f}]{marker}")
    
    print()
    print("* Baseline (canonical parameters)")
    print("=" * 90)

def print_uncertainty_summary(df):
    """
    Print summary of uncertainty propagation analysis.
    
    Parameters:
        df (DataFrame): Sensitivity grid results
    """
    uncertainty = analyze_uncertainty_propagation(df)
    
    print()
    print("=" * 90)
    print("UNCERTAINTY PROPAGATION ANALYSIS")
    print("=" * 90)
    print()
    print(f"Baseline Ψ_LLM (H={H_BASE}, V={V_BASE:.0e}):")
    print(f"  {uncertainty['baseline_psi']:.4e}  "
          f"({uncertainty['baseline_psi']*1e11:.4f} ×10⁻¹¹)")
    print()
    print("Parameter Variations:")
    print(f"  H range: {uncertainty['h_range'][0]:.1f} - "
          f"{uncertainty['h_range'][1]:.1f} bits/token")
    print(f"  V range: {uncertainty['v_range'][0]:.1e} - "
          f"{uncertainty['v_range'][1]:.1e} tokens")
    print()
    print("Uncertainty Propagation (relative to baseline):")
    print(f"  H uncertainty (±0.2 bits/token):   ±{uncertainty['h_variation_percent']:.1f}%")
    print(f"  V uncertainty (±0.3×10⁶ tokens):  ±{uncertainty['v_variation_percent']:.1f}%")
    print(f"  Combined (quadrature):             ±{uncertainty['combined_uncertainty_percent']:.1f}%")
    print()
    print("Physical Interpretation:")
    print("  • Linear scaling: Ψ ∝ H·V")
    print("  • H dominates entropy measurement uncertainty")
    print("  • V dominates corpus size choice uncertainty")
    print("  • CV remains constant (16.5%) → relative rankings preserved")
    print()
    print("Practical Implications:")
    print("  • Domain-specific applications: measure H = log₂(PPL_val) directly")
    print("  • Validation corpus: ensure V ≥ 10⁵ tokens for stable loss")
    print("  • Expected absolute uncertainty: ≈±0.4×10⁻¹¹ (32% relative)")
    print("=" * 90)

def print_manuscript_table(df):
    """
    Print table in manuscript LaTeX format (for copy-paste).
    
    Parameters:
        df (DataFrame): Sensitivity grid results
    """
    print()
    print("=" * 90)
    print("MANUSCRIPT TABLE (LaTeX FORMAT)")
    print("=" * 90)
    print()
    print("% Copy-paste into Supplementary Material Table 3")
    print()
    
    print("\\begin{tabular}{ccccc}")
    print("\\toprule")
    print("\\textbf{H} & \\textbf{V} & \\textbf{Mean} (\\tentotheminus{11}) & "
          "\\textbf{Std} (\\tentotheminus{11}) & \\textbf{CV (\\%)} \\\\")
    print("(bits/token) & (tokens) & & & \\\\")
    print("\\midrule")
    
    # Group by H
    for h in sorted(df["H_bits_per_token"].unique()):
        subset = df[df["H_bits_per_token"] == h].sort_values("V_tokens")
        
        for i, (_, row) in enumerate(subset.iterrows()):
            v = row["V_tokens"]
            mean = row["mean_psi"] * 1e11
            std = row["std"] * 1e11
            cv = row["cv_percent"]
            
            # Format V in scientific notation
            v_str = f"\\tentothe{{{v/1e6:.1f}}}{{6}}"
            
            # Bold baseline row
            if h == H_BASE and v == V_BASE:
                h_str = f"\\textbf{{{h:.1f}}}"
                mean_str = f"\\textbf{{{mean:.2f}}}"
                std_str = f"\\textbf{{{std:.2f}}}"
                cv_str = f"\\textbf{{{cv:.1f}}}"
            else:
                h_str = f"{h:.1f}"
                mean_str = f"{mean:.2f}"
                std_str = f"{std:.2f}"
                cv_str = f"{cv:.1f}"
            
            print(f"{h_str} & {v_str} & {mean_str} & {std_str} & {cv_str} \\\\")
        
        # Add midrule between H groups (except last)
        if h != sorted(df["H_bits_per_token"].unique())[-1]:
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print()
    print("=" * 90)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis of Ψ_LLM to H and V variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 3×3 grid (manuscript table)
  python sensitivity_analysis.py
  
  # High-resolution 5×5 grid
  python sensitivity_analysis.py --grid_points 5
  
  # Custom parameter ranges
  python sensitivity_analysis.py --H_min 1.5 --H_max 2.5 --V_min 0.5e6 --V_max 1.5e6
  
  # Generate plots
  python sensitivity_analysis.py --plot
  
  # Full analysis with all outputs
  python sensitivity_analysis.py --grid_points 5 --plot --latex
        """
    )
    
    parser.add_argument("--H_min", type=float, default=H_MIN_DEFAULT,
                       help=f"Minimum entropy (default: {H_MIN_DEFAULT})")
    parser.add_argument("--H_max", type=float, default=H_MAX_DEFAULT,
                       help=f"Maximum entropy (default: {H_MAX_DEFAULT})")
    parser.add_argument("--V_min", type=float, default=V_MIN_DEFAULT,
                       help=f"Minimum validation size (default: {V_MIN_DEFAULT:.1e})")
    parser.add_argument("--V_max", type=float, default=V_MAX_DEFAULT,
                       help=f"Maximum validation size (default: {V_MAX_DEFAULT:.1e})")
    parser.add_argument("--grid_points", type=int, default=GRID_POINTS_DEFAULT,
                       help=f"Grid resolution (default: {GRID_POINTS_DEFAULT})")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--latex", action="store_true",
                       help="Print LaTeX table format")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory (default: results/)")
    
    args = parser.parse_args()
    
    # Generate parameter ranges
    H_values = np.linspace(args.H_min, args.H_max, args.grid_points)
    V_values = np.linspace(args.V_min, args.V_max, args.grid_points)
    
    print()
    print("=" * 90)
    print("LIICS SENSITIVITY ANALYSIS - Ψ_LLM vs (H, V)")
    print("=" * 90)
    print(f"Parameter ranges:")
    print(f"  H: [{args.H_min:.2f}, {args.H_max:.2f}] bits/token  "
          f"({args.grid_points} points)")
    print(f"  V: [{args.V_min:.2e}, {args.V_max:.2e}] tokens     "
          f"({args.grid_points} points)")
    print(f"Grid size: {args.grid_points}×{args.grid_points} = "
          f"{args.grid_points**2} calculations")
    print()
    print(f"Models included (compute-optimal only, n={len(MODELS)}):")
    for name, params in MODELS.items():
        print(f"  • {name}: {params['N']/1e9:.0f}B params, "
              f"{params['D']/1e12:.2f}T tokens")
    print("=" * 90)
    
    # Calculate sensitivity grid
    df = calculate_psi_grid(MODELS, H_values, V_values)
    
    # Print formatted table
    print_sensitivity_table(df)
    
    # Print uncertainty analysis
    print_uncertainty_summary(df)
    
    # Print LaTeX table (if requested)
    if args.latex:
        print_manuscript_table(df)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # Export CSV
    csv_file = output_path / "sensitivity_grid.csv"
    df.to_csv(csv_file, index=False, float_format="%.6e")
    print()
    print(f"✅ Results exported: {csv_file}")
    
    # Generate plots (if requested)
    if args.plot:
        plot_sensitivity_heatmap(df, output_path / "sensitivity_heatmap.png")
        plot_uncertainty_bars(df, output_path / "uncertainty_bars.png")
    
    # Summary
    print()
    print("=" * 90)
    print("SUMMARY FOR MANUSCRIPT (Supplementary Material)")
    print("=" * 90)
    print("Key Findings:")
    print("  • Ψ_LLM scales linearly with H and V: Ψ ∝ H·V")
    print("  • H uncertainty (±10%) → Ψ uncertainty (±10%)")
    print("  • V uncertainty (±30%) → Ψ uncertainty (±30%)")
    print("  • Combined uncertainty: ≈±32% (quadrature sum)")
    print("  • Coefficient of Variation constant at 16.5%")
    print()
    print("Implications:")
    print("  • Relative model rankings preserved under normalization changes")
    print("  • Domain-specific calibration recommended for H and V")
    print("  • Absolute Ψ_LLM values sensitive to normalization scale")
    print("  • Architecture-specific trends (k ∝ 1/D) robust to H, V choice")
    print("=" * 90)
    print()

if __name__ == "__main__":
    main()
