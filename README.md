# Law of Information Incompleteness for Complex Systems (LIICS)

**🌐 Language:** **English** | [Русский](README.md)

![GitHub](https://img.shields.io/github/license/designhumanai/liics)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/designhumanai/liics)
![GitHub last commit](https://img.shields.io/github/last-commit/designhumanai/liics)
![GitHub Issues](https://img.shields.io/github/issues/designhumanai/liics)

**Repository for the manuscript:**  
*"An Empirical Invariant for Transformer Scaling: Towards an Information Incompleteness Hypothesis"*

**Author:** Viktor N. Savitskiy (ORCID: 0000-0003-1356-7260)  
**Affiliation:** DHAIE Research Initiative, DesignHumanAI.com

---

## 📄 Abstract

This repository contains all data, code, and supplementary materials for reproducing the calculations in the LIICS manuscript. We derive an empirical scaling invariant **Ψ_LLM ≈ 1.27×10⁻¹¹** for Transformer-based large language models, demonstrating that performance plateaus occur when the dimensionless quantity G_S(C) → 1.

---

## 🎯 Quick Start

### Reproduce Main Results

```bash
# Clone repository
git clone https://github.com/designhumanai/liics.git
cd liics

# Install dependencies
pip install -r requirements.txt

# Run canonical calculation
python scripts/compute_psi_canonical.py

# Run sensitivity analysis
python scripts/sensitivity_analysis.py
```

**Expected output:**
```
Model        Ψ(e-11)  Status
----------------------------------------
GPT-3           4.50  Undertrained
Chinchilla      1.34  Optimal
PaLM            1.03  Optimal
LLaMA-65B       1.44  Optimal

Mean (optimal): 1.27 ± 0.21 x 10^-11
Canonical: 1.27 x 10^-11
```

---

## 📁 Repository Structure

```
liics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── data/
│   └── master_table.csv              # Canonical model parameters
├── scripts/
│   ├── compute_psi_canonical.py      # Main calculation script
│   └── sensitivity_analysis.py       # H×V parameter sweep
├── results/
│   ├── results_psi_canonical.csv     # Individual model Ψ values
│   └── results_sensitivity.csv       # Sensitivity analysis output
└── docs/
    ├── manuscript.pdf                 # Main article (preprint)
    ├── supplementary.pdf              # Supplementary material
    └── CORRECTION_SUMMARY.md          # Change log (v2.0)
```

---

## 📊 Data Description

### `data/master_table.csv`

Canonical architectural parameters for analyzed models:

| Column | Description | Units |
|--------|-------------|-------|
| `Model` | Model name | - |
| `N_billions` | Parameters | Billions |
| `D_tokens_trillions` | Training tokens | Trillions |
| `L_layers` | Number of layers | - |
| `E_embedding` | Embedding dimension | - |
| `H_bits_per_token` | Domain entropy | bits/token |
| `V_validation_tokens` | Validation set size | tokens |
| `Source` | Original paper citation | - |
| `Status` | Training regime | Undertrained/Compute-optimal |

**Data Sources:**
- GPT-3: Brown et al., NeurIPS 2020
- Chinchilla: Hoffmann et al., 2022 (arXiv:2203.15556)
- PaLM: Chowdhery et al., 2022 (arXiv:2204.02311)
- LLaMA-65B: Touvron et al., 2023 (arXiv:2302.13971)

**Note:** LLaMA-65B training data corrected to 1.4T tokens (v2.0 update).

---

## 🧮 Core Formula

The Law of Information Incompleteness for Complex Systems:

```
G_S(C) = Ψ_LLM · (N·D) / (L·E·H·V)
```

Where:
- **G_S(C)** = Dimensionless efficiency invariant (→1 at plateau)
- **Ψ_LLM** = Empirical constant for Transformers (1.27×10⁻¹¹)
- **N** = Number of parameters
- **D** = Training tokens
- **L** = Number of layers
- **E** = Embedding dimension
- **H** = Domain entropy (bits/token)
- **V** = Validation set size (tokens)

**Predictive Power:**

Optimal training data for any architecture:
```
D_max = (L·E·H·V) / (Ψ_LLM · N)
```

---

## 🔬 Scripts Documentation

### `compute_psi_canonical.py`

Calculate Ψ_LLM for all models with statistical analysis.

**Usage:**
```bash
python scripts/compute_psi_canonical.py [--psi PSI_VALUE] [--H ENTROPY] [--V VAL_SIZE]
```

**Options:**
- `--psi`: Override canonical Ψ (default: 1.27e-11)
- `--H`: Domain entropy in bits/token (default: 2.0)
- `--V`: Validation set size in tokens (default: 1e6)

**Output files:**
- `results/results_psi_canonical.csv` - Per-model calculations
- Console output with statistics (mean, σ, 95% CI)

**Example with custom parameters:**
```bash
python scripts/compute_psi_canonical.py --H 2.2 --V 1.3e6
```

---

### `sensitivity_analysis.py`

Analyze Ψ_LLM sensitivity to H and V variations.

**Usage:**
```bash
python scripts/sensitivity_analysis.py [--H_min H_MIN] [--H_max H_MAX] [--V_min V_MIN] [--V_max V_MAX]
```

**Options:**
- `--H_min`, `--H_max`: Entropy range (default: 1.8-2.2)
- `--V_min`, `--V_max`: Validation size range (default: 0.7e6-1.3e6)

**Output:**
- `results/results_sensitivity.csv` - Full H×V grid
- Heatmap visualization (requires matplotlib)
- Combined uncertainty estimate

---

## 📈 Key Results

### Empirical Invariant (n=3 compute-optimal models)

| Statistic | Value |
|-----------|-------|
| **Mean Ψ_LLM** | 1.27×10⁻¹¹ |
| **Sample σ** | 0.21×10⁻¹¹ |
| **95% CI** | [0.87, 1.67]×10⁻¹¹ |
| **Coefficient of Variation** | 16.5% |

### Individual Model Values

| Model | Ψ_LLM (×10⁻¹¹) | Status |
|-------|----------------|--------|
| GPT-3 | 4.50 | Undertrained (control) |
| Chinchilla | 1.34 | Compute-optimal |
| PaLM | 1.03 | Compute-optimal |
| LLaMA-65B | 1.44 | Compute-optimal |

### Sensitivity Analysis

- **H uncertainty (±10%)** → ±10% change in Ψ_LLM
- **V uncertainty (±30%)** → ±30% change in Ψ_LLM
- **Combined uncertainty:** ≈±32% (matches observed CI)

---

## 🔧 Advanced Usage

### Adding New Models

Edit `data/master_table.csv` and add a row:

```csv
MyModel,500,2.0,100,16384,2.0,1000000,"MySource 2025",Compute-optimal
```

Re-run calculations:
```bash
python scripts/compute_psi_canonical.py
```

### Custom Domain Analysis

For code or math domains with different entropy:

```python
from compute_psi_canonical import compute_psi

# Code domain (H ≈ 2.5)
psi_code = compute_psi(N=70e9, D=1.4e12, L=80, E=8192, H=2.5, V=1e6)
print(f"Code domain Ψ: {psi_code:.2e}")
```

### Prediction for Future Models

```python
from compute_psi_canonical import predict_dmax

# 1T parameter model
N_future = 1e12
L_future = 200
E_future = 20480

dmax = predict_dmax(N_future, L_future, E_future)
print(f"Predicted D_max: {dmax:.2e} tokens")
```

---

## 📚 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{savitskiy2025liics,
  title={An Empirical Invariant for Transformer Scaling: Towards an Information Incompleteness Hypothesis},
  author={Savitskiy, Viktor N.},
  journal={[Journal Name]},
  year={2025},
  note={Available at: https://github.com/designhumanai/liics}
}
```

---

## 🔍 Reproducibility

### Environment

All calculations performed with:
- Python 3.9+
- NumPy 1.24.3
- Pandas 2.0.3
- SciPy 1.11.1

Install exact versions:
```bash
pip install -r requirements.txt
```

### Verification

Run test suite (requires pytest):
```bash
pytest tests/test_calculations.py
```

Expected: All tests pass, numerical differences < 1e-10.

### Commit Hash

This release corresponds to commit: `[INSERT_HASH_AFTER_UPLOAD]`

---

## 🐛 Known Issues & Limitations

1. **Sample size:** n=3 compute-optimal models limits statistical power
2. **Architecture scope:** Results specific to dense Transformers
3. **Domain normalization:** H and V approximate general text; domain-specific calibration needed
4. **Dynamic analysis:** Plateau detection during training not yet validated

See manuscript Section 6.5 for detailed discussion.

---

## 🗺️ Roadmap

### v2.1 (Planned)
- [ ] Add Mixture-of-Experts models (Mixtral, Qwen-MoE)
- [ ] State Space Model analysis (Mamba, RWKV)
- [ ] Domain-specific entropy measurements
- [ ] Interactive visualization dashboard

### v3.0 (Future)
- [ ] Dynamic G_S(C)(t) tracking during training
- [ ] Multi-modal Transformer analysis
- [ ] Theoretical derivation of Ψ_U (universal constant)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Add model data to `master_table.csv`
4. Run verification (`python scripts/compute_psi_canonical.py`)
5. Submit pull request with results

**Contribution guidelines:**
- Include source citation for new models
- Verify training reached plateau (see manuscript Definition 1)
- Update documentation if adding new analysis

---

<!-- UNIFIED CONTACTS BLOCK START -->
## 📞 Contacts

**General Inquiries:**
- 🌐 Website: [designhumanai.com](https://designhumanai.com) *(In development)*
- 📧 Email: `info@designhumanai.com`
- 💬 GitHub: [github.com/designhumanai](https://github.com/designhumanai)

**Scientific & Technical:**
- 📧 Email: `dhaie@designhumanai.com`
- 👨‍🔬 ORCID: [0000-0003-1356-7260](https://orcid.org/0000-0003-1356-7260)

**Community & Discussion:**
- 💬 GitHub Discussions: [Philosophical and technical discussions](https://github.com/designhumanai/liics/discussions)
- 💬 GitHub Issues: [Technical questions and bugs](https://github.com/designhumanai/liics/issues)

<!-- UNIFIED CONTACTS BLOCK END -->

For questions about:
- **Scientific content:** Email author directly
- **Code issues:** Open GitHub issue
- **Collaboration:** Contact via email

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

Code and data are freely available for academic and commercial use with attribution.

---

**Copyright © Viktor Savitskiy (Савицкий Виктор Николаевич), 1995–2025**  
**DHAIE Project — Design Human AI Engineering & Enhancement**  
All rights reserved under applicable international law.

**Last updated:** 2025-10-10  
**Version:** 2.0 (LLaMA correction + initial release)

---

## 🙏 Acknowledgments

- AI research assistants (ChatGPT, Gemini, DeepSeek, Claude) for computational verification
- Original authors of GPT-3, Chinchilla, PaLM, and LLaMA for publishing detailed architectural parameters
- Open-source community for Python scientific computing ecosystem

---

## 📊 Repository Statistics

![GitHub stars](https://img.shields.io/github/stars/designhumanai/liics)
![GitHub forks](https://img.shields.io/github/forks/designhumanai/liics)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)

**Last updated:** 2025-11-10  
**Version:** 2.0 (LLaMA correction + synchronization)

---

## 🔗 Related Links

- **Preprint:** [Link to arXiv/bioRxiv when available]
- **Supplementary Material:** [docs/supplementary.pdf](docs/supplementary.pdf)
- **Interactive Demo:** [Coming soon]
- **Discussions:** [GitHub Discussions](https://github.com/designhumanai/liics/discussions)

---

**⭐ If you find this work useful, please star the repository!**
