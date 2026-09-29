# Spectral Phasor Analysis

A Python desktop application for spectral and frequency-domain FLIM phasor analysis of fluorescent probes in solution (cuvette measurements).

Developed at the **Advanced Bioimaging Unit**, Institut Pasteur de Montevideo & Hospital de Clínicas, Universidad de la República (UdelaR), Uruguay.

---

## Features

- Load and preprocess fluorescence spectra (Cary Eclipse CSV format)
- Background subtraction (per-file or global)
- Optional LOESS smoothing (visualization only — not applied to phasor computation)
- Spectral phasor plot (G, S) computed from raw background-corrected signal
- Frequency-domain FLIM phasor from `.fbd` files (FLIMBox / SimFCS)
- Phasor calibration with reference fluorophore (`phasor_calibrate`, PhasorPy)
- Per-sample intensity threshold and median filtering
- Interactive cursor showing apparent lifetimes (τ_φ, τ_M) in real time
- Style editor for publication-quality figures
- Export individual plots as PNG/PDF/SVG (300 dpi)

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

> **Note:** `fbdfile` is required only for FLIM `.fbd` file support.

---

## Installation

```bash
git clone https://github.com/<your-username>/spectral-phasor-analysis.git
cd spectral-phasor-analysis
pip install -r requirements.txt
python spectral_phasor_gui.py
```

---

## Usage

### Spectral phasor (cuvette)
1. Click **➕ Agregar espectros** to load CSV files (Cary Eclipse format)
2. Set the spectral range (λ_min, λ_max) in the dialog that appears
3. Assign a background file via double-click on each file, or use **▶ Cargar BG único y restar a todos**
4. The spectral phasor is computed automatically from the background-corrected raw signal

### FLIM phasor (FLIMBox)
1. Switch to the **⏱ Phasor Lifetime** tab
2. Load a reference fluorophore `.fbd` file and enter its known lifetime (τ_ref)
3. Click **⚙ Calcular corrección** to calibrate
4. Load sample `.fbd` files and click **▶ Calcular y plotear**
5. Apply per-sample threshold and median filter as needed

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [PhasorPy](https://www.phasorpy.org) ≥ 0.9 | Phasor computation, calibration, filtering, plotting |
| NumPy | Numerical arrays |
| SciPy | Interpolation |
| pandas | CSV loading |
| matplotlib | Plot rendering |
| PyQt5 | GUI framework |
| statsmodels | LOESS smoothing |
| fbdfile | FLIMBox `.fbd` file reading |

---

## Citation

If you use this software, please cite:

Rossello J., Lopassio M., Rodriguez J., Möller M., Malacrida L.
*D-alanine analog HADA as an environment-sensitive probe: photophysical 
characterization and lifetime-based reporting of local polarity, H-bonding, 
and viscosity.* (2026) Manuscript in preparation.
---

## License

MIT License — see [LICENSE](LICENSE) for details.
