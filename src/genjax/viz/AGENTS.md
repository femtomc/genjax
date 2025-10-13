# Visualization Module Guide

`genjax.viz` provides reusable plotting helpers used across case studies, with an emphasis on particle diagnostics.

## Key Components
- `raincloud.py`: raincloud plots for summarising weight distributions.
  - `horizontal_raincloud(...)`: half-violin + box + strip plot for one or more groups.
  - `raincloud(...)`: pandas-friendly wrapper mirroring the PtitPrince interface.
  - `diagnostic_raincloud(...)`: particle-filter specific variant returning `(ess_value, ess_label)`.
- `standard.py`: GenJAX Research Visualization Standards (GRVS) helpers for fonts, colours, figure sizing, tick handling, and publication-quality exports.

## Usage
```python
from genjax.viz.standard import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, save_publication_figure,
)
from genjax.viz.raincloud import horizontal_raincloud

setup_publication_fonts()
fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
horizontal_raincloud([weights_a, weights_b], labels=["MH", "HMC"], ax=ax)
apply_grid_style(ax)
save_publication_figure(fig, "weights.pdf")
```

`horizontal_raincloud` accepts NumPy or JAX arrays; inputs are converted to NumPy internally for Matplotlib compatibility.

## Implementation Notes
- Keep styling consistent by routing all case-study figures through the utilities in `standard.py` rather than customising `matplotlib` globally.
- Raincloud helpers depend on `scipy.stats.gaussian_kde` for density estimation; ensure SciPy remains in the environment when extending features.
- When adding new diagnostic plots, expose colour palettes and figure-size presets via `standard.py` so other modules can reuse them.

## Testing
- `tests/test_viz_raincloud.py` (if present) or case-study-specific smoke tests should cover new visual components.
- For diagnostic helpers, unit test ESS thresholds and return values independent of Matplotlib rendering.
