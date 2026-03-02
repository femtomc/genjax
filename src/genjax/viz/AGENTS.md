# Visualization Module Guide

`genjax.viz` contains reusable plotting utilities shared across case studies.

## Module Map

- `standard.py`: GenJAX Research Visualization Standards (GRVS)
  - typography setup
  - color palettes
  - figure-size presets
  - publication export helpers
- `raincloud.py`
  - `horizontal_raincloud(...)`
  - `raincloud(...)`
  - `diagnostic_raincloud(...)`

## Recommended Usage

```python
from genjax.viz.standard import (
    setup_publication_fonts,
    FIGURE_SIZES,
    apply_grid_style,
    save_publication_figure,
)
from genjax.viz.raincloud import horizontal_raincloud

setup_publication_fonts()
fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
horizontal_raincloud([a, b], labels=["A", "B"], ax=ax)
apply_grid_style(ax)
save_publication_figure(fig, "weights.pdf")
```

## Styling Policy

- Prefer `standard.py` helpers over ad-hoc `matplotlib` globals.
- Keep method colors and sizing centralized for cross-figure consistency.
- Use figure helpers from case-study `figs.py` files when possible.

## Data/Rendering Notes

- Raincloud helpers convert inputs to NumPy for Matplotlib compatibility.
- KDE support relies on SciPy; keep this dependency intact when extending.
- Diagnostic plotting code should be testable independently of rendering (e.g., ESS computation paths).

## Tests / Validation

- Add focused tests for non-visual logic (thresholds, labels, returned diagnostics).
- Use smoke checks in case-study workflows for rendered outputs.
