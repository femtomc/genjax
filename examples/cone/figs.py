from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from genjax.viz.standard import (
    FIGURE_SIZES,
    apply_grid_style,
    apply_standard_ticks,
    get_method_color,
    save_publication_figure,
    setup_publication_fonts,
)

from .core import (
    DEFAULT_DATA,
    expressive_variational_family,
    naive_variational_family,
    sample_guide,
    sample_model_prior,
)


setup_publication_fonts()


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _add_observation_circle(ax):
    radius = float(jnp.sqrt(DEFAULT_DATA["z"]))
    circle = plt.Circle((0.0, 0.0), radius=radius, fill=False, linewidth=3, color="black")
    ax.add_patch(circle)


def save_prior_samples_plot(
    *,
    output_dir: str | Path = "figs",
    n_samples: int = 5000,
    seed_value: int = 0,
):
    output_path = _ensure_output_dir(output_dir)

    choices = sample_model_prior(n_samples=n_samples, seed_value=seed_value)
    x = choices["x"]
    y = choices["y"]
    z = choices["z"]

    mask = z < 30.0
    x = x[mask]
    y = y[mask]
    z = z[mask]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    ax.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        s=8,
        alpha=0.4,
    )
    _add_observation_circle(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Cone prior samples (coloured by z)")

    apply_grid_style(ax)
    apply_standard_ticks(ax)

    save_publication_figure(fig, str(output_path / "cone_prior_samples.pdf"))


def save_posterior_samples_plot(
    guide,
    params,
    *,
    title: str,
    filename: str,
    output_dir: str | Path = "figs",
    n_samples: int = 50000,
    seed_value: int = 0,
):
    output_path = _ensure_output_dir(output_dir)

    choices = sample_guide(
        guide,
        params,
        n_samples=n_samples,
        seed_value=seed_value,
    )

    x = choices["x"]
    y = choices["y"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    ax.scatter(
        x,
        y,
        color=get_method_color("genjax_is"),
        s=5,
        alpha=0.12,
    )
    _add_observation_circle(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(title)

    apply_grid_style(ax)
    apply_standard_ticks(ax)

    save_publication_figure(fig, str(output_path / filename))


def save_figure_suite(
    *,
    naive_elbo_params,
    naive_iwae5_params,
    expressive_diwhvi_params,
    output_dir: str | Path = "figs",
):
    save_prior_samples_plot(output_dir=output_dir)

    save_posterior_samples_plot(
        naive_variational_family,
        naive_elbo_params,
        title="Naive guide posterior (ELBO)",
        filename="cone_naive_elbo_posterior.pdf",
        output_dir=output_dir,
        seed_value=1,
    )

    save_posterior_samples_plot(
        naive_variational_family,
        naive_iwae5_params,
        title="Naive guide posterior (IWAE, K=5)",
        filename="cone_naive_iwae5_posterior.pdf",
        output_dir=output_dir,
        seed_value=2,
    )

    save_posterior_samples_plot(
        expressive_variational_family,
        expressive_diwhvi_params,
        title="Expressive guide posterior (DIWHVI, N=5, K=5)",
        filename="cone_expressive_diwhvi_posterior.pdf",
        output_dir=output_dir,
        seed_value=3,
    )
