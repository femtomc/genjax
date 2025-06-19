import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timing
from core import (
    onepoint_curve,
    npoint_curve,
    infer_latents,
    get_points_for_inference,
)
from jax import vmap


## Onepoint trace visualization ##
def visualize_onepoint_trace(trace, ylim=(-1.5, 1.5)):
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(-1, 10, 300)
    fig = plt.figure(figsize=(2, 2))
    plt.plot(xvals, jax.vmap(curve)(xvals), color="black")
    color = "green"  # Point color for visualization
    plt.scatter(pt[0], pt[1], color=color, s=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.ylim(ylim)
    plt.tight_layout(pad=0.5)
    return fig


def save_onepoint_trace_viz():
    print("Making and saving onepoint trace visualization.")
    trace = onepoint_curve.simulate(0.0)
    fig = visualize_onepoint_trace(trace)
    fig.savefig("figs/010_onepoint_trace.pdf")


## Multipoint trace visualization ##
def visualize_multipoint_trace(
    trace,
    figsize=(4, 2),
    yrange=None,
    show_ticks=True,
    ax=None,
    min_and_max_x=(-1, 11),
):
    curve, (xs, ys) = trace.get_retval()
    xvals = jnp.linspace(*min_and_max_x, 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.plot(xvals, jax.vmap(curve)(xvals), color="black")
    ax.scatter(
        xs,
        ys,
        color="green",  # Consistent point color for visualization
        s=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if yrange is not None:
        ax.set_ylim(yrange)
    if fig is not None:
        fig.tight_layout(pad=0.5)
    return fig


def save_multipoint_trace_viz():
    print("Making and saving multipoint trace visualization.")
    import jax.numpy as jnp

    xs = jnp.linspace(-1, 11, 10)  # 10 points covering the full visualization range
    trace = npoint_curve.simulate(xs)
    fig = visualize_multipoint_trace(trace, yrange=(-1.5, 1.5))
    fig.savefig("figs/020_multipoint_trace.pdf")


## 4 Multipoint trace visualization ##
def make_fig_with_centered_number(number):
    fig = plt.figure(figsize=(1.6, 0.8))
    plt.text(0.5, 0.5, f"{number:.4f}", fontsize=20, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig


def save_four_multipoint_trace_vizs():
    print(
        "Making and saving visualizations of traces generated from modular_vmap(simulate)."
    )
    import jax.numpy as jnp
    from genjax.pjax import modular_vmap

    xs = jnp.linspace(-1, 11, 10)  # 10 points covering the full visualization range

    # Use modular_vmap to generate 4 different traces
    traces = modular_vmap(npoint_curve.simulate, axis_size=4, in_axes=None)(xs)
    for i in range(4):
        trace = jax.tree.map(lambda x: x[i], traces)
        fig = visualize_multipoint_trace(
            trace, figsize=(1, 0.5), yrange=(-3, 3), show_ticks=False
        )
        fig.savefig(f"figs/03{i}_batched_multipoint_trace.pdf", pad_inches=0)

    print("Making and saving visualizations of trace densities.")
    densities = vmap(
        lambda chm: npoint_curve.log_density(chm, xs),
        in_axes=0,
    )(traces.get_choices())
    for i in range(4):
        density_val = jnp.asarray(densities[i]).item()
        fig = make_fig_with_centered_number(density_val)
        fig.savefig(f"figs/04{i}_batched_multipoint_trace_density.pdf")


## Inference-related figures ##
def save_inference_viz(n_curves_to_plot=100):
    print("Making and saving inference visualization.")

    xvals = jnp.linspace(-1, 11, 300)
    xs, ys = get_points_for_inference()
    from genjax.core import Const
    from genjax.pjax import seed

    # Apply seeding to inference function
    seeded_infer_latents = seed(infer_latents)
    samples, weights = seeded_infer_latents(
        jrand.key(1), xs, ys, Const(int(10_000_000))
    )
    order = jnp.argsort(weights, descending=True)
    samples, weights = jax.tree.map(
        lambda x: x[order[:n_curves_to_plot]], (samples, weights)
    )
    curves = [
        jax.tree.map(lambda x: x[i], samples.get_retval()[0])
        for i in range(n_curves_to_plot)
    ]
    alphas = jnp.sqrt(jax.nn.softmax(weights))
    fig = plt.figure(figsize=(2, 2))
    for i, curve in enumerate(curves):
        plt.plot(xvals, curve(xvals), color="blue", alpha=float(alphas[i]))
        plt.scatter(xs, ys, color="black", s=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig("figs/050_inference_viz.pdf")


## Inference time & quality scaling plots ##
def get_inference_scaling_data():
    from genjax.core import Const
    from genjax.pjax import seed

    n_samples = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    mean_lml_ests = []
    mean_times = []
    xs, ys = get_points_for_inference()

    # Apply seeding to inference function
    seeded_infer_latents = seed(infer_latents)

    for n in n_samples:
        lml_ests = []
        n_const = Const(n)

        # Warm-up run
        seeded_infer_latents(jrand.key(1), xs, ys, n_const)

        # Use shared timing utility
        def inference_task():
            samples, weights = seeded_infer_latents(jrand.key(1), xs, ys, n_const)
            return weights

        times_array, (mean_time, std_time) = timing(
            inference_task,
            repeats=200,
            inner_repeats=1,
            auto_sync=True,  # Let timing handle jax.block_until_ready
        )

        # Collect LML estimates separately (not part of timing)
        for _ in range(200):
            samples, weights = seeded_infer_latents(jrand.key(1), xs, ys, n_const)
            lml_est = jax.scipy.special.logsumexp(weights) - jnp.log(n)
            lml_ests.append(lml_est)

        print(n, mean_time, std_time)
        mean_times.append(mean_time * 1000)  # Convert to milliseconds
        mean_lml_ests.append(np.mean(lml_ests))
    return n_samples, mean_lml_ests, mean_times


def save_inference_scaling_viz():
    n_samples, mean_lml_ests, mean_times = get_inference_scaling_data()

    gold_standard_lml_est = mean_lml_ests[-1]
    lml_est_errors = [gold_standard_lml_est - lml_est for lml_est in mean_lml_ests]

    n_samples, lml_est_errors, mean_times = (
        n_samples[:-1],
        lml_est_errors[:-1],
        mean_times[:-1],
    )

    ## Plot 1: Inference time scaling ##
    print("Making and saving inference time scaling visualization.")
    fig = plt.figure(figsize=(3, 1.5))
    plt.plot(n_samples, mean_times, marker="o", color="black")
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Inference\ntime (ms)")
    plt.ylim((0, 3))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig("figs/060_inference_time_scaling.pdf")

    ## Plot 2: Inference quality scaling ##
    print("Making and saving inference quality scaling visualization.")
    fig = plt.figure(figsize=(3, 2.5))
    plt.plot(mean_times, lml_est_errors, marker="o", color="black", label="IS")
    plt.xscale("log")
    plt.xlabel("Mean wall clock time (ms)")
    plt.ylabel("Error in est.\nof log P(obs)")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout(pad=0.2)
    fig.savefig("figs/061_inference_quality_scaling.pdf")


def save_comprehensive_benchmark_figure(
    n_points=20,
    n_samples=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=30,
    figsize=(16, 10),
):
    """
    Create comprehensive benchmarking figure comparing all frameworks and methods.

    Top panel: Posterior approximations
    Bottom panel: Timing comparison bar chart
    """
    from core import run_comprehensive_benchmark, extract_posterior_samples
    from data import generate_test_dataset

    print("Running comprehensive benchmark...")

    # Run benchmarks
    benchmark_results = run_comprehensive_benchmark(
        n_points=n_points,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=seed,
        timing_repeats=timing_repeats,
    )

    # Extract posterior samples
    posterior_samples = extract_posterior_samples(benchmark_results)

    # Generate true data for reference
    data = generate_test_dataset(seed=seed, n_points=n_points)
    xs_fine = np.linspace(data["xs"].min(), data["xs"].max(), 200)
    true_freq = data["true_params"]["freq"]
    true_offset = data["true_params"]["offset"]
    true_curve = np.sin(2 * np.pi * true_freq * xs_fine + true_offset)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # Top panel: Posterior approximations
    ax_posterior = fig.add_subplot(gs[0])

    # Color scheme for different frameworks and methods
    colors = {
        "genjax_importance": "#1f77b4",  # Blue
        "genjax_hmc": "#ff7f0e",  # Orange
        "numpyro_importance": "#2ca02c",  # Green
        "numpyro_hmc": "#d62728",  # Red
    }

    method_labels = {
        "genjax_importance": "GenJAX IS",
        "genjax_hmc": "GenJAX HMC",
        "numpyro_importance": "NumPyro IS",
        "numpyro_hmc": "NumPyro HMC",
    }

    # Plot true curve
    ax_posterior.plot(
        xs_fine, true_curve, "k-", linewidth=3, label="True curve", zorder=10
    )

    # Plot observed data
    ax_posterior.scatter(
        data["xs"],
        data["ys"],
        c="red",
        s=60,
        alpha=0.8,
        label="Observed data",
        zorder=10,
        edgecolor="darkred",
    )

    # Plot posterior approximations from each method
    for method_name, samples in posterior_samples.items():
        if method_name not in colors:
            continue

        freq_samples = samples["freq"]
        offset_samples = samples["offset"]

        # Sample subset of curves to plot
        n_curves = min(100, len(freq_samples))
        indices = np.linspace(0, len(freq_samples) - 1, n_curves, dtype=int)

        color = colors[method_name]
        alpha = 0.05  # Low alpha for individual curves

        for i, idx in enumerate(indices):
            freq = freq_samples[idx]
            offset = offset_samples[idx]

            # Handle different data types (JAX, NumPy, PyTorch)
            if hasattr(freq, "item"):
                freq = freq.item()
            if hasattr(offset, "item"):
                offset = offset.item()

            curve_y = np.sin(2 * np.pi * freq * xs_fine + offset)

            # Only add label for first curve of each method
            label = method_labels[method_name] if i == 0 else None
            ax_posterior.plot(
                xs_fine, curve_y, color=color, alpha=alpha, linewidth=0.5, label=label
            )

    ax_posterior.set_xlabel("x", fontsize=18)
    ax_posterior.set_ylabel("y", fontsize=18)
    ax_posterior.set_title(
        "Posterior Approximations Comparison", fontsize=20, fontweight="bold", pad=20
    )
    ax_posterior.legend(fontsize=14, loc="upper right")
    ax_posterior.grid(True, alpha=0.3)
    ax_posterior.tick_params(labelsize=20)

    # Bottom panel: Timing comparison
    ax_timing = fig.add_subplot(gs[1])

    # Extract timing data
    method_names = []
    timing_means = []
    timing_stds = []
    method_colors = []

    for method_name, result in benchmark_results.items():
        if method_name in colors:
            method_names.append(method_labels[method_name])
            timing_mean = result["timing_stats"][0] * 1000  # Convert to milliseconds
            timing_std = result["timing_stats"][1] * 1000
            timing_means.append(timing_mean)
            timing_stds.append(timing_std)
            method_colors.append(colors[method_name])

    # Create horizontal bar chart
    y_pos = np.arange(len(method_names))
    bars = ax_timing.barh(
        y_pos, timing_means, xerr=timing_stds, color=method_colors, alpha=0.7, capsize=5
    )

    ax_timing.set_yticks(y_pos)
    ax_timing.set_yticklabels(method_names, fontsize=22)
    ax_timing.set_xlabel("Time (milliseconds)", fontsize=18)
    ax_timing.set_title(
        "Inference Time Comparison", fontsize=20, fontweight="bold", pad=20
    )
    ax_timing.tick_params(labelsize=20)
    ax_timing.grid(True, axis="x", alpha=0.3)

    # Add timing values as text on bars
    for i, (bar, mean, std) in enumerate(zip(bars, timing_means, timing_stds)):
        width = bar.get_width()
        # Ensure values are Python floats, not JAX tracers
        mean_val = float(mean)
        std_val = float(std)
        ax_timing.text(
            width + std_val + max(timing_means) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.1f}±{std_val:.1f}",
            ha="left",
            va="center",
            fontsize=20,
        )

    # Add benchmark parameters as text
    param_text = f"Parameters: {n_points} points, {n_samples} samples"
    if any("hmc" in name for name in benchmark_results.keys()):
        param_text += f", {n_warmup} warmup"
    fig.text(0.02, 0.02, param_text, fontsize=14, alpha=0.7)

    # Save figure
    filename = f"figs/benchmark_comparison_n{n_points}_s{n_samples}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comprehensive benchmark figure: {filename}")

    # Print summary statistics
    print("\n=== Benchmark Summary ===")
    print(f"{'Method':<20} {'Time (ms)':<15} {'ESS':<10} {'Log ML':<10}")
    print("-" * 60)

    for method_name, result in benchmark_results.items():
        if method_name in method_labels:
            label = method_labels[method_name]
            time_ms = float(result["timing_stats"][0] * 1000)

            # Extract quality metrics if available
            ess = result.get("ess", "N/A")
            log_ml = result.get("log_marginal", "N/A")

            if isinstance(ess, (int, float)):
                ess = f"{float(ess):.0f}"
            if isinstance(log_ml, (int, float)):
                log_ml = f"{float(log_ml):.2f}"

            print(f"{label:<20} {time_ms:<15.1f} {ess:<10} {log_ml:<10}")

    return benchmark_results, posterior_samples


def plot_inference_diagnostics(curve, xs, ys, samples, log_weights):
    """
    Create diagnostic plots for GenJAX inference results.

    Args:
        curve: Curve object with parameters
        xs: Input locations
        ys: Observed data
        samples: Trace samples from inference
        log_weights: Log importance weights
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # No overall title for research quality figure - individual panel titles provide context

    # Convert to numpy for plotting
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    log_weights_np = np.array(log_weights)

    # Plot 1: Data and posterior predictive samples
    ax1 = axes[0, 0]
    ax1.scatter(xs_np, ys_np, c="red", s=50, alpha=0.7, label="Observed data", zorder=3)

    # Plot true curve for reference
    x_fine = np.linspace(xs_np.min(), xs_np.max(), 100)
    true_freq = 0.3
    true_offset = 1.5
    y_true = np.sin(2 * np.pi * true_freq * x_fine + true_offset)
    ax1.plot(x_fine, y_true, "g-", linewidth=3, alpha=0.8, label="True curve", zorder=2)

    # Importance resample to get proper posterior samples using GenJAX categorical
    n_resample = min(500, len(samples.get_choices()["curve"]["freq"]))

    # Use GenJAX categorical with log weights directly
    # Generate a key for resampling
    resample_key = jrand.key(123)  # Use a fixed seed for reproducibility

    # Sample from categorical distribution with log weights as logits
    resampled_indices = jrand.categorical(
        resample_key, jnp.array(log_weights), shape=(n_resample,)
    )

    # Sample posterior curves using properly resampled indices
    for idx in resampled_indices:
        freq = float(samples.get_choices()["curve"]["freq"][idx])
        offset = float(samples.get_choices()["curve"]["off"][idx])
        y_curve = np.sin(2 * np.pi * freq * x_fine + offset)
        ax1.plot(
            x_fine, y_curve, "b-", alpha=0.05, linewidth=0.3
        )  # Lower alpha, thinner lines

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Data + Posterior Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter posterior distributions
    ax2 = axes[0, 1]
    freqs = np.array(samples.get_choices()["curve"]["freq"])
    offsets = np.array(samples.get_choices()["curve"]["off"])

    # Use resampled indices for proper posterior representation
    resampled_freqs = freqs[resampled_indices]
    resampled_offsets = offsets[resampled_indices]

    ax2.scatter(
        resampled_freqs, resampled_offsets, alpha=0.3, s=10, label="Resampled posterior"
    )
    ax2.scatter(
        true_freq,
        true_offset,
        c="green",
        s=100,
        marker="*",
        label="True parameters",
        zorder=3,
    )
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Offset")
    ax2.set_title("Parameter Posterior")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter traces
    ax3 = axes[1, 0]
    ax3.plot(freqs, alpha=0.7, label="Frequency")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Value")
    ax3.set_title("Parameter Trace (Frequency)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Log weights distribution
    ax4 = axes[1, 1]
    ax4.hist(log_weights_np, bins=30, alpha=0.7, density=True, color="blue")
    ax4.axvline(
        log_weights_np.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {log_weights_np.mean():.2f}",
    )
    ax4.set_xlabel("Log Weight")
    ax4.set_ylabel("Density")
    ax4.set_title("Importance Weights Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/genjax_diagnostics.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved diagnostic plot as 'figs/genjax_diagnostics.pdf'")


def save_genjax_scaling_benchmark(
    n_points=15, timing_repeats=3, seed=42, figsize=(16, 12)
):
    """
    GenJAX scaling benchmark: Performance and quality analysis across sample sizes and chain lengths.

    Creates a comprehensive 12-panel figure showing:
    - IS scaling (samples vs timing, samples vs quality)
    - HMC scaling (chain length vs timing, chain length vs quality)
    - Statistical diagnostics and posterior comparison
    """
    from core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
        log_marginal_likelihood,
        effective_sample_size,
    )
    from data import generate_test_dataset
    from utils import benchmark_with_warmup
    from genjax.core import Const
    import jax.random as jrand
    from matplotlib.gridspec import GridSpec

    print("=== GenJAX Scaling Benchmark ===")
    print(f"Data points: {n_points}, Timing repeats: {timing_repeats}, Seed: {seed}")

    # Generate test data
    data = generate_test_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_freq = data["true_params"]["freq"]
    true_offset = data["true_params"]["offset"]

    print(f"True parameters: freq={true_freq:.3f}, offset={true_offset:.3f}")

    # IS Scaling Analysis
    print("\n--- Importance Sampling Scaling ---")
    is_sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    is_timings = []
    is_qualities = []
    is_ess_values = []

    for n_samples in is_sample_sizes:
        print(f"  Testing {n_samples} samples...")

        # Timing benchmark
        def is_task():
            return infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples))

        times, timing_stats = benchmark_with_warmup(is_task, repeats=timing_repeats)
        is_timings.append(timing_stats)

        # Quality assessment
        samples, log_weights = infer_latents_jit(
            jrand.key(seed), xs, ys, Const(n_samples)
        )
        log_ml = log_marginal_likelihood(log_weights)
        ess = effective_sample_size(log_weights)
        is_qualities.append(float(log_ml))
        is_ess_values.append(float(ess))

        print(
            f"    Time: {timing_stats[0] * 1000:.1f}±{timing_stats[1] * 1000:.1f}ms, "
            f"LogML: {log_ml:.3f}, ESS: {ess:.0f}"
        )

    # HMC Scaling Analysis
    print("\n--- HMC Scaling ---")
    hmc_chain_lengths = [200, 500, 1000, 2000, 5000]
    hmc_warmup = 500
    hmc_timings = []
    hmc_qualities = []
    hmc_accept_rates = []

    for n_samples in hmc_chain_lengths:
        print(f"  Testing {n_samples} samples + {hmc_warmup} warmup...")

        # Timing benchmark
        def hmc_task():
            return hmc_infer_latents_jit(
                jrand.key(seed),
                xs,
                ys,
                Const(n_samples),
                Const(hmc_warmup),
                Const(0.01),
                Const(20),
            )

        times, timing_stats = benchmark_with_warmup(
            hmc_task, repeats=max(1, timing_repeats // 2)
        )
        hmc_timings.append(timing_stats)

        # Quality assessment
        samples, diagnostics = hmc_infer_latents_jit(
            jrand.key(seed),
            xs,
            ys,
            Const(n_samples),
            Const(hmc_warmup),
            Const(0.01),
            Const(20),
        )

        # Extract posterior statistics
        freq_samples = samples.get_choices()["curve"]["freq"]
        offset_samples = samples.get_choices()["curve"]["off"]

        freq_mean = float(jnp.mean(freq_samples))
        freq_bias = abs(freq_mean - true_freq)
        offset_mean = float(jnp.mean(offset_samples))
        offset_bias = abs(offset_mean - true_offset)
        accept_rate = float(diagnostics["acceptance_rate"])

        # Use combined bias as quality metric
        combined_bias = freq_bias + offset_bias
        hmc_qualities.append(combined_bias)
        hmc_accept_rates.append(accept_rate)

        print(
            f"    Time: {timing_stats[0] * 1000:.1f}±{timing_stats[1] * 1000:.1f}ms, "
            f"Freq bias: {freq_bias:.4f}, Offset bias: {offset_bias:.4f}, "
            f"Accept: {accept_rate:.3f}"
        )

    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.8, wspace=0.4)

    # Colors for methods
    is_color = "#1f77b4"
    hmc_color = "#ff7f0e"

    # Row 1: IS Performance vs Sample Size
    ax1 = fig.add_subplot(gs[0, 0])
    is_times_mean = [t[0] * 1000 for t in is_timings]
    is_times_std = [t[1] * 1000 for t in is_timings]
    ax1.errorbar(
        is_sample_sizes,
        is_times_mean,
        yerr=is_times_std,
        marker="o",
        color=is_color,
        capsize=5,
        linewidth=2,
    )
    ax1.set_xlabel("Number of Samples", fontsize=18)
    ax1.set_ylabel("Time (ms)", fontsize=18)
    ax1.set_title("IS Performance Scaling", fontsize=20, fontweight="bold")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=20)

    # Row 1: IS Quality vs Sample Size
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(is_sample_sizes, is_qualities, marker="o", color=is_color, linewidth=2)
    ax2.set_xlabel("Number of Samples", fontsize=18)
    ax2.set_ylabel("Log Marginal Likelihood", fontsize=18)
    ax2.set_title("IS Quality Scaling", fontsize=20, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=20)

    # Row 1: IS Effective Sample Size
    ax3 = fig.add_subplot(gs[0, 2])
    ess_efficiency = [ess / n for ess, n in zip(is_ess_values, is_sample_sizes)]
    ax3.plot(is_sample_sizes, ess_efficiency, marker="o", color=is_color, linewidth=2)
    ax3.set_xlabel("Number of Samples", fontsize=18)
    ax3.set_ylabel("ESS Efficiency", fontsize=18)
    ax3.set_title("IS Sampling Efficiency", fontsize=20, fontweight="bold")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=20)

    # Row 2: HMC Performance vs Chain Length
    ax4 = fig.add_subplot(gs[1, 0])
    hmc_times_mean = [t[0] * 1000 for t in hmc_timings]
    hmc_times_std = [t[1] * 1000 for t in hmc_timings]
    ax4.errorbar(
        hmc_chain_lengths,
        hmc_times_mean,
        yerr=hmc_times_std,
        marker="s",
        color=hmc_color,
        capsize=5,
        linewidth=2,
    )
    ax4.set_xlabel("Chain Length", fontsize=18)
    ax4.set_ylabel("Time (ms)", fontsize=18)
    ax4.set_title("HMC Performance Scaling", fontsize=20, fontweight="bold")
    ax4.set_xscale("log")
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=20)

    # Row 2: HMC Quality vs Chain Length
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(hmc_chain_lengths, hmc_qualities, marker="s", color=hmc_color, linewidth=2)
    ax5.set_xlabel("Chain Length", fontsize=18)
    ax5.set_ylabel("Parameter Bias", fontsize=18)
    ax5.set_title("HMC Quality Scaling", fontsize=20, fontweight="bold")
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=20)

    # Row 2: HMC Acceptance Rate
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(
        hmc_chain_lengths, hmc_accept_rates, marker="s", color=hmc_color, linewidth=2
    )
    ax6.axhline(y=0.65, color="red", linestyle="--", alpha=0.7, label="Target")
    ax6.set_xlabel("Chain Length", fontsize=18)
    ax6.set_ylabel("Acceptance Rate", fontsize=18)
    ax6.set_title("HMC Acceptance Rate", fontsize=20, fontweight="bold")
    ax6.set_xscale("log")
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=16)
    ax6.tick_params(labelsize=20)

    # Row 3: Performance Comparison (best configurations)
    ax7 = fig.add_subplot(gs[2, 0])
    best_is_idx = -1  # Use largest sample size
    best_hmc_idx = -1  # Use longest chain

    methods = ["IS (10k)", "HMC (5k)"]
    times = [is_times_mean[best_is_idx], hmc_times_mean[best_hmc_idx]]
    time_stds = [is_times_std[best_is_idx], hmc_times_std[best_hmc_idx]]
    colors = [is_color, hmc_color]

    bars = ax7.bar(methods, times, yerr=time_stds, color=colors, alpha=0.7, capsize=5)
    ax7.set_ylabel("Time (ms)", fontsize=18)
    ax7.set_title("Best Configuration Timing", fontsize=20, fontweight="bold")
    ax7.tick_params(labelsize=18)
    ax7.grid(True, axis="y", alpha=0.3)

    # Add values on bars
    for bar, time, std in zip(bars, times, time_stds):
        height = bar.get_height()
        ax7.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + max(times) * 0.01,
            f"{time:.1f}±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    # Row 3: Quality vs Time Trade-off
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(
        is_times_mean,
        is_qualities,
        "o-",
        color=is_color,
        linewidth=2,
        markersize=8,
        label="IS",
    )

    # For HMC, invert bias to make it comparable (higher is better)
    hmc_quality_inverted = [-q for q in hmc_qualities]
    ax8.plot(
        hmc_times_mean,
        hmc_quality_inverted,
        "s-",
        color=hmc_color,
        linewidth=2,
        markersize=8,
        label="HMC",
    )

    ax8.set_xlabel("Time (ms)", fontsize=18)
    ax8.set_ylabel("Quality Metric", fontsize=18)
    ax8.set_title("Quality vs Performance", fontsize=20, fontweight="bold")
    ax8.set_xscale("log")
    ax8.legend(fontsize=16)
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(labelsize=20)

    # Row 3: Statistical Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    # Best IS results
    best_is_time = is_times_mean[best_is_idx]
    best_is_quality = is_qualities[best_is_idx]
    best_is_ess = is_ess_values[best_is_idx]

    # Best HMC results
    best_hmc_time = hmc_times_mean[best_hmc_idx]
    best_hmc_quality = hmc_qualities[best_hmc_idx]
    best_hmc_accept = hmc_accept_rates[best_hmc_idx]

    summary_text = f"""Best Configuration Results:

IS (10k samples):
  Time: {best_is_time:.1f} ms
  LogML: {best_is_quality:.3f}
  ESS: {best_is_ess:.0f}

HMC (5k samples):
  Time: {best_hmc_time:.1f} ms
  Bias: {best_hmc_quality:.4f}
  Accept: {best_hmc_accept:.3f}

True Parameters:
  Freq: {true_freq:.3f}
  Offset: {true_offset:.3f}"""

    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=16,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Row 4: Posterior Comparison (span all columns)
    ax10 = fig.add_subplot(gs[3, :])

    # Generate fine grid for plotting
    x_fine = np.linspace(xs.min() - 1, xs.max() + 1, 200)
    true_curve = np.sin(2 * np.pi * true_freq * x_fine + true_offset)

    # Get samples from best configurations
    best_is_samples, _ = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(is_sample_sizes[best_is_idx])
    )
    best_hmc_samples, _ = hmc_infer_latents_jit(
        jrand.key(seed),
        xs,
        ys,
        Const(hmc_chain_lengths[best_hmc_idx]),
        Const(hmc_warmup),
        Const(0.01),
        Const(20),
    )

    # Plot IS posterior curves
    is_freq_samples = best_is_samples.get_choices()["curve"]["freq"]
    is_offset_samples = best_is_samples.get_choices()["curve"]["off"]

    # Importance resample IS curves
    log_marginal_likelihood(
        np.ones(len(is_freq_samples))
    )  # Equal weights approximation
    n_curves = min(50, len(is_freq_samples))
    is_indices = np.linspace(0, len(is_freq_samples) - 1, n_curves, dtype=int)

    for idx in is_indices:
        freq = float(is_freq_samples[idx])
        offset = float(is_offset_samples[idx])
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        ax10.plot(x_fine, curve_y, color=is_color, alpha=0.1, linewidth=0.5)

    # Plot HMC posterior curves
    hmc_freq_samples = best_hmc_samples.get_choices()["curve"]["freq"]
    hmc_offset_samples = best_hmc_samples.get_choices()["curve"]["off"]

    hmc_indices = np.linspace(0, len(hmc_freq_samples) - 1, n_curves, dtype=int)
    for idx in hmc_indices:
        freq = float(hmc_freq_samples[idx])
        offset = float(hmc_offset_samples[idx])
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        ax10.plot(x_fine, curve_y, color=hmc_color, alpha=0.1, linewidth=0.5)

    # Plot true curve and data
    ax10.plot(
        x_fine, true_curve, color="black", linewidth=3, label="True Curve", zorder=10
    )
    ax10.scatter(
        xs,
        ys,
        color="red",
        s=60,
        alpha=0.8,
        label="Observed Data",
        zorder=15,
        edgecolor="darkred",
    )

    # Plot posterior means
    is_freq_mean = float(jnp.mean(is_freq_samples))
    is_offset_mean = float(jnp.mean(is_offset_samples))
    is_mean_curve = np.sin(2 * np.pi * is_freq_mean * x_fine + is_offset_mean)
    ax10.plot(
        x_fine,
        is_mean_curve,
        color=is_color,
        linewidth=2,
        linestyle="--",
        label="IS Mean",
        zorder=12,
    )

    hmc_freq_mean = float(jnp.mean(hmc_freq_samples))
    hmc_offset_mean = float(jnp.mean(hmc_offset_samples))
    hmc_mean_curve = np.sin(2 * np.pi * hmc_freq_mean * x_fine + hmc_offset_mean)
    ax10.plot(
        x_fine,
        hmc_mean_curve,
        color=hmc_color,
        linewidth=2,
        linestyle="--",
        label="HMC Mean",
        zorder=12,
    )

    ax10.set_xlabel("x", fontsize=18)
    ax10.set_ylabel("y", fontsize=18)
    ax10.set_title(
        "Posterior Comparison (Best Configurations)", fontsize=20, fontweight="bold"
    )
    ax10.legend(fontsize=16)
    ax10.grid(True, alpha=0.3)
    ax10.tick_params(labelsize=20)

    # Save figure
    filename = f"figs/genjax_scaling_benchmark_n{n_points}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved GenJAX scaling benchmark figure: {filename}")
    print("=== GenJAX Scaling Benchmark Complete ===")


def save_genjax_posterior_comparison(
    n_points=15,
    n_samples_is=5000,
    n_samples_hmc=5000,
    n_warmup_hmc=3000,
    n_samples_numpyro_hmc=5000,
    n_warmup_numpyro_hmc=3000,
    seed=42,
    n_curves_to_plot=100,
    timing_repeats=3,
    figsize=(16, 12),
):
    """
    Comprehensive method comparison: GenJAX IS, GenJAX HMC, and NumPyro HMC.

    Creates a 4-row publication-quality figure with:
    - Individual method posterior plots + parameter estimates
    - Combined overlay comparison
    - Performance bar chart
    - Comprehensive shared legend
    """
    from core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
        numpyro_run_hmc_inference_jit,
    )
    from data import generate_test_dataset
    from utils import benchmark_with_warmup
    from genjax.core import Const
    import jax.random as jrand
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    print("=== GenJAX Posterior Comparison ===")
    print(f"Data points: {n_points}")
    print(
        f"IS samples: {n_samples_is}, HMC samples: {n_samples_hmc}, NumPyro HMC samples: {n_samples_numpyro_hmc}"
    )

    # Generate test data
    data = generate_test_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_freq = data["true_params"]["freq"]
    true_offset = data["true_params"]["offset"]

    print(f"True parameters: freq={true_freq:.3f}, offset={true_offset:.3f}")

    # Create fine grid for plotting curves
    x_fine = np.linspace(xs.min() - 1, xs.max() + 1, 200)
    true_curve = np.sin(2 * np.pi * true_freq * x_fine + true_offset)

    # Run GenJAX Importance Sampling
    print("\nRunning GenJAX Importance Sampling...")

    def is_task():
        return infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples_is))

    is_times, is_timing_stats = benchmark_with_warmup(is_task, repeats=timing_repeats)

    is_samples, is_log_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )

    # Extract IS posterior samples with importance resampling
    is_freq_samples = is_samples.get_choices()["curve"]["freq"]
    is_offset_samples = is_samples.get_choices()["curve"]["off"]

    # Resample according to importance weights
    n_resample = min(n_curves_to_plot * 2, n_samples_is)
    is_indices = jrand.categorical(jrand.key(123), is_log_weights, shape=(n_resample,))
    is_freq_resampled = is_freq_samples[is_indices]
    is_offset_resampled = is_offset_samples[is_indices]

    # Run GenJAX HMC
    print("Running GenJAX HMC...")

    def hmc_task():
        return hmc_infer_latents_jit(
            jrand.key(seed + 1),
            xs,
            ys,
            Const(n_samples_hmc),
            Const(n_warmup_hmc),
            Const(0.01),
            Const(20),
        )

    hmc_times, hmc_timing_stats = benchmark_with_warmup(
        hmc_task, repeats=max(1, timing_repeats // 2)
    )

    hmc_samples, hmc_diagnostics = hmc_infer_latents_jit(
        jrand.key(seed + 1),
        xs,
        ys,
        Const(n_samples_hmc),
        Const(n_warmup_hmc),
        Const(0.01),
        Const(20),
    )

    # Extract HMC posterior samples
    hmc_freq_samples = hmc_samples.get_choices()["curve"]["freq"]
    hmc_offset_samples = hmc_samples.get_choices()["curve"]["off"]

    # Subsample for plotting with thinning
    thin_factor = 5
    hmc_freq_thinned = hmc_freq_samples[::thin_factor]
    hmc_offset_thinned = hmc_offset_samples[::thin_factor]

    hmc_indices = np.linspace(
        0,
        len(hmc_freq_thinned) - 1,
        min(n_curves_to_plot, len(hmc_freq_thinned)),
        dtype=int,
    )
    hmc_freq_plot = hmc_freq_thinned[hmc_indices]
    hmc_offset_plot = hmc_offset_thinned[hmc_indices]

    # Take subset of IS samples for plotting
    is_plot_indices = np.arange(min(n_curves_to_plot, len(is_freq_resampled)))
    is_freq_plot = is_freq_resampled[is_plot_indices]
    is_offset_plot = is_offset_resampled[is_plot_indices]

    # Run NumPyro HMC
    print("Running NumPyro HMC...")

    def numpyro_hmc_task():
        return numpyro_run_hmc_inference_jit(
            jrand.key(seed + 2), xs, ys, n_samples_numpyro_hmc, n_warmup_numpyro_hmc
        )

    numpyro_hmc_times, numpyro_hmc_timing_stats = benchmark_with_warmup(
        numpyro_hmc_task, repeats=max(1, timing_repeats // 2)
    )

    numpyro_hmc_result = numpyro_run_hmc_inference_jit(
        jrand.key(seed + 2), xs, ys, n_samples_numpyro_hmc, n_warmup_numpyro_hmc
    )

    # Extract NumPyro HMC posterior samples
    numpyro_freq_samples = numpyro_hmc_result["samples"]["freq"]
    numpyro_offset_samples = numpyro_hmc_result["samples"]["offset"]

    # Subsample NumPyro HMC for plotting with thinning
    numpyro_freq_thinned = numpyro_freq_samples[::thin_factor]
    numpyro_offset_thinned = numpyro_offset_samples[::thin_factor]

    numpyro_indices = np.linspace(
        0,
        len(numpyro_freq_thinned) - 1,
        min(n_curves_to_plot, len(numpyro_freq_thinned)),
        dtype=int,
    )
    numpyro_freq_plot = numpyro_freq_thinned[numpyro_indices]
    numpyro_offset_plot = numpyro_offset_thinned[numpyro_indices]

    # Compute posterior statistics
    is_stats = {
        "freq_mean": float(jnp.mean(is_freq_resampled)),
        "freq_std": float(jnp.std(is_freq_resampled)),
        "offset_mean": float(jnp.mean(is_offset_resampled)),
        "offset_std": float(jnp.std(is_offset_resampled)),
    }

    hmc_stats = {
        "freq_mean": float(jnp.mean(hmc_freq_samples)),
        "freq_std": float(jnp.std(hmc_freq_samples)),
        "offset_mean": float(jnp.mean(hmc_offset_samples)),
        "offset_std": float(jnp.std(hmc_offset_samples)),
        "acceptance_rate": float(hmc_diagnostics["acceptance_rate"]),
    }

    # Compute NumPyro acceptance rate
    accept_probs = numpyro_hmc_result["diagnostics"]["accept_probs"]
    if len(accept_probs) > 0 and not jnp.all(jnp.isnan(accept_probs)):
        numpyro_acceptance_rate = float(
            jnp.mean(accept_probs[~jnp.isnan(accept_probs)])
        )
    else:
        numpyro_acceptance_rate = 0.0

    numpyro_hmc_stats = {
        "freq_mean": float(jnp.mean(numpyro_freq_samples)),
        "freq_std": float(jnp.std(numpyro_freq_samples)),
        "offset_mean": float(jnp.mean(numpyro_offset_samples)),
        "offset_std": float(jnp.std(numpyro_offset_samples)),
        "acceptance_rate": numpyro_acceptance_rate,
    }

    print(
        f"\nIS Posterior: freq={is_stats['freq_mean']:.3f}±{is_stats['freq_std']:.3f}, "
        f"offset={is_stats['offset_mean']:.3f}±{is_stats['offset_std']:.3f}"
    )
    print(
        f"IS Timing: {is_timing_stats[0] * 1000:.1f}±{is_timing_stats[1] * 1000:.1f}ms"
    )
    print(
        f"HMC Posterior: freq={hmc_stats['freq_mean']:.3f}±{hmc_stats['freq_std']:.3f}, "
        f"offset={hmc_stats['offset_mean']:.3f}±{hmc_stats['offset_std']:.3f}"
    )
    print(
        f"HMC Timing: {hmc_timing_stats[0] * 1000:.1f}±{hmc_timing_stats[1] * 1000:.1f}ms"
    )
    print(f"HMC Acceptance Rate: {hmc_stats['acceptance_rate']:.3f}")
    print(
        f"NumPyro HMC Posterior: freq={numpyro_hmc_stats['freq_mean']:.3f}±{numpyro_hmc_stats['freq_std']:.3f}, "
        f"offset={numpyro_hmc_stats['offset_mean']:.3f}±{numpyro_hmc_stats['offset_std']:.3f}"
    )
    print(
        f"NumPyro HMC Timing: {numpyro_hmc_timing_stats[0] * 1000:.1f}±{numpyro_hmc_timing_stats[1] * 1000:.1f}ms"
    )
    print(f"NumPyro HMC Acceptance Rate: {numpyro_hmc_stats['acceptance_rate']:.3f}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.9,
        wspace=0.4,
        width_ratios=[2, 2, 1.2],
        height_ratios=[1, 1, 0.6, 0.4],
    )

    # Colors
    is_color = "#1f77b4"  # Blue for IS
    hmc_color = "#ff7f0e"  # Orange for HMC
    numpyro_color = "#2ca02c"  # Green for NumPyro HMC
    true_color = "black"  # Black for true curve
    data_color = "red"  # Red for observed data

    # Plot 1: Importance Sampling Posterior Curves
    ax1 = fig.add_subplot(gs[0, 0])

    for i, (freq, offset) in enumerate(zip(is_freq_plot, is_offset_plot)):
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        alpha = 0.1 if len(is_freq_plot) > 50 else 0.2
        label = "IS Posterior" if i == 0 else None
        ax1.plot(
            x_fine, curve_y, color=is_color, alpha=alpha, linewidth=0.5, label=label
        )

    ax1.plot(
        x_fine, true_curve, color=true_color, linewidth=3, label="True Curve", zorder=10
    )
    ax1.scatter(
        xs,
        ys,
        color=data_color,
        s=60,
        alpha=0.8,
        label="Observed Data",
        zorder=15,
        edgecolor="darkred",
    )

    # Plot posterior mean curve
    mean_curve = np.sin(
        2 * np.pi * is_stats["freq_mean"] * x_fine + is_stats["offset_mean"]
    )
    ax1.plot(
        x_fine,
        mean_curve,
        color=is_color,
        linewidth=2,
        linestyle="--",
        label="IS Mean",
        zorder=12,
    )

    ax1.set_xlabel("x", fontsize=18)
    ax1.set_ylabel("y", fontsize=18)
    ax1.set_title("IS Posterior", fontsize=20, fontweight="bold")
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)

    # Plot 2: HMC Posterior Curves
    ax2 = fig.add_subplot(gs[0, 1])

    for i, (freq, offset) in enumerate(zip(hmc_freq_plot, hmc_offset_plot)):
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        alpha = 0.1 if len(hmc_freq_plot) > 50 else 0.2
        label = "HMC Posterior" if i == 0 else None
        ax2.plot(
            x_fine, curve_y, color=hmc_color, alpha=alpha, linewidth=0.5, label=label
        )

    ax2.plot(
        x_fine, true_curve, color=true_color, linewidth=3, label="True Curve", zorder=10
    )
    ax2.scatter(
        xs,
        ys,
        color=data_color,
        s=60,
        alpha=0.8,
        label="Observed Data",
        zorder=15,
        edgecolor="darkred",
    )

    # Plot posterior mean curve
    mean_curve = np.sin(
        2 * np.pi * hmc_stats["freq_mean"] * x_fine + hmc_stats["offset_mean"]
    )
    ax2.plot(
        x_fine,
        mean_curve,
        color=hmc_color,
        linewidth=2,
        linestyle="--",
        label="HMC Mean",
        zorder=12,
    )

    ax2.set_xlabel("x", fontsize=18)
    ax2.set_ylabel("y", fontsize=18)
    ax2.set_title("HMC Posterior", fontsize=20, fontweight="bold")
    ax2.tick_params(labelsize=20)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter Space Comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])

    ax3.scatter(
        is_freq_plot, is_offset_plot, alpha=0.6, s=20, color=is_color, label="IS"
    )
    ax3.scatter(
        hmc_freq_plot, hmc_offset_plot, alpha=0.6, s=20, color=hmc_color, label="HMC"
    )
    ax3.scatter(
        true_freq,
        true_offset,
        color=true_color,
        s=100,
        marker="*",
        label="True",
        zorder=10,
        edgecolor="white",
        linewidth=1,
    )

    # Plot posterior means
    ax3.scatter(
        is_stats["freq_mean"],
        is_stats["offset_mean"],
        color=is_color,
        s=80,
        marker="D",
        edgecolor="white",
        linewidth=1,
        label="IS Mean",
        zorder=8,
    )
    ax3.scatter(
        hmc_stats["freq_mean"],
        hmc_stats["offset_mean"],
        color=hmc_color,
        s=80,
        marker="D",
        edgecolor="white",
        linewidth=1,
        label="HMC Mean",
        zorder=8,
    )

    ax3.set_xlabel("Frequency", fontsize=18)
    ax3.set_ylabel("Offset", fontsize=18)
    ax3.set_title("Parameter Estimates", fontsize=20, fontweight="bold")
    ax3.tick_params(labelsize=20)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overlay Comparison (bottom spanning all 3 columns)
    ax4 = fig.add_subplot(gs[1, :])

    # Plot all three posteriors together with reduced alpha
    for i, (freq, offset) in enumerate(zip(is_freq_plot[:50], is_offset_plot[:50])):
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        label = "IS Posterior" if i == 0 else None
        ax4.plot(
            x_fine, curve_y, color=is_color, alpha=0.05, linewidth=0.5, label=label
        )

    for i, (freq, offset) in enumerate(zip(hmc_freq_plot[:50], hmc_offset_plot[:50])):
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        label = "HMC Posterior" if i == 0 else None
        ax4.plot(
            x_fine, curve_y, color=hmc_color, alpha=0.05, linewidth=0.5, label=label
        )

    for i, (freq, offset) in enumerate(
        zip(numpyro_freq_plot[:50], numpyro_offset_plot[:50])
    ):
        curve_y = np.sin(2 * np.pi * freq * x_fine + offset)
        label = "NumPyro HMC Posterior" if i == 0 else None
        ax4.plot(
            x_fine, curve_y, color=numpyro_color, alpha=0.05, linewidth=0.5, label=label
        )

    # Plot means and true curve
    is_mean_curve = np.sin(
        2 * np.pi * is_stats["freq_mean"] * x_fine + is_stats["offset_mean"]
    )
    hmc_mean_curve = np.sin(
        2 * np.pi * hmc_stats["freq_mean"] * x_fine + hmc_stats["offset_mean"]
    )
    numpyro_mean_curve = np.sin(
        2 * np.pi * numpyro_hmc_stats["freq_mean"] * x_fine
        + numpyro_hmc_stats["offset_mean"]
    )

    ax4.plot(
        x_fine,
        is_mean_curve,
        color=is_color,
        linewidth=3,
        linestyle="--",
        label="IS Mean",
        zorder=12,
    )
    ax4.plot(
        x_fine,
        hmc_mean_curve,
        color=hmc_color,
        linewidth=3,
        linestyle="--",
        label="HMC Mean",
        zorder=12,
    )
    ax4.plot(
        x_fine,
        numpyro_mean_curve,
        color=numpyro_color,
        linewidth=3,
        linestyle="--",
        label="NumPyro HMC Mean",
        zorder=12,
    )
    ax4.plot(
        x_fine, true_curve, color=true_color, linewidth=4, label="True Curve", zorder=15
    )
    ax4.scatter(
        xs,
        ys,
        color=data_color,
        s=80,
        alpha=0.9,
        label="Observed Data",
        zorder=20,
        edgecolor="darkred",
        linewidth=1,
    )

    ax4.set_xlabel("x", fontsize=18)
    ax4.set_ylabel("y", fontsize=18)
    ax4.set_title("IS vs HMC Posterior Comparison", fontsize=20, fontweight="bold")
    ax4.tick_params(labelsize=20)
    ax4.grid(True, alpha=0.3)

    # Plot 6: Timing Comparison (third row, spanning all columns)
    ax6 = fig.add_subplot(gs[2, :])

    # Timing data
    methods = ["IS", "GenJAX HMC", "HMC"]
    times = [
        is_timing_stats[0] * 1000,
        hmc_timing_stats[0] * 1000,
        numpyro_hmc_timing_stats[0] * 1000,
    ]
    time_stds = [
        is_timing_stats[1] * 1000,
        hmc_timing_stats[1] * 1000,
        numpyro_hmc_timing_stats[1] * 1000,
    ]
    colors = [is_color, hmc_color, numpyro_color]

    # Create horizontal bar chart
    y_pos = np.arange(len(methods))
    bars = ax6.barh(y_pos, times, xerr=time_stds, color=colors, alpha=0.7, capsize=5)

    ax6.set_yticks([])
    ax6.set_yticklabels([])
    ax6.set_xlabel("Time (milliseconds)", fontsize=18)
    ax6.set_title("Performance Comparison", fontsize=20, fontweight="bold")
    ax6.tick_params(labelsize=20)
    ax6.grid(True, axis="x", alpha=0.3)

    # Add timing values as text on bars
    for i, (bar, time, std) in enumerate(zip(bars, times, time_stds)):
        width = bar.get_width()
        ax6.text(
            width + std + max(times) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{time:.1f}±{std:.1f}ms",
            ha="left",
            va="center",
            fontsize=20,
        )

    # Plot 7: Shared Legend (bottom row, spanning all columns)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")

    # Create shared legend with all unique elements
    legend_elements = [
        Line2D([0], [0], color=true_color, linewidth=3, label="True Curve"),
        Line2D(
            [0],
            [0],
            color=is_color,
            alpha=0.3,
            linewidth=2,
            label="GenJAX IS Posterior",
        ),
        Line2D(
            [0],
            [0],
            color=is_color,
            linewidth=2,
            linestyle="--",
            label="GenJAX IS Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=true_color,
            markersize=12,
            label="True Parameters",
            markeredgecolor="white",
        ),
        Line2D(
            [0],
            [0],
            color=hmc_color,
            alpha=0.3,
            linewidth=2,
            label="GenJAX HMC Posterior",
        ),
        Line2D(
            [0],
            [0],
            color=hmc_color,
            linewidth=2,
            linestyle="--",
            label="GenJAX HMC Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=data_color,
            markersize=8,
            label="Observed Data",
            markeredgecolor="darkred",
        ),
        Line2D(
            [0],
            [0],
            color=numpyro_color,
            alpha=0.3,
            linewidth=2,
            label="NumPyro HMC Posterior",
        ),
        Line2D(
            [0],
            [0],
            color=numpyro_color,
            linewidth=2,
            linestyle="--",
            label="NumPyro HMC Mean",
        ),
    ]

    ax7.legend(
        handles=legend_elements,
        loc="center",
        ncol=3,
        fontsize=20,
        frameon=False,
        columnspacing=1.0,
    )

    # Save figure
    filename = f"figs/genjax_posterior_comparison_n{n_points}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved posterior comparison figure: {filename}")
    print("=== GenJAX Posterior Comparison Complete ===")
