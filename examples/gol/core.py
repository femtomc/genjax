from functools import partial

import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jax import jit
from jax.lax import dynamic_slice, scan
from jax.nn import softmax
from matplotlib import animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from genjax import Pytree, Trace, flip, gen, get_choices, seed, trace
from genjax import modular_vmap as vmap

neighbors_filter = jnp.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


@gen  #########            (3, 3)      ()
def get_cell_from_window(window, flip_prob):
    """
    Given a 3x3 window, generate the value taken by the
    cell at the center of the window at the next
    time step.

    Args:
        window: A (3, 3) array of zeros and ones.
        flip_prob: The probability of flipping the cell
            value at the next time step.
    """
    neighbors = jnp.sum(window * neighbors_filter)
    grid = window[1, 1]
    # Apply the rules of the Game of Life
    deterministic_next_bit = jnp.where(
        (grid == 1) & ((neighbors == 2) | (neighbors == 3)),
        1,
        jnp.where((grid == 0) & (neighbors == 3), 1, 0),
    )
    p_is_one = jnp.where(deterministic_next_bit == 1, 1 - flip_prob, flip_prob)
    bit = flip(p_is_one) @ "bit"
    return bit


def get_windows(state):
    """
    Returns a (n_y, n_x, 3, 3) array of windows.
    For each cell in the given state, this array will
    contain a (3, 3) window centered at that cell.
    """
    n_y, n_x = state.shape
    # pad the state with zeros
    state_padded = jnp.pad(state, 1)
    # windows of shape (n_y, n_x, 3, 3)
    return vmap(
        lambda i: vmap(lambda j: dynamic_slice(state_padded, (i, j), (3, 3)))(
            jnp.arange(n_x)
        )
    )(jnp.arange(n_y))


@gen
def generate_next_state(prev_state, p_flip):
    """
    Given a game of life state, generate the next state.
    """
    # windows of shape (n_y, n_x, 3, 3)
    windows = get_windows(prev_state)
    # construct next state; shape is (n_y, n_x)
    return trace(
        "cells",
        get_cell_from_window.vmap(
            in_axes=(0, None)  # map over axis 1
        ).vmap(
            in_axes=(0, None)  # map over axis 0
        ),
        (windows, p_flip),
    )


@gen
def generate_state_pair(n_y: int, n_x: int, p_flip):
    """
    Generate an initial game of life state uniformly at random,
    and generate the next state given the initial state.

    Args:
        n_y: The number of rows in the state. (Const)
        n_x: The number of columns in the state. (Const)
        p_flip: The probability of flipping a cell value.
    """
    init = flip.vmap().vmap()(0.5 * jnp.ones((n_y, n_x))) @ "init"
    step = generate_next_state(init, p_flip) @ "step"
    return (init, step)


### Single Cell Gibbs Update Implementation ###

AND = jnp.logical_and


def get_gibbs_probs_fast(i, j, current_state, future_state, p_flip):
    relevant_next_ys = jnp.array([i - 1, i, i + 1])
    relevant_next_xs = jnp.array([j - 1, j, j + 1])
    current_state_padded = jnp.pad(current_state, 1)

    def window_for_next_cell_at_offset(next_y, next_x, val_at_ij):
        window = dynamic_slice(current_state_padded, (next_y, next_x), (3, 3))
        a, b = i - next_y + 1, j - next_x + 1
        is_in_range = AND(AND(0 <= a, a < 3), AND(0 <= b, b < 3))
        val = jnp.where(is_in_range, val_at_ij, window[a, b])
        window = window.at[a, b].set(val)
        return window

    def get_score_for_val(val_at_ij):
        def _next_y(next_y):
            def _next_x(next_x):
                return get_cell_from_window.log_density(
                    (
                        window_for_next_cell_at_offset(
                            next_y,
                            next_x,
                            val_at_ij,
                        ),
                        p_flip,
                    ),
                    {"bit": future_state[next_y, next_x]},
                )

            return vmap(_next_x)(relevant_next_xs)

        scores = vmap(_next_y)(relevant_next_ys)

        mask = vmap(
            lambda next_y: vmap(
                lambda next_x: AND(
                    AND(0 <= next_y, next_y < future_state.shape[1]),
                    AND(0 <= next_x, next_x < future_state.shape[0]),
                )
            )(relevant_next_xs)
        )(relevant_next_ys)
        return jnp.sum(jnp.where(mask, scores, 0))

    scores = vmap(get_score_for_val)(jnp.array([0, 1]))

    return softmax(scores)


@gen
def gibbs_move_on_cell_fast(i, j, current_state, future_state, p_flip):
    """Gibbs sample a bit for the position (i, j). Runs in O(1)."""
    p_zero = get_gibbs_probs_fast(i, j, current_state, future_state, p_flip)[0]
    val = flip(1 - p_zero) @ "bit"
    return val


### Full Gibbs Sweep Implementation ###


@gen
def gibbs_move_on_all_cells_at_offset(
    oy: int,
    ox: int,
    current_state,
    future_state,
    p_flip,
):
    i_vals = jnp.arange(oy, current_state.shape[0], 3)
    j_vals = jnp.arange(ox, current_state.shape[1], 3)
    vals = (
        gibbs_move_on_cell_fast.vmap(
            in_axes=(None, 0, None, None, None),
        ).vmap(
            in_axes=(0, None, None, None, None),
        )(i_vals, j_vals, current_state, future_state, p_flip)
        @ "cells"
    )
    return current_state.at[*jnp.ix_(i_vals, j_vals)].set(vals)


def full_gibbs_sweep(
    current_state: jnp.ndarray,
    future_state: jnp.ndarray,
    p_flip: jnp.ndarray,
):
    for oy in range(3):
        for ox in range(3):
            current_state = gibbs_move_on_all_cells_at_offset(
                oy,
                ox,
                current_state,
                future_state,
                p_flip,
            )
    return current_state


@Pytree.dataclass
class GibbsSamplerState(Pytree):
    inferred_trace: Trace

    def trace_score(self):
        return self.inferred_trace.get_score()

    def predictive_posterior_score(self):
        return (
            -self.inferred_trace._choices["step"]
            ._choices["cells"]
            ._choices["bit"]
            .get_score()
        )

    def p_flip(self):
        return self.inferred_trace.get_args()[2]

    def inferred_prev_board(self):
        return get_choices(self.inferred_trace)["init"]

    def inferred_reconstructed_target(self):
        return generate_next_state(self.inferred_prev_board(), 0.0)


@Pytree.dataclass
class GibbsSampler(Pytree):
    target_image: jnp.ndarray
    p_flip: jnp.ndarray

    def get_initial_state(self):
        # Initialize a sample.
        def initialize(step, n_y, n_x, p_flip):
            @gen
            def proposal():
                (
                    flip.vmap().vmap()(
                        0.5 * jnp.ones((n_y, n_x)),
                    )
                    @ "init"
                )

            init_q_trace = proposal.simulate()
            p_trace, _ = generate_state_pair.generate(
                {**init_q_trace.get_choices(), **step},
                n_x,
                n_y,
                p_flip,
            )
            return GibbsSamplerState(p_trace)

        return initialize(
            {"step": {"cells": {"bit": self.target_image}}},
            self.target_image.shape[0],
            self.target_image.shape[1],
            self.p_flip,
        )

    def update_state(
        self,
        current_state: GibbsSamplerState,
    ) -> GibbsSamplerState:
        p_flip = current_state.inferred_trace.get_args()[2]
        choices = get_choices(current_state.inferred_trace)
        init_img = choices["init"]
        future_img = choices["step"]["cells"]["bit"]
        new_state = full_gibbs_sweep(
            init_img,
            future_img,
            p_flip,
        )
        new_trace, _, _ = current_state.inferred_trace.update(
            current_state.inferred_trace,
            {"init": new_state},
            self.target_image.shape[0],
            self.target_image.shape[1],
            self.p_flip,
        )
        return GibbsSamplerState(new_trace)


def unfold(f, init, n_steps: int):
    """
    Unfolds the function `f` for `n_steps` steps, starting from `init`.
    Returns final_state = f(f(f(...(f(init))))), and
    all_states = [f(init), f(f(init)), ...].
    """

    def unfold_kernel(state, _):
        next_state = f(state)
        return next_state, next_state

    final_state, all_states = scan(
        unfold_kernel,
        init,
        None,
        length=n_steps,
    )
    return final_state, all_states


@Pytree.dataclass
class GibbsRunSummary(Pytree):
    trace_scores: jnp.ndarray  # shape = (n_frames,)
    predictive_posterior_scores: jnp.ndarray  # shape = (n_frames,)
    p_flips: jnp.ndarray  # shape = (n_frames,)
    inferred_prev_boards: jnp.ndarray  # shape = (n_frames, n_y, n_x)
    inferred_reconstructed_targets: jnp.ndarray  # shape = (n_frames, n_y, n_x)

    def n_incorrect_bits_in_reconstructed_image(self, target_image):
        return (self.inferred_reconstructed_targets[-1] != target_image).sum()


def _run_sampler_and_get_summary(
    sampler: GibbsSampler,
    n_steps: int,
    n_steps_per_summary_frame: int,
):
    def run_n_steps(state):
        final_state, _ = unfold(sampler.update_state, state, n_steps_per_summary_frame)
        return final_state

    def inferred_state_to_score_and_reconstructed(state: GibbsSamplerState):
        return (
            state.trace_score(),
            state.predictive_posterior_score(),
            state.p_flip(),
            state.inferred_prev_board(),
            state.inferred_reconstructed_target(),
        )

    initial_state = sampler.get_initial_state()
    _, subsequent_states = unfold(
        run_n_steps, initial_state, n_steps // n_steps_per_summary_frame
    )
    all_states = jtu.tree_map(
        lambda x, y: jnp.concatenate([jnp.expand_dims(x, 0), y], axis=0),
        initial_state,
        subsequent_states,
    )

    return GibbsRunSummary(*vmap(inferred_state_to_score_and_reconstructed)(all_states))


@partial(jit, static_argnums=(2, 3))
def run_sampler_and_get_summary(
    key,
    sampler: GibbsSampler,
    n_steps: int,
    n_steps_per_summary_frame: int,
):
    return seed(
        lambda sampler: _run_sampler_and_get_summary(
            sampler, n_steps, n_steps_per_summary_frame
        )
    )(key, sampler)


### Animation ###


def get_gol_figure_and_updater(
    target_image,
    run_summary: GibbsRunSummary,
    n_steps_per_frame: int,
    *,
    include_time_line=True,
    grid_layout=True,
):
    n_frames = len(run_summary.predictive_posterior_scores)

    if grid_layout:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs_main = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
        gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 0], hspace=0.8)
        ax1 = fig.add_subplot(gs_sub[0, 0])
        ax2 = fig.add_subplot(gs_sub[1, 0])
        ax3 = fig.add_subplot(gs_main[0, 1])
        ax4 = fig.add_subplot(gs_main[1, 0])
        ax5 = fig.add_subplot(gs_main[1, 1])
    else:
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        gs_main = GridSpec(1, 4, figure=fig)
        gs_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 0], hspace=0.8)
        ax1 = fig.add_subplot(gs_sub[0, 0])
        ax2 = fig.add_subplot(gs_sub[1, 0])
        ax4 = fig.add_subplot(gs_main[0, 1])
        ax5 = fig.add_subplot(gs_main[0, 2])
        ax3 = fig.add_subplot(gs_main[0, 3])

    # Titles for each subplot
    ax1.set_title("Predictive Posterior Score")
    ax2.set_title("Softness Parameter")
    ax3.set_title("Target State to Reconstruct")
    ax4.set_title("Inferred Previous State")
    if grid_layout:
        ax5.set_title("One-Step Rollout from Inferred Previous State")
    else:
        ax5.set_title("One-Step Rollout from\nInferred Previous State")

    # Score and p_flip plots (animated)
    (score_line,) = ax1.plot([], [], lw=2, color="black")
    (p_flip_line,) = ax2.plot([], [], lw=2, color="black")
    if include_time_line:
        score_bar = ax1.axvline(x=0, color="r", linestyle="--")
        p_flip_bar = ax2.axvline(x=0, color="r", linestyle="--")

    # Target image (non-animated, static)
    ax3.imshow(target_image, cmap="gray", vmin=0, vmax=1)
    ax3.axis("off")

    # Inferred previous board state (animated)
    prev_board_img = ax4.imshow(
        run_summary.inferred_prev_boards[0], cmap="gray", vmin=0, vmax=1
    )
    ax4.axis("off")

    # Inferred reconstructed target (animated)
    reconstructed_img = ax5.imshow(
        run_summary.inferred_reconstructed_targets[0], cmap="gray", vmin=0, vmax=1
    )
    ax5.axis("off")

    # Set limits for the score plot, adjusting for n_steps_per_frame
    original_n_frames = len(run_summary.predictive_posterior_scores) * n_steps_per_frame
    ax1.set_xlim(0, original_n_frames - 1)
    ax1.set_ylim(
        min(run_summary.predictive_posterior_scores),
        max(run_summary.predictive_posterior_scores),
    )
    ax1.set_xlabel("# Gibbs Sweeps Performed")
    ax1.set_ylabel("Log P(Target | Inferred)")

    # Set limits for the p_flip plot, adjusting for n_steps_per_frame
    ax2.set_xlim(0, original_n_frames - 1)
    ax2.set_ylim(min(run_summary.p_flips), max(run_summary.p_flips))
    ax2.set_xlabel("# Gibbs Sweeps Performed")
    ax2.set_ylabel("Softness")

    plt.tight_layout()

    def update(frame):
        """Animation function to update the frames."""
        if frame >= n_frames:
            frame = n_frames - 1

        # Compute the original frame index
        original_frame = frame * n_steps_per_frame

        # Update the plots
        score_line.set_data(
            jnp.arange(
                0, n_steps_per_frame * len(run_summary.trace_scores), n_steps_per_frame
            ),
            run_summary.predictive_posterior_scores,
        )
        p_flip_line.set_data(
            jnp.arange(
                0, n_steps_per_frame * len(run_summary.p_flips), n_steps_per_frame
            ),
            run_summary.p_flips,
        )
        prev_board_img.set_array(run_summary.inferred_prev_boards[frame])
        reconstructed_img.set_array(run_summary.inferred_reconstructed_targets[frame])
        if include_time_line:
            score_bar.set_xdata([original_frame])
            p_flip_bar.set_xdata([original_frame])

        if include_time_line:
            return (
                score_line,
                p_flip_line,
                prev_board_img,
                reconstructed_img,
                score_bar,
                p_flip_bar,
            )
        else:
            return score_line, p_flip_line, prev_board_img, reconstructed_img

    return fig, update, n_frames


def get_gol_sampler_anim(
    target_image, run_summary: GibbsRunSummary, n_steps_per_frame: int, **anim_kwargs
) -> animation.FuncAnimation:
    """
    Creates an animation of the Gibbs sampler run, showing a subset
    of the states visited by the Gibbs sampler.

    - n_steps_per_frame: The number of sampler steps that occurred between each
        saved state in the run_summary.
    """
    fig, animate, n_frames = get_gol_figure_and_updater(
        target_image, run_summary, n_steps_per_frame, **anim_kwargs
    )
    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=200, blit=True
    )
    return ani


def get_gol_sampler_lastframe_figure(
    target_image,
    run_summary: GibbsRunSummary,
    n_steps_per_frame: int,
    **anim_kwargs,
):
    fig, update, n_frames = get_gol_figure_and_updater(
        target_image,
        run_summary,
        n_steps_per_frame,
        grid_layout=False,
        include_time_line=False,
        **anim_kwargs,
    )
    update(n_frames - 1)
    return fig
