import jax
import jax.numpy as jnp
from matplotlib import image as mpimg


def get_blinker_4x4():
    return jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )


def get_blinker_10x10():
    return jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def get_blinker_n(n: int):
    init = jnp.zeros((n, n), dtype=int)
    return init.at[n - 2, n - 2 : n + 1].set(1)


def get_popl_logo():
    img = jnp.array(mpimg.imread("examples/gol/assets/popl.png"))

    # Convert the image to grayscale, considering the alpha channel
    alpha_channel = img[:, :, 3]
    gray_img = jnp.mean(img[:, :, :3], axis=2)

    # Set transparent cells to white
    gray_img = jnp.where(alpha_channel == 0, 1, gray_img)

    # Apply a threshold to convert to black and white
    bw = jnp.where(gray_img < 0.7, 1, 0)

    # change dims to 512x512
    new_height = 512 * bw.shape[0] // bw.shape[1]
    with_right_dims = jax.image.resize(bw, (new_height, 512), method="nearest")

    # make a square
    pad = (512 - with_right_dims.shape[0]) // 2
    pad_top = pad
    pad_bottom = pad if with_right_dims.shape[0] + 2 * pad == 512 else pad + 1
    square = jnp.pad(
        with_right_dims,
        ((pad_top, pad_bottom), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    assert square.shape == (512, 512)
    return square


def get_mit_logo():
    img = mpimg.imread("examples/gol/assets/mit.png")

    # Convert the image to grayscale, considering the alpha channel
    alpha_channel = img[:, :, 3]
    gray_img = jnp.mean(img[:, :, :3], axis=2)

    # Set transparent cells to white
    gray_img = jnp.where(alpha_channel == 0, 1, gray_img)

    # Apply a threshold to convert to black and white
    return jnp.where(gray_img < 0.5, 1, 0)


def get_small_mit_logo(size=128):
    """Get a downsampled version of the MIT logo for faster computation."""
    full_logo = get_mit_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize


def get_small_popl_logo(size=128):
    """Get a downsampled version of the POPL logo for faster computation."""
    full_logo = get_popl_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize
