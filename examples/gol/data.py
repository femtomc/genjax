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
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "popl.png")
    img = jnp.array(mpimg.imread(img_path))

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


def get_popl_logo_white_lambda():
    """Load POPL logo with lambda thresholded to white instead of black."""
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "popl.png")
    img = jnp.array(mpimg.imread(img_path))

    # Convert the image to grayscale, considering the alpha channel
    alpha_channel = img[:, :, 3]
    gray_img = jnp.mean(img[:, :, :3], axis=2)

    # Set transparent cells to white
    gray_img = jnp.where(alpha_channel == 0, 1, gray_img)

    # Apply a lower threshold (0.3) to keep the lambda white
    # Only very dark parts become black (1), everything else stays white (0)
    bw = jnp.where(gray_img < 0.3, 1, 0)

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
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "mit.png")
    img = mpimg.imread(img_path)

    # Convert the image to grayscale, considering the alpha channel
    alpha_channel = img[:, :, 3]
    gray_img = jnp.mean(img[:, :, :3], axis=2)

    # Set transparent cells to white
    gray_img = jnp.where(alpha_channel == 0, 1, gray_img)

    # Apply a threshold to convert to black and white
    return jnp.where(gray_img < 0.5, 1, 0)


def get_small_mit_logo(size=128):
    """Get a downsampled version of the MIT logo for faster computation.

    Note: For sizes larger than 512, will upscale from the 512x512 original.
    """
    full_logo = get_mit_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize


def get_small_popl_logo(size=128):
    """Get a downsampled version of the POPL logo for faster computation."""
    full_logo = get_popl_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize


def get_small_popl_logo_white_lambda(size=128):
    """Get a downsampled version of the POPL logo with white lambda."""
    full_logo = get_popl_logo_white_lambda()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize


def get_hermes_logo():
    """Load high contrast Hermes logo."""
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "hermes.jpeg")
    img = jnp.array(mpimg.imread(img_path))

    # Convert to grayscale (already high contrast)
    if len(img.shape) == 3:
        gray_img = jnp.mean(img[:, :, :3], axis=2)
    else:
        gray_img = img

    # Apply threshold (lower for high contrast image)
    bw = jnp.where(gray_img < 128, 1, 0)

    # Resize to 512x512 maintaining aspect ratio
    height, width = bw.shape[:2]
    if height > width:
        new_height = 512
        new_width = int(512 * width / height)
    else:
        new_width = 512
        new_height = int(512 * height / width)

    resized = jax.image.resize(bw, (new_height, new_width), method="nearest")

    # Pad to make square
    pad_y = (512 - new_height) // 2
    pad_x = (512 - new_width) // 2

    pad_top = pad_y
    pad_bottom = 512 - new_height - pad_top
    pad_left = pad_x
    pad_right = 512 - new_width - pad_left

    square = jnp.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    assert square.shape == (512, 512)
    return square


def get_small_hermes_logo(size=128):
    """Get a downsampled version of the Hermes logo.

    Note: For sizes larger than 512, will upscale from the 512x512 original.
    """
    full_logo = get_hermes_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize


def get_wizards_logo():
    """Load wizards image and convert to binary pattern."""
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "wizards.jpg")
    img = jnp.array(mpimg.imread(img_path))

    # Convert to grayscale
    if len(img.shape) == 3:
        gray_img = jnp.mean(img[:, :, :3], axis=2)
    else:
        gray_img = img

    # Apply threshold
    bw = jnp.where(gray_img < 128, 1, 0)

    # Resize to 512x512 maintaining aspect ratio
    height, width = bw.shape[:2]
    if height > width:
        new_height = 512
        new_width = int(512 * width / height)
    else:
        new_width = 512
        new_height = int(512 * height / width)

    resized = jax.image.resize(bw, (new_height, new_width), method="nearest")

    # Pad to make square
    pad_y = (512 - new_height) // 2
    pad_x = (512 - new_width) // 2

    pad_top = pad_y
    pad_bottom = 512 - new_height - pad_top
    pad_left = pad_x
    pad_right = 512 - new_width - pad_left

    square = jnp.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    assert square.shape == (512, 512)
    return square


def get_small_wizards_logo(size=128):
    """Get a downsampled version of the wizards logo.

    Note: For sizes larger than 512, will upscale from the 512x512 original.
    """
    full_logo = get_wizards_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize
