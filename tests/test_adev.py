import jax.numpy as jnp
from genjax.adev import Dual, expectation, flip_enum
from jax.lax import cond


@expectation
def flip_exact_loss(p):
    b = flip_enum(p)
    return cond(
        b,
        lambda _: 0.0,
        lambda p: -p / 2.0,
        p,
    )


def test_flip_exact_loss_jvp():
    """Test that flip_exact_loss JVP estimates match expected values."""
    test_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for p in test_values:
        p_dual = flip_exact_loss.jvp_estimate(Dual(p, 1.0))
        expected_tangent = p - 0.5

        # Test that the tangent matches the expected value
        assert jnp.allclose(p_dual.tangent, expected_tangent, atol=1e-6)


def test_flip_exact_loss_symmetry():
    """Test that the loss function has expected symmetry properties."""
    # Test symmetry around p=0.5
    p1, p2 = 0.3, 0.7
    dual1 = flip_exact_loss.jvp_estimate(Dual(p1, 1.0))
    dual2 = flip_exact_loss.jvp_estimate(Dual(p2, 1.0))

    # The tangents should be symmetric around 0
    assert jnp.allclose(dual1.tangent, -dual2.tangent, atol=1e-6)


def test_flip_exact_loss_at_half():
    """Test the loss function at p=0.5."""
    p_dual = flip_exact_loss.jvp_estimate(Dual(0.5, 1.0))

    # At p=0.5, the tangent should be 0
    assert jnp.allclose(p_dual.tangent, 0.0, atol=1e-6)
