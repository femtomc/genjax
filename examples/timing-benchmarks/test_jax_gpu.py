import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")