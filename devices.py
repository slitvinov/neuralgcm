import jax
for d in jax.devices():
    print(f"- {d}")
