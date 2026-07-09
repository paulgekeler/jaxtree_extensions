from typing import Any

import jaxtree_extensions as jte


def test_tree_zip_mismatched_intersection() -> None:
    tree1 = {"a": [1, 2], "b": 3}
    tree2 = {"a": [10, 20, 30], "c": 4}

    res = jte.tree_zip_mismatched(lambda x, y: x + y, tree1, tree2, mode="intersection")
    assert res == {"a": [11, 22]}


def test_tree_zip_mismatched_padding() -> None:
    tree1 = {"a": [1, 2], "b": 3}
    tree2 = {"a": [10, 20, 30], "c": 4}

    res = jte.tree_zip_mismatched(
        lambda x, y: x + y, tree1, tree2, mode="padding", fill_value=0
    )
    assert res == {"a": [11, 22, 30], "b": 3, "c": 4}


def test_tree_zip_mismatched_broadcasting() -> None:
    tree = {"a": [1, 2], "b": 3}

    res = jte.tree_zip_mismatched(lambda x, y: x * y, tree, 10, mode="intersection")
    assert res == {"a": [10, 20], "b": 30}


def test_tree_zip_mismatched_equinox() -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    model = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    def scale_fn(x: Any, y: Any) -> Any:
        if not isinstance(x, (jnp.ndarray, float, int)):
            return x
        return x * y

    # Scale the model parameters by 2.0 using tree_zip_mismatched leaf broadcasting
    scaled_model = jte.tree_zip_mismatched(scale_fn, model, 2.0, mode="intersection")

    # Verify that it is still an MLP module
    assert isinstance(scaled_model, eqx.nn.MLP)

    # Verify that the values have been doubled
    orig_leaves = jax.tree_util.tree_leaves(model)
    scaled_leaves = jax.tree_util.tree_leaves(scaled_model)
    for orig, scaled in zip(orig_leaves, scaled_leaves, strict=True):
        if isinstance(orig, (jnp.ndarray, float, int)):
            assert jnp.allclose(scaled, orig * 2.0)
        else:
            assert scaled is orig
