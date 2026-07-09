from collections import namedtuple

import jaxtree_extensions as jte

Point = namedtuple("Point", ["x", "y"])


def test_tree_select_by_path() -> None:
    tree = {
        "layer1": {"weight": 10, "bias": 20},
        "layer2": Point(30, 40),
    }

    # Match exact suffix
    mask1 = jte.tree_select_by_path(tree, "weight")
    assert mask1 == {
        "layer1": {"weight": True, "bias": False},
        "layer2": Point(False, False),
    }

    # Match absolute path
    mask2 = jte.tree_select_by_path(tree, "/layer1/bias")
    assert mask2 == {
        "layer1": {"weight": False, "bias": True},
        "layer2": Point(False, False),
    }

    # Match recursive wildcard
    mask3 = jte.tree_select_by_path(tree, "**/y")
    assert mask3 == {
        "layer1": {"weight": False, "bias": False},
        "layer2": Point(False, True),
    }


def test_tree_at_path() -> None:
    tree = {
        "layer1": {"weight": 10, "bias": 20},
        "layer2": Point(30, 40),
    }

    # Replace weight
    res1 = jte.tree_at_path(tree, "**/weight", replace=100)
    assert res1 == {
        "layer1": {"weight": 100, "bias": 20},
        "layer2": Point(30, 40),
    }

    # Replace with function
    res2 = jte.tree_at_path(tree, "layer1/*", replace_fn=lambda x: x + 5)
    assert res2 == {
        "layer1": {"weight": 15, "bias": 25},
        "layer2": Point(30, 40),
    }


def test_path_select_equinox() -> None:
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

    # Biases in MLP are stored in model.layers[i].bias
    # We can select all biases via wildcard path "**/bias"
    mask = jte.tree_select_by_path(model, "**/bias")

    # Verify that bias leaves are True in the mask, and weight leaves are False
    for layer_mask in mask.layers:
        if hasattr(layer_mask, "bias") and layer_mask.bias is not None:
            assert jnp.all(layer_mask.bias)
        if hasattr(layer_mask, "weight") and layer_mask.weight is not None:
            assert not jnp.any(layer_mask.weight)

    # The leaves in an MLP are weights and biases
    # Let's check using tree_at_path: zero out all biases
    zero_bias_model = jte.tree_at_path(model, "**/bias", replace_fn=jnp.zeros_like)

    # Verify that all bias arrays in the zero_bias_model are indeed all zeros,
    # and weight arrays are not zeros.
    for layer in zero_bias_model.layers:
        if hasattr(layer, "bias") and layer.bias is not None:
            assert jnp.all(layer.bias == 0)
        if hasattr(layer, "weight") and layer.weight is not None:
            assert not jnp.all(layer.weight == 0)
