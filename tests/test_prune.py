from collections import namedtuple
from typing import Any

import jax.tree_util as jtu

import jaxtree_extensions as jte

Point = namedtuple("Point", ["x", "y"])


@jtu.register_pytree_node_class
class CustomContainer:
    def __init__(self, val1: Any, val2: Any) -> None:
        self.val1 = val1
        self.val2 = val2

    def tree_flatten(self) -> tuple[tuple[Any, Any], str]:
        return (self.val1, self.val2), "aux"

    @classmethod
    def tree_unflatten(cls, aux: str, children: tuple[Any, ...]) -> "CustomContainer":
        return cls(*children)


def test_tree_map_prune_basic() -> None:
    tree = {
        "a": [1, 2, 3],
        "b": Point(4, 5),
        "c": CustomContainer(6, 7),
    }

    # Prune even numbers (after mapping * 10)
    pruned, treedef = jte.tree_map_prune(
        lambda x: x * 10, tree, keep_fn=lambda x: x % 20 != 0
    )

    # Check pruned tree structure
    assert pruned["a"] == [10, 30]
    assert pruned["b"] == Point(None, 50)
    assert isinstance(pruned["c"], CustomContainer)
    assert pruned["c"].val1 is None
    assert pruned["c"].val2 == 70

    # Test reconstruction with None
    reconstructed_none = treedef.unflatten(pruned, fill_value=None)
    assert reconstructed_none["a"] == [10, None, 30]
    assert reconstructed_none["b"] == Point(None, 50)
    assert reconstructed_none["c"].val1 is None
    assert reconstructed_none["c"].val2 == 70

    # Test reconstruction with custom values
    fill_tree = {
        "a": [-1, -2, -3],
        "b": Point(-4, -5),
        "c": CustomContainer(-6, -7),
    }
    reconstructed_custom = treedef.unflatten(pruned, fill_value=fill_tree)
    assert reconstructed_custom["a"] == [10, -2, 30]
    assert reconstructed_custom["b"] == Point(-4, 50)
    assert reconstructed_custom["c"].val1 == -6
    assert reconstructed_custom["c"].val2 == 70


def test_tree_map_prune_equinox() -> None:
    import equinox as eqx
    import jax

    model = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=4,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    def keep_fn(x: Any) -> bool:
        import jax.numpy as jnp

        if not isinstance(x, (jnp.ndarray, float, int)):
            return True
        return bool(jnp.any(jnp.abs(x) >= 0.1))

    # Prune leaves where the absolute value is < 0.1
    pruned, treedef = jte.tree_map_prune(
        lambda x: x,
        model,
        keep_fn=keep_fn,
    )

    # Verify we can reconstruct it back to an MLP
    reconstructed = treedef.unflatten(pruned, fill_value=0.0)
    assert isinstance(reconstructed, eqx.nn.MLP)

    # Check that the model runs fine
    x = jax.numpy.array([1.0, 2.0])
    y = reconstructed(x)
    assert y.shape == (2,)
