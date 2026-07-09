from typing import Any

import jaxtree_extensions as jte


def test_tree_reduce_with_path() -> None:
    tree = {"a": [10, 20], "b": 30}

    # Sum all values, printing paths
    def reduce_fn(acc: list[str], path: tuple[Any, ...], leaf: Any) -> list[str]:
        from jaxtree_extensions.path_select import path_to_str

        acc.append(f"{path_to_str(path)}:{leaf}")
        return acc

    res = jte.tree_reduce_with_path(reduce_fn, tree, initializer=[])
    assert res == ["/a/0:10", "/a/1:20", "/b:30"]


def test_tree_reduce_with_path_equinox() -> None:
    import equinox as eqx
    import jax

    from jaxtree_extensions.path_select import path_to_str

    model = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    # Accumulate all parameter path strings
    def collect_paths(acc: list[str], path: tuple[Any, ...], leaf: Any) -> list[str]:
        acc.append(path_to_str(path))
        return acc

    paths = jte.tree_reduce_with_path(collect_paths, model, initializer=[])

    # Check that the collected paths contain MLP layers structures
    # (e.g., layers/0/weight, layers/0/bias, layers/1/weight, layers/1/bias)
    assert any("layers" in p for p in paths)
    assert any("weight" in p for p in paths)
    assert any("bias" in p for p in paths)
