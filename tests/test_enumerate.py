import jaxtree_extensions as jte


def test_tree_map_enumerate() -> None:
    tree = {"a": [10, 20], "b": 30}

    # Tag each leaf with its flat index
    res = jte.tree_map_enumerate(lambda idx, leaf: (idx, leaf), tree)
    assert res == {"a": [(0, 10), (1, 20)], "b": (2, 30)}


def test_tree_map_enumerate_equinox() -> None:
    import equinox as eqx
    import jax

    model = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    # Tag each leaf with its index
    tagged_model = jte.tree_map_enumerate(lambda idx, leaf: (idx, leaf), model)

    # Extract the tagged leaves, treating (idx, leaf) tuples as leaves
    tagged_leaves = jax.tree_util.tree_leaves(
        tagged_model,
        is_leaf=lambda x: (
            isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], int)
        ),
    )

    # Check that indices are strictly sequential starting from 0
    for idx, (leaf_idx, _orig_leaf) in enumerate(tagged_leaves):
        assert leaf_idx == idx
