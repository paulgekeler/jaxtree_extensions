import jax.numpy as jnp

import jaxtree_extensions as jte


def test_tree_compare_equal() -> None:
    tree_a = {"a": [1, 2], "b": jnp.ones((2, 2))}
    tree_b = {"a": [1, 2], "b": jnp.ones((2, 2))}

    diff = jte.tree_compare(tree_a, tree_b)
    assert not diff.has_changes
    assert diff.report() == "No differences found."


def test_tree_compare_mismatches() -> None:
    tree_a = {"a": [1, 2], "b": jnp.ones((2, 2)), "c": 3}
    # Type mismatch at a/0, value mismatch at c, missing key d
    tree_b = {"a": [1.0, 2], "b": jnp.ones((2, 2)), "c": 4, "d": 5}

    diff = jte.tree_compare(tree_a, tree_b)
    assert diff.has_changes

    report = diff.report()
    # Check that it identifies the problems
    assert "Type mismatch" in report or "type mismatch" in report.lower()
    assert "not equal" in report.lower() or "mismatch" in report.lower()
    assert "keys" in report.lower()


def test_tree_compare_equinox() -> None:
    import equinox as eqx
    import jax

    model1 = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        key=jax.random.PRNGKey(0),
    )
    model2 = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    # Identical models should match
    diff1 = jte.tree_compare(model1, model2)
    assert not diff1.has_changes

    # Change one bias value
    model2_perturbed = jax.tree_util.tree_map(
        lambda x: x + 1.0 if (hasattr(x, "ndim") and x.ndim == 1) else x, model2
    )
    diff2 = jte.tree_compare(model1, model2_perturbed)
    assert diff2.has_changes
    report2 = diff2.report()
    assert "bias" in report2 or "mismatch" in report2.lower()

    # Structural mismatch (different depth)
    model3 = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=2,
        key=jax.random.PRNGKey(0),
    )
    diff3 = jte.tree_compare(model1, model3)
    assert diff3.has_changes
    report3 = diff3.report()
    assert "mismatch" in report3.lower()
