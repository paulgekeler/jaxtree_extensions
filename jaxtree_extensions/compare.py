"""Comparison utilities for JAX PyTrees."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import jax.tree_util as jtu
import numpy as np

from jaxtree_extensions.path_select import path_to_str

__all__ = [
    "Mismatch",
    "StructureMismatch",
    "TypeMismatch",
    "ValueMismatch",
    "TreeDiff",
    "tree_compare",
]


class Mismatch:
    """Base class for PyTree comparison mismatches.

    Args:
        path: The key path tuple of the mismatched node.
        message: A descriptive message explaining the mismatch.
    """

    def __init__(self, path: tuple[Any, ...], message: str) -> None:
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"Path {path_to_str(self.path)}: {self.message}"


class StructureMismatch(Mismatch):
    """Represents a structural mismatch (e.g. key or length mismatch)."""


class TypeMismatch(Mismatch):
    """Represents a type mismatch between two nodes.

    Args:
        path: The key path tuple of the mismatched node.
        type_a: The type of the node in the first PyTree.
        type_b: The type of the node in the second PyTree.
    """

    def __init__(self, path: tuple[Any, ...], type_a: type, type_b: type) -> None:
        super().__init__(path, f"Type mismatch: {type_a.__name__} vs {type_b.__name__}")
        self.type_a = type_a
        self.type_b = type_b


class ValueMismatch(Mismatch):
    """Represents a leaf value mismatch (e.g. array values not close)."""


class TreeDiff:
    """Diagnostic report of differences between two PyTrees.

    Args:
        mismatches: A list of recorded Mismatch objects.
    """

    def __init__(self, mismatches: list[Mismatch]) -> None:
        self.mismatches = mismatches

    @property
    def has_changes(self) -> bool:
        """Return True if any differences were found."""
        return len(self.mismatches) > 0

    def report(self) -> str:
        """Return a formatted, readable string of all differences.

        Returns:
            A multi-line diagnostic report.
        """
        if not self.has_changes:
            return "No differences found."
        lines = [f"Found {len(self.mismatches)} mismatch(es):"]
        for idx, m in enumerate(self.mismatches, 1):
            lines.append(f"{idx}. {m}")
        return "\n".join(lines)


def tree_compare(
    tree_a: Any,
    tree_b: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
    is_leaf: Callable[[Any], bool] | None = None,
) -> TreeDiff:
    """Compare two PyTrees and return a detailed diagnostic diff report.

    Checks structure, node types, and leaf values (using array closeness for JAX
    and NumPy arrays, and strict equality for other leaf types).

    Args:
        tree_a: The first PyTree.
        tree_b: The second PyTree.
        rtol: Relative tolerance for numerical closeness.
        atol: Absolute tolerance for numerical closeness.
        equal_nan: Whether to treat NaNs as equal.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        A `TreeDiff` object representing the diagnostic report.
    """
    mismatches: list[Mismatch] = []
    registry = jtu.default_registry

    def check_is_leaf(x: Any) -> bool:
        if is_leaf is not None and is_leaf(x):
            return True
        return registry.flatten_one_level(x) is None

    def recurse(a: Any, b: Any, path: tuple[Any, ...]) -> None:
        is_leaf_a = check_is_leaf(a)
        is_leaf_b = check_is_leaf(b)

        if is_leaf_a != is_leaf_b:
            mismatches.append(
                StructureMismatch(
                    path,
                    "Leaf/Container mismatch: tree_a has"
                    f" {type(a).__name__}, tree_b has {type(b).__name__}",
                )
            )
            return

        if is_leaf_a:
            # Both are leaves
            if type(a) is not type(b):
                mismatches.append(TypeMismatch(path, type(a), type(b)))
                return

            # Check value closeness if array-like
            is_arr_a = hasattr(a, "shape") and hasattr(a, "dtype")
            is_arr_b = hasattr(b, "shape") and hasattr(b, "dtype")

            if is_arr_a or is_arr_b:
                if not (is_arr_a and is_arr_b):
                    mismatches.append(
                        TypeMismatch(
                            path,
                            type(a),
                            type(b),
                        )
                    )
                    return
                shape_a, shape_b = a.shape, b.shape
                if shape_a != shape_b:
                    mismatches.append(
                        ValueMismatch(
                            path,
                            f"Shape mismatch: {shape_a} vs {shape_b}",
                        )
                    )
                    return
                try:
                    arr_a = np.asarray(a)
                    arr_b = np.asarray(b)
                    if not np.allclose(
                        arr_a,
                        arr_b,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=equal_nan,
                    ):
                        mismatches.append(
                            ValueMismatch(
                                path,
                                "Array values are not close",
                            )
                        )
                except Exception as e:
                    mismatches.append(
                        ValueMismatch(
                            path,
                            f"Failed to check array closeness: {e}",
                        )
                    )
            else:
                # Standard leaf value comparison
                # Handle boolean array or standard comparisons
                try:
                    if not bool(np.all(a == b)):
                        mismatches.append(
                            ValueMismatch(
                                path,
                                f"Values not equal: {a} vs {b}",
                            )
                        )
                except Exception as e:
                    mismatches.append(
                        ValueMismatch(
                            path,
                            f"Failed to compare values: {e}",
                        )
                    )
            return

        # Both are containers
        if type(a) is not type(b):
            mismatches.append(TypeMismatch(path, type(a), type(b)))
            return

        is_dict = isinstance(a, dict)
        is_list = isinstance(a, list)
        is_tuple = isinstance(a, tuple) and not hasattr(a, "_fields")

        if is_dict:
            keys_a = set(a.keys())
            keys_b = set(b.keys())
            if keys_a != keys_b:
                only_a = keys_a - keys_b
                only_b = keys_b - keys_a
                if only_a:
                    mismatches.append(
                        StructureMismatch(
                            path, f"Keys present only in tree_a: {sorted(only_a)}"
                        )
                    )
                if only_b:
                    mismatches.append(
                        StructureMismatch(
                            path, f"Keys present only in tree_b: {sorted(only_b)}"
                        )
                    )

            for key in sorted(keys_a & keys_b):
                recurse(a[key], b[key], path + (jtu.DictKey(key),))

        elif is_list or is_tuple:
            len_a = len(a)
            len_b = len(b)
            if len_a != len_b:
                mismatches.append(
                    StructureMismatch(
                        path, f"Sequence length mismatch: {len_a} vs {len_b}"
                    )
                )

            for idx in range(min(len_a, len_b)):
                recurse(a[idx], b[idx], path + (jtu.SequenceKey(idx),))

        else:
            # Custom container or namedtuple
            flat_a = registry.flatten_one_level(a)
            flat_b = registry.flatten_one_level(b)
            if flat_a is not None and flat_b is not None:
                children_a, aux_a = flat_a
                children_b, aux_b = flat_b
                if aux_a != aux_b:
                    mismatches.append(
                        StructureMismatch(
                            path, f"Auxiliary data mismatch: {aux_a} vs {aux_b}"
                        )
                    )
                if len(children_a) != len(children_b):
                    mismatches.append(
                        StructureMismatch(
                            path,
                            "Children count mismatch:"
                            f" {len(children_a)} vs {len(children_b)}",
                        )
                    )
                for idx in range(min(len(children_a), len(children_b))):
                    recurse(
                        children_a[idx], children_b[idx], path + (jtu.SequenceKey(idx),)
                    )

    recurse(tree_a, tree_b, ())
    return TreeDiff(mismatches)
