from jaxtyping import install_import_hook

# setting up beartype with jaxtyping here
install_import_hook(
    modules=["jaxtree_extensions"],
    typechecker="beartype.beartype",
)
