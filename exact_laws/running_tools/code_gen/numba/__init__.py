from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("exact_laws.running_tools.code_gen.numba", package_path='templates'),
    autoescape=select_autoescape()
)
