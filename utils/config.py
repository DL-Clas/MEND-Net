import re
import yaml


# YAML 1.1 float pattern that also accepts scientific notation without a
# decimal point (e.g. "1e-6", "2.5e9"), which stock PyYAML parses as strings.
_FLOAT_RESOLVER = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |[-+]?\.(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)


class ConfigLoader(yaml.SafeLoader):
    """SafeLoader with proper scientific-notation float resolution."""


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _FLOAT_RESOLVER,
    list("-+0123456789."),
)


def load_config(path: str) -> dict:
    """Load a YAML configuration with scientific-notation support.

    Args:
        path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.load(f, Loader=ConfigLoader)
