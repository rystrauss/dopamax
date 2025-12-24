from collections.abc import Iterable, Mapping
from typing import Any, Union

from chex import ArrayTree, Numeric, Scalar

NumericTree = Union[Numeric, Iterable["NumericTree"], Mapping[Any, "NumericTree"]]

Observation = ArrayTree
Action = NumericTree

Metrics = dict[str, Scalar]
