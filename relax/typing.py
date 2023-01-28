from typing import Mapping, Any, Iterable

from chex import ArrayTree, Numeric

NumericTree = Numeric | Iterable["NumericTree"] | Mapping[Any, "NumericTree"]

Observation = ArrayTree
Action = NumericTree

Metrics = Mapping[str, Numeric]
