from typing import Dict, Mapping, Any, Iterable

from chex import ArrayTree, Numeric

NumericTree = Numeric | Iterable["NumericTree"] | Mapping[Any, "NumericTree"]

Observation = ArrayTree
Action = NumericTree

Metrics = Dict[str, Numeric]
