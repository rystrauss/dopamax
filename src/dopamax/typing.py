from typing import Dict, Mapping, Any, Iterable, Union

from chex import ArrayTree, Numeric, Scalar

NumericTree = Union[Numeric, Iterable["NumericTree"], Mapping[Any, "NumericTree"]]

Observation = ArrayTree
Action = NumericTree

Metrics = Dict[str, Scalar]
