import distrax


class Transformed(distrax.Transformed):
    """Thin wrapper that adds indexing functionality to distrax.Transformed."""

    def __getitem__(self, index):
        return Transformed(self._distribution.__getitem__(index), self._bijector)
