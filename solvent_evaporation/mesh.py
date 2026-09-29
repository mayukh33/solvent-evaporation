import numpy


class Grid:
    """n uniform cells; give `n` or `step`, which rounds to whole cells."""

    def __init__(self, n=None, step=None, start=0.0):
        if step is not None:
            n = numpy.max((2, numpy.round(1.0 / step))).astype(numpy.int64)
        if n is None:
            raise TypeError(
                "give the grid a cell count `n` or a mesh step `step`")

        self.n = n
        self.start = float(start)
        self.step = 1.0 / n
        self.faces = self.start + numpy.linspace(0.0, 1.0, n + 1)
        self.centers = 0.5 * (self.faces[:-1] + self.faces[1:])
