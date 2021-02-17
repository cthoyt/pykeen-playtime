# -*- coding: utf-8 -*-

"""Utilities for PyKEEN playtime."""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

from tqdm.contrib.itertools import product

__all__ = [
    'iter_configs_trials',
]


def iter_configs_trials(
    grid: Union[str, Path, Mapping[str, Sequence[Any]]],
    *,
    trials: int = 10,
    **kwargs,
) -> Iterable[Tuple[Mapping[str, Any], int]]:
    """Iterate over several configurations for a given number of trials.

    :param grid: Either a grid search dictionary or a str/path for a JSON
        file containing one.
    :param trials: The number of trials that should be conducted fon each configuration. Defaults to 10.
    :param kwargs: Keyword arguments to pass through to :func:`tqdm.tqdm`.
    :yields: Pairs of configurations and trial numbers
    """
    for config in _iter_configs(grid=grid, **kwargs):
        for trial in range(1, 1 + trials):
            yield config, trial


def _iter_configs(
    grid: Union[str, Path, Mapping[str, Sequence[Any]]],
    **kwargs,
) -> Iterable[Mapping[str, Any]]:
    if isinstance(grid, (str, Path)):
        with open(grid) as file:
            grid = json.load(file)
    keys, values = zip(*grid.items())  # type: ignore
    for v in product(*values, **kwargs):
        yield dict(zip(keys, v))
