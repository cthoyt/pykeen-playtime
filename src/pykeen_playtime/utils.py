# -*- coding: utf-8 -*-

"""Utilities for PyKEEN playtime."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from tabulate import tabulate
from tqdm.contrib.itertools import product

__all__ = [
    'fix_logging',
    'iter_configs_trials',
    'Runner',
    'GridType',
]

from pykeen.constants import PYKEEN_EXPERIMENTS

GridType = Mapping[str, Sequence[Any]]


def fix_logging() -> None:
    """Fix over-logging in PyKEEN."""
    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.ERROR)
    logging.getLogger('pykeen.stoppers.early_stopping').setLevel(logging.ERROR)
    logging.getLogger('pykeen.triples.triples_factory').setLevel(logging.ERROR)
    logging.getLogger('pykeen.models.cli').setLevel(logging.ERROR)


def iter_configs_trials(
    grid: Union[str, Path, Mapping[str, Sequence[Any]]],
    *,
    trials: Optional[int] = None,
    **kwargs,
):
    """Iterate over several configurations for a given number of trials.

    :param grid: Either a grid search dictionary or a str/path for a JSON
        file containing one.
    :param trials: The number of trials that should be conducted fon each configuration. Defaults to 10.
    :param kwargs: Keyword arguments to pass through to :func:`tqdm.tqdm`.
    :returns: An iterator for trials
    """
    config_iterator = _iter_configs(grid=grid, **kwargs)
    return _ConfigTrialIterator(config_iterator, trials)


def _iter_configs(
    grid: Union[str, Path, Mapping[str, Sequence[Any]]],
    *,
    order: Optional[Sequence[str]] = None,
    **kwargs,
):
    if isinstance(grid, (str, Path)):
        return _ConfigIterator.from_path(grid, order=order, **kwargs)
    return _ConfigIterator(grid, order=order, **kwargs)


class _ConfigTrialIterator:
    def __init__(self, config_iterator: _ConfigIterator, trials: Optional[int] = None):
        """Initialize the configuration/trial iterator.

        :param config_iterator: the configuration iterator to wrap
        :param trials: the number of trials, defaults to 10.
        """
        self.config_iterator = config_iterator
        self.trials = trials or 10

    @property
    def keys(self):
        """Return the keys of the wrapped iterator."""
        return self.config_iterator.keys

    def __iter__(self) -> Iterable[Tuple[Mapping[str, Any], int]]:
        for config in iter(self.config_iterator):
            for trial in range(1, 1 + self.trials):
                yield config, trial


class _ConfigIterator:
    def __init__(self, grid, *, order: Optional[Sequence[str]] = None, **kwargs):
        """Initialize the configuration iterator.

        :param grid: The grid to generate configurations over
        :param order: The optional ordering of the keys
        :param kwargs: keyword arguments to pass through to :func:`tqdm.tqdm`
        """
        if order:
            self.keys = order
            self.values = [grid[k] for k in order]
        else:
            self.keys, self.values = zip(*grid.items())  # type: ignore
        self.kwargs = kwargs

    @classmethod
    def from_path(cls, path: Union[str, Path], *, order: Optional[Sequence[str]] = None, **kwargs) -> _ConfigIterator:
        """Create a config iterator from a grid stored in a JSON file."""
        with open(path) as file:
            return cls(json.load(file), order=order, **kwargs)

    def __iter__(self):
        for v in product(*self.values, **self.kwargs):
            yield dict(zip(self.keys, v))


class Runner(ABC):
    """A harness for grid search experiment runners."""

    #: The name of the experiment
    name: ClassVar[str]
    #: The labels of the results returned by the run() function
    result_labels: ClassVar[Sequence[str]]
    #: A dictionary of reformatters for config values
    formatters: ClassVar[Mapping[str, Callable[[Any], str]]] = {}
    #: The grid to search
    grid: GridType

    def __init__(self, grid: GridType, trials: Optional[int] = None):
        """Initialize the runner.

        :param grid: The grid to check
        :param trials: The number of trials to run. Defaults to 10.
        :raises ValueError: if the ``result_labels`` variable is the wrong length
        """
        self.directory = PYKEEN_EXPERIMENTS / self.name
        self.directory.mkdir(exist_ok=True, parents=True)
        self.path = self.directory / 'results.tsv'

        self.grid = grid
        self.it = iter_configs_trials(
            grid,
            trials=trials,
            desc='Early Stopper HPO',
        )

        precalculated = {}
        if self.path.exists():
            for row in pd.read_csv(self.path, sep='\t').values:
                row = tuple(row)
                precalculated[row[:len(self.it.keys) + 1]] = row[len(self.it.keys) + 1:]

        self.rows = []
        for config, trial in self.it:
            row_start = tuple(self._format(key, config[key]) for key in self.it.keys) + (trial,)
            if row_start in precalculated:
                row_end = precalculated[row_start]
            else:
                row_end = self.run(config, trial)
                if len(row_end) != len(self.result_labels):
                    raise ValueError(
                        f'Not enough results returned. '
                        f'Got {len(row_end)}, should have got {len(self.result_labels)}',
                    )
            self.rows.append((*row_start, *row_end))

        self.df = pd.DataFrame(
            self.rows,
            columns=[*self.it.keys, 'trial', *self.result_labels],
        )
        self.df.to_csv(self.directory / 'results.tsv', sep='\t', index=False)

    def _format(self, key: str, value) -> str:
        formatter = self.formatters.get(key)
        if formatter is None:
            return value
        return formatter(value)

    @abstractmethod
    def run(self, config, trial) -> Sequence[str]:
        """Run the experiment."""
        raise NotImplementedError

    def print(self) -> None:
        """Print the results of all experiments."""
        print(tabulate(self.df.values, headers=self.df.columns))
