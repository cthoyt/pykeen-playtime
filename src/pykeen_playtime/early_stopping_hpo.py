# -*- coding: utf-8 -*-

"""Early Stopping HPO Experiment."""

from typing import Any, Mapping, Sequence

import pandas as pd
from tabulate import tabulate

from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.pipeline import pipeline
from pykeen_playtime.countries import Countries
from pykeen_playtime.utils import iter_configs_trials

DIRECTORY = PYKEEN_EXPERIMENTS / 'early_stopping_hpo'
DIRECTORY.mkdir(exist_ok=True, parents=True)

TSV_PATH = DIRECTORY / 'stopping_hpo.tsv'

GRID: Mapping[str, Sequence[Any]] = dict(
    dataset=[Countries],
    model=['TransE', 'ComplEx', 'RotatE'],
    frequency=[1, 2, 5, 10],
    patience=[3, 5, 7],
    relative_delta=[0.001, 0.002, 0.02],
)


def _main(trials: int = 5):
    rows = []
    for config, trial in iter_configs_trials(GRID, trials=trials, desc='Early Stopper HPO'):
        results = pipeline(
            dataset=config['dataset'],
            model=config['model'],
            random_seed=trial,
            device='cpu',
            stopper='early',
            stopper_kwargs=dict(
                metric='adjusted_mean_rank',
                frequency=config['frequency'],
                patience=config['patience'],
                relative_delta=config['relative_delta'],
            ),
            training_kwargs=dict(num_epochs=1000),
            evaluation_kwargs=dict(use_tqdm=False),
            automatic_memory_optimization=False,  # not necessary on CPU
        )
        rows.append((
            config['dataset'] if isinstance(config['dataset'], str) else config['dataset'].get_normalized_name(),
            config['model'],
            trial,
            config['frequency'],
            config['patience'],
            config['relative_delta'],
            len(results.losses),
            results.metric_results.get_metric('both.avg.adjusted_mean_rank'),
            results.metric_results.get_metric('hits@10'),
        ))

    df = pd.DataFrame(rows, columns=[
        'Dataset', 'Model', 'Trial', 'Frequency', 'Patience', 'Delta', 'Epochs', 'AMR', 'Hits@10',
    ])
    df.to_csv(TSV_PATH, sep='\t', index=False)

    print(tabulate(df, headers=df.columns))


if __name__ == '__main__':
    _main()
