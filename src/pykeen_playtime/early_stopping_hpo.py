# -*- coding: utf-8 -*-

"""Early Stopping HPO Experiment."""

from pykeen.pipeline import pipeline
from pykeen_playtime.countries import Countries
from pykeen_playtime.utils import GridType, Runner, fix_logging

GRID: GridType = dict(
    dataset=[Countries.get_normalized_name()],
    model=['transe', 'complex', 'rotate'],
    frequency=[1, 2, 5, 10],
    patience=[3, 5, 7],
    relative_delta=[0.001, 0.002, 0.02],
)


class ESRunner(Runner):
    """Runner for early stopping HPO experiments."""

    name = 'early_stopper_hpo'
    result_labels = ['epochs', 'amr', 'hits@10']

    def run(self, config, trial):
        """Run the early stopper HPO experiment."""
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
            training_kwargs=dict(
                num_epochs=1000,
                tqdm_kwargs=dict(leave=False),
            ),
            evaluation_kwargs=dict(use_tqdm=False),
            automatic_memory_optimization=False,  # not necessary on CPU
        )
        return (
            len(results.losses),
            results.metric_results.get_metric('both.avg.adjusted_mean_rank'),
            results.metric_results.get_metric('hits@10'),
        )


def _main(trials: int = 10):
    runner = ESRunner(GRID, trials=trials)
    runner.print()


if __name__ == '__main__':
    fix_logging()
    _main()
