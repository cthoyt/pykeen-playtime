# -*- coding: utf-8 -*-

"""The countries dataset."""

from pykeen.datasets.base import UnpackedRemoteDataset

BASE_URL = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Countries/Countries_S1/'

__all__ = [
    'Countries',
]


class Countries(UnpackedRemoteDataset):
    """The Countries dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Countries small dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=f'{BASE_URL}/train.txt',
            testing_url=f'{BASE_URL}/test.txt',
            validation_url=f'{BASE_URL}/valid.txt',
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


if __name__ == '__main__':
    Countries().summarize()
