import csv
import os
import time

from .file_adapter import FileAdapter
from .logger import LOG, AlgProtocol, LoggerAdapterFactory

__all__ = ["TydoxAdapter", "TydoxAdapterFactory"]


class TydoxAdapter(FileAdapter):
    """Adapter for Tydox-readable csv format. Also puts all metrics into one csv
    file."""

    _algo: AlgProtocol
    _logdir: str
    _metricdir: str
    _is_model_watched: bool

    def __init__(self, algo: AlgProtocol, logdir: str):
        self._algo = algo
        self._logdir = logdir
        self._metricdir = os.path.join(logdir, "metrics.csv")
        self._is_model_watched = False
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
            LOG.info(f"Directory is created at {self._logdir}")
        self.time = time.time()

    def write_params(self, params):
        return super().write_params(params)

    def before_write_metric(self, epoch, step):
        """Empty write buffers before new rows can be written.

        Parameters
        ----------
        epoch : int
        step : int
        """
        self.name_buffer = []
        self.value_buffer = []

    def write_metric(self, epoch, step, name, value):
        """Collect the metric and value instead of immediately writing.

        Writing occurs in the `after_write_metric` function after all metrics have been
        collected.

        Parameters
        ----------
        epoch : int
        step : int
        name : str
            name of the metric.
        value : Any
            value of the metric.
        """
        self.name_buffer.append(str(name))
        self.value_buffer.append(str(value))

    def watch_model(self, epoch, step):
        return super().watch_model(epoch, step)

    def after_write_metric(self, epoch, step):
        """Write all metrics in CSV format after collection.

        Will also write header and units if the file has not yet been created.

        Parameters
        ----------
        epoch : int
        step : int
        """
        if not os.path.isfile(self._metricdir):
            name_row = ["time", "epoch", "step"] + self.name_buffer
            unit_row = ["[s]"] + (["[-]"] * (len(self.name_buffer) + 2))
            with open(self._metricdir, 'a') as f:
                writer = csv.writer(f, delimiter=',', lineterminator='\n')
                writer.writerows((name_row, unit_row))
        value_row = [time.time() - self.time, epoch, step] + self.value_buffer
        with open(self._metricdir, 'a') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerow(value_row)

    def save_model(self, epoch, algo):
        return super().save_model(epoch, algo)

    def close(self):
        return super().close()


class TydoxAdapterFactory(LoggerAdapterFactory):
    r"""TydoxAdapterFactory class.

    This class instantiates ``TydoxAdapter`` object.
    Log directory will be created at ``<root_dir>/<experiment_name>``.

    Args:
        root_dir (str): Top-level log directory.
    """

    _root_dir: str

    def __init__(self, root_dir: str = "d3rlpy_logs"):
        self._root_dir = root_dir

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> TydoxAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return TydoxAdapter(algo, logdir)
