import os
import time
import pandas as pd
from datetime import datetime

from tensorboardX import SummaryWriter
import shiva.helpers.dir_handler as dh

class TimeProfiler:
    _ti = {}
    _tf = {}
    _data = []

    def __init__(self, configs, base_dir, filename_suffix=''):
        self.configs = configs
        self._base_dir = base_dir
        self._save_dir = dh.make_dir(os.path.join(self._base_dir, 'profiler', filename_suffix), use_existing=True)
        if 'profiler' not in self.configs['Admin']:
            self.configs['Admin']['profiler'] = False
        if self.configs['Admin']['profiler']:
            self.writer = SummaryWriter(logdir=self._save_dir, filename_suffix=filename_suffix)
        self.reset()

    def reset(self):
        self._data = []

    def start(self, metric_name):
        if type(metric_name) == str:
            self._ti[metric_name] = time.time()
        elif type(metric_name) == list:
            for _m in metric_name:
                self._ti[_m] = time.time()

    def time(self, metric_name, x_value, output_quantity=1):
        self._tf[metric_name] = time.time()
        self._output_quantity = output_quantity
        self._time_diff = self._tf[metric_name] - self._ti[metric_name]

        self._output_per_second = self._output_quantity / self._time_diff
        self._output_per_minute = 60 * (self._output_quantity / self._time_diff)
        self._output_per_hour = 60 * 60 * (self._output_quantity / self._time_diff)
        self._record(metric_name, x_value)
        self._ti[metric_name] = time.time()

    def _record(self, metric_name, x_value):
        _now = str(datetime.now())
        # self._data.append({
        #     'ts': _now,
        #     'metric': metric_name,
        #     'ti': self._ti[metric_name],
        #     'tf': self._tf[metric_name],
        #     'diff': self._time_diff,
        #     'output_quantity': self._output_quantity,
        #     'output_per_sec': self._output_per_second,
        #     'output_per_min': self._output_per_minute,
        #     'output_per_hour': self._output_per_hour
        # })

        y_val, _per_string = self._output_per_minute, 'per_minute'
        # outliers for minutes
        if y_val > 10000:
            return
        if self.configs['Admin']['profiler'] and self._output_per_minute:
            self.writer.add_scalar("Profiler/{}_{}".format(metric_name, _per_string), y_val, x_value)

    def show_results(self, range=(0, 5)):
        return pd.DataFrame(self._data, columns=['ts', 'ti', 'tf', 'diff', 'output_quantity', 'output_per_sec', 'output_per_min', 'output_per_hour']).sort_index(ascending=False).iloc[range[0]:range[1]]
