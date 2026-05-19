import os
import pickle
import re
import shutil
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf


class IOHelper:
    """
    Helper class for IO operations.

    Attributes
    ----------
    input_dir : str
        Input directory path.
    out_dir : str
        Root output directory path.
    read_only : bool
        Read-only flag.
    """

    def __init__(self, io_cfg: DictConfig, read_only: bool = False) -> None:
        self.input_dir = io_cfg.input_dir
        self.root_out_dir = io_cfg.root_out_dir
        self.out_dir = io_cfg.root_out_dir
        self.read_only = read_only

    def init_dir(self) -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        assert hasattr(self, 'out_dir'), "you didn't set 'out_dir'"
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def mkdir(self, path: str = '', abs: bool = False) -> None:
        assert hasattr(self, 'out_dir'), "you didn't set 'out_dir'"
        path = path if abs else self.out_dir + path
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def create_path(self, cfg: DictConfig) -> str:
        if 'input_dir' not in cfg.io:
            raise ValueError("'input_dir' not found in cfg.io")
        path = str(self.root_out_dir) + f'{cfg.io["input_dir"]}/'

        # Data configuration
        data_cfg = cfg.data
        for key in data_cfg.keys():
            if OmegaConf.is_missing(data_cfg, key):
                value = 'None'
            else:
                value = data_cfg[key]
            path += f'{str(key)}={str(value)}/'

        # Experiment configuration
        path += f'seed={str(cfg.exp.seed)}/iter={str(cfg.exp.iterations)}/'

        # Model configuration
        model_cfg = cfg.model
        for key in model_cfg.keys():
            if OmegaConf.is_missing(model_cfg, key):
                value = 'None'
            else:
                value = model_cfg[key]
            path += f'{str(key)}={str(value)}/'

        self.out_dir = path
        return path

    def load_data(self, cfg: DictConfig, seed: int, verbose: bool) -> pd.DataFrame:
        dataset_module = import_module(f'data.{self.input_dir}')
        data = dataset_module.load_data(**cfg, seed=seed, verbose=verbose)
        return data

    def build_results_df(
        self,
        date_range: list = [],
        data_name: str = 'synthetics',
        conditions: tuple = (),
    ) -> pd.DataFrame:
        param_dicts: List[Dict[str, Any]] = []
        for date in date_range:
            out_dir = os.path.join(self.root_out_dir, date, data_name)
            param_dicts.extend(self._collect_all_param_dicts(out_dir, conditions))
        df = pd.DataFrame(param_dicts)
        return df

    def savefig(self, fig: Figure, name: str = '') -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        fig.savefig(self.out_dir + name)
        plt.close()

    def savepkl(self, obj: Any, name: str = '', abs: bool = False) -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        if '.' not in name:
            name += '.pkl'
        file_path = name if abs else self.out_dir + name
        f = open(file_path, 'wb')
        pickle.dump(obj, f)
        f.close()

    def loadpkl(self, name: str = '', abs: bool = False) -> Any:
        if '.' not in name:
            name += '.pkl'
        file_path = name if abs else self.out_dir + name
        try:
            f = open(file_path, 'rb')
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: {file_path}')
        obj = pickle.load(f)
        f.close()

        return obj

    def _collect_all_param_dicts(
        self,
        root_dir: str = 'out',
        conditions: tuple = (),
    ) -> List[Dict[str, Any]]:
        param_dicts: List[Dict[str, Any]] = []
        key_to_index: Dict[Any, int] = {}
        key_to_date: Dict[Any, str] = {}

        def _extract_date_and_dataset(path: str) -> tuple[Optional[str], Optional[str]]:
            try:
                rel_to_root = os.path.relpath(path, self.root_out_dir)
            except ValueError:
                return None, None
            parts = [part for part in rel_to_root.split(os.sep) if part not in ('.', '')]
            date_str: Optional[str] = None
            dataset: Optional[str] = None
            for idx, part in enumerate(parts):
                if re.fullmatch(r'\d{8}', part):
                    date_str = part
                    if idx + 1 < len(parts):
                        dataset = parts[idx + 1]
                    break
            return date_str, dataset

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if dirnames:
                continue

            skip_flag = False
            for cond in conditions:
                if cond in dirpath:
                    skip_flag = True
                    break
            if skip_flag:
                continue

            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                continue

            params = self._extract_params_from_path(rel_path)
            results = self._extract_results(dirpath, filenames)

            if params and results:
                date_str, dataset = _extract_date_and_dataset(dirpath)
                key = (dataset, tuple(sorted(params.items())))
                record = dict(
                    **params,
                    **results,
                )

                if key in key_to_index:
                    prev_date = key_to_date.get(key, '')
                    current_date = date_str or ''
                    if current_date > prev_date:
                        param_dicts[key_to_index[key]] = record
                        key_to_date[key] = current_date
                else:
                    idx = len(param_dicts)
                    param_dicts.append(record)
                    key_to_index[key] = idx
                    key_to_date[key] = date_str or ''
        return param_dicts

    def _extract_params_from_path(self, path: str) -> Dict[str, Any]:
        pattern = re.compile(r'([^=/\\]+)=([^/\\]+)')
        return dict(pattern.findall(path))

    def _extract_results(self, dirpath: str, filenames: List[str]) -> Optional[Dict[str, Any]]:
        try:
            if 'metrics.pkl' in filenames:
                metrics = self.loadpkl(os.path.join(dirpath, 'metrics.pkl'), abs=True)
            return {'metrics': metrics}
        except UnboundLocalError as e:
            print(f'Error processing {dirpath}: {e}')
            return None
        except Exception as e:
            raise e


def backup() -> None:
    current_date = datetime.now()
    date_str = current_date.strftime('%Y%m%d')
    backup_path = f'backup/{date_str}/'
    if os.path.isdir(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree('out/', backup_path)


if __name__ == '__main__':
    backup()
