import importlib
import logging
import socket
import warnings
from datetime import datetime
from typing import Any, Callable, cast

import hydra
from omegaconf import DictConfig

from src.models.adakoop import AdaKoop
from src.utils.io_helper import IOHelper
from src.utils.preprocessor import preprocess
from utils import print_cfg, set_seed

log = logging.getLogger(__name__)
warnings.simplefilter('ignore')


def import_callable(name: str, component: str) -> Callable[..., Any]:
    module = importlib.import_module(f'src.models.{name}.{component}')
    func = cast(Callable[..., Any], getattr(module, component, None))
    if not callable(func):
        raise ValueError(f'Module src.models.{name}.{component} does not have a callable "{component}" function.')
    return func


@hydra.main(version_base=None, config_path='config', config_name='settings')
def main(cfg: DictConfig) -> None:
    print_cfg(
        obj={
            'Current time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Hostname': socket.gethostname(),
            'Model': cfg.model.name,
            'Input dir': cfg.io.input_dir,
            'Forecasting setup': (cfg.model.lcurr, cfg.model.lstep),
        },
        title='Experimental Metadata',
        show_types=False,
        unicode_box=True,
    )
    set_seed(cfg.exp.seed, use_gpu=cfg.use_gpu)
    ioh = IOHelper(io_cfg=cfg.io)

    # create output directory path
    cfg.io.out_dir = ioh.create_path(cfg)

    # prepare for simulation
    data = ioh.load_data(cfg=cfg.data, seed=cfg.exp.seed, verbose=cfg.verbose)
    data = preprocess(data=data, cfg=cfg)

    print(f'DATASETS: {cfg.io.input_dir}_{cfg.data.fn}')
    print(f'PREPROCESSED: {data.shape}')
    print(f'MODEL: {cfg.model.name}')

    # load model
    model = AdaKoop(verbose=cfg.verbose)

    # simulate
    run = import_callable(cfg.model.name, 'run')
    results = run(data, model, cfg)

    # save results
    if cfg.save and results is not None:
        ioh.mkdir()
        for key, value in results.items():
            ioh.savepkl(obj=value, name=key)


if __name__ == '__main__':
    main()
