import random
import shutil
import textwrap
from dataclasses import fields, is_dataclass
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

BOLD = '\033[1m'
BLACK = '\033[30m'
MAG_BG = '\033[45m'
END = '\033[0m'


def cprint(element, **kwargs):
    element = f'{BOLD}{MAG_BG}{BLACK} === {element} === {END}'
    if kwargs.get('end', False):
        print(element, end='')
    elif kwargs.get('sec', True):
        print('\n' + element)
    else:
        print(element)


def line() -> None:
    print('-' * 30)


def set_seed(seed: int = 42, use_gpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if use_gpu:
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            raise ImportError('PyTorch is not installed. Please install it to use this feature.')


def _fmt_value(v, float_sig: int = 6) -> str:
    if isinstance(v, float):
        return f'{v:.{float_sig}g}'
    if isinstance(v, (tuple, list)):
        open_b, close_b = ('(', ')') if isinstance(v, tuple) else ('[', ']')
        return open_b + ', '.join(_fmt_value(x, float_sig) for x in v) + close_b
    if isinstance(v, dict):
        items = ', '.join(f'{_fmt_value(k, float_sig)}: {_fmt_value(val, float_sig)}' for k, val in v.items())
        return '{' + items + '}'
    if isinstance(v, str):
        return repr(v)
    return str(v)


def format_config_box(
    obj: dict,
    title: str = 'Config',
    show_types: bool = False,
    width: Optional[int] = None,
    unicode_box: bool = True,
    *,
    max_val_width: int = 56,
    max_key_width: int = 24,
) -> str:
    if unicode_box:
        H, V = '─', '│'
        TL, TR, BL, BR = '┌', '┐', '└', '┘'
        LSEP, RSEP = '├', '┤'
        TSEP, BSEP = '┬', '┴'
    else:
        H, V = '-', '|'
        TL = TR = BL = BR = '+'
        LSEP = RSEP = '+'
        TSEP = BSEP = '+'

    items: list[tuple[str, Any]] = []
    if is_dataclass(obj):
        for f in fields(obj):
            items.append((f.name, getattr(obj, f.name)))
    elif isinstance(obj, dict):
        for k in obj:
            items.append((str(k), obj[k]))
    else:
        raise TypeError('format_config_box: Please provide `dataclass` or `dict`.')

    key_max = max((len(k) for k, _ in items), default=8)
    key_w = min(max(10, key_max), max_key_width)

    FIXED = 7

    if width is None:
        try:
            term_w = shutil.get_terminal_size((100, 20)).columns
        except Exception:
            term_w = 100
        width = min(term_w, FIXED + key_w + max_val_width)
    width = max(48, min(200, width))

    val_w = width - FIXED - key_w
    if val_w < 10:
        need = 10 - val_w
        shrinkable = max(0, key_w - 10)
        take = min(need, shrinkable)
        key_w -= take
        val_w = width - FIXED - key_w
        val_w = max(10, val_w)

    top_border = TL + H * (key_w + 2) + H * (val_w + 3) + TR
    header_sep = LSEP + H * (key_w + 2) + TSEP + H * (val_w + 2) + RSEP
    bottom_border = BL + H * (key_w + 2) + BSEP + H * (val_w + 2) + BR

    title_str = f' {title} '
    inside_total_w = width - 3
    left_inside_w = key_w + 2
    inside_total = title_str.center(inside_total_w, ' ')
    title_line = V + inside_total[:left_inside_w] + ' ' + inside_total[left_inside_w:] + V

    lines = [top_border, title_line, header_sep]
    for k, v in items:
        v_str = _fmt_value(v)
        if show_types:
            v_str = f'{v_str}    ({type(v).__name__})'
        wrapped = textwrap.wrap(v_str, width=val_w) or ['']
        lines.append(f'{V} {k:<{key_w}} {V} {wrapped[0]:<{val_w}} {V}')
        for cont in wrapped[1:]:
            lines.append(f'{V} {"":<{key_w}} {V} {cont:<{val_w}} {V}')

    lines.append(bottom_border)
    return '\n'.join(lines)


def print_cfg(obj: dict, **kwargs) -> None:
    print(format_config_box(obj, **kwargs))


def tabulate_wide(
    wide: pd.DataFrame,
    value_name: str = 'values',
    index_name: Optional[str] = None,
    style: Literal['multirow', 'flatten', 'long'] = 'multirow',
    sep: str = ' / ',
    dropna_values: bool = False,
    tablefmt: str = 'psql',
    floatfmt: str = 'g',
) -> str:
    idx_name = index_name if index_name is not None else (wide.index.name or 'time')

    # --- long style ---
    if style == 'long':
        if not isinstance(wide.columns, pd.MultiIndex):
            # Works with single-level columns: Treat column names as a single level
            cols = [str(c) for c in wide.columns]
            df = wide.copy()
            df.columns = pd.MultiIndex.from_arrays([cols], names=['col'])
        else:
            df = wide

        levels = list(range(df.columns.nlevels))
        try:
            s = df.stack(level=levels, future_stack=True)
            long = s.rename(value_name).reset_index()
            if dropna_values:
                long = long.dropna(subset=[value_name])
        except TypeError:
            s = df.stack(level=levels, dropna=dropna_values)
            long = s.rename(value_name).reset_index()

        long = long.rename(columns={long.columns[0]: idx_name})
        # Rearrange columns into [index] + column levels + [value]
        col_level_names = [nm if nm is not None else f'level_{i}' for i, nm in enumerate(df.columns.names)]
        out = long[[idx_name] + col_level_names + [value_name]]
        return tabulate(out, headers='keys', tablefmt=tablefmt, showindex=False, floatfmt=floatfmt)

    # --- flatten style ---
    if style == 'flatten':
        out = wide.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = pd.Index(
                [sep.join(map(str, tup)) for tup in out.columns.to_flat_index()],
                dtype='object',
            )
        else:
            out.columns = out.columns.map(str)
        return tabulate(out, headers='keys', tablefmt=tablefmt, showindex=True, floatfmt=floatfmt)

    # --- multirow style ---
    if not isinstance(wide.columns, pd.MultiIndex):
        # Works with single-level columns: Treat column names as a single level
        return tabulate(wide, headers='keys', tablefmt=tablefmt, showindex=True, floatfmt=floatfmt)

    nlv = wide.columns.nlevels
    header_rows: List[List[str]] = []
    # Create header rows for each level
    for level in range(nlv):
        row = []
        row.append(idx_name if level == 0 else '')  # index column
        for col in wide.columns:
            row.append(str(col[level]))
        header_rows.append(row)

    # Data body (index values + each column value)
    body_rows: List[List[str]] = []
    for idx, rowvals in wide.iterrows():
        row = [str(idx)]
        # rowvals is 1D (corresponding to column MultiIndex)
        for v in rowvals:
            if pd.isna(v):
                row.append('nan')
            else:
                if isinstance(v, (float, np.floating)):
                    row.append(('{:' + floatfmt + '}').format(v))
                else:
                    row.append(str(v))
        body_rows.append(row)

    # Concatenate all and render as "header-less table"
    table = header_rows + body_rows
    # Clear column headers and use the created header_rows as actual data
    return tabulate(table, headers=[], tablefmt=tablefmt, showindex=False)


def blockdiag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]), dtype=float)
    out[: A.shape[0], : A.shape[1]] = A
    out[A.shape[0] :, A.shape[1] :] = B
    return out
