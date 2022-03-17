import os, time
from collections.abc import MutableMapping
import pandas as pd


class Timer:

    def __init__(self, f, verbose=True, name=None, *args):
        time0 = time.time()
        self.res = f(*args)
        time1 = time.time()
        timeDelta = self.timeDelta = time1 - time0
        if verbose:
            if name:
                print(f'Function: {name}')
            print(
                f'\t\tTime: {timeDelta // 60} min {(timeDelta % 60):.2f} sec')

    def function_result(self):
        return self.res


def exportfig(fig, direc, file_name, SVG=True, dpi=150):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, F'{direc}' + os.sep)
    poster_dir = os.path.join(script_dir,
                              F'{direc}' + os.sep + 'poster' + os.sep)
    sample_file_name = file_name + '.png'
    svg_file_name = file_name + '.svg'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(poster_dir):
        os.makedirs(poster_dir)
    fig.savefig(results_dir + sample_file_name, dpi=dpi)
    fig.savefig(poster_dir + sample_file_name, dpi=dpi, facecolor=(1, 1, 1, 0))
    if SVG:
        fig.savefig(results_dir + svg_file_name)
        fig.savefig(poster_dir + svg_file_name, facecolor=(1, 1, 1, 0))


def flatten(d):
    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.extend(flatten(v).items())
        else:
            items.append((k, v))
    return dict(items)


def export_df(df, file_name, sheetname='Parameters', mode='w'):
    docCount = 1
    engine = "openpyxl"
    while 1:
        try:
            writer = pd.ExcelWriter(F'{file_name}{docCount}.xlsx',
                                    mode=mode,
                                    engine=engine)
            df.to_excel(writer, sheetname)
            writer.close()
            break
        except Exception as e:
            print(f'Problem writing to {file_name}.xlsx')
            print(e)
            docCount += 1


def printError(e, text=None):
    print('--' * 20)
    print('--' * 20)
    print(e)
    if text:
        print(text)
    print('--' * 20)
    print('--' * 20)


def k_to_perm(k, T):
    ##convert m^4/N-s (hydraulic Conductivity) to m^2 (permeability)
    u = (-0.0194444444 * T + 1.4194444436) * 1e-3
    return k * u


def find_encoding(fname):
    import chardet as char
    r_file = open(fname, 'rb').read()
    result = char.detect(r_file)
    charenc = result['encoding']
    return charenc


def merge_dicts(*args):
    result = {}
    for d in args:
        if isinstance(d, dict):
            result.update(d)
    return result