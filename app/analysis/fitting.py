from pubsub import pub
import wx
import time
from multiprocessing import freeze_support
import pandas as pd
import matplotlib.pyplot as plt
from app.utilities import *
from app.file_types import ExperimentFile
from .experiment import BiphasicExperiment
from tqdm import tqdm
from app.config import ConfigContainer as cc


def fit_experiment_files(files: list[ExperimentFile], selected_settings: dict):
    freeze_support()
    settings = flatten(cc.default_settings)
    num_files = len(files)
    object_counted = 'files' if num_files > 1 else 'file'
    print('--' * 20)
    print(F'{num_files} data {object_counted} recognized')
    print('--' * 20)
    ##endregion

    totalTime0 = time.time()

    compiledProps = []
    analysisfile_name = 'Compiled Properties'
    iter_files = files
    if not cc.log_messages:
        iter_files = tqdm(files)
    for i, experiment_file in enumerate(iter_files):
        file_name = experiment_file.getFileName()
        if not cc.log_messages:
            iter_files.set_description(f'{file_name} ({i+1}/{num_files})')
        wx.CallAfter(pub.sendMessage, "update", msg='file')

        experiment = BiphasicExperiment(experiment_file, settings)
        experiment.analyze()
        try:
            compiledProps.append(experiment.dfMatProps)
            dfCompiledProps = pd.concat(compiledProps, ignore_index=True)
            export_df(dfCompiledProps, analysisfile_name)
        except Exception as e:
            print(e)
            continue

        plt.close('all')

    totalRunTime = time.time() - totalTime0
    print(F'Run time: {totalRunTime//60} min {(totalRunTime%60):.2f} sec')
