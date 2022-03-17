import numpy as np
import os
import datetime as datetime
import time
import pandas as pd
import scipy.special as special
import scipy.optimize as optimize
from mpmath import invertlaplace
import scipy.signal as signal
import seaborn as sns
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
try:
    """figs showed downsampled signals using ulttb. default to no downsampling if not installed"""
    from ulttb import downsample
    ulttb = True
except:
    ulttb = False
from app.config import ConfigContainer as cc
from app.utilities import *
from app.file_types import cellscale_csv, custom_csv
from .vectorize_mp import *
from .analysis_helpers import *
from .biphasic_math import *

## define some config constants
label_font_size = cc.label_font_size
preload_pattern = cc.preload_pattern

file_type = cellscale_csv


class BiphasicExperiment:

    def __init__(self, fileObject, settings):
        self.samplingRate = settings['samplingRate']
        self.equiTime = settings['equiTime']
        self.settings = settings
        self.bounds = {
            'frequency': settings['freqBounds'],
            'fitting': settings['fittingBounds'],
            'graph': settings['graphBounds']
        }

        self.rad = settings['rad'] * 1e-3
        self.surfaceArea = np.pi * self.rad**2
        self.equil = self.samplingRate * self.equiTime * -1
        self.sampleTime = f'{(1 / self.samplingRate) * 1e3}ms'

        self.error = 0
        self.max_cycles = 0
        self.thickness = 0
        self.exportFolder = ''
        self.exportName = ''
        self.cycles = []
        self.fileData = fileObject.exportData()
        self.folder = fileObject.folder
        date = self.fileData['date']
        exp = self.fileData['exp']
        id = self.fileData['id']
        eye = self.fileData['eye']
        region = self.fileData['region']
        self.file_name = f"{date}-{exp}-{id}-{eye}-{region}"
        self.preload = int(preload_pattern.sub(
            '', self.fileData['preload'])) * 1e-6

        self._get_exp()
        self._get_cycles()

    def _get_exp(self):
        file = self.fileData['file']
        try:
            df = pd.read_csv(file,
                             usecols=file_type["columns_to_use"],
                             encoding='ISO-8859-1')
        except Exception as e:
            printError(e)
            try:
                df = pd.read_csv(file,
                                 usecols=file_type["columns_to_use"],
                                 encoding=find_encoding(file))

            except Exception as e:
                printError(e, f"{file}: \n File Not Read")
                df = pd.DataFrame()
                self.error = 1
        if not df.empty:
            df.columns = file_type['column_labels']
            df[file_type['numeric_columns']] = df[
                file_type['numeric_columns']].apply(pd.to_numeric)
            df.Set = df.Set.str.lower()
            df.Cycle = df.Cycle.str.lower()

            self.dfNoise = df[df.Set == 'noise'].copy()
            if self.dfNoise.empty == False:
                df = df[df.Set != 'noise']
                df[['Time', 'Force', 'Tip',
                    'Base']] -= df[['Time', 'Force', 'Tip', 'Base']].iloc[0]

            df.Time = df.Time.astype(int)
            df.Time *= 1e-3

            ##region Data Integrity

            ##Drop any duplicated points (not common, but sometimes occurs between set phases)
            try:
                df = df.set_index('Time')
                df = df[~df.index.duplicated(keep='first')].reset_index()
            except Exception as e:
                printError(e)

            ##look for abberant data points (common with microsquisher data)
            # while True:
            #     f = df.Force.iloc[:-1]
            #     f1 = df.Force.iloc[1:].reset_index(drop=True)
            #     result = f1 - f
            #     if not result[result > df.Force.mean()].empty:
            #         dropIndices = [x + 1 for x in result[result > 10000].index]
            #         df.drop(dropIndices, inplace=True)
            #     else:
            #         break

            ##Interpolate any missing datapoints
            df['datetime'] = pd.to_datetime(df.Time.values, unit='s')
            df = df.set_index('datetime').asfreq(self.sampleTime).interpolate(
                method='time').fillna(method='bfill').reset_index()

            ##endregion

            ##Calculate important quantities
            df[['Force', 'Tip', 'Base', 'Size']] *= 1e-6
            self.thickness = df.Size.iloc[0]
            df['Stress'] = df.Force / self.surfaceArea
            df['Strain'] = df.Tip / self.thickness
            self.dfData = df.drop('datetime', axis=1)

            ##Recognize stepwise stress-relaxation tests, get the equilibrium data for each step
            # yapf:disable
            try:
                self.dfEquil = (
                    df[(df.Cycle.str.contains('hold'))]
                    .groupby('Set')
                    .agg(lambda x: x.iloc[self.equil:].mean())
                                )
            # yapf:enable

            except Exception as e:
                self.dfEquil = pd.DataFrame()
                self.error = 1
                printError(
                    e,
                    'Ensure cycles are properly labeled under "Set" column (C_{1},C_{2},...C_{n})'
                )
            self.max_cycles = len(self.dfEquil)

    def _get_cycles(self):
        df = self.dfData
        cycles = []
        print(F'{self.max_cycles} stress relaxation steps detected from file')
        for cycle in range(1, self.max_cycles + 1):
            print(f'Cycle {cycle}')
            if df[(df.Set == F'c{cycle}')].empty == True:
                continue
            else:
                dfCycle = df[(df.Set == F'c{cycle}')].copy().reset_index(
                    drop=True)

                ##Make quantities relative to beginning of current step
                time0 = dfCycle.iloc[0]
                dfCycle.Time -= time0.Time
                dfCycle.Stress -= time0.Stress
                dfCycle.Strain -= time0.Strain
                dfCycle.Tip -= time0.Tip
                dfCycle['datetime'] = pd.to_datetime(dfCycle.Time.values,
                                                     unit='s')
                cycle = StressRelaxation(
                    self, cycle, dfCycle[[
                        'datetime', 'Time', 'Stress', 'Strain', 'Tip', 'Temp'
                    ]])
                cycles.append(cycle)

        self.cycles = cycles

    def analyze(self):

        ##endregion
        self.matProps = []
        analysisStartTime = time.time()
        self.fullfile_name = file_name = self.folder / self.file_name
        self.expName = expName = F"CLE-{self.bounds['fitting'][1]:.1E}"
        print('---' * 10)
        print('Analysis Start')
        print('---' * 10)
        print('---' * 10)
        print('Sample Properties:'
              F'\n\t\tRadius: {(self.rad * 1e3):.02f} mm'
              F'\n\t\tThickness: {(self.thickness * 1e6):.02f} um'
              F"\n\t\tPreload: {(self.preload * 1e6):.01f} uN")

        previousStress = 0
        for cycle in self.cycles:
            cycleTime0 = time.time()
            print(f'Step {cycle.cycleNum}:'
                  F'\n\t\tComp. Step: {(cycle.rampStep * 1e6):.02f} um'
                  F'\n\t\tRamp Time: {cycle.rampTime:.2f} seconds'
                  F'\n\t\tMax Freq: {cycle.omega:.2E} Hz'
                  F'\n\t\tTotal Time: {cycle.timeEnd} seconds'
                  F'\n\t\tEquilibrium Sample time: {self.equiTime} seconds'
                  F'\n\t\tEquilibrium Stress: {cycle.equilStress:.02f} Pa'
                  F'\n\t\tEquilibrium Strain: {cycle.equilStrain:.02E}'
                  F'\n\t\tEy: {(cycle.Ey*1e-3):.02f} kPa')

            print('---' * 10)
            if cycle.Ey < 0:
                print('Ey<0. Skipping analysis')
                cycle.analysisTime = time.time() - cycleTime0
                print('---' * 10)
                print(
                    F'Cycle Analysis Time: {cycle.analysisTime // 60} min {(cycle.analysisTime % 60):.2f} sec'
                )
                print('---' * 10 + '\n' + '---' * 10)
                continue

            cycleTF = BiphasicTF(cycle)
            cycleTF.fit_tf()
            GR2 = cycleTF.R2
            rad = self.rad
            preload = self.preload
            Hc = cycleTF.Hc
            Ht, lam2, k = cycleTF.matProps
            th = rad**2 / (Ht * k)
            s0 = 1 / th
            pois = lam2 / (Ht + lam2)
            fluidSupport = 1 / (1 + 2 * ((Hc - lam2) / (Ht - lam2)))
            appliedStress = (preload / (np.pi * rad**2)) + previousStress
            previousStress += cycle.equilStress

            print('---' * 10)
            print('Fitted CLE Properties:'
                  F'\n\t\tHc ={(Hc*1e-3):.03f} kPa'
                  F'\n\t\tHt = {(Ht*1e-3):.03f} kPa '
                  F'\n\t\tlam2 = {(lam2*1e-3):.03f} kPa '
                  F'\n\t\tk = {k:.04E} m^4/Ns')
            print('Fit Description:'
                  f'\n\t\tNumber of iterations: {cycleTF.fitSteps}'
                  f'\n\t\tObjective function value: {cycleTF.objMin}'
                  F'\n\t\tTransfer Function R2 = {GR2}')
            print('---' * 10)
            print(
                'Derived Values:'
                F'\n\t\tGel Time: {th:.02f} Seconds'
                F'\n\t\tCharacteristic Freq: {s0:.2E} Hz'
                F'\n\t\tPoisson Ratio: {pois:.04f}'
                F'\n\t\tInstantaneous Fluid Load support: {fluidSupport:.04f}')
            print('---' * 10)
            print('---' * 10)

            date = self.fileData['date']
            exp = self.fileData['exp']
            id = self.fileData['id']
            eye = self.fileData['eye']
            region = self.fileData['region']

            cycle.TF = cycleTF
            cycle.derivedProps = {
                'th': (th, 'sec'),
                's0': (s0, 'Hz'),
                'pois': (pois, ''),
                'fluidSupport': (fluidSupport, '')
            }
            self.sampleProperties = {
                'Radius': (self.rad * 1e3, 'mm'),
                'Thickness': (self.thickness * 1e6, 'um'),
                'Preload': (self.preload * 1e6, 'uN')
            }
            cycle.stepProperties = {
                'eps0': (cycle.rampStep * 1e6, 'um'),
                't0': (cycle.rampTime, 'sec'),
                'maxFreq': (cycle.omega, 'Hz'),
                'timeEnd': (cycle.timeEnd, 'sec'),
                'equilTime': (self.equiTime, 'sec'),
                'equiStress': (cycle.equilStress, 'Pa'),
                'maxStrain': (cycle.maxStrain, ''),
                'equiStrain': (cycle.equilStrain, ''),
                'temp': (cycle.temp[0], 'C')
            }
            cycle.matProps = {
                'Ey': (cycle.Ey, 'Pa'),
                'Hc': (Hc, 'Pa'),
                'Ht': (Ht, 'Pa'),
                'lam2': (lam2, 'Pa'),
                'k': (k, 'm^4/Ns')
            }

            cycle.visualize()

            cycle.sampleInfo = {
                'date': date,
                'group': self.fileData['group'],
                'exp': exp,
                'id': id,
                'eye': eye,
                'region': region,
                'cycle': cycle.cycleNum,
                'FreqR2': GR2,
                'TDR2': cycle.tdR2
            }

            self.export_cycle(cycle)
            if cc.export_cycle_raw_data:
                cycle.export_data()
            self.dfMatProps = pd.concat(self.matProps, ignore_index=True)
            matPropsfile_name = F'{self.folder}{os.sep}{self.file_name}-Params'

            export_df(self.dfMatProps, matPropsfile_name)

            plt.close()
            print('Graphs Exported')

            cycle.analysisTime = time.time() - cycleTime0
            print('---' * 10)
            print(
                F'Cycle Analysis Time: {cycle.analysisTime // 60} min {(cycle.analysisTime % 60):.2f} sec'
            )
            print('---' * 10 + '\n' + '---' * 10)

        self.visualize()
        self.analysisTime = time.time() - analysisStartTime
        print('---' * 10)
        print(F"{'--'*5}{'ANALYSIS COMPLETE'}{'--'*5}")
        print(
            F'File Time: {self.analysisTime // 60} min {(self.analysisTime % 60):.2f} sec'
        )
        print('---' * 10 + '\n' + '---' * 10)

    def export_cycle(self, cycle):

        data = [
            cycle.sampleInfo, self.sampleProperties, cycle.stepProperties,
            cycle.matProps, cycle.derivedProps
        ]
        dataDict = merge_dicts(*data)
        dfParams = pd.DataFrame([dataDict], index=[0])

        def firstEle(x):
            if type(x.values[0]) == tuple:
                return x.values[0][0]
            else:
                return x.values[0]

        dfParams = dfParams.apply(firstEle)
        dfParams = dfParams[[
            'group', 'date', 'exp', 'Preload', 'id', 'eye', 'region', 'cycle',
            'FreqR2', 'TDR2', 'Ey', 'Hc', 'Ht', 'lam2', 'k', 'fluidSupport',
            'maxFreq', 'pois', 's0', 'th', 'temp', 'Thickness', 'Radius', 't0',
            'maxStrain', 'eps0', 'equiStrain', 'equiStress', 'equilTime',
            'timeEnd'
        ]]
        dfParams = dfParams.to_frame().transpose()
        cycle.dfParams = dfParams
        self.matProps.append(dfParams)

    def _set_graph_design(self):
        flatui = ["#1b9e77", "#d95f02", "#7570b3"]
        sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2})
        plt.rcParams.update({
            'figure.dpi': self.settings['figDPI'],
            'lines.markersize': 4
            # ,'font.family':'sans-serif'
        })
        sns.set_palette(sns.color_palette(flatui))

    def expGraph(self, df):
        steps = []
        grouped = df[(df.Cycle.str.contains('compress'))].groupby('Set')
        for name, group in grouped:
            steps.append(group.Time.iloc[0] - 20)
        steps.append(df.Time.iloc[-1])
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(df.Time.values, df.Strain.values)
        ax[1].plot(df.Time.values, df.Stress.values * 1e-3)
        for i in range(1, len(steps)):
            if i == 1:
                ax[0].text(steps[i - 1] - 20,
                           df.Strain.max() + 0.01,
                           F'Step \n {i}',
                           fontsize=label_font_size - 4,
                           zorder=10)
            else:
                ax[0].text((steps[i - 1] + steps[i]) / 2.0,
                           df.Strain.max() + 0.01,
                           F'\n {i}',
                           horizontalalignment='center',
                           fontsize=label_font_size - 4)
        ax[0].set_ylim(0, 0.2)
        ax[0].set_ylabel('Strain', fontsize=label_font_size)
        ax[0].set_xlabel('Time (s)', fontsize=label_font_size)
        ax[1].set_ylabel('Stress (kPa)', fontsize=label_font_size)
        ax[1].set_xlabel('Time (s)', fontsize=label_font_size)
        ax[0].tick_params(axis='both', labelsize=label_font_size - 2)
        ax[1].tick_params(axis='both', labelsize=label_font_size - 2)
        ax[0].vlines(steps[1:-1],
                     0,
                     df.Strain.max() + 0.05,
                     linestyle='--',
                     zorder=3)
        ax[1].vlines(steps[1:-1],
                     0,
                     df.Stress.max() * 1e-3 * 1.2,
                     linestyle='--',
                     zorder=3)
        plt.subplots_adjust(top=0.97,
                            bottom=0.25,
                            right=0.97,
                            left=0.18,
                            wspace=0.4)
        fig.set_size_inches(7.5, 3, forward=True)
        return fig, ax

    def visualize(self):
        self.fullfile_name = file_name = self.folder / self.file_name
        self.expName = expName = F"CLE-{self.bounds['fitting'][1]:.1E}"
        self._set_graph_design()
        exportFigs = self.settings['exportFigs']
        # for cycle in self.cycles:
        #     cycle.visualize()
        if self.settings['gPlotsExp']:
            figExp, axExp = self.expGraph(self.dfData)
            if exportFigs:
                exportfig(figExp, file_name, '0-' + expName + '-Experiment')


class StressRelaxation():

    def __init__(self, BiphasicAnalysis, cycleNum, dfData):

        self.experiment = BiphasicAnalysis

        settings = self.settings = self.experiment.settings
        self.samplingRate = self.experiment.samplingRate
        self.samplingTime = (1 / self.samplingRate) * 1e3
        self.equil = self.experiment.equil
        self.rad = self.experiment.rad
        self.bounds = self.experiment.bounds
        self.preload = self.experiment.preload
        self.file_name = self.experiment.file_name
        self.folder = self.experiment.folder
        self.filter = settings['filterData']
        self.butterParams = settings['butterParams']
        self.upsampleTime = settings['upsampleTime']

        self.vFreq = 10**np.linspace(np.log10(self.bounds['frequency'][0]),
                                     np.log10(self.bounds['frequency'][1]),
                                     cc.mFreq)

        self.cycleNum = cycleNum
        self.dfTest = dfData.copy()
        self.temp = (self.dfTest.Temp.mean(), self.dfTest.Temp.std())
        self.timeEnd = self.dfTest.Time.iloc[-1]
        print(f'Step Run Time: {self.timeEnd:.1f} seconds')
        self.dfTest['StrainFilt'] = self.dfTest.Strain.values
        self.dfTest['StressFilt'] = self.dfTest.Stress.values
        if self.filter:
            self.filter_data()

        self.equilibrium = self.dfTest.drop(
            'datetime', axis=1).apply(lambda x: x.iloc[self.equil:].mean())
        self.rampStep = self.equilibrium.Tip
        self.equilStress = self.equilibrium.StressFilt  # relative equilibrium stress and strain for i cycle
        self.equilStrain = self.equilibrium.StrainFilt
        self.Ey = (
            self.equilStress / self.equilStrain
        )  # Calculate the Matrix modulus from equilibrium stress/strain

        self.normalize_data()
        self.upsample_data()

    def _df_log_reduce(self, df, nSamples=None):
        # region Sample nTime points in log10. [dfUniform -> dfLog]
        nLogSamples = cc.nTime
        if type(nSamples) == int:
            nLogSamples = nSamples

        f = np.log
        finv = np.exp

        epoch = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        dfLog = df.copy()

        try:
            dfLog.datetime = dfLog.datetime - epoch  # make time DEFAULT_FRAME_STYLE
            dfLog.Time = f(dfLog.datetime.dt.total_seconds())
        except Exception as e:
            printError(e)
            dfLog['datetime'] = pd.to_datetime(dfLog.Time.values, unit='s')
            dfLog.datetime = dfLog.datetime - epoch  # make time DEFAULT_FRAME_STYLE
            dfLog.Time = f(dfLog.datetime.dt.total_seconds())

        try:
            logRampTime = dfLog.iloc[dfLog.NStressFilt.idxmax()].Time
        except:
            logRampTime = 0
        uniLogTime = np.linspace(
            dfLog.Time.iloc[0], dfLog.Time.iloc[-1],
            nLogSamples)  # get uniform time array in log10 scale. n = nTime)

        if logRampTime != 0:
            uniLogTime = np.append(uniLogTime, logRampTime)

        uniLogTime = [f(round(finv(x), 3)) for x in uniLogTime]
        uniLogTime = sorted(list(set(uniLogTime)))

        logSample = pd.DataFrame(uniLogTime,
                                 columns=['Time'])  # convert to a dataframe
        logSample['log'] = 1  # 1 = keep these samples in the future
        dfLog = dfLog.append(
            logSample, ignore_index=True,
            sort=False)  # Combine dfLog and the desired sample dataframe
        dfLog = dfLog.sort_values('Time')  # Sort by log10(time)
        dfLog.Time = dfLog.Time.apply(lambda x: round(finv(
            x), 3))  # convert df.Time back to normal time (ms resolution)
        dfLog['datetime'] = pd.to_datetime(
            dfLog['Time'], unit='s')  # update df.datetime to new samples
        dfLog.log = dfLog.log.fillna(
            0)  # Fill NaNs -> 0 = throw away points in reduced dataframe
        dfLog = dfLog.set_index('datetime').interpolate(
            method='time')  # interpolate by time
        dfReduced = dfLog[dfLog.log ==
                          1].copy()  # only keep desired distribution of points
        del dfLog
        dfReduced.drop('log', axis=1, inplace=True)
        return dfReduced

    def filter_data(self):
        dfTest = self.dfTest
        b, a = signal.butter(self.butterParams[0],
                             self.butterParams[1])  # butterworth filter 1
        dfTest['StrainFilt'] = signal.filtfilt(b, a, dfTest.Strain.values)
        dfTest['StressFilt'] = signal.filtfilt(b, a, dfTest.Stress.values)
        dfTest.StressFilt -= dfTest.iloc[0].StressFilt
        dfTest.StrainFilt -= dfTest.StrainFilt.iloc[0]

    def normalize_data(self):
        dfTest = self.dfTest
        dfTest.loc[:, 'NormStrain'] = dfTest.Strain / self.equilStrain
        dfTest.loc[:, 'NormStress'] = dfTest.Stress / self.equilStress
        dfTest.loc[:, 'NStrainFilt'] = dfTest.StrainFilt / self.equilStrain
        dfTest.loc[:, 'NStressFilt'] = dfTest.StressFilt / self.equilStress
        self.maxStrain = dfTest.iloc[dfTest.NStrainFilt.idxmax()].NStrainFilt
        self.rampTime = dfTest.iloc[dfTest.NStressFilt.idxmax()].Time
        self.omega = 2 * np.pi / (self.rampTime)

        dfTest.loc[dfTest.index.max() +
                   1] = dfTest.loc[dfTest.index.max()].values
        dfTest.loc[dfTest.index.max(), 'Time'] += 1 / self.samplingRate
        dfTest.loc[dfTest.index.max(), 'datetime'] += datetime.timedelta(
            seconds=(1 / self.samplingRate))

        dfTest.loc[dfTest.index.max() +
                   1] = dfTest.loc[dfTest.index.max()].values
        dfTest.loc[dfTest.index.max(), 'Time'] += cc.extendTime
        dfTest.loc[dfTest.index.max(),
                   'datetime'] += datetime.timedelta(seconds=cc.extendTime)
        dfTest.loc[
            dfTest.index.max() - 1:dfTest.index.max(),
            ['NormStrain', 'NormStress', 'NStrainFilt', 'NStressFilt']] = [
                1, 1, 1, 1
            ]
        self.dfTest = dfTest.set_index('datetime').asfreq(
            f'{(self.samplingTime)}ms').interpolate(
                method='time').reset_index()

    def upsample_data(self):
        ## region Interpolation
        self.dfTest.drop(['Tip', 'Temp'], axis=1, inplace=True)
        self.dfTest = self.dfTest.set_index('datetime')
        upsampleFreq = (1 / (self.upsampleTime * 1e-3))
        print(F'Upsampling to {upsampleFreq} Hz...')
        print(
            F'df length: {int(round(upsampleFreq*(self.timeEnd+cc.extendTime)))}'
        )
        try:
            f = lambda: self.dfTest.asfreq(f'{(self.upsampleTime)}ms'
                                           ).interpolate(method='time'
                                                         ).reset_index()
            self.dfTest = f()
            print('Done')
            increment = self.upsampleTime * 1e-3
        except Exception as e:
            printError(e, 'Error upsampling data')
            increment = 1 / self.samplingRate

        self.dfTest.Time += increment
        self.dfTest.datetime += datetime.timedelta(seconds=increment)
        dfUniform = self.dfTest.copy()
        # endregion

        dfLogReduced = self._df_log_reduce(dfUniform)
        dfLogReduced = (dfLogReduced.reset_index()[[
            'Time', 'NormStrain', 'NStrainFilt', 'StrainFilt', 'NormStress',
            'NStressFilt', 'StressFilt'
        ]])  # only keep certain columns
        self.dfUniform = dfUniform
        self.dfLogReduced = dfLogReduced

    def export_data(self):

        file_name = self.folder / f"{self.file_name}-Cycle{self.cycleNum}-Data"
        export_df(self.dfReduced, file_name, sheetname='Stress')
        export_df(self.dfUniform, file_name, sheetname='Time Domain', mode='a')
        export_df(self.dfFreq, file_name, sheetname='Freq Domain', mode='a')

    def visualize(self):

        def stepSignals(df,
                        filt=False,
                        norm=False,
                        export=False,
                        log_scale=False):
            """df should have 'Time', 'Strain', 'Stress' columns at minimum """

            figFilt, (axf1, axf2) = plt.subplots(nrows=2, sharex=True)
            axf = (axf1, axf2)

            df = df[df.Time < self.timeEnd]
            t = df.Time.values
            strain = df.Strain.values
            if norm:
                strain = df.NormStrain.values
            stress = df.Stress.values
            if norm:
                stress = df.NormStress.values

            if ulttb:
                ##region Downsampling
                t1 = self.rampTime * 1.5
                desiredLen1 = 15
                dstrain = downsample(np.asarray(list(zip(t, strain))), 500)
                tstrain, rstrain = list(zip(*dstrain))

                tstress1 = []
                rstress1 = []
                tstress2 = []
                rstress2 = []
                tstress3 = []
                rstress3 = []

                df1 = df[(df.Time <= (t1))]
                t = df1.Time.values
                stress = df1.Stress.values

                n = round(len(t) / desiredLen1)
                tstress1 = t[::n]
                rstress1 = stress[::n]

                df3 = df[(df.Time >= (t1))]
                t = df3.Time.values
                stress = df3.Stress.values
                if norm:
                    stress = df3.NormStress.values
                dstress = downsample(np.asarray(list(zip(t, stress))), 20)
                tstress3, rstress3 = list(zip(*dstress))

                tstress = [*tstress1, *tstress2, *tstress3]
                rstress = [*rstress1, *rstress2, *rstress3]
                ##endregion
            else:
                tstrain = t[::10]
                rstrain = strain[::10]
                tstress = tstrain
                rstress = stress[::10]
            data = [tstress, rstress]
            dfOut = df[df.Time.isin(tstress)]
            axf[0].plot(tstrain,
                        rstrain,
                        linestyle='-',
                        linewidth=1,
                        color='k',
                        label='Applied strain')
            axf[1].plot(tstress,
                        rstress,
                        linestyle='none',
                        marker='o',
                        markersize=4,
                        markerfacecolor='none',
                        markeredgecolor='k',
                        label='Experimental Stress',
                        zorder=10)
            if filt == True:
                axf[0].plot(df[df.Time <= self.timeEnd].Time.values,
                            df[df.Time <= self.timeEnd].NStrainFilt.values,
                            label='Filtered')
                axf[1].plot(df[df.Time <= self.timeEnd].Time.values,
                            df[df.Time <= self.timeEnd].NStressFilt.values,
                            label='Filtered',
                            zorder=2)
            axf[0].legend(fontsize=label_font_size - 6, loc=4, frameon=False)
            axf[1].legend(fontsize=label_font_size - 6, frameon=False)
            axf[1].set_xlabel('Time(s)', fontsize=label_font_size)
            if norm:
                axf[1].set_ylabel('Norm Stress', fontsize=label_font_size)
                axf[0].set_ylabel('Norm Strain', fontsize=label_font_size)
            else:
                axf[1].set_ylabel('Stress (Pa)', fontsize=label_font_size)
                axf[0].set_ylabel('Strain', fontsize=label_font_size)
            axf[1].tick_params(axis='y', labelsize=label_font_size - 4)
            axf[1].tick_params(axis='x', labelsize=label_font_size - 4)
            axf[0].tick_params(axis='y', labelsize=label_font_size - 2)

            plt.subplots_adjust(top=0.99,
                                bottom=0.15,
                                left=0.2,
                                right=0.96,
                                hspace=0.1)
            if log_scale == True:
                axf[1].set_xscale('log')
                axf[1].set_xticks([1, 10, 100, 1000])

            axf[1].get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            if export & (not filt):
                plt.subplots_adjust(top=0.98,
                                    bottom=0.18,
                                    left=0.15,
                                    right=0.95,
                                    hspace=0.1)
                figFilt.set_size_inches(4, 5, forward=True)
                exportfig(
                    figFilt, file_name,
                    F'{cycle}-' + self.experiment.expName + '-NormalizedStep')
            if export & filt:
                figFilt.set_size_inches(6, 6, forward=True)
                exportfig(
                    figFilt, file_name,
                    F'{cycle}-' + self.experiment.expName + '-FilteredStep')
            figFilt.set_size_inches(3.5, 4, forward=True)
            figFilt.align_ylabels()
            return figFilt, (axf1, axf2), data, dfOut

        self.tdR2 = np.nan
        try:
            cycleTF = self.TF
            Ey = self.Ey
            Hc = self.TF.Hc
            Ht, lam2, k = self.TF.matProps

        except Exception as e:
            print(e)
            return

        exportFigs = self.settings['exportFigs']
        figName = F'C{self.cycleNum}-{self.experiment.expName}'
        file_name = self.experiment.fullfile_name

        dfUniform = self.dfUniform
        dfUniform = dfUniform[dfUniform.Time <= self.timeEnd]
        dfTest = self.dfLogReduced[self.dfLogReduced.Time <= self.timeEnd]
        dfFreq = cycleTF.dfFreq
        self.dfFreq = dfFreq
        dfCurve = dfFreq[(dfFreq.freq > self.bounds['fitting'][0])
                         & (dfFreq.freq < self.bounds['fitting'][1])]
        testMode = self.settings['testMode']
        graphBounds = self.settings['graphBounds']
        dfGraph = dfFreq[(dfFreq.freq > graphBounds[0])
                         & (dfFreq.freq < graphBounds[1])]

        ##Plot Exp TF
        if self.settings['gPlotsLaplace']:
            ##region Downsampling
            f = dfGraph.freq.values
            eTF = dfGraph.eTF.values
            if ulttb:
                d_freq = downsample(np.asarray(list(zip(f, eTF))), 25)
                f, eTF = list(zip(*d_freq))
            ##endregion

            figG, ax2 = plt.subplots()  # Create new plot objects
            ax2.plot(f,
                     eTF,
                     label='Experimental',
                     linestyle='none',
                     marker='o',
                     markersize=4,
                     markerfacecolor='none',
                     markeredgecolor='k',
                     zorder=1)
            ax2.set_xscale('log')
            ax2.set_ylabel('Transfer Function $G(\\omega)$',
                           fontsize=label_font_size)
            ax2.set_xlabel('Frequency (Hz)', fontsize=label_font_size)
            ax2.tick_params(axis='both', labelsize=label_font_size - 2)
            if testMode == 'c':
                ax2.set_ylim(top=1.1, bottom=-0.1)
            figG.set_size_inches(3.5, 3, forward=True)
            plt.subplots_adjust(top=0.94, bottom=0.23, right=0.98, left=0.24)
            if exportFigs:
                exportfig(figG, file_name, 'G-' + figName)

            # region Transfer function Fit
            if (self.settings['gPlotsLaplaceFit']):
                dfGraph = dfFreq[(dfFreq.freq > graphBounds[0])
                                 & (dfFreq.freq < graphBounds[1])]
                figG.set_size_inches(3.5, 3, forward=True)
                ax2.axvline(self.bounds['fitting'][0], color='gray', alpha=0.8)
                ax2.axvline(self.bounds['fitting'][1], color='gray', alpha=0.8)
                ax2.plot(dfGraph.freq.values,
                         dfGraph.mTF.values,
                         linestyle='--',
                         color='k',
                         linewidth=1,
                         label='Model Fit',
                         zorder=0)
                ax2.legend(fontsize=label_font_size - 6, loc=1, frameon=False)
                ax2.text(0.2,
                         0.15,
                         F'$R^2$ = \n {cycleTF.R2:.05f}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax2.transAxes,
                         fontsize=label_font_size - 4)
                if exportFigs:
                    exportfig(figG, file_name, 'GFit-' + figName)
        figFit, axFit, expStressData, dfStepReduced = stepSignals(
            dfUniform, filt=False, export=cc.export_ramp_data)
        ##region Bode Plots
        if self.settings['gPlotsBode']:
            f = list(dfGraph.freq.values)
            eGain = list(dfGraph.eGain.values)
            ePhase = list(dfGraph.ePhase.values)
            n = round(len(f) / 30)
            rfreq = f[::n]
            reGain = eGain[::n]
            rePhase = ePhase[::n]

            figBode, (axBode1, axBode2) = plt.subplots(nrows=2, sharex=True)
            axBode1.plot(rfreq,
                         reGain,
                         label='Experimental',
                         linestyle='none',
                         marker='o',
                         markersize=4,
                         markerfacecolor='none',
                         markeredgecolor='k',
                         zorder=1)
            axBode2.plot(rfreq,
                         rePhase,
                         label='Experimental',
                         linestyle='none',
                         marker='o',
                         markersize=4,
                         markerfacecolor='none',
                         markeredgecolor='k',
                         zorder=1)
            if self.settings['gPlotsBodeFit']:
                axBode1.plot(dfFreq.freq.values,
                             dfFreq.mGain.values,
                             label='Model',
                             linestyle='--',
                             color='k',
                             linewidth=1,
                             zorder=0)
                axBode2.plot(dfFreq.freq.values,
                             dfFreq.mPhase.values,
                             label='Model',
                             linestyle='--',
                             color='k',
                             linewidth=1,
                             zorder=0)

            axBode1.set_xscale('log')
            axBode1.set_ylabel('Gain \n $G(j\\omega) (dB)$',
                               fontsize=label_font_size)
            axBode2.set_xlabel('Frequency (Hz)', fontsize=label_font_size)
            axBode2.set_ylabel('Phase \n$\\angle G(j\\omega) (deg)$',
                               fontsize=label_font_size)
            if testMode == 'c':
                axBode1.set_ylim(top=10, bottom=-100)
                axBode2.set_ylim(top=50, bottom=-100)
                axBode1.legend(loc=3,
                               fontsize=label_font_size - 6,
                               frameon=False)
            else:
                axBode2.set_ylim(bottom=-50, top=100)

            axBode1.tick_params(axis='both', labelsize=label_font_size - 2)
            axBode1.get_xaxis().set_visible(False)
            axBode2.tick_params(axis='both', labelsize=label_font_size - 2)
            plt.subplots_adjust(top=0.98,
                                bottom=0.17,
                                right=0.98,
                                left=0.32,
                                hspace=0.19)
            figBode.set_size_inches(3.5, 5, forward=True)
            if exportFigs:
                exportfig(figBode, file_name, 'B-' + figName)
        ## endregion

        ##region Objective Function
        if self.settings['gPlotsMinima']:
            objCount = 0
            cycle = self.cycleNum
            dfCObj = pd.DataFrame(columns=['cycle', 'Ht', 'lam2', 'k', 'f'])
            objMC4 = cycleTF._obj_func4
            lam2L = np.linspace(-0.5 * min(Hc, Ht), min(Hc, Ht), 25)
            htL = 10**np.linspace(np.log10(0.01 * Ht), np.log10(100 * Ht), 25)
            kL = 10**np.linspace(np.log10(0.01 * k), np.log10(100 * k), 25)
            for i in range(len(lam2L)):
                Hc = Ey + 2 * lam2L[i]**2 / (Ht + lam2L[i])
                p = [cycle, Ht, lam2L[i], k]
                p.append(objMC4([Hc] + p[1:], dfCurve.freq, dfCurve.eTF))
                dfCObj.loc[objCount] = p
                objCount += 1

            for i in range(len(htL)):
                Hc = Ey + 2 * lam2**2 / (htL[i] + lam2)
                p = [cycle, htL[i], lam2, k]
                p.append(objMC4([Hc] + p[1:], dfCurve.freq, dfCurve.eTF))
                dfCObj.loc[objCount] = p
                objCount += 1

            Hc = Ey + 2 * lam2**2 / (Ht + lam2)
            for i in range(len(kL)):
                p = [cycle, Ht, lam2, kL[i]]
                p.append(objMC4([Hc] + p[1:], dfCurve.freq, dfCurve.eTF))
                dfCObj.loc[objCount] = p
                objCount += 1

            dfCObj.Ht *= (1e-3)
            dfCObj.lam2 *= (1e-3)
            b1 = len(lam2L)
            b2 = b1 + len(htL)
            b3 = b2 + len(kL)
            sliceLam2 = slice(0, b1)
            sliceHt = slice(b1, b2)
            sliceK = slice(b2, b3)

            figObj, axObj = plt.subplots(nrows=1, ncols=3, sharey='row')
            axObj[0].plot(dfCObj.lam2.values[sliceLam2],
                          dfCObj.f.values[sliceLam2],
                          linestyle='none',
                          marker='o')
            axObj[1].plot(dfCObj.Ht.values[sliceHt],
                          dfCObj.f.values[sliceHt],
                          linestyle='none',
                          marker='o')
            axObj[2].plot(dfCObj.k.values[sliceK],
                          dfCObj.f.values[sliceK],
                          linestyle='none',
                          marker='o')

            axObj[0].axvline(lam2 * 1e-3,
                             ymax=0.85,
                             color='black',
                             linestyle='--')
            axObj[1].axvline(Ht * 1e-3,
                             ymax=0.85,
                             color='black',
                             linestyle='--')
            axObj[2].axvline(k, ymax=0.85, color='black', linestyle='--')

            lamLim = axObj[0].get_xlim()
            lamPos = (lam2 * 1e-3 - lamLim[0]) / (lamLim[1] - lamLim[0])
            if (lamPos < 0.1) | (lamPos > 0.9):
                lamPos = 0.5

            axObj[0].text(lamPos,
                          0.92,
                          F'{(lam2 * 1e-3):.02f}',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axObj[0].transAxes,
                          fontsize=24)
            axObj[1].text(0.5,
                          0.92,
                          F'{(Ht * 1e-3):.02f}',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axObj[1].transAxes,
                          fontsize=24)
            axObj[2].text(0.5,
                          0.92,
                          F'{(k):.02E}',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axObj[2].transAxes,
                          fontsize=24)

            axObj[0].set_ylabel('Obj Function', fontsize=24)
            axObj[0].set_xlabel('$\lambda_2$ (kPa)', fontsize=24)
            axObj[1].set_xlabel('$H_t$ (kPa)', fontsize=24)
            axObj[2].set_xlabel('k $\\frac{m^4}{Ns}$', fontsize=24)

            axObj[1].set_xscale('log')
            axObj[2].set_xscale('log')
            plt.subplots_adjust(top=0.99,
                                bottom=0.19,
                                right=0.99,
                                left=0.14,
                                wspace=0.04,
                                hspace=0.45)
            figObj.set_size_inches(10, 6.6, forward=True)
            axObj[0].tick_params(axis='both', labelsize=20)
            axObj[1].tick_params(axis='both', labelsize=20)
            axObj[2].tick_params(axis='both', labelsize=20)
            self.dfObj = dfCObj
            if exportFigs:
                exportfig(figObj, file_name, 'O-' + figName)
        ##endregion

        if self.settings['gModelFit'] | self.settings['gSimpleModelFit']:

            Aexp = A(Ht, lam2)
            Bexp = B(Hc, Ht, lam2)
            Cexp = C(Hc, Ht, lam2)
            th = self.derivedProps['th'][0]
            p = [Aexp, Bexp, Cexp, th]
            rootList = roots(1000)
            scale = calcScale(p, rootList)
            rootList = rootList[:100]

        if self.settings['gModelFit']:
            dfUniform = dfUniform[dfUniform.Time <= self.timeEnd]
            self.dfUniform = dfUniform

            if not cc.invLap:
                phi = dfUniform.iloc[-1].Stress / dfUniform.iloc[-1].NStressFilt
                print('Generate Model Stress by Impulse Response Convolution')
                ### Faster, but amplifies high-frequency noise

                dt = dfUniform.Time.iloc[1] - dfUniform.Time.iloc[0]
                dfUniform['impulse'] = impulse(dfUniform, rootList, p)
                conv = signal.fftconvolve(dfUniform.impulse.values,
                                          dfUniform.NStrainFilt.values,
                                          mode='full') * dt
                conv = conv[0:len(dfUniform.Time.values)] + scale * (
                    dfUniform.NStrainFilt.values)
                dfUniform['NStressFiltFitting'] = conv
                dfUniform[
                    'modelStressFit'] = dfUniform['NStressFiltFitting'] * phi
                dfUniform.drop('impulse', axis=1, inplace=True)
                self.dfUniform = dfUniform
                self.tdR2 = np.nan
                if ulttb:
                    dfReduced = dfUniform[dfUniform.Time.isin(
                        expStressData[0])]
                else:
                    dfReduced = dfUniform[(dfUniform.index % 10 == 0) |
                                          (dfUniform.index == 0)]

            else:

                print('Generate Model Stress by Numerical Inverse Laplace')
                dfReduced = dfStepReduced
                t = dfReduced.Time.values

                # yapf: disable
                f = lambda s: self.TF.modelDynamicModulus(s) * self.TF.lapStrain(s)
                StressFilt = map(
                    lambda x: phi * invertlaplace(f, x, method='dehoog', dps=10, degree=20), t)
                # yapf: enable

                print('Inverse Laplace Tranforming signal...')
                lapStartTime = time.time()
                downsampModelStress = list(StressFilt)
                dfReduced['modelStressFit'] = downsampModelStress

                print(f'Done... {time.time()-lapStartTime} seconds')
                self.tdR2 = calculate_r2(dfReduced.modelStressFit.values,
                                         dfReduced.Stress.values)
                print(f'Model R2: {self.tdR2}')
            self.dfReduced = dfReduced

            axFit[1].plot(dfReduced.Time.values,
                          dfReduced.modelStressFit.values,
                          linestyle='--',
                          color='gray',
                          linewidth=1,
                          label='Model Fit',
                          zorder=0)
            axFit[0].legend(loc=4, fontsize=label_font_size - 6, frameon=False)
            axFit[1].legend(loc=1, fontsize=label_font_size - 6, frameon=False)
            if exportFigs:
                exportfig(figFit, file_name, 'M-' + figName)
            simpleModelFit = False

        if self.settings['gSimpleModelFit']:

            dfUniform = dfUniform[dfUniform.Time <= self.timeEnd]
            dfUniform['modelFit'] = ucc_ramp_hold_model(
                dfUniform.Time.values, p, rootList, self.rampTime)
            self.dfUniform = dfUniform
            reduced = dfUniform[(dfUniform.index % 10 == 0) |
                                (dfUniform.index == 0)]

            tnorm = reduced.Time.values / self.rampTime

            axFit[0].plot(reduced.Time.values,
                          norm_ramp_hold(tnorm) * self.equilStrain,
                          linestyle='-')
            axFit[1].plot(reduced.Time.values,
                          reduced.modelFit.values,
                          linestyle='-',
                          zorder=1)
            if exportFigs:
                exportfig(figFit, file_name, 'SM-' + figName)


class BiphasicTF():

    def __init__(self, StressRelaxation):
        self.matProps = []
        self.mTF = []
        self.eTF = []
        self.eGain = []
        self.mGain = []
        self.ePhase = []
        self.mPhase = []
        self.R2 = 0
        self.FreqR2 = 0
        self.Hc = 0
        self.mode = StressRelaxation.settings['testMode']
        self.Ey = StressRelaxation.Ey
        self._fitting_bounds = StressRelaxation.settings['fittingBounds']
        self.rad = StressRelaxation.settings['rad'] * 1e-3
        self.verbose = StressRelaxation.settings['verbose']
        self.freq = StressRelaxation.vFreq
        self.time = StressRelaxation.dfLogReduced.Time.values
        self.nstress = StressRelaxation.dfLogReduced.NStressFilt.values
        self.nstrain = StressRelaxation.dfLogReduced.NStrainFilt.values
        timeExtend = StressRelaxation.extendTime
        self.timeEnd = self.time[-1] - timeExtend
        self.rampTime = self.time[np.argmax(self.nstress)]

        self.eTF = self._calc_exp_tf()

        self.eGain, self.ePhase = self._calc_exp_bode()

    def set_fitting_bounds(self, lower, upper):
        if (type(lower) == float) & (type(upper) == float):
            if lower < upper:
                self._fitting_bounds = [lower, upper]
            else:
                self._fitting_bounds = [upper, lower]
        else:
            print('set bounds as floats')

    def _calc_exp_tf(self):
        mode = self.mode
        s = self.freq * 2 * np.pi
        lapStrain = interpolation_signal(self.time, self.nstrain)
        lapStress = interpolation_signal(self.time, self.nstress)

        eps = list(map(lambda x: lapStrain(x), s))
        phi = list(map(lambda x: lapStress(x), s))
        eps = np.asarray(eps)
        phi = np.asarray(phi)
        if mode == 'c':
            gExp = np.divide(eps, phi)
        else:
            gExp = np.divide(phi, eps)
        return gExp

    def _calc_exp_bode(self):
        gain, phase = experimental_to_bode(self.freq, self.time, self.nstrain,
                                           self.nstress, self.mode)
        return gain, phase

    def _obj_func4(self, x0, freq, Gexp, f=None):
        if not f:
            f = self._ucc_cle4
        freq = freq.values
        Gmodel = f(freq, x0)[0]
        c3 = (np.dot(
            Gexp, Gmodel))**2 / (np.dot(Gexp, Gexp) * np.dot(Gmodel, Gmodel))
        return np.log(1 - c3)

    def _obj_func3(self, x0, freq, Gexp, Ey, f, w=[1, 1, 1, 1, 1, 1]):
        params = [x / y for x, y in zip(x0, w)]
        Ht, lam2 = params[0], params[1]
        Hc = Ey + 2 * lam2**2 / (Ht + lam2)
        if (lam2 > min(Hc, Ht)) | (lam2 < -0.5 * min(Hc, Ht)) | (
                np.abs(Hc - (Ey + (2 * lam2**2) / (Ht + lam2))) > 1000):
            res = 1000
        else:
            Gmodel = f(freq, params, Ey)[0]
            c3 = (np.dot(Gexp, Gmodel))**2 / (np.dot(Gexp, Gexp) *
                                              np.dot(Gmodel, Gmodel))
            res = np.log(1 - c3)
        return res

    def _roots(self, n, p):
        Av = p[0]

        def f(r):
            return r * special.jn(0, (r)) - Av * special.jn(1, (r))

        count = 1
        rL = [0]
        aL = [.01]
        bL = [.3]
        while count < n + 1:
            freq = np.linspace(bL[count - 1] + .01, bL[count - 1] + 10, 100)
            values = f(freq)
            for i in range(1, len(freq)):
                if (values[i - 1] * values[i]) < 0:
                    a = freq[i - 1]
                    b = freq[i]
                    break
                else:
                    continue
            aL.append(a)
            bL.append(b)
            rL.append(optimize.bisect(f, a, b))
            count += 1
        return np.asarray(rL)

    def _calc_p(self, x0):
        return calculate_biphasic_params(self.Ey, self.rad, x0)

    def _obj_func3_time(self, x0, time, Ey):
        num1 = 20
        Ht, lam2, k = x0[0], x0[1], x0[2]
        Hc = Ey + 2 * lam2**2 / (Ht + lam2)
        if (lam2 > min(Hc, Ht)) | (lam2 < -0.5 * min(Hc, Ht)) | (
                np.abs(Hc - (Ey + (2 * lam2**2) / (Ht + lam2))) > 1000):
            res = 1e3
        else:
            p = self._calc_p(x0)
            roots = self._roots(num1, p)[1:]
            modelPhi = self._modelRamp(time, p, roots)
            # c3 = (np.dot(self.nstress, modelPhi)) ** 2 / (np.dot(self.nstress, self.nstress) * np.dot(modelPhi, modelPhi))
            res = sum((modelPhi - self.nstress)**2)
            # o = 1 - c3
        return res

    def _modelRamp(self, time, p, root):
        return ucc_ramp_hold_model(time, p, root, self.rampTime)

    def dynamic_modulus(self, params, Ey) -> Callable:
        H_t, lam2, kr = params
        return create_dynamic_modulus(self.rad, Ey, H_t, lam2, kr)

    def _ucc_cle3(self, freq, params, Ey, imagin=1) -> int:
        mode = self.mode
        return ucc_cle_3param(mode, self.rad, freq, params, Ey, imagin)

    def _ucc_cle4(self, freq, params, imagin=1):
        mode = self.mode
        return ucc_cle_4param(mode, self.rad, freq, params, imagin)

    def fit_tf(self):
        print('Differential Evolution Fitting:')
        print('Starting...')

        weights = [1, 1, 1]
        self.fittingObj = obj = obj_func_3param
        matModel = self._ucc_cle3
        fitMask = (self.freq > self._fitting_bounds[0]) & (
            self.freq < self._fitting_bounds[1])
        self.Gexp = self.eTF[fitMask]

        bounds = [(1e3 * weights[0], 1e7 * weights[0]),
                  (-1 * self.Ey * weights[1], self.Ey * 2 * weights[1]),
                  (1e-18 * weights[2], 1e-5 * weights[2])]
        self.paramBounds = bounds
        arr = (self.freq[fitMask], self.Gexp, self.Ey, matModel)
        fitCount = 0
        while True:

            resDE = optimize.differential_evolution(obj,
                                                    bounds,
                                                    popsize=40,
                                                    tol=1e-10,
                                                    recombination=0.7,
                                                    mutation=(0.75, 1.75),
                                                    updating='deferred',
                                                    workers=-1,
                                                    disp=self.verbose,
                                                    args=(arr))
            fitCount += 1
            print(resDE.fun)
            if resDE.fun < -5:
                break
            elif fitCount > 200:
                print('\nFit Failed')
                break
        try:
            self.matProps = resDE.x
            self.objMin = resDE.fun
            self.fitSteps = resDE.nit
            self.fitResult = resDE
            print('Done')
            self.Hc = self.Ey + 2 * self.matProps[1]**2 / (self.matProps[0] +
                                                           self.matProps[1])
            self.modelDynamicModulus = self.dynamic_modulus(
                self.matProps, self.Ey)
            self.lapStrain = interpolation_signal(self.time, self.nstrain)
            self.mTF = self._ucc_cle3(self.freq, self.matProps, self.Ey)[0]
            self.mGain, self.mPhase = self._ucc_cle3(self.freq, self.matProps,
                                                     self.Ey, 1j)
            self.R2 = calculate_r2(self.mTF[fitMask], self.Gexp)

            self.dfFreq = pd.DataFrame({
                'freq': self.freq,
                'eTF': self.eTF,
                'mTF': self.mTF,
                'eGain': self.eGain,
                'mGain': self.mGain,
                'ePhase': self.ePhase,
                'mPhase': self.mPhase
            })
            return True
        except:
            return False

    def fit_stress(self):
        print('Differential Evolution Fitting:')
        print('Starting...')
        obj = self._obj_func3_time
        fitMask = (self.freq > self._fitting_bounds[0]) & (
            self.freq < self._fitting_bounds[1])
        self.Gexp = self.eTF[fitMask]
        # weights = [scaleHt, scaleLam2, scalek, scaleC, scaleT1, scaleT2]
        weights = [1, 1, 1, 1, 1, 1]
        bounds = [(1e3 * weights[0], 1e7 * weights[0]),
                  (-1 * self.Ey * weights[1], self.Ey * 2 * weights[1]),
                  (1e-18 * weights[2], 1e-10 * weights[2])]
        arr = (self.time, self.Ey)
        fitCount = 0
        while True:
            resDE = optimize.differential_evolution(obj,
                                                    bounds,
                                                    popsize=50,
                                                    tol=.01,
                                                    mutation=(0.5, 1.25),
                                                    updating='deferred',
                                                    workers=-1,
                                                    disp=True,
                                                    args=(arr))
            fitCount += 1
            if resDE.fun != 1000:
                print(resDE.x, )
                break
            elif fitCount > 200:
                print('\nFit Failed')
                break
        self.matProps = resDE.x
        self.Hc = self.Ey + 2 * self.matProps[1]**2 / (self.matProps[0] +
                                                       self.matProps[1])
        self.objMin = resDE.fun
        self.mTF = self._ucc_cle3(self.freq, self.matProps, self.Ey)[0]
        self.mGain, self.mPhase = self._ucc_cle3(self.freq, self.matProps,
                                                 self.Ey, 1j)
        p = self._calc_p(self.matProps)
        self.mStress = self._modelRamp(self.time, p, self._roots(20, p)[1:])
        self.FreqR2 = calculate_r2(self.mTF[fitMask], self.Gexp)
        self.R2 = calculate_r2(self.mStress, self.nstress)

    def get_data(self):
        columns = ['t0']
        if not self.Hc == 0:
            columns.append('Hc')
            if len(self.matProps) == 3:
                columns.extend(['Ht', 'lam2', 'k'])
            elif len(self.matProps) == 6:
                columns.extend(['Ht', 'lam2', 'k', 'c', 'T1', 'T2'])
            else:
                columns = []
        if len(columns) > 1:
            data = [self.rampTime, self.Hc]
            data.extend(self.matProps)
            df = pd.DataFrame(columns=columns)
            df.loc[0] = data
            return df