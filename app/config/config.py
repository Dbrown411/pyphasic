import re as r
from pathlib import Path


class ConfigContainer:

    nTime = 1000  # number of datapoints in time domain
    mFreq = 1000  # number of datapoints in laplace domain
    nParam = 100  # number of parameters to try for ChiSquare fitting
    extendTime = 2000  # number of datapoints to extend the equilibrium values

    invLap = True  ##Method used to get time domain prediction of stress
    log_messages = True
    export_ramp_data = False
    label_font_size = 14
    preload_pattern = r.compile(r'[A-Za-z]*')
    export_cycle_raw_data = False
    default_dir = Path(__file__).parent.parent.parent / "Example_Data"
    settings_experiment = {
        'cycleCount': 3,
        'samplingRate': 5,
        'rad': 0.5,
        'equiTime': 50
    }
    settings_analysis = {
        'freqBounds': (1e-8, 1),
        'upsampleTime': 10,
        'fittingBounds': (1e-7, .1),
        'testMode': 'c',
        'filterData': False,
        'butterParams': (7, 0.2),
        'verbose': False
    }
    settings_export = {
        'color': 'dark',
        'graphBounds': (1e-7, 1),
        'exportFigs': True,
        'figDPI': 90,
        'gPlotsExp': True,
        'gPlotsMinima': True,
        'gPlotsLaplace': True,
        'gModelFit': True,
        'gPlotsLaplaceFit': True,
        'gSimpleModelFit': False,
        'gPlotsBode': True,
        'gPlotsMatProps': True,
        'gPlotsBodeFit': True,
        'gPlotsCompMod': True
    }

    default_settings = {
        'experiment': settings_experiment,
        'analysis': settings_analysis,
        'export': settings_export
    }
    settings_labels = {
        'cycleCount': 'Max Cycles:',
        'samplingRate': 'Sampling Rate (Hz):',
        'rad': 'Sample Radius (mm)',
        'equiTime': 'Equilibrium (seconds from last point):',
        'freqBounds': 'Frequency Bounds (Hz)',
        'upsampleTime': 'Analysis Upsampling (ms):',
        'fittingBounds': 'Fitting Bounds (Hz)',
        'testMode': 'Control Mode:',
        'filterData': 'Filter?',
        'butterParams': 'Filter Parameters:',
        'verbose': 'Verbose fitting?',
        'graphBounds': 'Graph Bounds (Hz):',
        'exportFigs': 'Export Graphs?',
        'figDPI': 'Figure DPI:',
        'gPlotsExp': 'Experiment:',
        'gPlotsLaplace': 'Transfer Function:',
        'gPlotsLaplaceFit': 'Fit:',
        'gPlotsBode': 'Bode:',
        'gPlotsBodeFit': 'Fit:',
        'gPlotsMinima': 'Objective Function:',
        'gModelFit': 'Time Domain Fit:',
        'gSimpleModelFit': 'Simple model:',
        'gPlotsMatProps': 'Measured:',
        'gPlotsCompMod': 'Fitted:',
        'color': 'Color Scheme:'
    }
