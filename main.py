from app.config import ConfigContainer as cc
from app.file_browser import selectDirectory
from app.analysis import AnalysisProgress
import wx

if __name__ == '__main__':
    app = wx.App(False)

    ##UI for file selection
    ex = selectDirectory(cc.default_settings, cc.settings_labels)
    app.SetTopWindow(ex)
    ex.Show()
    app.MainLoop()
    ##

    selected_settings = ex.settings
    selectedFiles = ex.selectedFiles
    if selectedFiles:
        ##UI for Biphasic Analysis
        app2 = wx.App()
        a = AnalysisProgress(selectedFiles, selected_settings)
        app2.SetTopWindow(a)
        a.Show()
        app2.MainLoop()
    ##
    else:
        print('No Files Selected')