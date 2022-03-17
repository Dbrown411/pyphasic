import sys
import wx
import time
from threading import Thread
from pubsub import pub
from app.config import ConfigContainer as cc
from .fitting import fit_experiment_files


#################  PROGRESS UI    #################################
class AnalysisThread(Thread):
    """Test Worker Thread Class."""

    def __init__(self, files, selected_settings):
        """Init Worker Thread Class."""
        Thread.__init__(self)

        self.files = files
        self.selected_settings = selected_settings

        self.start()  # start the thread

    # ----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        wx.CallAfter(pub.sendMessage, "update", msg='start')
        fit_experiment_files(self.files, self.selected_settings)
        wx.CallAfter(pub.sendMessage, "update", msg="end")


class AnalysisProgress(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self, files: list, selected_settings: dict):
        ##Current locations of Data
        self.files = files
        self.num_files = len(files)
        self.selected_settings = selected_settings

        self.startTime = time.time()
        self.time = time.time()
        self.elapsed = 0
        self.file_counter = 0

        ##
        wx.Frame.__init__(self,
                          None,
                          wx.ID_ANY,
                          "Analysis in progress",
                          style=wx.DEFAULT_FRAME_STYLE)
        self.panel = self._init_gui()

        AnalysisThread(self.files, self.selected_settings)
        if cc.log_messages:
            redir = RedirectText(self.log)
            sys.stdout = redir

    def _init_gui(self) -> wx.Panel:
        panel = wx.Panel(self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        logStyle = wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL
        hBox = wx.BoxSizer(wx.HORIZONTAL)
        self.progress = wx.Gauge(panel)
        emptycell = (0, 0)
        self.textElapsed = wx.StaticText(panel, style=wx.ALIGN_CENTER)
        self.textElapsed.SetLabel(f'Runtime: {0} sec')
        self.currentFile = wx.StaticText(panel, style=wx.ALIGN_CENTER)
        self.currentFile.SetLabel(f'File {self.file_counter}/{self.num_files}')
        self.log = wx.TextCtrl(panel,
                               wx.ID_ANY,
                               size=(400, 400),
                               style=logStyle)
        self.timer = wx.Timer(self, -1)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.btn = wx.Button(panel, label="Close")
        self.btn.Bind(wx.EVT_BUTTON, self.close)
        sizer.Add(self.progress, 0, wx.EXPAND, 10)
        hBox.Add(self.textElapsed, 1, wx.EXPAND, 10)
        hBox.Add(emptycell, 1, wx.EXPAND, 10)
        hBox.Add(self.currentFile, 1, wx.EXPAND, 10)
        sizer.Add(hBox)
        sizer.Add(self.log, 1, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.btn, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 5)
        panel.SetSizer(sizer)
        sizer.SetSizeHints(self)
        self.btn.Disable()
        # create a pubsub receiver
        pub.subscribe(self.updateProgress, "update")
        self.timer.Start(200)
        return panel

    def onTimer(self, evt):
        t = time.time()
        self.elapsed = t - self.startTime
        if (t - self.time) > 1:
            self.textElapsed.SetLabel(str(round(self.elapsed)))
            self.time = t

    def updateProgress(self, msg):
        """"""
        if msg == 'end':
            self.progress.SetValue(100)
            self.timer.Stop()
            self.btn.Enable()
        elif msg == 'file':
            self.file_counter += 1
            self.currentFile.SetLabel(
                f'File {self.file_counter}/{self.num_files}')

    def close(self, event):
        self.btn = event.GetEventObject()
        self.btn.Disable()
        self.Destroy()


class RedirectText(object):

    def __init__(self, aWxTextCtrl):
        self.out = aWxTextCtrl

    def write(self, string):
        self.out.WriteText(str(string))


###################################################################
