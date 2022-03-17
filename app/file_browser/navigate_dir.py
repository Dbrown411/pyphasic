import os
import wx, wx.lib.scrolledpanel, wx.lib.newevent, wx.lib.colourchooser, wx.lib.agw.floatspin
try:
    import ObjectListView2 as olv
except:
    import ObjectListView as olv
from pathlib import Path
from app.file_types import ExperimentFile
from .settings_dialogs import SettingsDialog
from app.config import ConfigContainer as cc


class selectDirectory(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self, settings, labels):
        self.screenSize = wx.DisplaySize()
        self.frameWidth = 0.6 * self.screenSize[0]
        self.frameHeight = 0.5 * self.screenSize[1]
        wx.Frame.__init__(self,
                          None,
                          title='Select Data Directory',
                          style=wx.DEFAULT_FRAME_STYLE,
                          size=(self.frameWidth, self.frameHeight))
        self.objects = []
        self.settings = settings
        self.settingsLabels = labels
        self.selectedFiles = []
        self.main_panel = wx.Panel(self, wx.ID_ANY)
        self.init_UI()
        self.Center()

    # ----------------------------------------------------------------------
    def init_UI(self):
        panel = wx.Panel(
            self.main_panel,
            style=wx.SIMPLE_BORDER,
        )
        panel2 = wx.lib.scrolledpanel.ScrolledPanel(self.main_panel,
                                                    -1,
                                                    style=wx.SIMPLE_BORDER)
        panel2.SetupScrolling(scroll_x=False)
        panel2.SetBackgroundColour('#FFFFFF')

        sizer = wx.BoxSizer(wx.VERTICAL)
        bSizer = wx.BoxSizer(wx.VERTICAL)
        v_psizer = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.btn = wx.Button(panel, -1, label="Analyze")
        self.btn.Bind(wx.EVT_BUTTON, self.onGo)
        self.btn2 = wx.Button(panel, label="Settings")
        self.btn2.Bind(wx.EVT_BUTTON, self.onSettings)

        hbox1.Add(self.btn, 4, wx.LEFT | wx.RIGHT | wx.EXPAND, 5)
        hbox1.Add(self.btn2, 4, wx.LEFT | wx.RIGHT | wx.EXPAND, 5)
        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.updateList)

        self.dirCtrl = wx.DirPickerCtrl(panel, wx.ID_ANY,
                                        cc.default_dir.as_posix(),
                                        u"Select a folder", wx.DefaultPosition,
                                        wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)

        ##Initialize OLV
        self.dataOlv = olv.FastObjectListView(
            panel2,
            wx.ID_ANY,
            style=wx.LC_REPORT | wx.SUNKEN_BORDER | wx.LIST_ALIGN_LEFT)

        self.columnDate = olv.ColumnDefn(title='Date',
                                         align='left',
                                         width=self.frameWidth * 0.0625,
                                         valueGetter='getDate',
                                         valueSetter='setDate',
                                         minimumWidth=60,
                                         isEditable=True)

        self.columnExp = olv.ColumnDefn('Experiment',
                                        'left',
                                        self.frameWidth * 0.125,
                                        valueGetter='getExp',
                                        minimumWidth=60,
                                        valueSetter='setExp',
                                        isEditable=True)
        self.columnPreload = olv.ColumnDefn('Preload',
                                            'left',
                                            self.frameWidth * 0.0625,
                                            valueGetter='getPreload',
                                            valueSetter='setPreload',
                                            minimumWidth=60,
                                            isEditable=True)
        self.columnID = olv.ColumnDefn('ID',
                                       'left',
                                       self.frameWidth * 0.0625,
                                       valueGetter='getID',
                                       minimumWidth=60,
                                       valueSetter='setID',
                                       isEditable=True)
        self.columnEye = olv.ColumnDefn('Eye',
                                        'left',
                                        self.frameWidth * 0.0625,
                                        valueGetter='getEye',
                                        valueSetter='setEye',
                                        minimumWidth=60,
                                        isEditable=True)
        self.columnRegion = olv.ColumnDefn('Region',
                                           'left',
                                           self.frameWidth * 0.0625,
                                           valueGetter='getRegion',
                                           valueSetter='setRegion',
                                           minimumWidth=60,
                                           isEditable=True)
        self.columnDetails = olv.ColumnDefn('Details',
                                            'left',
                                            self.frameWidth * 0.09,
                                            valueGetter='getDetails',
                                            valueSetter='setDetails',
                                            minimumWidth=60,
                                            isEditable=True)
        self.columnGroup = olv.ColumnDefn('Grouping (optional)',
                                          'left',
                                          self.frameWidth * 0.09,
                                          valueGetter='getGroup',
                                          valueSetter='setGroup',
                                          minimumWidth=60,
                                          isEditable=True)
        self.columnOrig = olv.ColumnDefn(title='File Name',
                                         align='left',
                                         width=self.frameWidth * 0.125,
                                         valueGetter='getFileName',
                                         minimumWidth=60,
                                         isEditable=False)
        self.olvCols = [
            self.columnDate, self.columnExp, self.columnPreload, self.columnID,
            self.columnEye, self.columnRegion, self.columnDetails,
            self.columnGroup, self.columnOrig
        ]
        self.dataOlv.SetColumns(self.olvCols)
        self.dataOlv.CreateCheckStateColumn()
        self.columnOrig.isSpaceFilling = True
        self.dataOlv.cellEditMode = 'CELLEDIT_DOUBLECLICK'
        self.dataOlv.SortBy(len(self.olvCols), ascending=False)
        self.dataOlv.SetObjects(self.objects)
        self.dataOlv.AutoSizeColumns()
        self.dataOlv.HeaderUsesThemes = False
        self.dataOlv.OwnerDraw = True
        ##Add OLV to UI
        sizer.Add(hbox1, 0, wx.EXPAND, 5)
        sizer.Add(self.dirCtrl, 0, wx.EXPAND, 5)
        panel.SetSizer(sizer)
        v_psizer.Add(panel, 0.3, wx.EXPAND)

        bSizer.Add(self.dataOlv, 1, wx.EXPAND, 5)
        panel2.SetSizer(bSizer)

        v_psizer.Add(panel2, 1, wx.EXPAND)
        self.main_panel.SetSizer(v_psizer)
        self.updateList()

    def updateList(self, evt=None):
        self.folder = Path(self.dirCtrl.GetPath())
        os.chdir(self.folder)
        self.objects = [
            ExperimentFile(file) for file in self.folder.glob("*.csv")
        ]
        self.dataOlv.SetObjects(self.objects)

    def onGo(self, event):
        btn = event.GetEventObject()
        btn.Disable()
        self.selectedFiles = self.dataOlv.GetCheckedObjects()
        btn.Enable()
        self.Close()

    def onSettings(self, event):
        btn = event.GetEventObject()
        btn.Disable()
        btn.Refresh()

        dlg = SettingsDialog(self.settings, self.settingsLabels)
        dlg.ShowModal()
        self.settings = dlg.settings
        dlg.Destroy()
        btn.Enable()