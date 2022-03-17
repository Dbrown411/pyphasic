import wx, wx.lib.scrolledpanel, wx.lib.newevent, wx.lib.colourchooser, wx.lib.agw.floatspin


class CustomFloatSpin(wx.lib.agw.floatspin.FloatSpin):

    def __init__(self, *args, **kwargs):
        wx.lib.agw.floatspin.FloatSpin.__init__(self, *args, **kwargs)

    def update_increment(self, dir):
        if dir == 'down':
            self._increment = 0.9 * self._value
        if dir == 'up':
            self._increment = 9 * self._value

    def OnSpinUp(self, event):
        """
		Handles the ``wx.EVT_SPIN_UP`` event for :class:`FloatSpin`.

		:param `event`: a :class:`SpinEvent` event to be processed.
		"""
        if self.GetFormat() == '%E':
            self.update_increment('up')
        if self._textctrl and self._textctrl.IsModified():
            self.SyncSpinToText(False)

        if self.InRange(self._value + self._increment * self._spinmodifier):
            self._value = self._value + self._increment * self._spinmodifier
            self.SetValue(self._value)
            self.DoSendEvent()

    def OnSpinDown(self, event):
        """
		Handles the ``wx.EVT_SPIN_DOWN`` event for :class:`FloatSpin`.

		:param `event`: a :class:`SpinEvent` event to be processed.
		"""
        if self.GetFormat() == '%E':
            self.update_increment('down')
        if self._textctrl and self._textctrl.IsModified():
            self.SyncSpinToText(False)

        if self.InRange(self._value - self._increment * self._spinmodifier):
            self._value = self._value - self._increment * self._spinmodifier
            self.SetValue(self._value)
            self.DoSendEvent()


class SettingsDialog(wx.Dialog):

    def __init__(self, settings, labels):
        self.dictMode = {
            'c': 'Force',
            'sr': 'Displacement',
            'Force': 'c',
            'Displacement': 'sr'
        }
        self.screenSize = wx.DisplaySize()
        self.settings = settings
        self.labels = labels
        wx.Dialog.__init__(self,
                           None,
                           title="Settings",
                           size=(0.4 * self.screenSize[0],
                                 0.4 * self.screenSize[1]))
        # Create a panel and notebook (tabs holder)
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # Create the tab windows
        self.tab1 = SettingsExperimentTab(nb, settings['experiment'], labels)
        self.tab2 = SettingsAnalysisTab(nb, settings['analysis'], labels)
        self.tab3 = SettingsExportTab(nb, settings['export'], labels)

        # Add the windows to tabs and name them.
        nb.AddPage(self.tab1, "Experiment")
        nb.AddPage(self.tab2, "Analysis")
        nb.AddPage(self.tab3, "Exports")

        # Set noteboook in a sizer to create the layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nb, 1, wx.EXPAND)
        hBox = wx.BoxSizer(wx.HORIZONTAL)

        btn1 = wx.Button(p, label='Okay')
        btn2 = wx.Button(p, label='Cancel')
        self.Bind(wx.EVT_BUTTON, self.btnOkay, btn1)
        self.Bind(wx.EVT_BUTTON, self.btnCancel, btn2)
        hBox.Add(btn1, 1, wx.EXPAND)
        hBox.Add(btn2, 1, wx.EXPAND)

        sizer.Add(hBox)
        p.SetSizer(sizer)

    # ----------------------------------------------------------------------
    def btnOkay(self, msg):
        """"""
        experiment = {}
        analysis = {}
        export = {}
        for k, v in self.tab1.controls.items():
            if type(v) != tuple:
                newVal = v.GetValue()
                if type(newVal) == str:
                    newVal = self.dictMode[newVal]
            else:
                newVal = (v[0].GetValue(), v[1].GetValue())

            experiment[k] = newVal
        for k, v in self.tab2.controls.items():
            if type(v) != tuple:
                newVal = v.GetValue()
                if type(newVal) == str:
                    newVal = self.dictMode[newVal]
            else:
                newVal = (v[0].GetValue(), v[1].GetValue())

            analysis[k] = newVal
        for k, v in self.tab3.controls.items():
            if type(v) != tuple:
                newVal = v.GetValue()
                if type(newVal) == str:
                    newVal = self.dictMode[newVal]
            else:
                newVal = (v[0].GetValue(), v[1].GetValue())

            export[k] = newVal
        updatedSettings = {
            'experiment': experiment,
            'analysis': analysis,
            'export': export
        }
        self.settings = updatedSettings
        self.Close()

    def btnCancel(self, msg):
        """"""
        self.Close()


class SettingsExperimentTab(wx.Panel):

    def __init__(self, parent, settings, labels):
        wx.Panel.__init__(self, parent)

        self.dictMode = {
            'c': 'Force',
            'sr': 'Displacement',
            'Force': 'c',
            'Displacement': 'sr'
        }
        self.screenSize = wx.DisplaySize()
        self.settings = settings
        self.labels = labels
        cols = 5
        gs = wx.GridBagSizer(5, 5)
        emptyCell = (0, 0)
        self.controls = {}
        widgetCount = 0
        for k, v in settings.items():
            if widgetCount == 0:
                gs.Add(emptyCell,
                       pos=(widgetCount, 0),
                       span=(widgetCount, cols))
                widgetCount += 1
            if k == 'testMode':
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.ComboBox(
                    self,
                    wx.ID_ANY,
                    value=self.dictMode[v],
                    choices=['Force', 'Displacement'],
                    style=wx.ALIGN_CENTER)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == bool:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.CheckBox(self,
                                               wx.ID_ANY,
                                               label='',
                                               style=wx.ALIGN_CENTER)
                self.controls[k].SetValue(v)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == int):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.SpinCtrl(self,
                                               wx.ID_ANY,
                                               initial=v,
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min=0)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == float):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = CustomFloatSpin(self,
                                                   wx.ID_ANY,
                                                   value=v,
                                                   size=wx.DefaultSize,
                                                   style=wx.ALIGN_CENTER,
                                                   min_val=0)
                self.controls[k].SetDigits(1)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == tuple:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                if k == 'butterParams':
                    tupCtrl = (wx.SpinCtrl(self,
                                           wx.ID_ANY,
                                           initial=v[0],
                                           size=wx.DefaultSize,
                                           style=wx.ALIGN_CENTER,
                                           min=1),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[1].SetDigits(4)
                else:
                    tupCtrl = (CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[0],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[0].SetFormat('%E')
                    tupCtrl[1].SetFormat('%E')
                    tupCtrl[0].SetDigits(2)
                    tupCtrl[1].SetDigits(2)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(tupCtrl[0], pos=(widgetCount, 1))
                gs.Add(tupCtrl[1], pos=(widgetCount, 2))
                self.controls[k] = tupCtrl
            widgetCount += 1

        self.gs = gs
        self.SetSizer(gs)
        gs.SetSizeHints(self)


class SettingsAnalysisTab(wx.Panel):

    def __init__(self, parent, settings, labels):
        wx.Panel.__init__(self, parent)

        self.dictMode = {
            'c': 'Force',
            'sr': 'Displacement',
            'Force': 'c',
            'Displacement': 'sr'
        }
        self.screenSize = wx.DisplaySize()
        self.settings = settings
        self.labels = labels
        cols = 5
        gs = wx.GridBagSizer(5, 5)
        emptyCell = (0, 0)
        self.controls = {}
        widgetCount = 0
        for k, v in settings.items():
            if widgetCount == 0:
                gs.Add(emptyCell,
                       pos=(widgetCount, 0),
                       span=(widgetCount, cols))
                widgetCount += 1
            if k == 'testMode':
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.ComboBox(
                    self,
                    wx.ID_ANY,
                    value=self.dictMode[v],
                    choices=['Force', 'Displacement'],
                    style=wx.ALIGN_CENTER)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == bool:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.CheckBox(self,
                                               wx.ID_ANY,
                                               label='',
                                               style=wx.ALIGN_CENTER)
                self.controls[k].SetValue(v)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == int):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.SpinCtrl(self,
                                               wx.ID_ANY,
                                               initial=v,
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min=0)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == float):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = CustomFloatSpin(self,
                                                   wx.ID_ANY,
                                                   value=v,
                                                   size=wx.DefaultSize,
                                                   style=wx.ALIGN_CENTER,
                                                   min_val=0)
                self.controls[k].SetDigits(1)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == tuple:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                if k == 'butterParams':
                    tupCtrl = (wx.SpinCtrl(self,
                                           wx.ID_ANY,
                                           initial=v[0],
                                           size=wx.DefaultSize,
                                           style=wx.ALIGN_CENTER,
                                           min=1),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[1].SetDigits(4)
                else:
                    tupCtrl = (CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[0],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[0].SetFormat('%E')
                    tupCtrl[1].SetFormat('%E')
                    tupCtrl[0].SetDigits(2)
                    tupCtrl[1].SetDigits(2)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(tupCtrl[0], pos=(widgetCount, 1))
                gs.Add(tupCtrl[1], pos=(widgetCount, 2))
                self.controls[k] = tupCtrl
            widgetCount += 1

        self.gs = gs
        self.SetSizer(gs)
        gs.SetSizeHints(self)


class SettingsExportTab(wx.Panel):

    def __init__(self, parent, settings, labels):
        wx.Panel.__init__(self, parent)

        self.dictMode = {
            'c': 'Force',
            'sr': 'Displacement',
            'Force': 'c',
            'Displacement': 'sr'
        }
        self.screenSize = wx.DisplaySize()
        self.settings = settings
        self.labels = labels
        cols = 5
        gs = wx.GridBagSizer(5, 5)
        emptyCell = (0, 0)
        self.controls = {}
        widgetCount = 0
        for k, v in settings.items():
            if widgetCount == 0:
                gs.Add(emptyCell,
                       pos=(widgetCount, 0),
                       span=(widgetCount, cols))
                widgetCount += 1
            if k == 'testMode':
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.ComboBox(
                    self,
                    wx.ID_ANY,
                    value=self.dictMode[v],
                    choices=['Force', 'Displacement'],
                    style=wx.ALIGN_CENTER)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == bool:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.CheckBox(self,
                                               wx.ID_ANY,
                                               label='',
                                               style=wx.ALIGN_CENTER)
                self.controls[k].SetValue(v)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == int):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = wx.SpinCtrl(self,
                                               wx.ID_ANY,
                                               initial=v,
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min=0)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif (type(v) == float):
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                self.controls[k] = CustomFloatSpin(self,
                                                   wx.ID_ANY,
                                                   value=v,
                                                   size=wx.DefaultSize,
                                                   style=wx.ALIGN_CENTER,
                                                   min_val=0)
                self.controls[k].SetDigits(1)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(self.controls[k], pos=(widgetCount, 1))
            elif type(v) == tuple:
                lbl1 = wx.StaticText(self,
                                     -1,
                                     style=wx.ALIGN_CENTER,
                                     label=f'{self.labels[k]}')
                if k == 'butterParams':
                    tupCtrl = (wx.SpinCtrl(self,
                                           wx.ID_ANY,
                                           initial=v[0],
                                           size=wx.DefaultSize,
                                           style=wx.ALIGN_CENTER,
                                           min=1),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[1].SetDigits(4)
                else:
                    tupCtrl = (CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[0],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0),
                               CustomFloatSpin(self,
                                               wx.ID_ANY,
                                               value=v[1],
                                               size=wx.DefaultSize,
                                               style=wx.ALIGN_CENTER,
                                               min_val=0))
                    tupCtrl[0].SetFormat('%E')
                    tupCtrl[1].SetFormat('%E')
                    tupCtrl[0].SetDigits(2)
                    tupCtrl[1].SetDigits(2)
                gs.Add(lbl1, pos=(widgetCount, 0), flag=wx.ALIGN_RIGHT)
                gs.Add(tupCtrl[0], pos=(widgetCount, 1))
                gs.Add(tupCtrl[1], pos=(widgetCount, 2))
                self.controls[k] = tupCtrl
            widgetCount += 1

        self.gs = gs
        self.SetSizer(gs)
        gs.SetSizeHints(self)
