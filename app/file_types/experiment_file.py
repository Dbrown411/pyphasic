import re as r
from pathlib import Path

valid_delimiters = ['_', '-', ' ']


class ExperimentFile:

    def __init__(self, file: Path):
        self._file = file
        self.file_name = file.stem
        self.folder = file.parent
        expParams = r.split('|'.join(valid_delimiters), self.file_name)

        funs = [
            self.setDate, self.setExp, self.setPreload, self.setID,
            self.setEye, self.setRegion, self.setDetails
        ]
        if len(funs) <= len(expParams):
            for i in range(len(funs)):
                funs[i](expParams[i])
        else:
            [x() for x in funs]

        self.setGroup()

    def exportData(self) -> dict:
        return {
            'date': self.getDate(),
            'exp': self.getExp(),
            'preload': self.getPreload(),
            'id': self.getID(),
            'eye': self.getEye(),
            'region': self.getRegion(),
            'details': self.getDetails(),
            'group': self.getGroup(),
            'file_name': self.getFileName(),
            'file': self.getFile()
        }

    def setDate(self, newValue=''):
        self._date = newValue

    def getDate(self) -> str:
        return self._date

    def setExp(self, newValue=''):
        self._exp = newValue

    def getExp(self) -> str:
        return self._exp

    def setPreload(self, newValue=''):
        self._preload = newValue

    def getPreload(self) -> str:
        return self._preload

    def setID(self, newValue=''):
        self._id = newValue

    def getID(self) -> str:
        return self._id

    def setEye(self, newValue=''):
        self._eye = newValue

    def getEye(self) -> str:
        return self._eye

    def setRegion(self, newValue=''):
        self._region = newValue

    def getRegion(self) -> str:
        return self._region

    def setDetails(self, newValue=''):
        self._details = newValue

    def getDetails(self) -> str:
        return self._details

    def getFile(self) -> Path:
        return self._file

    def getFileName(self) -> str:
        return self.file_name

    def setGroup(self, newValue=''):
        self._group = newValue

    def getGroup(self) -> str:
        return self._group
