#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from PyQt5.QtCore import pyqtSlot, QEventLoop
from PyQt5.QtGui import QFont, QColor, QIcon

from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout

from SigMarkerViewPage import SigContentView
from PageTabWidget import PageTabWidget
from FileListView import FileListView

__Author__ = "By: Yu Qiang"
__Copyright__ = "Copyright (c) 2021 NJU BME 313."
__Version__ = "Version 1.0"
icon_loading = "./resources/images/icn_loading.svg"


class Window(QWidget):
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.resize(1200, 800)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(246, 248, 250))
        self.setPalette(palette)
        self.base_url = './data/training/'
        self.label_path = "./data/custom_labels/"
        self.fileListView = FileListView(self, path=self.base_url + "RECORDS", label_path=self.label_path)
        self.fileListView.load(True)
        self.fileListView.item_click_sig.connect(self.change_sig_file)
        self.file = self.fileListView.action_next(0)
        self.header = PageTabWidget(self, path=self.base_url + self.file)
        self.main_page = SigContentView(self, path=self.base_url + self.file)
        self.main_page.load(True)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.fileListView)
        hlayout.addWidget(self.main_page, 1)
        hlayout.setContentsMargins(0, 0, 0, 0)
        view_widget = QWidget(self)
        view_widget.setLayout(hlayout)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.header)
        vlayout.addWidget(view_widget, 1)
        self.setLayout(vlayout)
        self.header.cb_flip.stateChanged.connect(self.change_sig_flip)
        self.header.cb_norm.stateChanged.connect(self.change_sig_norm)
        self.header.cb_noise.stateChanged.connect(self.change_sig_noise)
        self.header.btn_record.clicked.connect(self.slot_to_btn_record)
        self.header.btn_pre.clicked.connect(self.slot_to_btn_pre)
        self.header.btn_next.clicked.connect(self.slot_to_btn_next)
        self.header.btn_range.clicked.connect(self.slot_change_range)
        self.main_page.item_make_label.connect(self.refresh_file_list)
        return

    @pyqtSlot(str)
    def change_sig_file(self, file):
        self.file = file
        path = self.base_url + file
        self.main_page.load(reload=True, file=path)
        self.header.changeInfo(path=path)
        return

    def change_sig_flip(self):
        if self.header.cb_flip.isChecked():
            self.main_page.sig_flip(True)
        else:
            self.main_page.sig_flip(False)
        return

    def change_sig_norm(self):
        if self.header.cb_norm.isChecked():
            self.main_page.sig_norm(True)
        else:
            self.main_page.sig_norm(False)
        return

    def change_sig_noise(self):
        if self.header.cb_noise.isChecked():
            self.main_page.sig_noise(True)
        else:
            self.main_page.sig_noise(False)
        return

    @pyqtSlot(bool)
    def slot_to_btn_record(self, clicked):
        self.main_page.action_make_label(self.label_path + self.file + '.csv')
        return

    @pyqtSlot(bool)
    def slot_to_btn_next(self, clicked):
        file = self.fileListView.action_next(1)
        self.file = file
        path = self.base_url + file
        self.main_page.load(reload=True, file=path)
        self.header.changeInfo(path=path)
        return

    @pyqtSlot(bool)
    def slot_to_btn_pre(self, clicked):
        file = self.fileListView.action_next(-1)
        self.file = file
        path = self.base_url + file
        self.main_page.load(reload=True, file=path)
        self.header.changeInfo(path=path)
        return

    @pyqtSlot(bool)
    def slot_change_range(self, clicked):
        start_time = self.header.te_start.time()
        end_time = self.header.te_end.time()
        self.main_page.change_range(start_time, end_time)
        return

    @pyqtSlot(bool)
    def refresh_file_list(self, complete):
        self.fileListView.item_change_color(self.file, complete)
        return


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    app = QApplication(sys.argv)
    customFont = QFont("Times", 14, QFont.Bold)
    w = Window()
    w.setWindowTitle("Make Label")
    w.show()
    app.processEvents(QEventLoop.ExcludeUserInputEvents)
    app.setFont(customFont)
    sys.exit(app.exec_())
