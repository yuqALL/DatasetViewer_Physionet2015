#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from PyQt5.QtCore import Qt, QTime

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, \
    QHBoxLayout, QPushButton, QGridLayout, QCheckBox, QTimeEdit

from datasets import data_read_utils as dutils


class PageTabWidget(QWidget):
    def __init__(self, parent, path):
        super(PageTabWidget, self).__init__(parent)
        self.path = path
        self.setFixedHeight(80)

        self.all_alarm_name = {'Ventricular_Tachycardia': "VTA", 'Tachycardia': "ETC",
                               'Ventricular_Flutter_Fib': "VFB", 'Bradycardia': "EBR",
                               'Asystole': "ASY"}

        self.record = dutils.load_record(path)
        self.file_path = QLabel("File Path: " + os.path.abspath(path))
        self.record_name = QLabel("Record Name: %5s" % self.record.record_name)
        self.fs = QLabel("Fs: %4d Hz" % self.record.fs)
        self.alarm_type = QLabel("Alarm Type: %4s" % self.all_alarm_name[self.record.comments[0]])
        self.alabel = QLabel("Label: %11s" % self.record.comments[1])
        self.cb_norm = QCheckBox("Normalization", self)
        self.cb_noise = QCheckBox("Add Noise", self)
        self.cb_flip = QCheckBox("Flip signal", self)
        self.btn_record = QPushButton("Make Label", self)
        self.btn_pre = QPushButton("Previous", self)
        self.btn_next = QPushButton("Next", self)

        self.change_range = QLabel("Load Signal Range: ", self)
        start = QTime(0, 4, 50)
        end = QTime(0, 5, 0)
        self.te_start = QTimeEdit(start, self)
        self.te_end = QTimeEdit(end, self)
        self.te_start.setDisplayFormat("mm:ss")
        self.te_end.setDisplayFormat("mm:ss")
        self.te_start.setMinimumTime(QTime(0, 0, 0))
        self.te_start.setMaximumTime(QTime(0, 5, 0))
        self.te_end.setMinimumTime(QTime(0, 0, 0))
        self.te_end.setMaximumTime(QTime(0, 5, 0))

        self.btn_range = QPushButton("Change", self)

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.file_path, 1)
        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.cb_flip)
        hlayout2.addWidget(self.cb_norm)
        hlayout2.addWidget(self.cb_noise)
        hlayout2.setContentsMargins(0, 0, 0, 0)
        hlayout1.addLayout(hlayout2)
        hlayout1.setContentsMargins(0, 0, 0, 0)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.record_name)
        hlayout3.addWidget(self.fs)
        hlayout3.addWidget(self.alarm_type)
        hlayout3.addWidget(self.alabel)

        hlayout4 = QHBoxLayout()
        hlayout4.addWidget(self.change_range)
        hlayout4.addWidget(self.te_start)
        hlayout4.addWidget(self.te_end)
        hlayout4.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hlayout4.setContentsMargins(0, 0, 0, 0)
        hlayout3.addLayout(hlayout4, 1)
        hlayout3.setContentsMargins(0, 0, 0, 0)

        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout1)
        vlayout.addLayout(hlayout3)
        vlayout.setContentsMargins(0, 10, 0, 0)

        btn_grid = QGridLayout()
        btn_grid.addWidget(self.btn_pre, 0, 0, 1, 1)
        btn_grid.addWidget(self.btn_next, 0, 1, 1, 1)
        btn_grid.addWidget(self.btn_range, 1, 0, 1, 1)
        btn_grid.addWidget(self.btn_record, 1, 1, 1, 1)
        btn_grid.setContentsMargins(0, 0, 0, 14)

        hlayout = QHBoxLayout()
        hlayout.addLayout(vlayout, 1)
        hlayout.addLayout(btn_grid)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setAlignment(Qt.AlignVCenter)

        self.setLayout(hlayout)
        self.setContentsMargins(0, 0, 0, 0)

        normal_style = "QPushButton{border-style:none;" \
                       "padding:8px;" \
                       "border-radius:5px;" \
                       "color:#FFFFFF;" \
                       "background:#428BCB;}" \
                       "QPushButton:hover{" \
                       "color:#F0F0F0;" \
                       "background:#49A3CA;}" \
                       "QPushButton:pressed{" \
                       "color:#FFFFFF;" \
                       "background:#1385CA;}"
        record_style = "QPushButton{border-style:none;" \
                       "padding:8px;" \
                       "border-radius:5px;" \
                       "color:#FFFFFF;" \
                       "background:#C44223;}" \
                       "QPushButton:hover{" \
                       "color:#F0F0F0;" \
                       "background:#C44A27;}" \
                       "QPushButton:pressed{" \
                       "color:#FFFFFF;" \
                       "background:#C43A21;}"
        self.btn_pre.setStyleSheet(normal_style)
        self.btn_next.setStyleSheet(normal_style)
        self.btn_range.setStyleSheet(normal_style)
        self.btn_record.setStyleSheet(record_style)
        self.btn_pre.setFixedSize(90, 32)
        self.btn_next.setFixedSize(90, 32)
        self.btn_range.setFixedSize(90, 32)
        self.btn_record.setFixedSize(90, 32)

        self.record_name.setFixedWidth(140)
        self.fs.setFixedWidth(80)
        self.alarm_type.setFixedWidth(140)
        self.alabel.setFixedWidth(140)
        self.te_start.setFixedWidth(80)
        self.te_end.setFixedWidth(80)
        self.cb_flip.setFixedWidth(140)
        self.cb_norm.setFixedWidth(140)
        self.cb_noise.setFixedWidth(140)
        return

    def changeInfo(self, path):
        self.record = dutils.load_record(path)
        self.file_path.setText("File Path: " + os.path.abspath(path))
        self.record_name.setText("Record Name: %5s" % self.record.record_name)
        self.fs.setText("Fs: %4d Hz" % self.record.fs)
        self.alarm_type.setText("Alarm Type: %4s" % self.all_alarm_name[self.record.comments[0]])
        self.alabel.setText("Label: %11s" % self.record.comments[1])
        return
