from PyQt5.QtCore import QEvent, pyqtSignal, pyqtSlot, QObject, QTime

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QAbstractSlider, QListWidget, QListWidgetItem, \
    QRadioButton, QGroupBox
from datasets import data_read_utils as dutils
import pyqtgraph as pg
import csv


##### Override class #####
class NonScientific(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super(NonScientific, self).__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [int(value * 1) for value in values]  # This line return the NonScientific notation value


class SigItemWidget(QWidget):
    check_item = pyqtSignal(str)

    def __init__(self, parent, sig_name, sig_data, sig_units, interest_area):
        super(SigItemWidget, self).__init__(parent)
        self.sig_name = sig_name
        if self.sig_name == 'PLETH':
            self.sig_name = 'PPG'
        self.sig_data = sig_data
        self.interest_area = interest_area

        self.item = QListWidgetItem()
        self._layout = QHBoxLayout(self)
        # self.sig_label = QLabel(self.sig_name, self)
        # self.sig_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # self.sig_label.setFixedWidth(40)

        self.sig_plot = pg.PlotWidget(self, title="The signal curve of channel {}".format(self.sig_name))
        self.sig_plot.showGrid(x=True, y=True)  # 显示图形网格
        self.sig_plot.setBackground([255, 255, 255, 255])
        self.sig_plot.setLabel('left', 'Amplitude', units=sig_units)
        self.sig_plot.setLabel('bottom', 'Time(min:sec)', units=None)

        x_axis = self.sig_plot.getAxis("bottom")
        start_time = (interest_area[0] + 1) // 250
        ticks = [i for i in range(0, interest_area[1] - interest_area[0], 250)]
        x_axis.setTicks([[(v, self.get_tick_time(start_time + v // 250)) for v in ticks]])

        self.sig_plot.setFixedHeight(260)
        # pen = pg.mkPen('b', width=5)[0, 0, 0, 255]
        self.sig_plot.plot().setData(self.sig_data, pen=pg.mkPen('b', width=5))

        self.info_widget = QWidget(self)

        # 信号Marker
        self.rb_group = QGroupBox(title="Is this signal abnormal?")
        self.rb_uncertain = QRadioButton(text="Uncertain")  # -1
        self.rb_true_alarm = QRadioButton(text="Yes")  # 1
        self.rb_false_alarm = QRadioButton(text="No")  # 0
        self.rb_invalid = QRadioButton(text="Invalid")  # 0
        self.rb_uncertain.setChecked(True)
        check_layout = QVBoxLayout()
        check_layout.addWidget(self.rb_uncertain)
        check_layout.addWidget(self.rb_true_alarm)
        check_layout.addWidget(self.rb_false_alarm)
        check_layout.addWidget(self.rb_invalid)
        check_layout.addStretch(1)
        check_layout.setContentsMargins(20, 20, 10, 10)

        self.rb_group.setLayout(check_layout)
        self.rb_group.setFixedWidth(160)

        # self._layout.addWidget(self.sig_label)
        self._layout.addWidget(self.sig_plot, 1)
        self._layout.addWidget(self.rb_group)

        self._layout.setContentsMargins(5, 5, 15, 5)
        self.sig_plot.installEventFilter(self)
        self.installEventFilter(self)
        self.item.setSizeHint(self.sizeHint())

    def get_status(self):
        start = self.get_tick_time((self.interest_area[0] + 1) // 250)
        end = self.get_tick_time(self.interest_area[1] // 250)
        if self.rb_uncertain.isChecked():
            return self.sig_name, 'uncertain', start, end
        if self.rb_true_alarm.isChecked():
            return self.sig_name, 'true', start, end
        if self.rb_false_alarm.isChecked():
            return self.sig_name, 'false', start, end
        if self.rb_invalid.isChecked():
            return self.sig_name, 'invalid', start, end

    def get_tick_time(self, seconds):
        cur_min = seconds // 60
        cur_sec = seconds % 60
        return "{}:{:02d}".format(cur_min, cur_sec)

    # def sizeHint(self):
    #     # 每个item控件的大小
    #     return QSize(650, 300)

    def eventFilter(self, obj: 'QObject', event: 'QEvent') -> bool:
        if obj == self.sig_plot and event.type() == QEvent.MouseButtonDblClick:
            self.sig_plot.autoRange()
            return True
        return False


class SigContentView(QWidget):
    item_click_sig = pyqtSignal(str)
    item_make_label = pyqtSignal(bool)

    def __init__(self, parent, path):
        super(SigContentView, self).__init__(parent)
        self.setUpdatesEnabled(True)
        self._layout = QHBoxLayout(self)
        self.path = path
        self.record = None
        self.norm = False
        self.noise = False
        self.flip = False
        self.interest_area = (72499, 75000)

        self.item_list = QListWidget(self)
        self._layout.addWidget(self.item_list, 1)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.cur_pos = 0

        # 连接竖着的滚动条滚动事件
        self.item_list.installEventFilter(self)
        self.item_list.verticalScrollBar().actionTriggered.connect(self.onActionTriggered)
        self.items = []
        self.item_list.setStyleSheet("QListView::item:selected {border-right: 5px solid #f44336;}"
                                     )
        # "QListView::item:hover {border-right: 5px solid #3B3E40;}")
        # self.setAutoFillBackground(True)
        # palette = self.palette()
        # palette.setColor(self.backgroundRole(), Qt.blue)
        # self.setPalette(palette)

    def load(self, reload=True, file=None):
        if file is not None:
            self.path = file
        self._load(reload=reload)
        return

    def sig_norm(self, norm=False):
        self.norm = norm
        self._load(reload=True)
        return

    def sig_flip(self, flip=False):
        self.flip = flip
        self._load(reload=True)
        return

    def sig_noise(self, noise=False):
        self.noise = noise
        self._load(reload=True)
        return

    def _load(self, reload=True):
        if reload:
            self.clear()
            if self.path:
                self.record = dutils.load_record(self.path)
            else:
                print("Error, can't load images")

        n = self.record.n_sig
        if self.cur_pos >= n:
            return

        self._makeItem()
        return

    def _makeItem(self):
        sig = dutils.read_interest_area(self.record.p_signal, *self.interest_area)
        if self.flip:
            sig = dutils.sig_flip(sig)
        if self.norm:
            sig = dutils.minmax_scale(sig)
        if self.noise:
            sig = dutils.gaussion_noise(sig)

        N = self.record.n_sig
        for i in range(N):
            self.cur_pos += 1
            iwidget = SigItemWidget(self.item_list, self.record.sig_name[i], sig[:, i], self.record.units[i],
                                    self.interest_area)
            self.item_list.addItem(iwidget.item)
            self.item_list.setItemWidget(iwidget.item, iwidget)
            # iwidget.like_item.connect(self.slot_item_click)
            self.items.append(iwidget)
        return

    @pyqtSlot(str)
    def slot_item_click(self, img_path):
        self.item_click_sig.emit(img_path)
        return

    def onActionTriggered(self, action):
        # 这里要判断action=QAbstractSlider.SliderMove，可以避免窗口大小改变的问题
        if action != QAbstractSlider.SliderMove:
            return
        # 使用sliderPosition获取值可以同时满足鼠标滑动和拖动判断
        if self.item_list.verticalScrollBar().sliderPosition() == self.item_list.verticalScrollBar().maximum():
            # 可以下一页了
            self.load(reload=False)
        return

    def clear(self):
        self.item_list.clear()
        while self.item_list.count():
            item = self.item_list.takeItem(0)
            del item
        self.items.clear()
        self.record = None
        self.cur_pos = 0

    def action_make_label(self, save_path):

        with open(save_path, 'w+') as log_file:
            writer = csv.writer(log_file)
            have_complete = True
            cur_data = []
            all_uncertain = True
            for item in self.items:
                ch, label, start, end = item.get_status()
                cur_data.append([ch, start, end, label])
                print((ch, start, end, label))
                if label == 'uncertain':
                    have_complete = False
                else:
                    all_uncertain = False
            if all_uncertain:
                return
            if have_complete:
                self.item_make_label.emit(True)
                writer.writerow(["complete"])
            else:
                self.item_make_label.emit(False)
                writer.writerow(["not complete"])
            writer.writerows(cur_data)

        return

    def change_range(self, start: QTime, end: QTime):
        start_pos = (start.minute() * 60 + start.second()) * 250 - 1
        end_pos = (end.minute() * 60 + end.second()) * 250
        self.interest_area = (start_pos, end_pos)
        self._load(reload=True)
        return


class SigMarkerMainView(QWidget):
    def __init__(self, parent, path):
        super(SigMarkerMainView, self).__init__(parent)
        self.setUpdatesEnabled(True)
        self.path = path
        vlayout = QVBoxLayout()
        self.sig_list_view = SigContentView(self, path=path)
        vlayout.addWidget(self.sig_list_view, 1)
        vlayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vlayout)
        self.sig_list_view.load(True)
        return
