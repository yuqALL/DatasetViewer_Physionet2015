from PyQt5.QtCore import Qt, QEvent, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, \
    QHBoxLayout, QAbstractSlider, QListWidget, QListWidgetItem
from datasets import data_read_utils as dutils


class FileItemWidget(QWidget):
    choose_item = pyqtSignal(str)

    def __init__(self, parent, file):
        super(FileItemWidget, self).__init__(parent)
        self.file = file
        self.flabel = QLabel(self, text=file)
        self.flabel.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.flabel.setContentsMargins(0, 0, 10, 0)
        self.item = QListWidgetItem()
        self._layout = QHBoxLayout(self)
        self.installEventFilter(self)
        self.item.setSizeHint(self.sizeHint())

    # def sizeHint(self):
    #     # 每个item控件的大小
    #     return QSize(650, 300)

    def eventFilter(self, obj: 'QObject', event: 'QEvent') -> bool:
        if obj == self and event.type() == QEvent.MouseButtonRelease:
            self.choose_item.emit(self.file)
            return True
        return False

    def change_color(self, complete=False):
        if not complete:
            self.flabel.setStyleSheet("color:rgb(246, 146, 61)")
        else:
            self.flabel.setStyleSheet("color:rgb(63, 182, 66)")
        return


class FileListView(QWidget):
    item_click_sig = pyqtSignal(str)

    def __init__(self, parent, path, label_path):
        super(FileListView, self).__init__(parent)
        self.path = path
        self.files = dutils.load_record_id(self.path)
        self.complete_labels, self.uncomplete_labels = dutils.load_label_path(label_path)
        self.item_list = QListWidget(self)
        self.cur_pos = 0
        self.index = 0
        self.files = []
        # 连接竖着的滚动条滚动事件
        self.item_list.installEventFilter(self)
        self.item_list.verticalScrollBar().actionTriggered.connect(self.onActionTriggered)
        self.item_list.setContentsMargins(0, 0, 0, 0)
        self.items = {}
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.item_list, 1)
        vlayout.setContentsMargins(0, 0, 0, 0)
        self.setFixedWidth(100)
        self.setLayout(vlayout)

        self.item_list.setStyleSheet(
            "QListView::item:selected {border: 1px solid #498BA7;background: #E8EAED;}"
            "QListView::item:hover {background: #F6F8FA;}")
        # self.item_list.setAutoFillBackground(True)
        # palette = self.item_list.palette()
        # palette.setColor(self.item_list.backgroundRole(), QColor(246,248,250))
        # self.item_list.setPalette(palette)

    def load(self, reload=True):
        self._load(reload=reload)

    def _load(self, reload=True):
        if reload:
            self.clear()
            if self.path:
                self.files = dutils.load_record_id(self.path)
            else:
                print("Error, can't load images")

        n = len(self.files)
        if self.cur_pos >= n:
            return

        self._makeItem()
        return

    def _makeItem(self):
        for f in self.files:
            self.cur_pos += 1
            iwidget = FileItemWidget(self.item_list, file=f)
            iwidget.choose_item.connect(self.slot_item_click)
            if f in self.complete_labels:
                iwidget.change_color(complete=True)
            elif f in self.uncomplete_labels:
                iwidget.change_color(complete=False)
            self.item_list.addItem(iwidget.item)
            self.item_list.setItemWidget(iwidget.item, iwidget)
            self.items[f] = iwidget
        return

    def item_change_color(self, file, compute):
        item = self.items[file]
        if compute:
            item.change_color(complete=True)
        else:
            item.change_color(complete=False)
        return

    @pyqtSlot(str)
    def slot_item_click(self, file):
        self.item_click_sig.emit(file)
        # print(file)
        return

    def onActionTriggered(self, action):
        # 这里要判断action=QAbstractSlider.SliderMove，可以避免窗口大小改变的问题
        if action != QAbstractSlider.SliderMove:
            return
        # 使用sliderPosition获取值可以同时满足鼠标滑动和拖动判断
        if self.item_list.verticalScrollBar().sliderPosition() == self.item_list.verticalScrollBar().maximum():
            # 可以下一页了
            self.load(reload=False)

    def clear(self):
        self.item_list.clear()
        while self.item_list.count():
            item = self.item_list.takeItem(0)
            del item
        self.items.clear()
        self.record = None

    def action_next(self, step):
        n = len(self.files)
        self.index = (self.index + step) % n
        self.item_list.setCurrentRow(self.index)

        return self.files[self.index]
