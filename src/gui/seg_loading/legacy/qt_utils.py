import io

from PyQt6.QtCore import QBuffer, QEvent, QObject, QPoint, pyqtSignal
from PyQt6.QtGui import QImage
from PIL import Image

def q_im_to_pil(q_im: QImage) -> Image:
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    q_im.save(buffer, "PNG")
    return Image.open(io.BytesIO(buffer.data()))

class MouseTracker(QObject):
    position_changed = pyqtSignal(QPoint)
    position_clicked = pyqtSignal(QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if obj is self.widget and event.type() == QEvent.Type.MouseMove:
            self.position_changed.emit(event.pos())
        elif obj is self.widget and event.type() == QEvent.Type.MouseButtonPress:
            self.position_clicked.emit(event.pos())
        return super().eventFilter(obj, event)
