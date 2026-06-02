import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QShortcut
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
import pyqtgraph as pg

class PyQt4DSTEMViewer(QMainWindow):
    def __init__(self, data_stack):
        super().__init__()
        self.fdata = data_stack
        self.sy, self.sx, self.dsy, self.dsx = data_stack.shape
        self.ind = [self.sy // 2, self.sx // 2]  # Current probe position (y, x)
        
        # Calculate initial real-space integration image
        self.int_img = np.sum(self.fdata, axis=(2, 3))
        
        # Set up UI
        self.init_ui()
        self.update_diffraction_pattern()
        
    def init_ui(self):
        self.setWindowTitle("PyQt + PyQtGraph 4D-STEM Viewer")
        self.resize(1200, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 1. Real Space Panel (Left)
        real_layout = QVBoxLayout()
        self.real_label = QLabel(f"Real Space Map - Cursor: (x={self.ind[1]}, y={self.ind[0]})")
        self.real_plot = pg.PlotWidget(title="Real Space Integrated Intensity")
        self.real_img_item = pg.ImageItem(self.int_img)
        self.real_plot.addItem(self.real_img_item)
        self.real_plot.setAspectLocked(True)
        
        # Add interactive crosshair for probe position
        self.probe_cursor = pg.TargetItem(pos=(self.ind[1], self.ind[0]), size=15, pen='r', movable=True)
        self.real_plot.addItem(self.probe_cursor)
        self.probe_cursor.sigDragged.connect(self.on_cursor_dragged)
        
        real_layout.addWidget(self.real_label)
        real_layout.addWidget(self.real_plot)
        layout.addLayout(real_layout, stretch=1)
        
        # 2. Reciprocal Space Panel (Right)
        recip_layout = QVBoxLayout()
        self.recip_label = QLabel("Diffraction Pattern (DP) - Press 'L' for log scale")
        self.recip_plot = pg.PlotWidget(title="Diffraction Pattern at Probe")
        self.recip_img_item = pg.ImageItem()
        self.recip_plot.addItem(self.recip_img_item)
        self.recip_plot.setAspectLocked(True)
        
        # Add virtual detector ROI (Rectangle ROI)
        self.by, self.bx = self.dsy // 2, self.dsx // 2
        self.w, self.h = self.dsy // 10, self.dsx // 10
        self.detector_roi = pg.RectROI([self.bx, self.by], [self.w, self.h], pen='r', movable=True)
        self.recip_plot.addItem(self.detector_roi)
        self.detector_roi.sigRegionChanged.connect(self.on_roi_changed)
        
        recip_layout.addWidget(self.recip_label)
        recip_layout.addWidget(self.recip_plot)
        layout.addLayout(recip_layout, stretch=1)
        
        # State variables
        self.log_scale = False
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.move_probe(0, -1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.move_probe(0, 1))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self.move_probe(-1, 0))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self.move_probe(1, 0))
        QShortcut(QKeySequence(Qt.Key_L), self, self.toggle_log_scale)
        
    def move_probe(self, dy, dx):
        self.ind[0] = np.clip(self.ind[0] + dy, 0, self.sy - 1)
        self.ind[1] = np.clip(self.ind[1] + dx, 0, self.sx - 1)
        self.probe_cursor.setPos((self.ind[1], self.ind[0]))
        self.real_label.setText(f"Real Space Map - Cursor: (x={self.ind[1]}, y={self.ind[0]})")
        self.update_diffraction_pattern()
        
    def on_cursor_dragged(self):
        pos = self.probe_cursor.pos()
        x = int(np.clip(pos.x(), 0, self.sx - 1))
        y = int(np.clip(pos.y(), 0, self.sy - 1))
        self.ind = [y, x]
        self.real_label.setText(f"Real Space Map - Cursor: (x={x}, y={y})")
        self.update_diffraction_pattern()
        
    def on_roi_changed(self):
        # Calculate virtual STEM image based on current detector ROI
        pos = self.detector_roi.pos()
        size = self.detector_roi.size()
        
        bx = int(np.clip(pos.x(), 0, self.dsx - 1))
        by = int(np.clip(pos.y(), 0, self.dsy - 1))
        w = int(np.clip(size.x(), 1, self.dsx - bx))
        h = int(np.clip(size.y(), 1, self.dsy - by))
        
        # Calculate virtual image
        virtual_img = np.sum(self.fdata[:, :, by:by+h, bx:bx+w], axis=(2, 3))
        self.real_img_item.setImage(virtual_img)
        
    def toggle_log_scale(self):
        self.log_scale = not self.log_scale
        self.update_diffraction_pattern()
        
    def update_diffraction_pattern(self):
        dp = self.fdata[self.ind[0], self.ind[1]].astype(np.float32)
        if self.log_scale:
            dp = np.log(np.clip(dp, 1e-9, None))
        self.recip_img_item.setImage(dp)


class PyQt3DViewer(QMainWindow):
    def __init__(self, data_stack, x_scale=1.0, x_unit="channel"):
        super().__init__()
        self.fdata = data_stack
        self.sy, self.sx, self.sz = data_stack.shape
        self.x_scale = x_scale
        self.x_unit = x_unit
        self.ind = [self.sy // 2, self.sx // 2]
        
        # Calculate initial integration map
        self.int_img = np.sum(self.fdata, axis=2)
        self.x_range = np.arange(self.sz) * self.x_scale
        
        self.init_ui()
        self.update_spectrum()
        
    def init_ui(self):
        self.setWindowTitle("PyQt + PyQtGraph 3D Spectrum Image Viewer")
        self.resize(1200, 500)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 1. Real Space Map (Left)
        real_layout = QVBoxLayout()
        self.real_label = QLabel(f"Spatial Map - Cursor: (x={self.ind[1]}, y={self.ind[0]})")
        self.real_plot = pg.PlotWidget(title="Integrated Intensity")
        self.real_img_item = pg.ImageItem(self.int_img)
        self.real_plot.addItem(self.real_img_item)
        self.real_plot.setAspectLocked(True)
        
        self.probe_cursor = pg.TargetItem(pos=(self.ind[1], self.ind[0]), size=15, pen='r', movable=True)
        self.real_plot.addItem(self.probe_cursor)
        self.probe_cursor.sigDragged.connect(self.on_cursor_dragged)
        
        real_layout.addWidget(self.real_label)
        real_layout.addWidget(self.real_plot)
        layout.addLayout(real_layout, stretch=1)
        
        # 2. Virtual Bandpass Image (Middle)
        band_layout = QVBoxLayout()
        self.band_label = QLabel("Virtual Bandpass Image")
        self.band_plot = pg.PlotWidget(title="Sum over Selected Range")
        self.band_img_item = pg.ImageItem(self.int_img)
        self.band_plot.addItem(self.band_img_item)
        self.band_plot.setAspectLocked(True)
        
        band_layout.addWidget(self.band_label)
        band_layout.addWidget(self.band_plot)
        layout.addLayout(band_layout, stretch=1)
        
        # 3. 1D Spectrum Plot (Right)
        spec_layout = QVBoxLayout()
        self.spec_label = QLabel("Spectrum - Press 'L' for log scale, 'T' for total sum")
        self.spec_plot = pg.PlotWidget(title="Local Spectrum")
        self.spec_curve = self.spec_plot.plot(pen='w')
        self.spec_plot.setLabel('bottom', self.x_unit)
        self.spec_plot.showGrid(x=True, y=True)
        
        # Linear Region Selector (virtual bandpass selection)
        self.energy_roi = pg.LinearRegionItem([0, self.sz * self.x_scale], movable=True)
        self.spec_plot.addItem(self.energy_roi)
        self.energy_roi.sigRegionChanged.connect(self.on_roi_changed)
        
        spec_layout.addWidget(self.spec_label)
        spec_layout.addWidget(self.spec_plot)
        layout.addLayout(spec_layout, stretch=1)
        
        # State variables
        self.log_scale = False
        self.whole_sum = False
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.move_probe(0, -1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.move_probe(0, 1))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self.move_probe(-1, 0))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self.move_probe(1, 0))
        QShortcut(QKeySequence(Qt.Key_L), self, self.toggle_log_scale)
        QShortcut(QKeySequence(Qt.Key_T), self, self.toggle_whole_sum)
        
    def move_probe(self, dy, dx):
        self.ind[0] = np.clip(self.ind[0] + dy, 0, self.sy - 1)
        self.ind[1] = np.clip(self.ind[1] + dx, 0, self.sx - 1)
        self.probe_cursor.setPos((self.ind[1], self.ind[0]))
        self.real_label.setText(f"Spatial Map - Cursor: (x={self.ind[1]}, y={self.ind[0]})")
        self.update_spectrum()
        
    def on_cursor_dragged(self):
        pos = self.probe_cursor.pos()
        x = int(np.clip(pos.x(), 0, self.sx - 1))
        y = int(np.clip(pos.y(), 0, self.sy - 1))
        self.ind = [y, x]
        self.real_label.setText(f"Spatial Map - Cursor: (x={x}, y={y})")
        self.update_spectrum()
        
    def on_roi_changed(self):
        min_x, max_x = self.energy_roi.getRegion()
        idx_min = int(np.clip(min_x / self.x_scale, 0, self.sz - 1))
        idx_max = int(np.clip(max_x / self.x_scale, 1, self.sz))
        if idx_min >= idx_max:
            idx_max = idx_min + 1
            
        # Update middle panel image with sum over energy/channel slice
        bandpass_img = np.sum(self.fdata[:, :, idx_min:idx_max], axis=2)
        self.band_img_item.setImage(bandpass_img)
        self.band_label.setText(f"Virtual Bandpass Image [{idx_min * self.x_scale:.2f} to {idx_max * self.x_scale:.2f} {self.x_unit}]")
        
    def toggle_log_scale(self):
        self.log_scale = not self.log_scale
        self.update_spectrum()
        
    def toggle_whole_sum(self):
        self.whole_sum = not self.whole_sum
        self.update_spectrum()
        
    def update_spectrum(self):
        if self.whole_sum:
            spec = np.sum(self.fdata, axis=(0, 1))
            self.spec_label.setText("Spectrum - Total Sum Mode")
        else:
            spec = self.fdata[self.ind[0], self.ind[1]]
            self.spec_label.setText(f"Spectrum - Local Mode at ({self.ind[1]}, {self.ind[0]})")
            
        spec = spec.astype(np.float32)
        if self.log_scale:
            spec = np.log(np.clip(spec, 1e-9, None))
            
        self.spec_curve.setData(self.x_range, spec)


class PyQtSliceViewer(QMainWindow):
    def __init__(self, data_stack):
        super().__init__()
        self.X = data_stack
        self.slices, self.rows, self.cols = data_stack.shape
        self.ind = 0
        
        self.init_ui()
        self.update_slice()
        
    def init_ui(self):
        self.setWindowTitle("PyQt + PyQtGraph Slice Viewer")
        self.resize(600, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.slice_label = QLabel(f"Slice No. {self.ind + 1} / {self.slices}")
        self.plot = pg.PlotWidget(title="Slice Display")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.plot.setAspectLocked(True)
        
        layout.addWidget(self.slice_label)
        layout.addWidget(self.plot)
        
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.change_slice(-1))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self.change_slice(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.change_slice(1))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self.change_slice(1))
        
    def change_slice(self, delta):
        self.ind = (self.ind + delta) % self.slices
        self.slice_label.setText(f"Slice No. {self.ind + 1} / {self.slices}")
        self.update_slice()
        
    def update_slice(self):
        self.img_item.setImage(self.X[self.ind])


def launch_pyqt_viewer(data_stack):
    """
    Launch the PyQt + pyqtgraph 4D-STEM viewer app.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    viewer = PyQt4DSTEMViewer(data_stack)
    viewer.show()
    app.exec_()

def launch_pyqt_3d_viewer(data_stack, x_scale=1.0, x_unit="channel"):
    """
    Launch the PyQt + pyqtgraph 3D spectrum image viewer app.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    viewer = PyQt3DViewer(data_stack, x_scale=x_scale, x_unit=x_unit)
    viewer.show()
    app.exec_()

def launch_pyqt_slice_viewer(data_stack):
    """
    Launch the PyQt + pyqtgraph slice viewer app.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    viewer = PyQtSliceViewer(data_stack)
    viewer.show()
    app.exec_()
