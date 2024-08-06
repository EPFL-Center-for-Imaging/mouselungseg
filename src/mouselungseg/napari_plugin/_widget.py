import os
import sys

import napari.layers
from napari.qt.threading import thread_worker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
    QSizePolicy,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from mouselungseg.predict import LungsPredictor

class LungsSegmentationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.predictor = LungsPredictor()

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)

        # Compute button
        self.btn = QPushButton("Segment lungs", self)
        self.btn.clicked.connect(self._start_segmentation)
        grid_layout.addWidget(self.btn, 3, 0, 1, 2)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 4, 0, 1, 2)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                if x.data.ndim in [2, 3]:
                    self.cb_image.addItem(x.name, x.data)

    @thread_worker
    def _tumor_prediction_thread(self):
        return self.predictor.predict(self.selected_image)

    def _start_segmentation(self):
        self.selected_image = self.cb_image.currentData()
        if self.selected_image is None:
            return

        self.pbar.setMaximum(0)

        worker = self._tumor_prediction_thread()
        worker.returned.connect(self._load_in_viewer)
        worker.start()

    def _load_in_viewer(self, segmentation):
        """Callback from thread returning."""
        if segmentation is not None:
            segmentation_layer = self.viewer.add_labels(segmentation, name="Lungs (Yolo)")
            segmentation_layer.opacity = 0.2
            segmentation_layer.blending = "additive"
        
        self.pbar.setMaximum(1)
