#!/usr/bin/env python3
"""
VEP Desktop Application
=======================
Configuration GUI + Native PyVista Viewer (separate window).
This architecture avoids pyvistaqt embedding issues on macOS.
"""

import sys
import os

# Ensure VEP modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar,
    QListWidget, QListWidgetItem, QGroupBox, QFormLayout, QSpinBox,
    QStatusBar, QAbstractItemView, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import numpy as np


class SimulationWorker(QThread):
    """Background thread for running simulations."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, str)
    status = pyqtSignal(str)
    
    def __init__(self, atlas, cortex, duration, ez_indices):
        super().__init__()
        self.atlas = atlas
        self.cortex = cortex
        self.duration = duration
        self.ez_indices = ez_indices
        
    def run(self):
        try:
            from vep.anatomy import BrainAnatomy
            from vep.simulator import Simulator
            
            # Load anatomy
            self.status.emit("Loading brain anatomy...")
            self.progress.emit(10)
            
            anatomy = BrainAnatomy()
            anatomy.load_connectivity(n_regions=self.atlas)
            anatomy.load_cortex(resolution=self.cortex)
            self.progress.emit(30)
            
            # Run simulation
            self.status.emit(f"Simulating {self.duration}ms...")
            simulator = Simulator(anatomy)
            time, data, onset_times = simulator.run(self.ez_indices, duration=self.duration)
            x0_values = simulator.model.x0.copy()
            self.progress.emit(90)
            
            self.status.emit("Complete!")
            self.progress.emit(100)
            
            # Package results
            result = {
                'anatomy': anatomy,
                'time': time,
                'data': data,
                'onset_times': onset_times,
                'x0_values': x0_values
            }
            
            self.finished.emit(result, None, "")
            
        except Exception as e:
            import traceback
            self.status.emit(f"Error: {str(e)}")
            self.finished.emit(None, None, traceback.format_exc())


class VEPConfigWindow(QMainWindow):
    """Configuration window for VEP simulations."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VEP - Virtual Epileptic Patient")
        self.setMinimumSize(400, 700)
        self.setMaximumWidth(500)
        
        self.result = None
        self._setup_ui()
        
        # Load regions after window appears
        QTimer.singleShot(200, self._load_region_list)
        
    def _setup_ui(self):
        """Build the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("VEP Configuration")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Configure and run epilepsy simulations")
        subtitle.setStyleSheet("color: #888; font-size: 12px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Atlas Group
        atlas_group = QGroupBox("Brain Atlas")
        atlas_form = QFormLayout(atlas_group)
        
        self.atlas_combo = QComboBox()
        self.atlas_combo.addItems(["66", "68", "76 (Default)", "80", "96", "192", "998", "84 (Macaque)"])
        self.atlas_combo.setCurrentText("76 (Default)")
        self.atlas_combo.currentTextChanged.connect(self._load_region_list)
        atlas_form.addRow("Regions:", self.atlas_combo)
        
        self.cortex_combo = QComboBox()
        self.cortex_combo.addItems(["16k", "80k"])
        atlas_form.addRow("Cortex Mesh:", self.cortex_combo)
        
        layout.addWidget(atlas_group)
        
        # Simulation Group
        sim_group = QGroupBox("Simulation")
        sim_form = QFormLayout(sim_group)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(500, 10000)
        self.duration_spin.setValue(4000)
        self.duration_spin.setSuffix(" ms")
        self.duration_spin.setSingleStep(500)
        sim_form.addRow("Duration:", self.duration_spin)
        
        layout.addWidget(sim_group)
        
        # EZ Regions Group
        ez_group = QGroupBox("Epileptogenic Zones (EZ)")
        ez_layout = QVBoxLayout(ez_group)
        
        ez_hint = QLabel("Select regions to mark as seizure onset:")
        ez_hint.setStyleSheet("color: gray; font-size: 11px;")
        ez_layout.addWidget(ez_hint)
        
        self.ez_list = QListWidget()
        self.ez_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.ez_list.setMinimumHeight(200)
        ez_layout.addWidget(self.ez_list)
        
        layout.addWidget(ez_group)
        
        # Run Button
        self.run_btn = QPushButton("▶  Run Simulation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.run_btn.clicked.connect(self._run_simulation)
        layout.addWidget(self.run_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Configure settings and click Run.")
        
    def _load_region_list(self):
        """Load region labels based on selected atlas."""
        self.ez_list.clear()
        self.status_bar.showMessage("Loading regions...")
        
        atlas = self._get_atlas_value()
        
        try:
            from vep.anatomy import BrainAnatomy
            anatomy = BrainAnatomy()
            anatomy.load_connectivity(n_regions=atlas)
            
            for i, label in enumerate(anatomy.labels):
                item = QListWidgetItem(f"{i}: {label}")
                self.ez_list.addItem(item)
                
                # Pre-select temporal/limbic regions
                if any(kw in label for kw in ['AMYG', 'HC', 'PHC']):
                    item.setSelected(True)
                    
            self.status_bar.showMessage(f"Loaded {len(anatomy.labels)} regions. Select EZ and click Run.")
                    
        except Exception as e:
            self.status_bar.showMessage(f"Error loading regions: {e}")
            
    def _get_atlas_value(self):
        """Parse atlas dropdown value."""
        text = self.atlas_combo.currentText()
        return int(text.split()[0])
    
    def _get_cortex_value(self):
        """Get cortex resolution."""
        return self.cortex_combo.currentText()
    
    def _get_ez_indices(self):
        """Get selected EZ region indices."""
        indices = []
        for item in self.ez_list.selectedItems():
            idx = int(item.text().split(":")[0])
            indices.append(idx)
        return indices if indices else [0, 1, 2]
    
    def _run_simulation(self):
        """Start simulation in background thread."""
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        atlas = self._get_atlas_value()
        cortex = self._get_cortex_value()
        duration = self.duration_spin.value()
        ez_indices = self._get_ez_indices()
        
        self.worker = SimulationWorker(atlas, cortex, duration, ez_indices)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self._on_simulation_done)
        self.worker.start()
        
    def _on_simulation_done(self, result, _, error):
        """Handle simulation completion."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if error:
            QMessageBox.critical(self, "Error", f"Simulation failed:\n{error}")
            return
        
        self.result = result
        
        # Count recruited regions
        recruited = np.sum(result['onset_times'] > 0)
        total = result['anatomy'].n_regions
        
        self.status_bar.showMessage(f"Done! {recruited}/{total} regions recruited. Launching viewer...")
        
        # Launch native PyVista viewer in a separate process
        QTimer.singleShot(500, self._launch_viewer)
        
    def _launch_viewer(self):
        """Launch the native PyVista viewer in a separate process."""
        if self.result is None:
            return
        
        import subprocess
            
        # Save checkpoint with metadata
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'vep_checkpoint.npz')
        np.savez(
            checkpoint_path,
            time=self.result['time'],
            data=self.result['data'],
            onset_times=self.result['onset_times'],
            x0_values=self.result['x0_values'],
            atlas=self._get_atlas_value(),
            cortex=self._get_cortex_value()
        )
        
        self.status_bar.showMessage(f"Launching viewer... (checkpoint saved)")
        
        # Launch viewer in separate process (avoids Qt/VTK conflict)
        viewer_script = os.path.join(os.path.dirname(__file__), 'viewer.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(__file__)
        
        subprocess.Popen(
            [sys.executable, viewer_script, checkpoint_path],
            env=env,
            start_new_session=True
        )
        
        self.status_bar.showMessage("Viewer launched! You can run another simulation.")
        

def main():
    # Set environment for Qt
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark theme
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = VEPConfigWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
