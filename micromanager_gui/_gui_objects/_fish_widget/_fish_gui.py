from fonticon_mdi6 import MDI6
from typing import Optional
from qtpy import QtWidgets as QtW
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from superqt.fonticon import icon


class FishWidgetGui(QWidget):
    """Just the UI portion of the Fish widget. Runtime logic in FishWidget."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(10, 10, 10, 10)

        # general scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fish_wdg = self._create_gui()
        self._scroll.setWidget(self.fish_wdg)
        self.layout().addWidget(self._scroll)

        # acq order and buttons wdg
        self.bottom_wdg = self._create_bottom_wdg()
        self.layout().addWidget(self.bottom_wdg)

    def _create_gui(self):
        wdg = QWidget()
        wdg_layout = QVBoxLayout()
        wdg_layout.setSpacing(20)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg.setLayout(wdg_layout)
        
        self.fluidics_cfg_wdg = self._create_fluidics_config() #FluidicsConfigurationWidget()
        wdg_layout.addWidget(self.fluidics_cfg_wdg)

        self.save_groupBox = self._create_save_group()
        wdg_layout.addWidget(self.save_groupBox)

        self.channel_groupBox = self._create_channel_group()
        wdg_layout.addWidget(self.channel_groupBox)

        self.stage_pos_groupBox = self._create_stage_pos_groupBox()
        wdg_layout.addWidget(self.stage_pos_groupBox)

        self.stack_groupBox = self._create_stack_groupBox()
        wdg_layout.addWidget(self.stack_groupBox)

        self.dpc_groupBox = self._create_dpc_groupBox()
        wdg_layout.addWidget(self.dpc_groupBox)

        return wdg

    def _create_save_group(self):
        group = QGroupBox(title="Save FISH experiment")
        group.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        group.setCheckable(False)
        # group.setChecked(True)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group.setLayout(group_layout)

        # directory
        dir_group = QWidget()
        dir_group_layout = QHBoxLayout()
        dir_group_layout.setSpacing(5)
        dir_group_layout.setContentsMargins(0, 10, 0, 5)
        dir_group.setLayout(dir_group_layout)
        lbl_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        min_lbl_size = 80
        btn_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        dir_lbl = QLabel(text="Directory:")
        dir_lbl.setMinimumWidth(min_lbl_size)
        dir_lbl.setSizePolicy(lbl_sizepolicy)
        self.fish_dir_lineEdit = QLineEdit()
        self.browse_save_Button = QPushButton(text="...")
        self.browse_save_Button.setSizePolicy(btn_sizepolicy)
        dir_group_layout.addWidget(dir_lbl)
        dir_group_layout.addWidget(self.fish_dir_lineEdit)
        dir_group_layout.addWidget(self.browse_save_Button)

        # filename
        fname_group = QWidget()
        fname_group_layout = QHBoxLayout()
        fname_group_layout.setSpacing(5)
        fname_group_layout.setContentsMargins(0, 5, 0, 10)
        fname_group.setLayout(fname_group_layout)
        fname_lbl = QLabel(text="File Name: ")
        fname_lbl.setMinimumWidth(min_lbl_size)
        fname_lbl.setSizePolicy(lbl_sizepolicy)
        self.fish_fname_lineEdit = QLineEdit()
        self.fish_fname_lineEdit.setText("Experiment")
        fname_group_layout.addWidget(fname_lbl)
        fname_group_layout.addWidget(self.fish_fname_lineEdit)

        # checkbox
        self.checkBox_save_pos = QCheckBox(
            text="Save XY Positions in separate files (ImageJ compatibility)"
        )

        group_layout.addWidget(dir_group)
        group_layout.addWidget(fname_group)
        group_layout.addWidget(self.checkBox_save_pos)

        return group
    
    def _create_fluidics_config(self):
        group = QGroupBox(title="Fluidics experiment")
        group.setCheckable(True)
        group.setChecked(True)
        group.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        group_layout = QVBoxLayout()
        fluidics_cfg_layout = QHBoxLayout()
        # group_layout.setSpacing(10)
        # group_layout.setContentsMargins(10, 10, 10, 10)
        group.setLayout(group_layout)

        # fish config
        self.fluidics_cfg = QLineEdit()
        self.fluidics_cfg.setPlaceholderText("IterativeFish.csv")

        self.browse_cfg_Button = QPushButton("...")

        # fluidics config table
        self.fluidics_tableWidget = QTableWidget()
        self.fluidics_tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fluidics_tableWidget.setMinimumHeight(5)
        hdr = self.fluidics_tableWidget.horizontalHeader()
        hdr.setSectionResizeMode(hdr.Stretch)
        self.fluidics_tableWidget.verticalHeader().setVisible(False)
        self.fluidics_tableWidget.setTabKeyNavigation(True)
        self.fluidics_tableWidget.setColumnCount(5)
        self.fluidics_tableWidget.setRowCount(0)
        self.fluidics_tableWidget.setHorizontalHeaderLabels(["use", "round", "source", "time", "pump"])
        
        self.run_fluidics_Button = QPushButton("Run fluidics")

        fluidics_cfg_layout.addWidget(self.fluidics_cfg)
        fluidics_cfg_layout.addWidget(self.browse_cfg_Button)
        group_layout.addLayout(fluidics_cfg_layout)
        group_layout.addWidget(self.fluidics_tableWidget)
        group_layout.addWidget(self.run_fluidics_Button)

        return group


    def _create_channel_group(self):
        group = QGroupBox(title="Channels")
        group.setMinimumHeight(230)
        group_layout = QGridLayout()
        group_layout.setHorizontalSpacing(15)
        group_layout.setVerticalSpacing(0)
        group_layout.setContentsMargins(10, 0, 10, 0)
        group.setLayout(group_layout)

        # table
        self.channel_tableWidget = QTableWidget()
        self.channel_tableWidget.setMinimumHeight(90)
        hdr = self.channel_tableWidget.horizontalHeader()
        hdr.setSectionResizeMode(hdr.Stretch)
        self.channel_tableWidget.verticalHeader().setVisible(False)
        self.channel_tableWidget.setTabKeyNavigation(True)
        self.channel_tableWidget.setColumnCount(2)
        self.channel_tableWidget.setRowCount(0)
        self.channel_tableWidget.setHorizontalHeaderLabels(
            ["Channel", "Exposure Time (ms)"]
        )
        group_layout.addWidget(self.channel_tableWidget, 0, 0, 3, 1)

        # buttons
        btn_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        min_size = 100
        self.add_ch_Button = QPushButton(text="Add")
        self.add_ch_Button.setMinimumWidth(min_size)
        self.add_ch_Button.setSizePolicy(btn_sizepolicy)
        self.remove_ch_Button = QPushButton(text="Remove")
        self.remove_ch_Button.setMinimumWidth(min_size)
        self.remove_ch_Button.setSizePolicy(btn_sizepolicy)
        self.clear_ch_Button = QPushButton(text="Clear")
        self.clear_ch_Button.setMinimumWidth(min_size)
        self.clear_ch_Button.setSizePolicy(btn_sizepolicy)

        # checkbox
        self.checkBox_split_channels = QCheckBox(text="Split Channels")


        group_layout.addWidget(self.add_ch_Button, 0, 1, 1, 1)
        group_layout.addWidget(self.remove_ch_Button, 1, 1, 1, 2)
        group_layout.addWidget(self.clear_ch_Button, 2, 1, 1, 2)
        group_layout.addWidget(self.checkBox_split_channels, 3, 0, 1, 1)

        return group

    # skip creation of a time_group

    def _create_stage_pos_groupBox(self):
        group = QGroupBox(title="Stage Positions (double-click to move to position)")
        group.setCheckable(True)
        group.setChecked(False)
        group.setMinimumHeight(230)
        group_layout = QGridLayout()
        group_layout.setHorizontalSpacing(15)
        group_layout.setVerticalSpacing(0)
        group_layout.setContentsMargins(10, 0, 10, 0)
        group.setLayout(group_layout)

        # table
        self.stage_tableWidget = QTableWidget()
        self.stage_tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stage_tableWidget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_tableWidget.setMinimumHeight(90)
        hdr = self.stage_tableWidget.horizontalHeader()
        hdr.setSectionResizeMode(hdr.Stretch)
        self.stage_tableWidget.verticalHeader().setVisible(False)
        self.stage_tableWidget.setTabKeyNavigation(True)
        self.stage_tableWidget.setColumnCount(3)
        self.stage_tableWidget.setRowCount(0)
        self.stage_tableWidget.setHorizontalHeaderLabels(["X", "Y", "Z"])
        group_layout.addWidget(self.stage_tableWidget, 0, 0, 3, 1)

        # buttons
        btn_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        min_size = 100
        self.add_pos_Button = QPushButton(text="Add")
        self.add_pos_Button.setMinimumWidth(min_size)
        self.add_pos_Button.setSizePolicy(btn_sizepolicy)
        self.remove_pos_Button = QPushButton(text="Remove")
        self.remove_pos_Button.setMinimumWidth(min_size)
        self.remove_pos_Button.setSizePolicy(btn_sizepolicy)
        self.clear_pos_Button = QPushButton(text="Clear")
        self.clear_pos_Button.setMinimumWidth(min_size)
        self.clear_pos_Button.setSizePolicy(btn_sizepolicy)

        self.rect_roi_checkBox = QCheckBox(text="Make rectangle ROI")

        group_layout.addWidget(self.add_pos_Button, 0, 1, 1, 1)
        group_layout.addWidget(self.remove_pos_Button, 1, 1, 1, 2)
        group_layout.addWidget(self.clear_pos_Button, 2, 1, 1, 2)
        group_layout.addWidget(self.rect_roi_checkBox, 3, 1, 1, 2)

        return group

    def _create_stack_groupBox(self):
        group = QGroupBox(title="Z Stacks")
        group.setCheckable(True)
        group.setChecked(False)
        group.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group.setLayout(group_layout)

        # tab
        self.z_tabWidget = QTabWidget()
        z_tab_layout = QVBoxLayout()
        z_tab_layout.setSpacing(0)
        z_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.z_tabWidget.setLayout(z_tab_layout)
        group_layout.addWidget(self.z_tabWidget)

        # top bottom
        tb = QWidget()
        tb_layout = QGridLayout()
        tb_layout.setContentsMargins(10, 10, 10, 10)
        tb.setLayout(tb_layout)

        self.set_top_Button = QPushButton(text="Set Top")
        self.set_bottom_Button = QPushButton(text="Set Bottom")

        lbl_range_tb = QLabel(text="Range (µm):")
        lbl_range_tb.setAlignment(Qt.AlignCenter)

        self.z_top_doubleSpinBox = QDoubleSpinBox()
        self.z_top_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.z_top_doubleSpinBox.setMinimum(0.0)
        self.z_top_doubleSpinBox.setMaximum(100000)
        self.z_top_doubleSpinBox.setDecimals(2)

        self.z_bottom_doubleSpinBox = QDoubleSpinBox()
        self.z_bottom_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.z_bottom_doubleSpinBox.setMinimum(0.0)
        self.z_bottom_doubleSpinBox.setMaximum(100000)
        self.z_bottom_doubleSpinBox.setDecimals(2)

        self.z_range_topbottom_doubleSpinBox = QDoubleSpinBox()
        self.z_range_topbottom_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.z_range_topbottom_doubleSpinBox.setMaximum(10000000)
        self.z_range_topbottom_doubleSpinBox.setButtonSymbols(
            QAbstractSpinBox.NoButtons
        )
        self.z_range_topbottom_doubleSpinBox.setReadOnly(True)

        tb_layout.addWidget(self.set_top_Button, 0, 0)
        tb_layout.addWidget(self.z_top_doubleSpinBox, 1, 0)
        tb_layout.addWidget(self.set_bottom_Button, 0, 1)
        tb_layout.addWidget(self.z_bottom_doubleSpinBox, 1, 1)
        tb_layout.addWidget(lbl_range_tb, 0, 2)
        tb_layout.addWidget(self.z_range_topbottom_doubleSpinBox, 1, 2)

        self.z_tabWidget.addTab(tb, "TopBottom")

        # range around
        ra = QWidget()
        ra_layout = QHBoxLayout()
        ra_layout.setSpacing(10)
        ra_layout.setContentsMargins(10, 10, 10, 10)
        ra.setLayout(ra_layout)

        lbl_range_ra = QLabel(text="Range (µm):")
        lbl_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl_range_ra.setSizePolicy(lbl_sizepolicy)

        self.zrange_spinBox = QSpinBox()
        self.zrange_spinBox.setValue(5)
        self.zrange_spinBox.setAlignment(Qt.AlignCenter)
        self.zrange_spinBox.setMaximum(100000)

        self.range_around_label = QLabel(text="-2.5 µm <- z -> +2.5 µm")
        self.range_around_label.setAlignment(Qt.AlignCenter)

        ra_layout.addWidget(lbl_range_ra)
        ra_layout.addWidget(self.zrange_spinBox)
        ra_layout.addWidget(self.range_around_label)

        self.z_tabWidget.addTab(ra, "RangeAround")

        # above below wdg
        ab = QWidget()
        ab_layout = QGridLayout()
        ab_layout.setContentsMargins(10, 0, 10, 15)
        ab.setLayout(ab_layout)

        lbl_above = QLabel(text="Above (µm):")
        lbl_above.setAlignment(Qt.AlignCenter)
        self.above_doubleSpinBox = QDoubleSpinBox()
        self.above_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.above_doubleSpinBox.setMinimum(0.05)
        self.above_doubleSpinBox.setMaximum(10000)
        self.above_doubleSpinBox.setSingleStep(0.5)
        self.above_doubleSpinBox.setDecimals(2)

        lbl_below = QLabel(text="Below (µm):")
        lbl_below.setAlignment(Qt.AlignCenter)
        self.below_doubleSpinBox = QDoubleSpinBox()
        self.below_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.below_doubleSpinBox.setMinimum(0.05)
        self.below_doubleSpinBox.setMaximum(10000)
        self.below_doubleSpinBox.setSingleStep(0.5)
        self.below_doubleSpinBox.setDecimals(2)

        lbl_range = QLabel(text="Range (µm):")
        lbl_range.setAlignment(Qt.AlignCenter)
        self.z_range_abovebelow_doubleSpinBox = QDoubleSpinBox()
        self.z_range_abovebelow_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.z_range_abovebelow_doubleSpinBox.setMaximum(10000000)
        self.z_range_abovebelow_doubleSpinBox.setButtonSymbols(
            QAbstractSpinBox.NoButtons
        )
        self.z_range_abovebelow_doubleSpinBox.setReadOnly(True)

        ab_layout.addWidget(lbl_above, 0, 0)
        ab_layout.addWidget(self.above_doubleSpinBox, 1, 0)
        ab_layout.addWidget(lbl_below, 0, 1)
        ab_layout.addWidget(self.below_doubleSpinBox, 1, 1)
        ab_layout.addWidget(lbl_range, 0, 2)
        ab_layout.addWidget(self.z_range_abovebelow_doubleSpinBox, 1, 2)

        self.z_tabWidget.addTab(ab, "AboveBelow")

        # step size wdg
        step_wdg = QWidget()
        step_wdg_layout = QHBoxLayout()
        step_wdg_layout.setSpacing(15)
        step_wdg_layout.setContentsMargins(0, 10, 0, 0)
        step_wdg.setLayout(step_wdg_layout)

        s = QWidget()
        s_layout = QHBoxLayout()
        s_layout.setSpacing(0)
        s_layout.setContentsMargins(0, 0, 0, 0)
        s.setLayout(s_layout)
        lbl = QLabel(text="Step Size (µm):")
        lbl_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl.setSizePolicy(lbl_sizepolicy)
        self.step_size_doubleSpinBox = QDoubleSpinBox()
        self.step_size_doubleSpinBox.setAlignment(Qt.AlignCenter)
        self.step_size_doubleSpinBox.setMinimum(0.05)
        self.step_size_doubleSpinBox.setMaximum(10000)
        self.step_size_doubleSpinBox.setSingleStep(0.5)
        self.step_size_doubleSpinBox.setDecimals(2)
        s_layout.addWidget(lbl)
        s_layout.addWidget(self.step_size_doubleSpinBox)

        self.n_images_label = QLabel(text="Number of Images:")

        step_wdg_layout.addWidget(s)
        step_wdg_layout.addWidget(self.n_images_label)
        group_layout.addWidget(step_wdg)

        return group

    def _create_dpc_groupBox(self):
        group = QGroupBox(title="DPC")
        group.setCheckable(True)
        group.setChecked(False)
        group.setMinimumHeight(230)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group.setLayout(group_layout)

        # checkbox
        self.dpc_autofocus_checkBox = QCheckBox(text="Use DPC autofocus")
        group_layout.addWidget(self.dpc_autofocus_checkBox)
        # Exposure time
        expo = QHBoxLayout()
        self.dpc_expo_label = QLabel(text="DPC expo:")
        self.dpc_expo_time = QSpinBox()
        self.dpc_expo_time.setMinimum(0)
        self.dpc_expo_time.setMaximum(10000)
        self.dpc_expo_time.setAlignment(Qt.AlignCenter)
        self.dpc_expo_time.setValue(500)
        self.dpc_expo_ms_label = QLabel(text="ms")
        expo.addWidget(self.dpc_expo_label)
        expo.addWidget(self.dpc_expo_time)
        expo.addWidget(self.dpc_expo_ms_label)
        group_layout.addLayout(expo)
        # Hot pixel correction
        hotpix = QHBoxLayout()
        self.dpc_hotpix_checkBox = QCheckBox(text="Correct hot pixels")
        # load image file
        self.dpc_dark = QLineEdit()
        self.dpc_dark.setPlaceholderText("dark.tiff")
        self.dpc_dark_Button = QPushButton("...")
        # parameters correction
        self.hotpix_thresh = QLineEdit()
        self.hotpix_thresh.setPlaceholderText("threshold")
        # compute correction
        self.dpc_compute_Button = QPushButton("Compute")
        self.dpc_visualize_Button = QPushButton("Visualize")
        hotpix.addWidget(self.dpc_hotpix_checkBox)
        hotpix.addWidget(self.dpc_dark)
        hotpix.addWidget(self.dpc_dark_Button)
        hotpix.addWidget(self.hotpix_thresh)
        hotpix.addWidget(self.dpc_compute_Button)
        hotpix.addWidget(self.dpc_visualize_Button)
        group_layout.addLayout(hotpix)
        # Denoising
        denoise = QHBoxLayout()
        self.dpc_denoise_checkBox = QCheckBox(text="Denoise")
        # method
        self.dpc_denoise_method = QComboBox()
        self.dpc_denoise_method.addItems(["Gaussian", "Median", "Bin"])
        # parameters correction
        self.dpc_denoise_label = QLabel(text="size:")
        self.dpc_denoise_param = QLineEdit()
        self.dpc_denoise_param.setText("1")
        denoise.addWidget(self.dpc_denoise_checkBox)
        denoise.addWidget(self.dpc_denoise_method)
        denoise.addWidget(self.dpc_denoise_label)
        denoise.addWidget(self.dpc_denoise_param)
        group_layout.addLayout(denoise)
        # Save DPC stack
        save_dpc = QHBoxLayout()
        self.dpc_save_checkBox = QCheckBox(text="Save acquisition stack")
        self.dpc_save_lineEdit = QLineEdit()
        self.dpc_save_lineEdit.setPlaceholderText("Select file path")
        self.dpc_save_Button = QPushButton("...")
        save_dpc.addWidget(self.dpc_save_checkBox)
        save_dpc.addWidget(self.dpc_save_lineEdit)
        save_dpc.addWidget(self.dpc_save_Button)
        group_layout.addLayout(save_dpc)
        # Acquire DPC images
        acquire = QHBoxLayout()
        self.dpc_snap_Button = QPushButton("Snap curent position")
        self.dpc_acquire_Button = QPushButton("Acquire all positions")
        acquire.addWidget(self.dpc_snap_Button)
        acquire.addWidget(self.dpc_acquire_Button)
        group_layout.addLayout(acquire)

        return group

    def _create_bottom_wdg(self):

        wdg = QWidget()
        wdg.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        wdg_layout = QHBoxLayout()
        wdg_layout.setAlignment(Qt.AlignVCenter)
        wdg_layout.setSpacing(10)
        wdg_layout.setContentsMargins(10, 15, 10, 10)
        wdg.setLayout(wdg_layout)

        acq_wdg = QWidget()
        acq_wdg_layout = QHBoxLayout()
        acq_wdg_layout.setSpacing(0)
        acq_wdg_layout.setContentsMargins(0, 0, 0, 0)
        acq_wdg.setLayout(acq_wdg_layout)
        acquisition_order_label = QLabel(text="Acquisition Order:")
        lbl_sizepolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        acquisition_order_label.setSizePolicy(lbl_sizepolicy)
        self.acquisition_order_comboBox = QComboBox()
        self.acquisition_order_comboBox.setMinimumWidth(100)
        self.acquisition_order_comboBox.addItems(["pzc", "pcz"])
        acq_wdg_layout.addWidget(acquisition_order_label)
        acq_wdg_layout.addWidget(self.acquisition_order_comboBox)

        btn_sizepolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        min_width = 130
        icon_size = 40
        self.run_Button = QPushButton(text="Run")
        self.run_Button.setMinimumWidth(min_width)
        self.run_Button.setStyleSheet("QPushButton { text-align: center; }")
        self.run_Button.setSizePolicy(btn_sizepolicy)
        self.run_Button.setIcon(icon(MDI6.play_circle_outline, color=(0, 255, 0)))
        self.run_Button.setIconSize(QSize(icon_size, icon_size))
        self.pause_Button = QPushButton("Pause")
        self.pause_Button.setStyleSheet("QPushButton { text-align: center; }")
        self.pause_Button.setSizePolicy(btn_sizepolicy)
        self.pause_Button.setIcon(icon(MDI6.pause_circle_outline, color="green"))
        self.pause_Button.setIconSize(QSize(icon_size, icon_size))
        self.cancel_Button = QPushButton("Cancel")
        self.cancel_Button.setStyleSheet("QPushButton { text-align: center; }")
        self.cancel_Button.setSizePolicy(btn_sizepolicy)
        self.cancel_Button.setIcon(icon(MDI6.stop_circle_outline, color="magenta"))
        self.cancel_Button.setIconSize(QSize(icon_size, icon_size))

        spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Expanding)

        wdg_layout.addWidget(acq_wdg)
        wdg_layout.addItem(spacer)
        wdg_layout.addWidget(self.run_Button)
        wdg_layout.addWidget(self.pause_Button)
        wdg_layout.addWidget(self.cancel_Button)

        return wdg


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = FishWidgetGui()
    win.show()
    sys.exit(app.exec_())
