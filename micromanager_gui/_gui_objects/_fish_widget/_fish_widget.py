from __future__ import annotations

import warnings
from pathlib import Path
import json
from typing import TYPE_CHECKING

import useq
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt
from useq import MDASequence
from micromanager_gui import _mda
from ..._core import get_core_singleton
from ._fish_gui import FishWidgetGui


import numpy as np
import zarr
import time
from tqdm import trange
from skimage.registration import phase_cross_correlation
from distutils.util import strtobool
import pandas as pd
from hardware.HamiltonMVP import HamiltonMVP
from hardware.APump import APump
from hardware.Arduino import init_arduino, set_state
from utils.fluidics_control import run_fluidic_program


if TYPE_CHECKING:
    from pymmcore_plus.mda import PMDAEngine

# TODO: add run fludics only, select / modify fluidic steps

class FishWidget(FishWidgetGui):
    """Multi-dimensional acquisition Widget."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pause_Button.hide()
        self.cancel_Button.hide()

        self._mmc = get_core_singleton()

        self.pause_Button.released.connect(lambda: self._mmc.mda.toggle_pause())
        self.cancel_Button.released.connect(lambda: self._mmc.mda.cancel())

        # connect buttons
        self.add_pos_Button.clicked.connect(self.add_position)
        self.remove_pos_Button.clicked.connect(self.remove_position)
        self.clear_pos_Button.clicked.connect(self.clear_positions)
        self.add_ch_Button.clicked.connect(self._add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)

        self.browse_save_Button.clicked.connect(self.set_multi_d_acq_dir)
        self.run_Button.clicked.connect(self._on_run_clicked)

        # connect for z stack
        self.set_top_Button.clicked.connect(self._set_top)
        self.set_bottom_Button.clicked.connect(self._set_bottom)
        self.z_top_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)
        self.z_bottom_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)

        self.zrange_spinBox.valueChanged.connect(self._update_rangearound_label)

        self.above_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)
        self.below_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)

        self.z_range_abovebelow_doubleSpinBox.valueChanged.connect(
            self._update_n_images
        )
        self.zrange_spinBox.valueChanged.connect(self._update_n_images)
        self.z_range_topbottom_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.step_size_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.z_tabWidget.currentChanged.connect(self._update_n_images)
        self.stack_groupBox.toggled.connect(self._update_n_images)

        # toggle connect
        self.save_groupBox.toggled.connect(self.toggle_checkbox_save_pos)
        self.stage_pos_groupBox.toggled.connect(self.toggle_checkbox_save_pos)

        # connect position table double click
        self.stage_tableWidget.cellDoubleClicked.connect(self.move_to_position)

        # events
        self._mmc.mda.events.sequenceStarted.connect(self._on_mda_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_mda_finished)
        self._mmc.mda.events.sequencePauseToggled.connect(self._on_mda_paused)
        self._mmc.events.mdaEngineRegistered.connect(self._update_mda_engine)

        # fluidics parameters
        self.fluidics_file_path = None
        self.fluidics_program = None
        self.n_iterative_rounds = 0
        self.fluidics_loaded = False

        self.pump_COM_port = "COM4"
        self.valve_COM_port = "COM6"
        self.pump_parameters = {
            "pump_com_port": self.pump_COM_port,
            "pump_ID": 30,
            "verbose": True,
            "simulate_pump": False,
            "serial_verbose": False,
            "flip_flow_direction": False,
        }

        # arduino controller parameters
        self.arduino_port = "COM11"
        # we don't need all LEDs ON (ALIGN command) to compute DPC image
        self.DPC_commands = ["DPC0", "DPC1", "DPC2", "DPC3"]  # , 'ALIGN\n']
        self.n_DPC_illuminations = 4
        self.codebook = {}
        self.n_active_channels = 1  # we have 1 epifluo LED

        # tiling parameters
        self.stage_volume_set = False
        self.overlap = 0.2  # TODO: add choice of % tiles overlap
        self.pixel_size = 0.065
        self._update_n_images()

    def _update_mda_engine(self, newEngine: PMDAEngine, oldEngine: PMDAEngine):
        oldEngine.events.sequenceStarted.disconnect(self._on_mda_started)
        oldEngine.events.sequenceFinished.disconnect(self._on_mda_finished)
        oldEngine.events.sequencePauseToggled.disconnect(self._on_mda_paused)

        newEngine.events.sequenceStarted.connect(self._on_mda_started)
        newEngine.events.sequenceFinished.connect(self._on_mda_finished)
        newEngine.events.sequencePauseToggled.connect(self._on_mda_paused)

    def _set_enabled(self, enabled: bool):
        self.save_groupBox.setEnabled(enabled)
        self.acquisition_order_comboBox.setEnabled(enabled)
        self.channel_groupBox.setEnabled(enabled)

        if not self._mmc.getXYStageDevice():
            self.stage_pos_groupBox.setChecked(False)
            self.stage_pos_groupBox.setEnabled(False)
        else:
            self.stage_pos_groupBox.setEnabled(enabled)

        if not self._mmc.getFocusDevice():
            self.stack_groupBox.setChecked(False)
            self.stack_groupBox.setEnabled(False)
        else:
            self.stack_groupBox.setEnabled(enabled)

    def _set_top(self):
        self.z_top_doubleSpinBox.setValue(self._mmc.getZPosition())

    def _set_bottom(self):
        self.z_bottom_doubleSpinBox.setValue(self._mmc.getZPosition())

    def _update_topbottom_range(self):
        self.z_range_topbottom_doubleSpinBox.setValue(
            abs(self.z_top_doubleSpinBox.value() - self.z_bottom_doubleSpinBox.value())
        )

    def _update_rangearound_label(self, value):
        self.range_around_label.setText(f"-{value/2} µm <- z -> +{value/2} µm")

    def _update_abovebelow_range(self):
        self.z_range_abovebelow_doubleSpinBox.setValue(
            self.above_doubleSpinBox.value() + self.below_doubleSpinBox.value()
        )

    def _update_n_images(self):
        if self.stack_groupBox.isChecked():
            step = self.step_size_doubleSpinBox.value()
            # set what is the range to consider depending on the z_stack mode
            if self.z_tabWidget.currentIndex() == 0:
                _range = self.z_range_topbottom_doubleSpinBox.value()
            if self.z_tabWidget.currentIndex() == 1:
                _range = self.zrange_spinBox.value()
            if self.z_tabWidget.currentIndex() == 2:
                _range = self.z_range_abovebelow_doubleSpinBox.value()
            self.n_images_label.setText(
                f"Number of Images: {round((_range / step) + 1)}"
            )
        else:
            self.n_images_label.setText(f"Number of Images: 1")

    def _on_mda_started(self, sequence):
        self._set_enabled(False)
        self.pause_Button.show()
        self.cancel_Button.show()
        self.run_Button.hide()

    def _on_mda_finished(self, sequence):
        self._set_enabled(True)
        self.pause_Button.hide()
        self.cancel_Button.hide()
        self.run_Button.show()

    def _on_mda_paused(self, paused):
        self.pause_Button.setText("GO" if paused else "PAUSE")

    def _add_channel(self) -> bool:
        """Add, remove or clear channel table.  Return True if anyting was changed."""
        if len(self._mmc.getLoadedDevices()) <= 1:
            return False

        channel_group = self._mmc.getChannelGroup()
        if not channel_group:
            return False

        idx = self.channel_tableWidget.rowCount()
        self.channel_tableWidget.insertRow(idx)

        # create a combo_box for channels in the table
        channel_comboBox = QtW.QComboBox(self)
        channel_exp_spinBox = QtW.QSpinBox(self)
        channel_exp_spinBox.setRange(0, 10000)
        channel_exp_spinBox.setValue(100)

        if channel_group := self._mmc.getChannelGroup():
            channel_list = list(self._mmc.getAvailableConfigs(channel_group))
            channel_comboBox.addItems(channel_list)

        self.channel_tableWidget.setCellWidget(idx, 0, channel_comboBox)
        self.channel_tableWidget.setCellWidget(idx, 1, channel_exp_spinBox)
        return True

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.channel_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.channel_tableWidget.clearContents()
        self.channel_tableWidget.setRowCount(0)

    def toggle_checkbox_save_pos(self):
        if (
            self.stage_pos_groupBox.isChecked()
            and self.stage_tableWidget.rowCount() > 0
        ):
            self.checkBox_save_pos.setEnabled(True)

        else:
            self.checkBox_save_pos.setCheckState(Qt.CheckState.Unchecked)
            self.checkBox_save_pos.setEnabled(False)

    # add, remove, clear, move_to positions table
    def add_position(self):

        if not self._mmc.getXYStageDevice():
            return

        if len(self._mmc.getLoadedDevices()) > 1:
            idx = self._add_position_row()

            for c, ax in enumerate("XYZ"):
                if not self._mmc.getFocusDevice() and ax == "Z":
                    continue
                cur = getattr(self._mmc, f"get{ax}Position")()
                item = QtW.QTableWidgetItem(str(cur))
                item.setTextAlignment(int(Qt.AlignHCenter | Qt.AlignVCenter))
                self.stage_tableWidget.setItem(idx, c, item)

            self.toggle_checkbox_save_pos()

    def _add_position_row(self) -> int:
        idx = self.stage_tableWidget.rowCount()
        self.stage_tableWidget.insertRow(idx)
        return idx

    def remove_position(self):
        # remove selected position
        rows = {r.row() for r in self.stage_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.stage_tableWidget.removeRow(idx)
        self.toggle_checkbox_save_pos()

    def clear_positions(self):
        # clear all positions
        self.stage_tableWidget.clearContents()
        self.stage_tableWidget.setRowCount(0)
        self.toggle_checkbox_save_pos()

    def move_to_position(self):
        if not self._mmc.getXYStageDevice():
            return
        curr_row = self.stage_tableWidget.currentRow()
        x_val = self.stage_tableWidget.item(curr_row, 0).text()
        y_val = self.stage_tableWidget.item(curr_row, 1).text()
        z_val = self.stage_tableWidget.item(curr_row, 2).text()
        self._mmc.setXYPosition((float(x_val), float(y_val)))
        self._mmc.setPosition(self._mmc.getFocusDevice(), float(z_val))

    def set_multi_d_acq_dir(self):
        # set the directory
        self.dir = QtW.QFileDialog(self)
        self.dir.setFileMode(QtW.QFileDialog.DirectoryOnly)
        self.save_dir = QtW.QFileDialog.getExistingDirectory(self.dir)
        self.fish_dir_lineEdit.setText(self.save_dir)
        self.parent_path = Path(self.save_dir)

    def _calculate_scan_volume(self):

        # set experiment exposure
        self._mmc.setExposure(10.0)
        # snap image
        self._mmc.snap()
        # do we need to reset the exposure?
        self._mmc.setExposure(self.fish_expo)
        # grab ROI
        current_ROI = self._mmc.getROI()
        self.x_pixels = current_ROI[2]
        self.y_pixels = current_ROI[3]
        self.n_xy_positions = self.stage_tableWidget.rowCount()

        if self.rect_roi_checkBox.isChecked():
            # grab stage positions from widget
            x_grid = np.zeros([self.n_xy_positions], dtype=np.float32)
            y_grid = np.zeros([self.n_xy_positions], dtype=np.float32)
            z_grid = np.zeros([self.n_xy_positions], dtype=np.float32)

            for pos_idx in range(self.n_xy_positions):
                print(type(self.stage_tableWidget.item(pos_idx, 0).text()))
                x_grid[pos_idx] = self.stage_tableWidget.item(pos_idx, 0).text()
                y_grid[pos_idx] = self.stage_tableWidget.item(pos_idx, 1).text()
                z_grid[pos_idx] = self.stage_tableWidget.item(pos_idx, 2).text()

            x_min = np.min(x_grid)
            x_max = np.max(x_grid)
            y_min = np.min(y_grid)
            y_max = np.min(y_grid)

            # calculate number of X,Y positions assuming 20% overlap
            x_positions = np.linspace(
                x_min, x_max, self.pixel_size * self.x_pixels * self.overlap
            )
            y_positions = np.linspace(
                y_min, y_max, self.pixel_size * self.y_pixels * self.overlap
            )
            z_positions = np.mean(z_grid) * np.ones(x_positions.shape[0])

            # calculate actual XY tile positions
            x_grid, y_grid, z_grid = np.meshgrid(
                x_positions, y_positions, z_positions, indexing="xy"
            )
            self.xyz_stage_positions = np.vstack(
                [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
            )
        else:
            self.xyz_stage_positions = np.zeros([self.n_xy_positions, 3])
            for pos_idx in range(self.n_xy_positions):
                self.xyz_stage_positions[pos_idx, 0] = self.stage_tableWidget.item(
                    pos_idx, 0
                ).text()
                self.xyz_stage_positions[pos_idx, 1] = self.stage_tableWidget.item(
                    pos_idx, 1
                ).text()
                self.xyz_stage_positions[pos_idx, 2] = self.stage_tableWidget.item(
                    pos_idx, 2
                ).text()

        # calculate Z positions
        if self.stack_groupBox.isChecked():
            self.z_step = self.step_size_doubleSpinBox.value()
            if self.z_tabWidget.currentIndex() == 0:
                self.z_start = self.z_bottom_doubleSpinBox.value()
                self.z_end = self.z_top_doubleSpinBox.value()
            elif self.z_tabWidget.currentIndex() == 1:
                self.z_start = -self.zrange_spinBox.value() / 2
                self.z_end = self.zrange_spinBox.value() / 2
            elif self.z_tabWidget.currentIndex() == 2:
                self.z_start = self.below_doubleSpinBox.value()
                self.z_end = self.above_doubleSpinBox.value()

            self.n_z_positions = int(
                np.ceil(np.abs(self.z_end - self.z_start) / self.z_step)
            )
            self.z_displacements = np.linspace(
                self.z_end, self.z_start, self.n_z_positions
            )
        else:
            self.z_start = 0
            self.z_end = 0
            self.z_step = 0
            self.n_z_positions = 1
            self.z_displacements = np.array([0])

        print(f"n_z_positions {self.n_z_positions}")
        print(f"z_end {self.z_end}")
        print(f"z_start {self.z_start}")
        print(f"z_step {self.z_step}")

        # to save experiment parameters latter
        self.x_min_position = self.xyz_stage_positions[:, 0].min()
        self.x_max_position = self.xyz_stage_positions[:, 0].max()
        self.y_min_position = self.xyz_stage_positions[:, 1].min()
        self.y_max_position = self.xyz_stage_positions[:, 1].max()
        self.n_x_positions = np.unique(self.xyz_stage_positions[:, 0]).size
        self.n_y_positions = np.unique(self.xyz_stage_positions[:, 1]).size
        self.n_xy_tiles = len(self.xyz_stage_positions)

    # load fluidics and codebook files
    def _load_fluidics(self):
        try:
            # self.df_fluidics = self._read_fluidics_program(Path(self.fluidics_cfg.text()))
            file_path = Path(
                r"C:\Users\qi2lab\Documents\GitHub\napari-micromanager\micromanager_gui\_gui_objects\_fish_widget\no_fluids.csv"
            )
            # self.df_fluidics = self._read_fluidics_program(file_path))
            self.df_fluidics = pd.read_csv(file_path)
            # self.codebook = self._read_config_file(self.codebook_file_path)
            self.n_iterative_rounds = self.df_fluidics["round"].max()
            self.fluidics_loaded = True
            print("Fluidics program loaded")
            print(self.df_fluidics)
        except:
            raise Exception("Error in loading fluidics and/or codebook files.")

    # # generate summary of fluidics and codebook files
    # def _generate_fluidics_summary(self):
    #     # TODO: finish adapting this summary from the OPM setup
    #     self.n_iterative_rounds = int(self.codebook['n_rounds'])
    #     if (self.n_iterative_rounds == int(self.df_fluidics['round'].max())):

    #         self.n_active_channels_readout = int(self.codebook['channels_per_round'])
    #         self.channel_states_readout = [
    #             ch_name for ch_name, ch_dic in self.codebook["channel_states_readout"] \
    #             if ch_dic["used"] == "True"]

    #         if not(self.codebook['nuclei_round']==-1):
    #             self.n_active_channels_nuclei = 2
    #             self.channel_states_nuclei = [
    #                 True,
    #                 True]

    #         fluidics_data = (f"Experiment type: {str(self.codebook['type'])} \n"
    #                         f"Number of iterative rounds: {str(self.codebook['n_rounds'])} \n\n"
    #                         f"Number of targets: {str(self.codebook['n_targets'])} \n"
    #                         f"Channels per round: {str(self.codebook['dyes_per_round'])} \n"
    #                         f"DPC fiduciual: {str(self.codebook['alexa488'])} \n"
    #                         f"Cy5 readout: {str(self.codebook['alexa647'])} \n"
    #                         f"Nuclear marker round: {str(self.codebook['nuclei_round'])} \n\n")
    #         self.fluidics_summary = fluidics_data
    #     else:
    #         raise Exception('Number of rounds in codebook file and fluidics file do not match.')

    # # generate summary of experimental setup
    # TODO: finish adapting summaries from the OPM control code
    # def _generate_experiment_summary(self):

    #     exp_data = (f"Number of iterative rounds: {str(self.n_iterative_rounds)} \n\n"
    #                 f"X start: {str(self.x_min_position)}  \n"
    #                 f"X end:  {str(self.x_max_position)} \n"
    #                 f"Number of X tiles:  {str(self.n_x_positions)} \n"
    #                 f"Y start: {str(self.y_min_position)}  \n"
    #                 f"Y end:  {str(self.y_max_position)} \n"
    #                 f"Number of Y tiles:  {str(self.n_y_positions)} \n"
    #                 f"Z start: {str(self.z_bottom_doubleSpinBox)}  \n"
    #                 f"Z end:  {str(self.z_top_doubleSpinBox)} \n"
    #                 f"Number of Z positions:  {str(self.z_step)} \n\n"
    #                 f"--------Readout rounds------- \n"
    #                 f"Number of channels:  {str(self.n_active_channels_readout)} \n"
    #                 f"Active channels: {str(self.channel_states_readout)} \n\n"
    #                 f"--------Nuclei rounds------- \n"
    #                 f"Number of channels: {str(self.n_active_channels_nuclei)} \n"
    #                 f"Active channels: {str(self.channel_states_nuclei)} \n\n")
    #     self.experiment_summary = exp_data

    # def _save_round_metadata(self,r_idx):
    #     """
    #     Construct round metadata dictionary and save
    #     :param r_idx: int
    #         round index
    #     :return None:
    #     """

    #     scan_param_data = [{'root_name': str("WF_stage_metadata"),
    #                         'scan_type': 'WF',
    #                         'exposure_ms': float(self._mmc.getExposure()),
    #                         'pixel_size': float(self.pixel_size),
    #                         'r_idx': int(r_idx),
    #                         'num_r': int(self.n_iterative_rounds),
    #                         'num_xy': int(self.n_xy_tiles),
    #                         'num_z': int(self.n_z_positions),
    #                         'num_ch': int(self.n_active_channels_readout),
    #                         'y_pixels': int(self.y_pixels),
    #                         'x_pixels': int(self.x_pixels),
    #                         'DPC_active': bool(self.channel_states[0]),
    #                         'Cy5_active': bool(self.channel_states[1])
    #                         }]

    #     self._write_metadata(scan_param_data[0], self.metadata_dir_path / f'scan_metadata_r{r_idx:03}.csv'

    def _save_stage_positions(self, r_idx, tile_xy_idz, current_stage_data, z_offsets):
        """
        Construct stage position metadata dictionary and save
        :param r_idx: int
            round index
        :param tile_xy_idx: int
            xy tile index
        :param tile_xy_idx: int
            z tile index
        :param current_stage_data: dict
            dictionary of stage positions
        :return None:
        """

        file_path = self.metadata_dir_path / f"WF_stage_positions_r{r_idx:03}.json"
        dico_metadata = {
            "r_idx": r_idx,
            "xy_idx": tile_xy_idz,
            "actual_stage_positions": current_stage_data[r_idx, :].tolist(),
            "z_offset": z_offsets[r_idx, :].tolist(),
        }
        print(dico_metadata)
        # pd.DataFrame(dico_metadata).to_csv(file_path)
        try:
            with open(file_path, "w") as write_file:
                json.dump(dico_metadata, write_file, indent=4)
        except TypeError as err:
            print("TypeError:", err, "\n dumping a text file instead")
            with open(file_path, "w") as write_file:
                json.dumps(str(dico_metadata), write_file)

    def _read_fluidics_program(program_path):
        """
        Read fluidics program from CSV file as pandas dataframe
        :param program_path: Path
            location of fluidics program
        :return df_program: Dataframe
            dataframe containing fluidics program
        """

        df_program = pd.read_csv(program_path)
        if "Unnamed: 0" in df_program.columns:
            df_program.index = df_program["Unnamed: 0"]
            df_program.drop(columns=["Unnamed: 0"], inplace=True)
        return df_program

    def _on_run_clicked(self):

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a MM cfg file first.")

        # issue with FishWidget has no attribute self.fluidics_cfg
        # if self.fluidics_cfg.text() is None:
        #     raise ValueError("Load a fluidics cfg first.")

        if self.channel_tableWidget.rowCount() <= 0:
            raise ValueError("Select at least one channel.")

        if self.stage_pos_groupBox.isChecked() and (
            self.stage_tableWidget.rowCount() <= 0
        ):
            raise ValueError(
                "Select at least one position" "or deselect the position groupbox."
            )

        if not (
            self.fish_fname_lineEdit.text()
            and Path(self.fish_dir_lineEdit.text()).is_dir()
        ):
            raise ValueError("Select a filename and a valid directory.")

        # load fluidics program
        self._load_fluidics()

        # set output path
        output_dir_path = Path(self.fish_dir_lineEdit.text())

        if self.fluidics_loaded:
            # connect to pump
            self.pump_controller = APump(self.pump_parameters)
            # set pump to remote control
            self.pump_controller.enableRemoteControl(True)

            # connect to valves
            self.valve_controller = HamiltonMVP(com_port=self.valve_COM_port)
            # initialize valves
            self.valve_controller.autoAddress()
            print("Fluidics initialized succesfully")

        else:
            raise Exception("Configure fluidics first.")

        # get FISH exposure to alternate with DPC exposure if used and different
        self.fish_expo = self._mmc.getExposure()
        self.run_DPC = self.checkBox_dpc_autofocus.isChecked()
        self.dpc_expo = float(self.dpc_expo_edit.text())
        # create stage tiling positions
        print("calculate scan volume")
        self._calculate_scan_volume()

        # create metadata directory in output directory
        self.metadata_dir_path = output_dir_path / "metadata"
        self.metadata_dir_path.mkdir(parents=True, exist_ok=True)

        # create zarr data directory in output directory
        zarr_dir_path = output_dir_path / "raw_data"
        zarr_dir_path.mkdir(parents=True, exist_ok=True)

        # setup circular buffer to be large
        self._mmc.clearCircularBuffer()
        circ_buffer_mb = 8000
        self._mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))
        self._mmc.setTimeoutMs(120000)

        # arrays to track xyz stage position and z offsets
        actual_stage_positions = np.zeros(
            [self.n_iterative_rounds, self.n_xy_positions, 3], dtype=np.float32
        )
        z_offsets = np.zeros(
            [self.n_iterative_rounds, self.n_xy_positions], dtype=np.float32
        )

        # loop through rounds
        print(f"Starting iterating over {self.n_iterative_rounds} rounds")
        for r_idx in range(self.n_iterative_rounds):

            # run fluidics for this round
            success_fluidics = False
            print(f"run fluidics round {r_idx}")
            success_fluidics = run_fluidic_program(
                r_idx, self.df_fluidics, self.valve_controller, self.pump_controller
            )
            if not (success_fluidics):
                raise Exception("Error in fluidics unit.")

            # On the first round, acquire z stack using DPC only to create "ground truth" map of tissue
            self._mmc.setExposure(self.dpc_expo)
            if r_idx == 0 and self.run_DPC:
                print("make first DPC images")
                # set zarr path for this round
                # filename = f'DPC_fiducial_r{r_idx:03}_xy{xy_idx:03}.zarr'
                filename = f"DPC_fiducial_r{r_idx:03}.zarr"
                dpc_fiducial_zarr_output_path = zarr_dir_path / filename

                # create and open zarr file
                dpc_fiducial_data = zarr.open(
                    str(dpc_fiducial_zarr_output_path),
                    mode="w",
                    shape=(
                        self.n_xy_positions,
                        (self.n_DPC_illuminations + self.n_DPC_illuminations // 2),
                        self.n_z_positions,
                        self.y_pixels,
                        self.x_pixels,
                    ),
                    chunks=(1, 1, 1, self.y_pixels, self.x_pixels),
                    dtype=np.float32,
                )

                for xy_idx in trange(self.n_xy_tiles, desc="xy tile", position=0):

                    # set XY stage position
                    self._mmc.setXYPosition(
                        self.xyz_stage_positions[xy_idx, 0],
                        self.xyz_stage_positions[xy_idx, 1],
                    )

                    # move to middle of z stack
                    self._mmc.setPosition(
                        self._mmc.getFocusDevice(), self.xyz_stage_positions[xy_idx, 2]
                    )

                    for z_idx in trange(
                        self.n_z_positions, desc="z position", position=1, leave=False
                    ):

                        # set Z stage position taking offset into account
                        current_z_position = (
                            self.xyz_stage_positions[xy_idx, 2]
                            + self.z_displacements[z_idx]
                        )
                        self._mmc.setPosition(
                            self._mmc.getFocusDevice(), current_z_position
                        )

                        raw_dpc_images = np.zeros(
                            [4, self.y_pixels, self.x_pixels], dtype=np.uint16
                        )
                        for dpc_idx in range(self.n_DPC_illuminations):
                            # set channel to current DPC illumination
                            self._mmc.setConfig("Arduino", self.DPC_commands[dpc_idx])

                            # snap image
                            raw_dpc_images[dpc_idx, :] = self._mmc.snap()
                            time.sleep(0.5)
                            # set channel to OFF
                            self._mmc.setConfig("Arduino", "OFF")

                        dpc_fiducial_data[xy_idx, 0, z_idx, :, :] = (
                            raw_dpc_images[1] - raw_dpc_images[0]
                        ) / (raw_dpc_images[1] + raw_dpc_images[0])
                        dpc_fiducial_data[xy_idx, 1, z_idx, :, :] = (
                            raw_dpc_images[3] - raw_dpc_images[2]
                        ) / (raw_dpc_images[3] + raw_dpc_images[2])
                        dpc_fiducial_data[xy_idx, 2:6, z_idx, :, :] = raw_dpc_images

            for xy_idx in trange(self.n_xy_positions, desc="xy tile", position=0):
                # create and open zarr file
                if self.run_DPC:
                    dpc_zarr_output_path = (
                        zarr_dir_path / f"DPC_data_r{r_idx:03}_xy{xy_idx:03}.zarr"
                    )
                    dpc_round_data = zarr.open(
                        str(dpc_zarr_output_path),
                        mode="w",
                        shape=(
                            self.n_xy_positions,
                            (self.n_DPC_illuminations + self.n_DPC_illuminations // 2),
                            self.n_z_positions,
                            self.y_pixels,
                            self.x_pixels,
                        ),
                        chunks=(1, 1, 1, self.y_pixels, self.x_pixels),
                        dtype=np.float32,
                    )

                flr_zarr_output_path = (
                    zarr_dir_path / f"FLR_data_r{r_idx:03}_xy{xy_idx:03}.zarr"
                )
                flr_round_data = zarr.open(
                    str(flr_zarr_output_path),
                    mode="w",
                    shape=(
                        self.n_xy_positions,
                        self.n_z_positions,
                        self.y_pixels,
                        self.x_pixels,
                    ),
                    chunks=(1, 1, self.y_pixels, self.x_pixels),
                    dtype=np.uint16,
                )

                # set XY stage position
                self._mmc.setXYPosition(
                    self.xyz_stage_positions[xy_idx, 0],
                    self.xyz_stage_positions[xy_idx, 1],
                )

                # move to middle of z stack
                self._mmc.setPosition(
                    self._mmc.getFocusDevice(), self.xyz_stage_positions[xy_idx, 2]
                )

                # capture DPC image at middle of stack
                raw_dpc_images = np.zeros(
                    [4, self.y_pixels, self.x_pixels], dtype=np.uint16
                )
                if self.run_DPC:
                    self._mmc.setExposure(self.dpc_expo)
                    dpc_images = np.zeros(
                        [2, self.y_pixels, self.x_pixels], dtype=np.float32
                    )
                    for dpc_idx in range(self.n_DPC_illuminations):
                        # set channel to current DPC illumination
                        self._mmc.setConfig("Arduino", self.DPC_commands[dpc_idx])

                        # snap image
                        raw_dpc_images[dpc_idx, :] = self._mmc.snap()
                        time.sleep(0.5)

                        # set channel to OFF
                        self._mmc.setConfig("Arduino", "OFF")

                    dpc_images[0, :] = (raw_dpc_images[1] - raw_dpc_images[0]) / (
                        raw_dpc_images[1] + raw_dpc_images[0]
                    )
                    dpc_images[1, :] = (raw_dpc_images[3] - raw_dpc_images[2]) / (
                        raw_dpc_images[3] + raw_dpc_images[2]
                    )

                    # run 2D cross-correlation for each Z plane
                    try:
                        # we can have several different types of exception because of low SNR or testing
                        # conditions, it's more efficient to just ignore them and keep running the experiment
                        shifts = np.zeros([2, self.n_z_positions, 2], dtype=np.float32)
                        errors = np.zeros([2, self.n_z_positions, 1], dtype=np.float32)
                        for z_idx in range(self.n_z_positions):
                            print(f"compute phase cross correlation for z_idx {z_idx}")
                            (
                                shifts[0, z_idx, :],
                                errors[0, z_idx],
                                _,
                            ) = phase_cross_correlation(
                                dpc_fiducial_data[xy_idx, 0, z_idx, :],
                                dpc_images[0, :],
                                upsample_factor=10,
                                return_error=True,
                                # reference_mask=~np.isnan(dpc_fiducial_data[xy_idx, 0, z_idx, :]), 
                                # moving_mask=~np.isnan(dpc_images[0, :]),
                            )
                            (
                                shifts[1, z_idx, :],
                                errors[1, z_idx],
                                _,
                            ) = phase_cross_correlation(
                                dpc_fiducial_data[xy_idx, 1, z_idx, :],
                                dpc_images[1, :],
                                upsample_factor=10,
                                return_error=True,
                                # reference_mask=~np.isnan(dpc_fiducial_data[xy_idx, 1, z_idx, :]), 
                                # moving_mask=~np.isnan(dpc_images[1, :]),
                            )
                            # find best Z plane
                            best_z_idx = np.amin(np.sum(errors, 0))
                    except Exception as e:
                        print("An exception occured, no z shift will be applied:\n", e)
                        best_z_idx = None

                    # calculate shift from fiducial stack
                    current_z_offset = 0
                    if best_z_idx is not None:
                        if best_z_idx > self.n_z_positions // 2:
                            current_z_offset = (
                                -1
                                * (self.n_z_positions // 2 - best_z_idx)
                                * self.z_step
                            )
                        elif best_z_idx < self.n_z_positions // 2:
                            current_z_offset = (
                                1 * (self.n_z_positions // 2 - best_z_idx) * self.z_step
                            )
                else:
                    current_z_offset = 0

                # store Z offset for this position
                z_offsets[r_idx, xy_idx] = current_z_offset

                # grab actual stage positions
                current_x, current_y = np.asarray(self._mmc.getXYPosition())
                current_z = self._mmc.getZPosition()
                actual_stage_positions[r_idx, xy_idx, 0] = current_x
                actual_stage_positions[r_idx, xy_idx, 1] = current_y
                actual_stage_positions[r_idx, xy_idx, 2] = current_z

                for z_idx in trange(
                    self.n_z_positions, desc="z position", position=1, leave=False
                ):

                    # set Z stage position taking offset into account
                    current_z_position = (
                        self.xyz_stage_positions[xy_idx, 2]
                        + self.z_displacements[z_idx]
                        + z_offsets[r_idx, xy_idx]
                    )
                    self._mmc.setPosition(
                        self._mmc.getFocusDevice(), current_z_position
                    )

                    # capture DPC image at this position
                    if self.run_DPC:
                        self._mmc.setExposure(self.dpc_expo)
                        raw_dpc_images = np.zeros(
                            [4, self.y_pixels, self.x_pixels], dtype=np.uint16
                        )
                        for dpc_idx in range(self.n_DPC_illuminations):

                            # set channel to current DPC illumination
                            self._mmc.setConfig("Arduino", self.DPC_commands[dpc_idx])

                            # snap image
                            raw_dpc_images[dpc_idx, :] = self._mmc.snap()
                            time.sleep(0.05)

                            # set channel to OFF
                            self._mmc.setConfig("Arduino", "OFF")

                        # TODO: use numba to use JIT or GPU function
                        dpc_round_data[xy_idx, 0, z_idx, :, :] = (
                            raw_dpc_images[1] - raw_dpc_images[0]
                        ) / (raw_dpc_images[1] + raw_dpc_images[0])
                        dpc_round_data[xy_idx, 1, z_idx, :, :] = (
                            raw_dpc_images[3] - raw_dpc_images[2]
                        ) / (raw_dpc_images[3] + raw_dpc_images[2])
                        dpc_round_data[xy_idx, 2:6, z_idx, :, :] = raw_dpc_images

                    # capture red LED fluorescence image at this xyz position
                    # set channel to LED
                    self._mmc.setExposure(self.fish_expo)
                    self._mmc.setConfig("Arduino", "LED")
                    # snap image
                    flr_round_data[xy_idx, z_idx, :, :] = self._mmc.snap()
                    time.sleep(0.05)
                    # set channel to OFF
                    self._mmc.setConfig("Arduino", "OFF")

            self._save_stage_positions(r_idx, xy_idx, actual_stage_positions, z_offsets)
            dpc_round_data = None
            flr_round_data = None
            del dpc_round_data, flr_round_data
        print("Iteration over rounds completed")

        # write full metadata
        # self._save_full_metadata()

        return