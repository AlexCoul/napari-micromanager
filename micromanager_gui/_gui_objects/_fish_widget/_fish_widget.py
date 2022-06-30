from __future__ import annotations
from email.mime import image

import warnings
from pathlib import Path
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
from skimage.registration import phase_cross_correlation
from distutils.util import strtobool
import pandas as pd
from hardware.HamiltonMVP import HamiltonMVP
from hardware.APump import APump
from utils.fluidics_control import run_fluidic_program

if TYPE_CHECKING:
    from pymmcore_plus.mda import PMDAEngine


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

        self.pump_COM_port = 'COM5'
        self.valve_COM_port = 'COM6'
        self.pump_parameters = {'pump_com_port': self.pump_COM_port,
                                'pump_ID': 30,
                                'verbose': True,
                                'simulate_pump': False,
                                'serial_verbose': False,
                                'flip_flow_direction': False}


        # tiling parameters

        self.stage_volume_set = False

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
        step = self.step_size_doubleSpinBox.value()
        # set what is the range to consider depending on the z_stack mode
        if self.z_tabWidget.currentIndex() == 0:
            _range = self.z_range_topbottom_doubleSpinBox.value()
        if self.z_tabWidget.currentIndex() == 1:
            _range = self.zrange_spinBox.value()
        if self.z_tabWidget.currentIndex() == 2:
            _range = self.z_range_abovebelow_doubleSpinBox.value()

        self.n_images_label.setText(f"Number of Images: {round((_range / step) + 1)}")

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
        self._mmc.setXYPosition(float(x_val), float(y_val))
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
        self._mmc.setExposure(self.exposure_ms)

        # snap image
        self._mmc.snapImage()

        # grab exposure
        true_exposure = self._mmc.getExposure()

        # grab ROI
        current_ROI =self._mmc.getROI()
        self.x_pixels = current_ROI[2]
        self.y_pixels = current_ROI[3]

        # need to get stage positions from widget -> how?

        n_total_positions = self.stage_tableWidget.rowCount()
        for pos_idx in range(n_total_positions):
            x_grid[pos_idx]= self.stage_tableWidget.item(pos_idx,0)
            y_grid[pos_idx]= self.stage_tableWidget.item(pos_idx,1)
            z_grid[pos_idx]= self.stage_tableWidget.item(pos_idx,2)


        # calculate number of X,Y positions assuming 20% overlap
        self.overlap = 0.2
        y_extent = np.abs(y_max_coord - y_min_coord)
        n_y_positions = int(np.ceil(y_extent / (self.y_pixels/self.overlap)))

        x_extent = np.abs(x_max_coord - x_min_coord)
        n_x_positions = int(np.ceil(x_extent / (self.x_pixels/self.overlap)))

        # calculate actual XY tile positions

        # calculate Z positions
        n_z_positions = np.abs(self.z_top_doubleSpinBox-self.z_bottom_doubleSpinBox)/self.step_size_doubleSpinBox


    # load fluidics and codebook files
    def _load_fluidics(self):
        try:
            self.df_fluidics = data_io.read_fluidics_program(self.fluidics_file_path)
            self.codebook = data_io.read_config_file(self.codebook_file_path)
            self.fluidics_loaded = True
        except:
            raise Exception('Error in loading fluidics and/or codebook files.')

    # generate summary of fluidics and codebook files
    def _generate_fluidics_summary(self):

        self.n_iterative_rounds = int(self.codebook['n_rounds'])
        if (self.n_iterative_rounds == int(self.df_fluidics['round'].max())):

            self.n_active_channels_readout = int(self.codebook['channels_per_round'])
            self.channel_states_readout = [
                bool(strtobool(self.codebook['DPC'])),
                bool(strtobool(self.codebook['cy5']))]

            if not(self.codebook['nuclei_round']==-1):
                self.n_active_channels_nuclei = 2
                self.channel_states_nuclei = [
                    True,
                    True]

            fluidics_data = (f"Experiment type: {str(self.codebook['type'])} \n"
                            f"Number of iterative rounds: {str(self.codebook['n_rounds'])} \n\n"
                            f"Number of targets: {str(self.codebook['targets'])} \n"
                            f"Channels per round: {str(self.codebook['dyes_per_round'])} \n"
                            f"DPC fidicual: {str(self.codebook['alexa488'])} \n"
                            f"Cy5 readout: {str(self.codebook['alexa647'])} \n"
                            f"Nuclear marker round: {str(self.codebook['nuclei_round'])} \n\n")
            self.fluidics_summary.value = fluidics_data
        else:
            raise Exception('Number of rounds in codebook file and fluidics file do not match.')

    # generate summary of experimental setup
    def _generate_experiment_summary(self):

        exp_data = (f"Number of iterative rounds: {str(self.n_iterative_rounds)} \n\n"
                    f"X start: {str(self.x_min_position)}  \n"
                    f"X end:  {str(self.x_max_position)} \n"
                    f"Number of X tiles:  {str(self.n_x_positions)} \n"
                    f"Y start: {str(self.y_min_position)}  \n"
                    f"Y end:  {str(self.y_max_position)} \n"
                    f"Number of Y tiles:  {str(self.n_y_positions)} \n"
                    f"Z start: {str(self.z_bottom_doubleSpinBox)}  \n"
                    f"Z end:  {str(self.z_top_doubleSpinBox)} \n"
                    f"Number of Z positions:  {str(self.z_step)} \n\n"
                    f"--------Readout rounds------- \n"
                    f"Number of channels:  {str(self.n_active_channels_readout)} \n"
                    f"Active channels: {str(self.channel_states_readout)} \n\n"
                    f"--------Nuclei rounds------- \n"
                    f"Number of channels: {str(self.n_active_channels_nuclei)} \n"
                    f"Active channels: {str(self.channel_states_nuclei)} \n\n")
        self.experiment_summary.value = exp_data

    def _save_round_metadata(self,r_idx):
        """
        Construct round metadata dictionary and save
        :param r_idx: int
            round index
        :return None:
        """

        scan_param_data = [{'root_name': str("WF_stage_metadata"),
                            'scan_type': 'WF',
                            'exposure_ms': float(self.exposure_ms),
                            'pixel_size': float(self.camera_pixel_size_um),
                            'x_axis_start': float(self.scan_axis_start_um),
                            'x_axis_end': float(self.scan_axis_end_um),
                            'x_axis_step': float(self.scan_axis_step_um), 
                            'y_axis_start': float(self.tile_axis_start_um),
                            'y_axis_end': float(self.tile_axis_end_um),
                            'y_axis_step': float(self.tile_axis_step_um),
                            'z_axis_start': float(self.height_axis_start_um),
                            'z_axis_end': float(self.height_axis_end_um),
                            'z_axis_step': float(self.height_axis_step_um),
                            'r_idx': int(r_idx),
                            'num_r': int(self.n_iterative_rounds),
                            'num_xy': int(self.n_xy_tiles), 
                            'num_z': int(self.n_z_tiles),
                            'num_ch': int(self.n_active_channels),
                            'y_pixels': int(self.y_pixels),
                            'x_pixels': int(self.x_pixels),
                            'DPC_active': bool(self.channel_states[0]),
                            'Cy5_active': bool(self.channel_states[1])
                            }]
        
        write_metadata(scan_param_data[0], self.metadata_dir_path / Path('scan_'+str(r_idx).zfill(3)+'_metadata.csv'))

    def _save_stage_positions(self,r_idx,tile_xy_idz,tile_z_idx,current_stage_data):
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

        write_metadata(current_stage_data[0], self.metadata_dir_path / Path('stage_r'+str(r_idx).zfill(3)+'_xy'+str(tile_xy_idz).zfill(3)+'_z'+str(tile_z_idx).zfill(3)+'_metadata.csv'))
        
    def _read_fluidics_program(program_path):
        """
        Read fluidics program from CSV file as pandas dataframe
        :param program_path: Path
            location of fluidics program
        :return df_program: Dataframe
            dataframe containing fluidics program 
        """

        df_program = pd.read_csv(program_path)
        return df_program
    
    def write_metadata(data_dict, save_path):
        """
        Write metadata file as csv
        :param data_dict: dict
            dictionary of metadata entries
        :param save_path: Path
            path for file
        :return None:
        """

        pd.DataFrame([data_dict]).to_csv(save_path)

    
    def _on_run_clicked(self):

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a MM cfg file first.")
            
        if self.fluidics_cfg is None:
            raise ValueError("Load a fluidics cfg first.")

        if self.channel_tableWidget.rowCount() <= 0:
            raise ValueError("Select at least one channel.")

        if self.stage_pos_groupBox.isChecked() and (
            self.stage_tableWidget.rowCount() <= 0
        ):
            raise ValueError(
                "Select at least one position" "or deselect the position groupbox."
            )

        if not (
            self.fish_fname_lineEdit.text() and Path(self.fish_dir_lineEdit.text()).is_dir()
        ):
            raise ValueError("Select a filename and a valid directory.")


        # load fluidics program
        self._load_fluidics()


        if self.fluidics_loaded:
            # connect to pump
            self.pump_controller = APump(self.pump_parameters)
            # set pump to remote control
            self.pump_controller.enableRemoteControl(True)

            # connect to valves
            self.valve_controller = HamiltonMVP(com_port=self.valve_COM_port)
            # initialize valves
            self.valve_controller.autoAddress()

        else:
            raise Exception('Configure fluidics first.')
        

        # create metadata directory in output directory
        self.metadata_dir_path = output_dir_path / Path('metadata')
        self.metadata_dir_path.mkdir(parents=True, exist_ok=True)

        # create zarr data directory in output directory
        zarr_dir_path = output_dir_path / Path('raw_data')
        zarr_dir_path.mkdir(parents=True, exist_ok=True)


        # setup circular buffer to be large
        self._mmc.clearCircularBuffer()
        circ_buffer_mb = 8000
        self._mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))
        self._mmc.setTimeoutMs(120000)


        # loop through rounds
        for r_idx in range(self.n_iterative_rounds):

            # run fluidics for this round
            success_fluidics = False          
            success_fluidics = run_fluidic_program(r_idx,self.df_fluidics,self.valve_controller,self.pump_controller)
            if not(success_fluidics):
                raise Exception('Error in fluidics unit.')

            # On the first round, acquire z stack using DPC only to create "ground truth" map of tissue
            if r_idx ==0:

                # set zarr path for this round
                dpc_fidicual_zarr_output_path = zarr_dir_path / Path('DPC_fidicual_r'+str(r_idx).zfill(3)+'_xy'+str(xy_idx).zfill(3)+'.zarr')
                
                # create and open zarr file
                dpc_fidicual_data = zarr.open(
                    str(dpc_fidicual_zarr_output_path), 
                    mode="w", 
                    shape=(self.n_xy_positions, self.n_z_positions, self.y_pixels, self.x_pixels), 
                    chunks=(1, 1, 1, self.y_pixels, self.x_pixels),
                    dtype=np.float32)

                for xy_idx in trange(self.n_xy_tiles,desc="xy tile",position=0):

                # set XY stage position

                    for z_idx in trange(self.n_z_tiles,desc="z position",position=1,leave=False):

                        # set Z stage position

                        raw_dpc_images = np.zeros([4,self.y_pixels,self.x_pixels],dtype=np.uint16)
                        dpc_images = np.zeros([2,self.y_pixels,self.x_pixels],dtype=np.float32)
                        final_dpc_image = np.zeros([self.y_pixels,self.x_pixels],dtype=np.float32)
                        for ch_idx in range(self.n_DPC_illuminations):

                            # set Arduino command
                            command = self.DPC_illumination_commands[dpc_idx]

                            # snap image
                            raw_dpc_images[ch_idx,:] = self._mmc.snapImage()

                        dpc_images[0,:] = (raw_dpc_images[1]-raw_dpc_images[0])/(raw_dpc_images[1]+raw_dpc_images[0])
                        dpc_images[1,:] = (raw_dpc_images[3]-raw_dpc_images[2])/(raw_dpc_images[3]+raw_dpc_images[2])
                        final_dpc_image = np.max(dpc_images,0) # is this the right way to merge DPC images? Or should we take mean?

                        dpc_fidicual_data[xy_idx, 0, z_idx, :, :] = final_dpc_image

            # set zarr path for this round
            dpc_zarr_output_path = zarr_dir_path / Path('DPC_data_r'+str(r_idx).zfill(3)+'_xy'+str(xy_idx).zfill(3)+'.zarr')
            flr_zarr_output_path = zarr_dir_path / Path('FLR_data_r'+str(r_idx).zfill(3)+'_xy'+str(xy_idx).zfill(3)+'.zarr')
            
            # create and open zarr file

            dpc_round_data = zarr.open(
                str(dpc_zarr_output_path), 
                mode="w", 
                shape=(self.n_xy_positions, self.n_z_positions, self.y_pixels, self.x_pixels), 
                chunks=(1, 1, 1, self.y_pixels, self.x_pixels),
                dtype=np.float32)

            flr_round_data = zarr.open(
                str(flr_zarr_output_path), 
                mode="w", 
                shape=(self.n_xy_positions, self.n_z_positions, self.y_pixels, self.x_pixels), 
                chunks=(1, 1, 1, self.y_pixels, self.x_pixels),
                dtype=np.uint16)

            for xy_idx in trange(self.n_xy_tiles,desc="xy tile",position=0):

                # set XY stage position


                # move to middle of z stack


                # capture DPC image at middle of stack
                raw_dpc_images = np.zeros([4,self.y_pixels,self.x_pixels],dtype=np.uint16)
                dpc_images = np.zeros([2,self.y_pixels,self.x_pixels],dtype=np.float32)
                final_dpc_image = np.zeros([self.y_pixels,self.x_pixels],dtype=np.float32)
                for dpc_idx in range(self.n_DPC_illuminations):

                    # set Arduino command
                    command = self.DPC_illumination_commands[dpc_idx]

                    # snap image
                    raw_dpc_images[ch_idx,:] = self._mmc.snapImage()

                dpc_images[0,:] = (raw_dpc_images[1]-raw_dpc_images[0])/(raw_dpc_images[1]+raw_dpc_images[0])
                dpc_images[1,:] = (raw_dpc_images[3]-raw_dpc_images[2])/(raw_dpc_images[3]+raw_dpc_images[2])
                final_dpc_image = np.max(dpc_images,0) # is this the right way to merge DPC images? Or should we take mean?

                # run 2D cross-correlation for each Z plane
                shifts = np.zeros([self.n_z_positions,2],dtype=np.float32)
                errors = np.zer([self.n_z_positions,1],dtype=np.float32)
                for z_idx in range(self.n_z_positions):
                    shifts[z_idx,:], errors[z_idx], _ = phase_cross_correlation(dpc_fidicual_data[xy_idx,z_idx,:],
                                                                               final_dpc_image,
                                                                               upsample_factor=10,
                                                                               return_error=True)

                # calculate XY drift
                x_drift = np.mean(shifts[0,:])
                y_drift = np.mean(shifts[1,:])

                # find best Z plane
                best_z_idx = np.amin(errors)

                # calculate shift from fidicual stack
                if best_z_idx > n_z_positions//2:
                    current_z_offset = -1 * (n_z_positions//2 - best_z_idx) * self.step_size_doubleSpinBox
                elif best_z_idx < n_z_positions//2:
                    current_z_offset = 1 * (n_z_positions//2 - best_z_idx) * self.step_size_doubleSpinBox
                else:
                    current_z_offset = 0

                # store Z offset for this position
                z_offsets[r_idx,xy_idx] = current_z_offset

                for z_idx in trange(self.n_z_tiles,desc="z position",position=1,leave=False):

                    # set Z stage position taking offset into account

                    # capture DPC image at this position
                    raw_dpc_images = np.zeros([4,self.y_pixels,self.x_pixels],dtype=np.uint16)
                    dpc_images = np.zeros([2,self.y_pixels,self.x_pixels],dtype=np.float32)
                    final_dpc_image = np.zeros([self.y_pixels,self.x_pixels],dtype=np.float32)
                    for dpc_idx in range(self.n_DPC_illuminations):

                        # set Arduino command
                        command = self.DPC_illumination_commands[dpc_idx]

                        # snap image
                        raw_dpc_images[ch_idx,:] = self._mmc.snapImage()

                    dpc_images[0,:] = (raw_dpc_images[1]-raw_dpc_images[0])/(raw_dpc_images[1]+raw_dpc_images[0])
                    dpc_images[1,:] = (raw_dpc_images[3]-raw_dpc_images[2])/(raw_dpc_images[3]+raw_dpc_images[2])
                    final_dpc_image = np.max(dpc_images,0) # is this the right way to merge DPC images? Or should we take mean?

                    dpc_round_data[xy_idx, 0, z_idx, :, :] = final_dpc_image

                    # capture red LED fluorescence image at this xyz position

                    # set Arduino command
                    command = self.LED_illumination_command

                    # snap image
                    flr_round_data[xy_idx, z_idx, :, :] = self._mmc.snapImage()
                    time.sleep(.05)

            self._save_stage_positions(r_idx,xy_idx,z_idx,round_stage_data,z_offsets[r_idx,:])
            dpc_round_data = None
            flr_round_data = None
            del opm_round_data, flr_round_data

        # write full metadata
        self._save_full_metadata()


        # Iterative FISH experiment:
        # process fluidics cfg file
        # generate stage positions, XY and Z
        # save metadata
        # prepare empty zarr file?
        # run first round
        # if DPC, acquire whole 3D ROI reference image (same time as fluidics?)
        # fluo acquisition:
        #     for pos_id in positions:
        #         # run DPC autofocus:
        #         for z_id in z_levels:
        #             for LED_side in [left, right, both]:
        #                 get image
        #             make DPC image
        #         stack images in z
        #         perform fourier cross-correlation with reference image at this XY position
        #         save X,Y,Z shifts
        #         # run fluo imaging
        #         for z_id in z_levels:
        #             for c_id in channels:
        #                 get image
        #                 save in zarr
        # 


        # experiment = self.get_state()
        # Alexis: I think we can't use MDA experiment per round because we need to perform
        # DPC-based autofocus, and that requires (for  now) some manual handling of acquisitions
        # SEQUENCE_META[experiment] = SequenceMeta(
        #     mode="mda",
        #     split_channels=self.checkBox_split_channels.isChecked(),
        #     should_save=self.save_groupBox.isChecked(),
        #     file_name=self.fish_fname_lineEdit.text(),
        #     save_dir=self.fish_dir_lineEdit.text(),
        #     save_pos=self.checkBox_save_pos.isChecked(),
        # )
        # self._mmc.run_mda(experiment)  # run the MDA experiment asynchronously
        return
