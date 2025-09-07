import os
import pickle
from pathlib import Path
from contextlib import suppress

import numpy as np
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.interpolate as interpolate
from matplotlib.widgets import RectangleSelector, Cursor
import matplotlib.patches as patches

from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QPoint, QLine, pyqtSlot
from PyQt6.uic.load_ui import loadUi

from quantus.seg_loading.options import get_seg_loaders
from quantus.entrypoints import seg_loading_step
from quantus.data_objs import UltrasoundRfImage, BmodeSeg
from quantus.gui.seg_loading.advancedRoi import AdvancedRoiDrawGUI
from quantus.gui.seg_loading.qt_utils import MouseTracker, q_im_to_pil


def select_seg_helper(path_input, file_exts):
    file_exts = " ".join([f"*{ext}" for ext in file_exts])
    if not os.path.exists(path_input.text()):  # check if file path is manually typed
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", filter=file_exts)
        if file_name != "":  # If valid file is chosen
            path_input.setText(file_name)
        else:
            return

from .voi_selection_ui import Ui_constructRoi
class SegSelectionGUI(QDialog, Ui_constructRoi):
    def __init__(self, image_data: UltrasoundRfImage):
        super(SegSelectionGUI, self).__init__()
        super().__init__()
        self.setupUi(self)
        # loadUi(str(Path("quantus/gui/seg_loading/voi_selection.ui")), self)

        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.voi_layout)
        self.hide_voi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        
        self.seg_loaders = get_seg_loaders()
        seg_loaders = [s.replace("_", " ").capitalize() for s in self.seg_loaders.keys()]
        self.seg_loaders_list = [s.replace("rf", "RF").replace("iq", "IQ").replace("roi", "ROI").replace("voi", "VOI") for s in seg_loaders]
        self.seg_loaders_list.insert(0, "Draw New")
        self.seg_type_dropdown.addItems(self.seg_loaders_list)
        
        self.image_path_input.setText(image_data.scan_name)
        self.phantom_path_input.setText(image_data.phantom_name)
        self.back_button.clicked.connect(self.set_go_back)
        self.accept_type_button.clicked.connect(self.accept_seg_type)
        self.choose_seg_path_button.clicked.connect(self.select_seg_helper)
        self.clear_seg_path_button.clicked.connect(self.seg_path_input.clear)
        self.accept_seg_path_button.clicked.connect(self.load_segmentation)
        
        self.image_data = image_data
        self.seg_drawing_screen_2d = None
        self.seg_data: BmodeSeg = None
        self.go_back = False; self.frame = 0
        self.im_array = image_data.sc_bmode if image_data.sc_bmode is not None else image_data.bmode
        
    def hide_voi_layout(self):
        self.sag_adv_roi_edit_button.hide(); self.sag_frame_num.hide()
        self.sag_of_label.hide(); self.sag_total_frames.hide()
        self.sag_plane.hide(); self.sag_plane_label.hide()
        self.ax_adv_roi_edit_button.hide(); self.ax_frame_num.hide()
        self.ax_of_label.hide(); self.ax_total_frames.hide()
        self.ax_plane.hide(); self.ax_plane_label.hide()
        self.cor_adv_roi_edit_button.hide(); self.cor_frame_num.hide()
        self.cor_of_label.hide(); self.cor_total_frames.hide()
        self.cor_plane.hide(); self.cor_plane_label.hide()
        self.cur_slice_of_label.hide(); self.cur_slice_slider.hide()
        self.cur_slice_spin_box.hide(); self.cur_slice_total.hide()
        self.cur_slice_label.hide(); self.voi_alpha_of_label.hide()
        self.voi_alpha_spin_box.hide(); self.voi_alpha_status.hide()
        self.voi_alpha_total.hide(); self.voi_alpha_label.hide()
        self.back_to_prev_voi_button.hide(); self.draw_neg_voi_button.hide()
        self.draw_roi_button.hide(); self.interpolate_voi_button.hide()
        self.multiuse_roi_button.hide(); self.undo_last_pt_button.hide()
        self.back_from_draw_button.hide(); self.restart_voi_button.hide()
        self.save_voi_button.hide(); self.choose_save_folder_button.hide()
        self.clear_save_folder_button.hide(); self.back_from_save_button.hide()
        self.save_roi_button.hide(); self.dest_folder_label.hide()
        self.roi_name_label.hide(); self.save_folder_input.hide()
        self.save_name_input.hide(); self.construct_voi_label.hide()
        self.voi_advice_label.hide()
        
    def hide_type_selection_layout(self):
        self.select_type_label.hide()
        self.seg_type_dropdown.hide()
        self.accept_type_button.hide()
        
    def show_type_selection_layout(self):
        self.select_type_label.show()
        self.seg_type_dropdown.show()
        self.accept_type_button.show()
        
    def hide_seg_loading_layout(self):
        self.select_seg_label.hide()
        self.seg_path_label.hide()
        self.seg_path_input.hide()
        self.choose_seg_path_button.hide()
        self.clear_seg_path_button.hide()
        self.seg_kwargs_label.hide()
        self.seg_kwargs_box.hide()
        self.accept_seg_path_button.hide()
        self.loading_screen_label.hide()
        self.select_seg_error_msg.hide()
        
    def show_seg_loading_layout(self):
        self.select_seg_label.show()
        self.seg_path_label.show()
        self.seg_path_input.show()
        self.choose_seg_path_button.show()
        self.clear_seg_path_button.show()
        self.seg_kwargs_label.show()
        self.seg_kwargs_box.show()
        self.accept_seg_path_button.show()
        
    def set_go_back(self):
        self.go_back = True
        
    def select_seg_helper(self):
        select_seg_helper(self.seg_path_input, self.file_exts)
        self.select_seg_error_msg.hide()
        
    def back_to_select(self):
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.removeItem(self.frame_preview_layout)
        self.hide_frame_preview_layout()
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.removeItem(self.confirmation_layout)
        self.hide_seg_confirmation_layout()
        self.full_screen_layout.addItem(self.select_type_layout)
        self.show_type_selection_layout()
        try:
            self.seg_drawing_screen_2d.undo_last_roi()
            if hasattr(self, "spline_x"):
                del self.spline_x; del self.spline_y
        except AttributeError:
            pass
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.set_go_back)
        
    def accept_seg_type(self):
        self.seg_type = list(self.seg_loaders.keys())[self.seg_type_dropdown.currentIndex()-1] if self.seg_type_dropdown.currentIndex() else "Draw New"
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.back_to_select)
        
        if self.seg_type != "Draw New":
            self.file_exts = self.seg_loaders[self.seg_type]["exts"]
            
            self.full_screen_layout.removeItem(self.select_type_layout)
            self.hide_type_selection_layout()
            self.full_screen_layout.addItem(self.seg_loading_layout)
            self.show_seg_loading_layout()
            self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
            self.full_screen_layout.setStretchFactor(self.seg_loading_layout, 10)
            self.seg_path_label.setText(f"Input Path to Segmentation File\n ({', '.join(self.file_exts)})")
            
            self.seg_kwargs_box.setText("{\n\t'assert_scan': False,\t\t# Checks if the seg is initially from the same scan\n\t'assert_phantom': False,\t\t# Checks if the seg is initially from the same phantom\n}")
        else:
            self.full_screen_layout.removeItem(self.select_type_layout)
            self.hide_type_selection_layout()
            self.full_screen_layout.addItem(self.frame_preview_layout)
            self.show_frame_preview_layout()
            self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
            self.full_screen_layout.setStretchFactor(self.frame_preview_layout, 10)
            
            
    def load_segmentation(self):
        if [self.seg_path_input.text().endswith(ext) for ext in self.file_exts].count(True) == 0:
            self.select_seg_error_msg.setText(f"Please select a valid {self.seg_type} file.\nIt must end with one of the following extensions: {', '.join(self.file_exts)}")
            self.select_seg_error_msg.show()
            return
        
        kwargs_text = self.seg_kwargs_box.toPlainText()
        if kwargs_text == "":
            kwargs_text = "{\n}"
        try:
            scan_loader_kwargs = eval(kwargs_text)
            assert isinstance(scan_loader_kwargs, dict), "Options must be in JSON format."
        except (SyntaxError, AssertionError):
            self.select_seg_error_msg.setText("Invalid options format. Please use a JSON format.")
            self.select_seg_error_msg.show()
            return
        except Exception as e:
            self.select_seg_error_msg.setText(f"Error in options: {e}")
            self.select_seg_error_msg.show()
            return
        
        self.seg_data = seg_loading_step(self.seg_type, self.image_data, self.seg_path_input.text(), self.image_data.scan_path, self.image_data.phantom_path, **scan_loader_kwargs)
        
    class DrawNewSeg3d:
        """Class to draw new segmentation on 2D image.
        """
        
        def __init__(self, image_data: UltrasoundRfImage, displayed_im: np.ndarray, seg_gui: "SegSelectionGUI"):
            self.image_data = image_data
            self.displayed_im = displayed_im
            self.seg_gui = seg_gui
            self.spline_x = np.array([]); self.spline_y = np.array([])
            self.points_plotted_x = []; self.points_plotted_y = []
            self.scattered_points = []; self.rect_coords = []
            self.cur_points_plotted_x = []; self.cur_points_plotted_y = []
            self.im_array = image_data.sc_bmode if image_data.sc_bmode is not None else image_data.bmode
            self.cur_alpha = 255; self.adv_roi_draw_gui = AdvancedRoiDrawGUI()
            
            self.drawing_neg = False; self.cur_slice_index = 0
            self.painted = "none"; self.points_plotted = []; self.planes_drawn = []
            
            if self.im_array.ndim == 3:
                self.im_array = np.array(self.im_array).reshape(self.im_array.shape + (1,))
            assert self.im_array.ndim == 4, "Image data must be 4D (x, y, z, t) or 3D (x, y, z)."
            self.displayed_im = self.im_array[:, :, :, self.seg_gui.frame]
            self.x, self.y, self.z, self.num_slices = self.displayed_im.shape
            self.cur_slice_index = self.seg_gui.frame
            
            self.mask_cover_img = np.zeros([self.x, self.y, self.z, 4])
            self.seg_gui.cur_slice_slider.setMaximum(self.num_slices - 1)

            self.slice_array = np.array(range(self.num_slices))
            self.seg_gui.cur_slice_spin_box.setMaximum(self.slice_array[-1])
            self.seg_gui.cur_slice_total.setText(str(self.slice_array[-1]))

            self.seg_gui.cur_slice_spin_box.setValue(self.slice_array[self.cur_slice_index])
            self.seg_gui.cur_slice_slider.setValue(self.cur_slice_index)
            self.seg_gui.cur_slice_slider.valueChanged.connect(self.cur_slice_slider_value_changed)
            self.seg_gui.cur_slice_spin_box.valueChanged.connect(self.cur_slice_spin_box_value_changed)

            self.seg_gui.ax_total_frames.setText(str(self.z))
            self.seg_gui.sag_total_frames.setText(str(self.x))
            self.seg_gui.cor_total_frames.setText(str(self.y))
            self.seg_gui.ax_frame_num.setText(str(self.cur_slice_index + 1))
            self.seg_gui.sag_frame_num.setText(str(self.cur_slice_index + 1))
            self.seg_gui.cor_frame_num.setText(str(self.cur_slice_index + 1))

            self.new_x_val = 0; self.new_y_val = 0; self.new_z_val = 0
            self.update_crosshairs()

            ax_tracker = MouseTracker(self.seg_gui.ax_plane)
            ax_tracker.positionChanged.connect(self.ax_coord_changed)
            ax_tracker.positionClicked.connect(self.ax_plane_clicked)
            sag_tracker = MouseTracker(self.seg_gui.sag_plane)
            sag_tracker.positionChanged.connect(self.sag_coord_changed)
            sag_tracker.positionClicked.connect(self.sag_plane_clicked)
            cor_tracker = MouseTracker(self.seg_gui.cor_plane)
            cor_tracker.positionChanged.connect(self.cor_coord_changed)
            cor_tracker.positionClicked.connect(self.cor_plane_clicked)
            
        @pyqtSlot(QPoint)
        def ax_plane_clicked(self, pos):
            if self.seg_gui.draw_roi_button.isChecked():
                if self.painted == "none":
                    self.painted = "ax"
                    self.painted_slice = [self.new_z_val, self.cur_slice_index]
                if self.painted == "ax":
                    self.ax_coord_changed(pos)
                    if self.drawing_neg:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 255, 0, int(self.cur_alpha)]
                    else:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 0, 255, int(self.cur_alpha)]
                    self.cur_points_plotted_x.append(self.new_x_val)
                    self.cur_points_plotted_y.append(self.new_y_val)
                    self.update_crosshairs()
            elif not self.seg_gui.draw_roi_button.isHidden() and self.painted == "ax":
                self.scroll_paused = not self.scroll_paused
                
        @pyqtSlot(QPoint)
        def ax_coord_changed(self, pos):
            if not self.scroll_paused and ((self.observing_label.isHidden() and self.painted == "none") or self.painted == "ax"):
                x_diff = self.seg_gui.ax_plane.width() - self.seg_gui.ax_plane.pixmap().width()
                y_diff = self.seg_gui.ax_plane.height() - self.seg_gui.ax_plane.pixmap().height()
                x_coord = pos.x() - x_diff / 2
                y_coord = pos.y() - y_diff / 2

                if x_coord < 0 or y_coord < 0 or x_coord >= self.seg_gui.ax_plane.pixmap().width() or y_coord >= self.seg_gui.ax_plane.pixmap().height():
                    return
                self.new_y_val = int((x_coord / self.seg_gui.ax_plane.pixmap().width()) * self.y)
                self.new_x_val = int((y_coord / self.seg_gui.ax_plane.pixmap().height()) * self.x)
                self.update_crosshairs()
                
        @pyqtSlot(QPoint)
        def sag_plane_clicked(self, pos):
            if self.seg_gui.draw_roi_button.isChecked():
                if self.painted == "none":
                    self.painted = "sag"
                    self.painted_slice = [self.new_x_val, self.cur_slice_index]
                if self.painted == "sag":
                    self.sag_coord_changed(pos)
                    if self.drawing_neg:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 255, 0, int(self.cur_alpha)]
                    else:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 0, 255, int(self.cur_alpha)]
                    self.cur_points_plotted_x.append(self.new_x_val)
                    self.cur_points_plotted_y.append(self.new_z_val)
                    self.update_crosshairs()
            elif not self.seg_gui.draw_roi_button.isHidden() and self.painted == "sag":
                self.scroll_paused = not self.scroll_paused

        @pyqtSlot(QPoint)
        def sag_coord_changed(self, pos):
            if not self.scroll_paused and ((self.observing_label.isHidden() and self.painted == "none") or self.painted == "sag"):
                x_diff = self.seg_gui.sag_plane.width() - self.seg_gui.sag_plane.pixmap().width()
                y_diff = self.seg_gui.sag_plane.height() - self.seg_gui.sag_plane.pixmap().height()
                x_coord = pos.x() - x_diff / 2
                y_coord = pos.y() - y_diff / 2

                if x_coord < 0 or y_coord < 0 or x_coord >= self.seg_gui.sag_plane.pixmap().width() or y_coord >= self.seg_gui.sag_plane.pixmap().height():
                    return
                self.new_z_val = int((x_coord / self.seg_gui.sag_plane.pixmap().width()) * self.z)
                self.new_x_val = int((y_coord / self.seg_gui.sag_plane.pixmap().height()) * self.x)
                self.update_crosshairs()

        @pyqtSlot(QPoint)
        def cor_plane_clicked(self, pos):
            if self.seg_gui.draw_roi_button.isChecked():
                if self.painted == "none":
                    self.painted = "cor"
                    self.painted_slice = [self.new_y_val, self.cur_slice_index]
                if self.painted == "cor":
                    self.cor_coord_changed(pos)
                    if self.drawing_neg:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 255, 0, int(self.cur_alpha)]
                    else:
                        self.mask_cover_img[self.new_x_val, self.new_y_val, self.new_z_val] = [0, 0, 255, int(self.cur_alpha)]
                    self.cur_points_plotted_x.append(self.new_y_val)
                    self.cur_points_plotted_y.append(self.new_z_val)
                    self.update_crosshairs()
            elif not self.seg_gui.draw_roi_button.isHidden() and self.painted == "cor":
                self.scroll_paused = not self.scroll_paused

        @pyqtSlot(QPoint)
        def cor_coord_changed(self, pos):
            if not self.scroll_paused and ((self.observing_label.isHidden() and self.painted == "none") or self.painted == "cor"):
                x_diff = self.seg_gui.cor_plane.width() - self.seg_gui.cor_plane.pixmap().width()
                y_diff = self.seg_gui.cor_plane.height() - self.seg_gui.cor_plane.pixmap().height()
                x_coord = pos.x() - x_diff / 2
                y_coord = pos.y() - y_diff / 2

                if x_coord < 0 or y_coord < 0 or x_coord >= self.seg_gui.cor_plane.pixmap().width() or y_coord >= self.seg_gui.cor_plane.pixmap().height():
                    return
                self.new_z_val = int((x_coord / self.seg_gui.cor_plane.pixmap().width()) * self.z)
                self.new_y_val = int((y_coord / self.seg_gui.cor_plane.pixmap().height()) * self.y)
                self.update_crosshairs()
                
        def cur_slice_spin_box_value_changed(self):
            if not self.slice_slider_changed:
                self.slice_spin_box_changed = True
                self.slice_value_changed()

        def cur_slice_slider_value_changed(self):
            if not self.slice_spin_box_changed:
                self.slice_slider_changed = True
                self.slice_value_changed()
                
        def slice_value_changed(self):
            if self.slice_spin_box_changed and self.slice_slider_changed:
                self.sliceSpinBoxChanged = False
                self.slice_slider_changed = False
                print("Error tracking slices")
                return
            if self.slice_spin_box_changed:
                self.cur_slice_index = int(self.seg_gui.cur_slice_spin_box.value())
                self.seg_gui.cur_slice_slider.setValue(self.cur_slice_index)
                self.slice_spin_box_changed = False
            if self.slice_slider_changed:
                self.cur_slice_index = int(self.seg_gui.cur_slice_slider.value())
                self.seg_gui.cur_slice_spin_box.setValue(self.slice_array[self.cur_slice_index])
                self.slice_slider_changed = False
            self.update_crosshairs()
                
        def change_axial_slices(self):
            self.seg_gui.ax_frame_num.setText(str(self.new_z_val + 1))

            data_2d_ax = self.im_array[:, :, self.new_z_val, self.cur_slice_index]
            data_2d_ax = np.require(data_2d_ax, np.uint8, "C")
            height_ax, width_ax = data_2d_ax.shape  # getting height and width for each plane
            bytes_line_ax, _ = data_2d_ax.strides

            q_img_ax = QImage(data_2d_ax, width_ax, height_ax, bytes_line_ax, QImage.Format.Format_Grayscale8)
            q_img_ax = q_img_ax.convertToFormat(QImage.Format.Format_ARGB32)

            temp_ax = self.mask_cover_img[:, :, self.new_z_val, :]  # 2D data for axial
            temp_ax = np.require(temp_ax, np.uint8, "C")
            mask_ax_h, mask_ax_w = temp_ax[:, :, 0].shape
            mask_bytes_line_ax, _ = temp_ax[:, :, 0].strides

            cur_mask_ax_im = QImage(temp_ax, mask_ax_w, mask_ax_h, mask_bytes_line_ax, QImage.Format.Format_ARGB32)

            self.im_ax_pil = q_im_to_pil(q_img_ax)
            mask_ax = q_im_to_pil(cur_mask_ax_im)
            self.im_ax_pil.paste(mask_ax, mask=mask_ax)
            self.pixmap_ax = QPixmap.fromImage(ImageQt(self.im_ax_pil))
            self.seg_gui.ax_plane.setPixmap(self.pixmap_ax.scaled(
                self.seg_gui.ax_plane.width(), self.seg_gui.ax_plane.height(), Qt.AspectRatioMode.KeepAspectRatio))
            
        def change_sag_slices(self):
            self.seg_gui.sag_frame_num.setText(str(self.new_y_val + 1))

            data_2d_sag = self.im_array[:, self.new_y_val, :, self.cur_slice_index].astype(np.uint8)
            data_2d_sag = np.require(data_2d_sag, np.uint8, "C")
            height_sag, width_sag = data_2d_sag.shape
            bytes_line_sag, _ = data_2d_sag.strides

            q_img_sag = QImage(data_2d_sag, width_sag, height_sag, bytes_line_sag, QImage.Format.Format_Grayscale8)
            q_img_sag = q_img_sag.convertToFormat(QImage.Format.Format_ARGB32)

            temp_sag = self.mask_cover_img[:, self.new_y_val, :, :]
            temp_sag = np.require(temp_sag, np.uint8, "C")
            mask_sag_h, mask_sag_w = temp_sag[:, :, 0].shape
            mask_bytes_line_sag, _ = temp_sag[:, :, 0].strides

            cur_mask_sag_im = QImage(temp_sag, mask_sag_w, mask_sag_h, mask_bytes_line_sag, QImage.Format.Format_ARGB32)

            self.im_sag_pil = q_im_to_pil(q_img_sag)
            mask_sag = q_im_to_pil(cur_mask_sag_im)
            self.im_sag_pil.paste(mask_sag, mask=mask_sag)
            self.pixmap_sag = QPixmap.fromImage(ImageQt(self.im_sag_pil))
            self.seg_gui.sag_plane.setPixmap(self.pixmap_sag.scaled(
                self.seg_gui.sag_plane.width(), self.seg_gui.sag_plane.height(), Qt.AspectRatioMode.KeepAspectRatio))

        def change_cor_slices(self):
            self.seg_gui.cor_frame_num.setText(str(self.new_x_val + 1))

            data_2d_cor = self.im_array[self.new_x_val, :, :, self.cur_slice_index]
            data_2d_cor = np.require(data_2d_cor, np.uint8, "C")
            height_cor, width_cor = data_2d_cor.shape
            bytes_line_cor, _ = data_2d_cor.strides

            q_img_cor = QImage(data_2d_cor, width_cor, height_cor, bytes_line_cor, QImage.Format.Format_Grayscale8)
            q_img_cor = q_img_cor.convertToFormat(QImage.Format.Format_ARGB32)

            temp_cor = self.mask_cover_img[self.new_x_val, :, :, :]
            temp_cor = np.require(temp_cor, np.uint8, "C")
            mask_cor_h, mask_cor_w = temp_cor[:, :, 0].shape
            mask_bytes_line_cor, _ = temp_cor[:, :, 0].strides

            cur_mask_cor_im = QImage(temp_cor, mask_cor_w, mask_cor_h, mask_bytes_line_cor, QImage.Format.Format_ARGB32)

            self.im_cor_pil = q_im_to_pil(q_img_cor)
            mask_cor = q_im_to_pil(cur_mask_cor_im)
            self.im_cor_pil.paste(mask_cor, mask=mask_cor)
            self.pixmap_cor = QPixmap.fromImage(ImageQt(self.im_cor_pil))
            self.seg_gui.cor_plane.setPixmap(self.pixmap_cor.scaled(
                self.seg_gui.cor_plane.width(), self.seg_gui.cor_plane.height(), Qt.AspectRatioMode.KeepAspectRatio))
            
        def update_crosshairs(self):
            self.updated_adv_roi_edit_buttons()
            self.change_axial_slices(); self.change_sag_slices(); self.change_cor_slices()
                        
        def updated_adv_roi_edit_buttons(self):
            if not np.amax(self.mask_cover_img): # if 3D VOI interpolation is complete
                with suppress(TypeError): self.seg_gui.ax_adv_roi_edit_button.clicked.disconnect()
                with suppress(TypeError): self.seg_gui.sag_adv_roi_edit_button.clicked.disconnect()
                with suppress(TypeError): self.seg_gui.cor_adv_roi_edit_button.clicked.disconnect()
                self.seg_gui.ax_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
                self.seg_gui.sag_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
                self.seg_gui.cor_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
                return

            if len(self.planes_drawn) and self.painted == "none" and "ax" in np.array(self.planes_drawn, dtype=object)[:,0]:
                self.seg_gui.ax_adv_roi_edit_button.clicked.connect(self.ax_advanced_roi_draw)
                self.seg_gui.ax_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(0, 255, 71); border-radius: 15px;")
            else:
                with suppress(TypeError): self.seg_gui.ax_adv_roi_edit_button.clicked.disconnect()
                self.seg_gui.ax_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
            if len(self.planes_drawn) and self.painted == "none" and "sag" in np.array(self.planes_drawn, dtype=object)[:,0]:
                self.seg_gui.sag_adv_roi_edit_button.clicked.connect(self.sag_advanced_roi_draw)
                self.seg_gui.sag_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(0, 255, 71); border-radius: 15px;")
            else:
                with suppress(TypeError): self.seg_gui.sag_adv_roi_edit_button.clicked.disconnect()
                self.seg_gui.sag_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
            if len(self.planes_drawn) and self.painted == "none" and "cor" in np.array(self.planes_drawn, dtype=object)[:,0]:
                self.seg_gui.cor_adv_roi_edit_button.clicked.connect(self.cor_advanced_roi_draw)
                self.seg_gui.cor_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(0, 255, 71); border-radius: 15px;")
            else:
                with suppress(TypeError): self.seg_gui.cor_adv_roi_edit_button.clicked.disconnect()
                self.seg_gui.cor_adv_roi_edit_button.setStyleSheet("color: white; font-size: 16px; background: rgb(255, 37, 14); border-radius: 15px;")
                
        def ax_advanced_roi_draw(self):
            ax_drawings = np.array([plane_drawn[1] for plane_drawn in self.planes_drawn if plane_drawn[0] == "ax"])
            drawing_z_slices = ax_drawings[:,0]
            closest_z_ix = np.argmin(abs(drawing_z_slices - self.new_z_val))
            closest_ax_drawing = ax_drawings[closest_z_ix]

            plane_ix = [i for i, plane_drawn in enumerate(self.planes_drawn) if plane_drawn[0] == "ax" and np.all(plane_drawn[1] == closest_ax_drawing)][0]
            points_plotted_x, points_plotted_y = self.points_plotted[plane_ix]

            data_2d_ax = self.im_array[:, :, closest_ax_drawing[0], closest_ax_drawing[1]]
            data_2d_ax = np.rot90(np.flipud(data_2d_ax), 3)
            data_2d_ax = np.require(data_2d_ax, np.uint8, "C")
            self.new_z_val = closest_ax_drawing[0]; 
            self.seg_gui.cur_slice_slider.setValue(closest_ax_drawing[1])
            self.update_crosshairs()
            self.start_advanced_roi_draw(data_2d_ax, points_plotted_x, points_plotted_y, plane_ix)
            
        def sag_advanced_roi_draw(self):
            sag_drawings = np.array([plane_drawn[1] for plane_drawn in self.planes_drawn if plane_drawn[0] == "sag"])
            drawing_x_slices = sag_drawings[:,0]
            closest_x_ix = np.argmin(abs(drawing_x_slices - self.new_x_val))
            closest_sag_drawing = sag_drawings[closest_x_ix]
            
            plane_ix = [i for i, plane_drawn in enumerate(self.planes_drawn) if plane_drawn[0] == "sag" and np.all(plane_drawn[1] == closest_sag_drawing)][0]
            points_plotted_x, points_plotted_y = self.points_plotted[plane_ix]

            data_2d_sag = self.im_array[closest_sag_drawing[0], :, :, closest_sag_drawing[1]]
            data_2d_sag = np.require(data_2d_sag, np.uint8, "C")
            self.new_x_val = closest_sag_drawing[0]; 
            self.seg_gui.cur_slice_slider.setValue(closest_sag_drawing[1]) 
            self.update_crosshairs()
            self.start_advanced_roi_draw(data_2d_sag, points_plotted_x, points_plotted_y, plane_ix)
            
        def cor_advanced_roi_draw(self):
            cor_drawings = np.array([plane_drawn[1] for plane_drawn in self.planes_drawn if plane_drawn[0] == "cor"])
            drawing_y_slices = cor_drawings[:, 0]
            closest_y_index = np.argmin(abs(drawing_y_slices - self.new_y_val))
            closest_cor_drawing = cor_drawings[closest_y_index]

            plane_idx = [i for i, plane_drawn in enumerate(self.planes_drawn)
                        if plane_drawn[0] == "cor" and np.all(plane_drawn[1] == closest_cor_drawing)][0]
            points_plotted_x, points_plotted_y = self.points_plotted[plane_idx]

            data_2d_cor = self.im_array[:, closest_cor_drawing[0], :, closest_cor_drawing[1]]
            data_2d_cor = np.fliplr(np.rot90(data_2d_cor, 3))
            data_2d_cor = np.require(data_2d_cor, np.uint8, "C")
            
            self.new_y_val = closest_cor_drawing[0]
            self.seg_gui.cur_slice_slider.setValue(closest_cor_drawing[1])
            self.update_crosshairs()
            self.start_advanced_roi_draw(data_2d_cor, points_plotted_x, points_plotted_y, plane_idx)

            
        def start_advanced_roi_draw(self, image, points_plotted_x, points_plotted_y, drawing_ix):
            self.adv_roi_draw_gui.voiSelectionGUI = self
            self.adv_roi_draw_gui.drawingIdx = drawing_ix
            self.adv_roi_draw_gui.curPlane = self.planes_drawn[drawing_ix]
            self.adv_roi_draw_gui.ax.clear()
            self.adv_roi_draw_gui.x = points_plotted_x
            self.adv_roi_draw_gui.y = points_plotted_y
            self.adv_roi_draw_gui.ax.imshow(image, cmap="Greys_r", aspect="auto")
            self.adv_roi_draw_gui.prepPlot()
            self.adv_roi_draw_gui.canvas.draw()
            self.adv_roi_draw_gui.resize(self.size())
            self.adv_roi_draw_gui.show()
            
            
            
        def interpolate_points(self, event):  # Update ROI being drawn using spline using 2D interpolation
            if len(self.points_plotted_x) > 0 and self.points_plotted_x[-1] == int(event.xdata) and self.points_plotted_y[-1] == int(event.ydata):
                return

            self.points_plotted_x.append(int(event.xdata))
            self.points_plotted_y.append(int(event.ydata))
            plotted_points = len(self.points_plotted_x)

            if plotted_points > 1:
                if plotted_points > 2:
                    old_spline = self.spline.pop(0)
                    old_spline.remove()

                x_spline, y_spline = calculate_spline(
                    np.array(self.points_plotted_x) / self.displayed_im.shape[1], np.array(self.points_plotted_y) / self.displayed_im.shape[0]
                )
                x_spline *= self.displayed_im.shape[1]
                y_spline *= self.displayed_im.shape[0]
                x_spline = np.clip(x_spline, a_min=0, a_max=self.displayed_im.shape[1]-1)
                y_spline = np.clip(y_spline, a_min=0, a_max=self.displayed_im.shape[0]-1)
                self.spline = self.seg_gui.ax_draw.plot(
                    x_spline, y_spline, color="cyan", zorder=1, linewidth=0.75
                )
                self.seg_gui.figure_draw.subplots_adjust(
                    left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
                )
                self.seg_gui.ax_draw.tick_params(bottom=False, left=False)
            self.scattered_points.append(
                self.seg_gui.ax_draw.scatter(
                    self.points_plotted_x[-1],
                    self.points_plotted_y[-1],
                    marker="o", # type: ignore
                    s=0.5,
                    c="red",
                    zorder=500,
                )
            )
            self.seg_gui.canvas_draw.draw()
            
        def clear_rect(self, event):
            if len(self.seg_gui.ax_draw.patches) > 0:
                rect = self.seg_gui.ax_draw.patches[0]
                rect.remove()
                self.seg_gui.canvas_draw.draw()
                
        def plot_patch(self):
            if len(self.rect_coords) > 0:
                left, bottom, right, top = self.rect_coords
                rect = patches.Rectangle(
                    (left, bottom),
                    (right - left),
                    (top - bottom),
                    linewidth=1,
                    edgecolor="cyan",
                    facecolor="none",
                )
                if len(self.seg_gui.ax_draw.patches) > 0:
                    self.seg_gui.ax_draw.patches.pop()

                self.seg_gui.ax_draw.add_patch(rect)

                mpl_pix_width = abs(right - left)
                lateral_res = self.image_data.lateral_res if self.image_data.sc_bmode is None else self.image_data.sc_lateral_res
                cm_width = mpl_pix_width * lateral_res / 10
                self.seg_gui.physical_rect_width_val.setText(str(np.round(cm_width, decimals=2)))

                mpl_pix_height = abs(top - bottom)
                axial_res = self.image_data.axial_res if self.image_data.sc_bmode is None else self.image_data.sc_axial_res
                cm_height = mpl_pix_height * axial_res / 10
                self.seg_gui.physical_rect_height_val.setText(str(np.round(cm_height, decimals=2)))

                self.seg_gui.figure_draw.subplots_adjust(
                    left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
                )
                self.seg_gui.ax_draw.tick_params(bottom=False, left=False)
                self.seg_gui.canvas_draw.draw()
                
        def undo_last_pt(self):  # When drawing ROI, undo last point plotted
            if len(self.points_plotted_x) > 0 and self.seg_gui.draw_roi_button.isCheckable():
                scattered_point = self.scattered_points.pop()
                scattered_point.remove()
                self.points_plotted_x.pop(); self.points_plotted_y.pop()
                if len(self.points_plotted_x) > 0:
                    old_spline = self.spline.pop(0)
                    old_spline.remove()
                    if len(self.points_plotted_x) > 1:
                        x_spline, y_spline = calculate_spline(
                            np.array(self.points_plotted_x) / self.displayed_im.shape[1], np.array(self.points_plotted_y) / self.displayed_im.shape[0]
                        )
                        x_spline *= self.displayed_im.shape[1]
                        y_spline *= self.displayed_im.shape[0]
                        x_spline = np.clip(x_spline, a_min=0, a_max=self.displayed_im.shape[1]-1)
                        y_spline = np.clip(y_spline, a_min=0, a_max=self.displayed_im.shape[0]-1)
                        self.spline_x = x_spline
                        self.spline_y = y_spline
                        self.spline = self.seg_gui.ax_draw.plot(
                            self.spline_x,
                            self.spline_y,
                            color="cyan",
                            linewidth=0.75,
                        )
                self.seg_gui.canvas_draw.draw()
                self.seg_gui.draw_roi_button.setChecked(True)
                self.draw_freehand_clicked()
                
        def close_interpolation(self):  # Finish drawing ROI
            if len(self.points_plotted_x) > 2:
                if self.points_plotted_x[0] != self.points_plotted_x[-1] and self.points_plotted_y[0] != self.points_plotted_y[-1]:
                    self.points_plotted_x.append(self.points_plotted_x[0])
                    self.points_plotted_y.append(self.points_plotted_y[0])
                x_spline, y_spline = calculate_spline(
                    np.array(self.points_plotted_x) / self.displayed_im.shape[1], np.array(self.points_plotted_y) / self.displayed_im.shape[0]
                )
                x_spline *= self.displayed_im.shape[1]
                y_spline *= self.displayed_im.shape[0]
                x_spline = np.clip(x_spline, a_min=0, a_max=self.displayed_im.shape[1]-1)
                y_spline = np.clip(y_spline, a_min=0, a_max=self.displayed_im.shape[0]-1)
                self.spline_x = x_spline
                self.spline_y = y_spline
                
                self.spline_x = np.clip(self.spline_x, a_min=0, a_max=self.displayed_im.shape[1]-1)
                self.spline_y = np.clip(self.spline_y, a_min=0, a_max=self.displayed_im.shape[0]-1)

                self.seg_gui.draw_roi_button.setChecked(False)
                self.seg_gui.draw_roi_button.setCheckable(False)
                self.seg_gui.redraw_roi_button.show()
                self.seg_gui.close_roi_button.hide()
                self.cid = self.seg_gui.figure_draw.canvas.mpl_disconnect(self.cid)
                self.plot_on_canvas()
            
        def undo_last_roi(self):  # Remove previously drawn roi and prepare user to draw a new one
            self.spline_x = np.array([])
            self.spline_y = np.array([])
            self.points_plotted_x = []
            self.points_plotted_y = []
            self.seg_gui.draw_roi_button.setChecked(False)
            self.seg_gui.draw_roi_button.setCheckable(True)
            self.seg_gui.close_roi_button.show()
            self.seg_gui.redraw_roi_button.hide()
            self.plot_on_canvas()
            
        def start_draw_rect(self):
            self.hide_initial_buttons()
            self.show_rect_buttons()
                
        def plot_on_canvas(self):
            self.seg_gui.ax_draw.clear()
            if self.image_data.sc_bmode is not None:
                width = self.displayed_im.shape[1]*self.image_data.sc_lateral_res
                height = self.displayed_im.shape[0]*self.image_data.sc_axial_res
            else:
                width = self.displayed_im.shape[1]*self.image_data.lateral_res
                height = self.displayed_im.shape[0]*self.image_data.axial_res
            aspect = width/height
            im = self.seg_gui.ax_draw.imshow(self.displayed_im, cmap="gray")
            extent = im.get_extent()
            self.seg_gui.ax_draw.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
            self.seg_gui.figure_draw.set_facecolor((0, 0, 0, 0)) #type: ignore
            self.seg_gui.ax_draw.axis("off")
            
            if hasattr(self, 'spline_x') and len(self.spline_x):
                self.spline = self.seg_gui.ax_draw.plot(self.spline_x, self.spline_y, 
                                        color="cyan", zorder=1, linewidth=0.75)
            elif len(self.points_plotted_x) > 0:
                self.scattered_points.append(
                    self.seg_gui.ax_draw.scatter(
                        self.points_plotted_x[-1],
                        self.points_plotted_y[-1],
                        marker="o", #type: ignore
                        s=0.5,
                        c="red",
                        zorder=500,
                    )
                )
                if len(self.points_plotted_x) > 1:
                    x_spline, y_spline = calculate_spline(
                        np.array(self.points_plotted_x) / self.displayed_im.shape[1], np.array(self.points_plotted_y) / self.displayed_im.shape[0]
                    )
                    x_spline *= self.displayed_im.shape[1]
                    y_spline *= self.displayed_im.shape[0]
                    x_spline = np.clip(x_spline, a_min=0, a_max=self.displayed_im.shape[1]-1)
                    y_spline = np.clip(y_spline, a_min=0, a_max=self.displayed_im.shape[0]-1)
                    self.spline = self.seg_gui.ax_draw.plot(
                        x_spline, y_spline, color="cyan", zorder=1, linewidth=0.75
                    )

            self.seg_gui.figure_draw.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
            )
            plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            self.seg_gui.canvas_draw.draw()  # Refresh canvas
            
        def draw_freehand_clicked(self):
            if self.seg_gui.draw_roi_button.isChecked():  # Set up b-mode to be drawn on
                self.cid = self.seg_gui.figure_draw.canvas.mpl_connect(
                    "button_press_event", self.interpolate_points
                )
            else:  # No longer let b-mode be drawn on
                if hasattr(self, "cid"):
                    self.cid = self.seg_gui.figure_draw.canvas.mpl_disconnect(self.cid)
            self.seg_gui.canvas_draw.draw()
            
        def draw_rect_clicked(self):
            if self.seg_gui.draw_rectangle_button.isChecked():  # Set up b-mode to be drawn on
                self.selector.set_active(True)
                self.cid = self.seg_gui.figure_draw.canvas.mpl_connect(
                    "button_press_event", self.clear_rect
                )
            else:  # No longer let b-mode be drawn on
                self.cid = self.seg_gui.figure_draw.canvas.mpl_disconnect(self.cid)
                self.selector.set_active(False)
            self.seg_gui.canvas_draw.draw()
            
        def back_from_rect(self):
            self.seg_gui.physical_rect_height_val.setText("0")
            self.seg_gui.physical_rect_width_val.setText("0")
            self.seg_gui.draw_rectangle_button.setChecked(False)
            self.undo_last_roi(); self.seg_gui.close_roi_button.hide()
            self.hide_rect_buttons()
            self.seg_gui.show_draw_seg_start()
            self.rect_coords = []
            self.selector.set_active(False)
            if len(self.seg_gui.ax_draw.patches) > 0:
                self.seg_gui.ax_draw.patches.pop()
            self.seg_gui.canvas_draw.draw()
            
        def hide_rect_buttons(self):
            self.seg_gui.draw_rect_buttons.hide()
            self.seg_gui.draw_rectangle_button.hide()
            self.seg_gui.back_from_rectangle_button.hide()
            self.seg_gui.save_rect_button.hide()
            self.seg_gui.physical_rect_dims_label.hide()
            self.seg_gui.physical_rect_dims_label.hide()
            self.seg_gui.physical_rect_height_label.hide()
            self.seg_gui.physical_rect_width_label.hide()
            self.seg_gui.physical_rect_height_val.hide()
            self.seg_gui.physical_rect_width_val.hide()

        def show_rect_buttons(self):
            self.seg_gui.draw_rectangle_button.show()
            self.seg_gui.draw_rect_buttons.show()
            self.seg_gui.back_from_rectangle_button.show()
            self.seg_gui.save_rect_button.show()
            self.seg_gui.physical_rect_dims_label.show()
            self.seg_gui.physical_rect_height_label.show()
            self.seg_gui.physical_rect_width_label.show()
            self.seg_gui.physical_rect_height_val.show()
            self.seg_gui.physical_rect_width_val.show()
            
        def hide_freehand_buttons(self):
            self.seg_gui.undo_last_pt_button.hide()
            self.seg_gui.close_roi_button.hide()
            self.seg_gui.save_freehand_button.hide()
            self.seg_gui.back_from_freehand_button.hide()
            self.seg_gui.draw_roi_button.hide()
            self.seg_gui.redraw_roi_button.hide()
            self.seg_gui.draw_freehand_buttons.hide()
            
        def back_from_freehand(self):
            self.undo_last_roi()
            self.hide_freehand_buttons()
            self.seg_gui.draw_freehand_button.show(); self.seg_gui.user_draw_rectangle_button.show()
            self.seg_gui.draw_roi_button.setChecked(False)
            self.draw_freehand_clicked()
            
        def save_roi_clicked(self):
            self.hide_rect_buttons()
            self.hide_freehand_buttons()
            self.show_save_roi_grid()
            
        def back_from_save(self):
            self.undo_last_roi()
            self.back_from_freehand()
            self.back_from_rect()
            self.hide_save_roi_grid()
            
        def show_save_roi_grid(self):
            self.seg_gui.save_roi_button.show(); self.seg_gui.back_from_save_button.show()
            self.seg_gui.dest_folder_label.show(); self.seg_gui.save_folder_input.show()
            self.seg_gui.roi_name_label.show(); self.seg_gui.save_name_input.show()
            self.seg_gui.choose_save_folder_button.show(); self.seg_gui.clear_save_folder_button.show()
            
        def hide_save_roi_grid(self):
            self.seg_gui.save_roi_button.hide(); self.seg_gui.back_from_save_button.hide()
            self.seg_gui.dest_folder_label.hide(); self.seg_gui.save_folder_input.hide()
            self.seg_gui.roi_name_label.hide(); self.seg_gui.save_name_input.hide()
            self.seg_gui.choose_save_folder_button.hide(); self.seg_gui.clear_save_folder_button.hide()
            self.seg_gui.save_folder_input.clear(); self.seg_gui.save_name_input.clear()
            
        def start_draw_freehand(self):
            self.hide_initial_buttons()
            self.show_freehanded_buttons()
            
        def show_freehanded_buttons(self):
            self.seg_gui.undo_last_pt_button.show()
            self.seg_gui.close_roi_button.show()
            self.seg_gui.save_freehand_button.show()
            self.seg_gui.back_from_freehand_button.show()
            self.seg_gui.draw_roi_button.show()
            self.seg_gui.draw_freehand_buttons.show()
            
        def hide_initial_buttons(self):
            self.seg_gui.draw_freehand_button.hide()
            self.seg_gui.user_draw_rectangle_button.hide()
            
        def select_save_folder(self):
            folder_name = QFileDialog.getExistingDirectory(None, "Select Folder")
            if folder_name != "":  # If valid folder is chosen
                self.seg_gui.save_folder_input.setText(folder_name)
            else:
                return
            
        def save_segmentation(self):
            if self.seg_gui.save_name_input.text() == "":
                self.seg_gui.select_seg_error_msg.setText("Please enter a name for the segmentation file.")
                self.seg_gui.select_seg_error_msg.show()
                return
            
            save_folder = Path(self.seg_gui.save_folder_input.text())
            save_folder.mkdir(parents=True, exist_ok=True)
            dest_path = save_folder / (self.seg_gui.save_name_input.text() + ".pkl")
            
            if not len(self.spline_x): # Rectangle ROI case
                left, bottom = self.seg_gui.ax_draw.patches[0].get_xy()
                left = int(left)
                bottom = int(bottom)
                width = int(self.seg_gui.ax_draw.patches[0].get_width())
                height = int(self.seg_gui.ax_draw.patches[0].get_height())
                self.points_plotted_x = (
                    list(range(left, left + width))
                    + list(np.ones(height).astype(int) * (left + width - 1))
                    + list(range(left + width - 1, left - 1, -1))
                    + list(np.ones(height).astype(int) * left)
                )
                self.points_plotted_y = (
                    list(np.ones(width).astype(int) * bottom)
                    + list(range(bottom, bottom + height))
                    + list(np.ones(width).astype(int) * (bottom + height - 1))
                    + list(range(bottom + height - 1, bottom - 1, -1))
                )
                self.spline_x = np.array(self.points_plotted_x)
                self.spline_y = np.array(self.points_plotted_y)
            
            saved_dict = {
                "Spline X": self.spline_x,
                "Spline Y": self.spline_y,
                "Scan Name": self.image_data.scan_name,
                "Phantom Name": self.image_data.phantom_name,
                "Frame": self.seg_gui.frame,
            }
            
            with open(dest_path, "wb") as f:
                pickle.dump(saved_dict, f)
                
            self.seg_gui.seg_path_input.setText(str(dest_path))
            self.seg_gui.seg_type = "pkl_roi"
            self.seg_gui.file_exts = [".pkl", ".pickle"]
            self.seg_gui.load_segmentation()
        
        
        
def calculate_spline(xpts, ypts):  # 2D spline interpolation
    cv = []
    for i in range(len(xpts)):
        cv.append([xpts[i], ypts[i]])
    cv = np.array(cv)
    if len(xpts) == 2:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=1)
    elif len(xpts) == 3:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=2)
    else:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=3)
    x, y = np.array(interpolate.splev(np.linspace(0, 1, 1000), tck))
    return x, y
