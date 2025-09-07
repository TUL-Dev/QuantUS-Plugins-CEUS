import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.interpolate as interpolate
from matplotlib.widgets import RectangleSelector, Cursor
import matplotlib.patches as patches

from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.uic.load_ui import loadUi

from quantus.seg_loading.options import get_seg_loaders
from quantus.entrypoints import seg_loading_step
from quantus.data_objs import UltrasoundRfImage, BmodeSeg


def select_seg_helper(path_input, file_exts):
    file_exts = " ".join([f"*{ext}" for ext in file_exts])
    if not os.path.exists(path_input.text()):  # check if file path is manually typed
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", filter=file_exts)
        if file_name != "":  # If valid file is chosen
            path_input.setText(file_name)
        else:
            return

from .roi_selection_ui import Ui_constructRoi
class SegSelectionGUI(QDialog, Ui_constructRoi):
    def __init__(self, image_data: UltrasoundRfImage):
        super(SegSelectionGUI, self).__init__()
        super().__init__()
        self.setupUi(self)
        # loadUi(str(Path("quantus/gui/seg_loading/roi_selection.ui")), self)

        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.removeItem(self.frame_preview_layout)
        self.hide_frame_preview_layout()
        self.full_screen_layout.removeItem(self.confirmation_layout)
        self.hide_seg_confirmation_layout()
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
        self.back_from_confirm_button.clicked.connect(self.back_to_select)
        self.frame_slider.valueChanged.connect(self.preview_frame_changed)
        self.accept_frame_button.clicked.connect(self.accept_preview_frame)
        
        self.image_data = image_data
        self.seg_drawing_screen_2d = None
        self.seg_data: BmodeSeg = None
        self.go_back = False
        self.frame = 0
        self.im_array = image_data.sc_bmode if image_data.sc_bmode is not None else image_data.bmode
        
        # Prepare B-Mode display plot
        self.bmode_confirmation_layout = QHBoxLayout(self.display_confirmed_frame)
        self.bmode_confirmation_layout.setObjectName("bmode_confirmation_layout")
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.bmode_confirmation_layout.addWidget(self.canvas)
        
        # Prepare B-Mode display plot
        self.bmode_draw_layout = QHBoxLayout(self.im_display_frame)
        self.bmode_draw_layout.setObjectName("bmode_draw_layout")
        self.figure_draw = plt.figure()
        self.canvas_draw = FigureCanvas(self.figure_draw)
        self.ax_draw = self.figure_draw.add_subplot(111)
        self.bmode_draw_layout.addWidget(self.canvas_draw)
            
    def hide_seg_confirmation_layout(self):
        self.segmentation_confirmation_label.hide()
        self.segmentation_name_val.hide()
        self.segmentation_name_label.hide()
        self.seg_confirmation_info.hide()
        self.confirm_seg_button.hide()
        self.back_from_confirm_button.hide()
        self.seg_options_buttons.hide()
        self.display_confirmed_frame.hide()
        self.confirmation_frame_label.hide()
        
    def show_seg_confirmation_layout(self):
        self.segmentation_confirmation_label.show()
        self.segmentation_name_val.show()
        self.segmentation_name_label.show()
        self.seg_confirmation_info.show()
        self.confirm_seg_button.show()
        self.back_from_confirm_button.show()
        self.seg_options_buttons.show()
        self.display_confirmed_frame.show()
        self.confirmation_frame_label.show()
        
    def hide_frame_preview_layout(self):
        self.select_frame_label.hide()
        self.im_preview.hide()
        self.frame_slider.hide()
        self.cur_frame_label.hide()
        self.of_frames_label.hide()
        self.total_frames_label.hide()
        self.accept_frame_button.hide()
        
    def show_frame_preview_layout(self):
        self.select_frame_label.show()
        self.im_preview.show()
        self.frame_slider.show()
        self.cur_frame_label.show()
        self.of_frames_label.show()
        self.total_frames_label.show()
        self.accept_frame_button.show()
            
    def hide_frame_selection_layout(self):
        self.select_frame_label.hide()
        
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
        
    def hide_draw_roi_layout(self):
        self.physical_depth_label.hide(); self.physical_depth_val.hide()
        self.physical_width_label.hide(); self.physical_width_val.hide()
        self.physical_dims_label.hide(); self.pixel_depth_label.hide()
        self.pixel_depth_val.hide(); self.pixel_width_label.hide()
        self.pixel_width_val.hide(); self.pixel_dims_label.hide()
        self.construct_roi_label.hide(); self.edit_image_display_button.hide()
        self.physical_rect_dims_label.hide(); self.physical_rect_height_label.hide()
        self.physical_rect_height_val.hide(); self.physical_rect_width_label.hide()
        self.physical_rect_width_val.hide(); self.draw_roi_button.hide()
        self.save_rect_button.hide(); self.back_from_freehand_button.hide()
        self.back_from_rectangle_button.hide(); self.close_roi_button.hide()
        self.draw_freehand_button.hide(); self.draw_rectangle_button.hide()
        self.redraw_roi_button.hide(); self.undo_last_pt_button.hide()
        self.user_draw_rectangle_button.hide(); self.draw_freehand_buttons.hide()
        self.im_display_frame.hide(); self.save_freehand_button.hide()
        self.draw_options_buttons.hide(); self.draw_rect_buttons.hide()
        self.save_roi_button.hide(); self.back_from_save_button.hide()
        self.dest_folder_label.hide(); self.save_folder_input.hide()
        self.roi_name_label.hide(); self.save_name_input.hide()
        self.choose_save_folder_button.hide(); self.clear_save_folder_button.hide()
        
    def show_draw_seg_start(self):
        self.physical_depth_label.show(); self.physical_depth_val.show()
        self.physical_width_label.show(); self.physical_width_val.show()
        self.physical_dims_label.show(); self.pixel_depth_label.show()
        self.pixel_depth_val.show(); self.pixel_width_label.show()
        self.pixel_width_val.show(); self.pixel_dims_label.show()
        self.draw_freehand_button.show(); self.user_draw_rectangle_button.show()
        self.im_display_frame.show(); self.draw_options_buttons.show()
        self.construct_roi_label.show()
        
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
            if self.im_array.ndim == 3: # need to select frame
                self.full_screen_layout.removeItem(self.select_type_layout)
                self.hide_type_selection_layout()
                self.full_screen_layout.addItem(self.frame_preview_layout)
                self.show_frame_preview_layout()
                self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
                self.full_screen_layout.setStretchFactor(self.frame_preview_layout, 10)
                
                self.displayed_im = np.array(self.im_array[self.frame]).reshape(self.im_array.shape[1], self.im_array.shape[2])
                self.displayed_im = np.require(self.displayed_im, np.uint8, 'C')
                self.bytes_line = self.displayed_im.strides[0]; self.ar_height, self.ar_width = self.displayed_im.shape
                self.q_im = QImage(self.displayed_im, self.ar_width, self.ar_height, self.bytes_line, QImage.Format.Format_Grayscale8)
                self.im_preview.setPixmap(QPixmap.fromImage(self.q_im).scaled(self.im_preview.width(), self.im_preview.height(), Qt.AspectRatioMode.IgnoreAspectRatio))
                
                self.total_frames_label.setText(str(self.im_array.shape[0]-1))
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(self.im_array.shape[0]-1)
                self.cur_frame_label.setText(str(self.frame))
                
            elif self.im_array.ndim == 2: # only need to draw ROI
                self.full_screen_layout.removeItem(self.select_type_layout)
                self.hide_type_selection_layout()
                self.displayed_im = self.im_array
                self.start_2d_seg_drawing()
                
    def accept_preview_frame(self):
        self.full_screen_layout.removeItem(self.frame_preview_layout)
        self.hide_frame_preview_layout()
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.back_to_preview)
        self.displayed_im = np.array(self.im_array[self.frame]).reshape(self.im_array.shape[1], self.im_array.shape[2])
        self.start_2d_seg_drawing()
        
    def back_to_preview(self):
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.accept_seg_type()
                
    def start_2d_seg_drawing(self):
        self.full_screen_layout.addItem(self.draw_roi_layout)
        self.show_draw_seg_start()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.draw_roi_layout, 10)
        
        if self.image_data.sc_bmode is not None:
            lateral_res = self.image_data.sc_lateral_res
            axial_res = self.image_data.sc_axial_res
        else:
            lateral_res = self.image_data.lateral_res
            axial_res = self.image_data.axial_res
        
        self.physical_width_val.setText(
            str(np.round(lateral_res*self.displayed_im.shape[1]/10, decimals=2))
        )
        self.physical_depth_val.setText(
            str(np.round(axial_res*self.displayed_im.shape[0]/10, decimals=2))
        )
        self.pixel_width_val.setText(str(self.displayed_im.shape[1]))
        self.pixel_depth_val.setText(str(self.displayed_im.shape[0]))
        
        if self.seg_drawing_screen_2d is not None:
            self.draw_freehand_button.clicked.disconnect(); self.user_draw_rectangle_button.clicked.disconnect()
            self.draw_rectangle_button.clicked.disconnect(); self.draw_roi_button.clicked.disconnect()
            self.back_from_rectangle_button.clicked.disconnect(); self.undo_last_pt_button.clicked.disconnect()
            self.close_roi_button.clicked.disconnect(); self.redraw_roi_button.clicked.disconnect()
            self.back_from_freehand_button.clicked.disconnect(); self.save_freehand_button.clicked.disconnect()
            self.save_rect_button.clicked.disconnect(); self.back_from_save_button.clicked.disconnect()
            self.choose_save_folder_button.clicked.disconnect(); self.clear_save_folder_button.clicked.disconnect()
            self.save_roi_button.clicked.disconnect()
            del self.seg_drawing_screen_2d
            self.seg_drawing_screen_2d = None
        
        self.seg_drawing_screen_2d = SegSelectionGUI.DrawNewSeg2d(self.image_data, self.displayed_im, self)
        self.seg_drawing_screen_2d.plot_on_canvas()
        
                
    def preview_frame_changed(self, value):
        self.frame = value
        self.cur_frame_label.setText(str(value))
        self.displayed_im = np.array(self.im_array[self.frame]).reshape(self.im_array.shape[1], self.im_array.shape[2])
        self.displayed_im = np.require(self.displayed_im, np.uint8, 'C')
        self.bytes_line = self.displayed_im.strides[0]; self.ar_height, self.ar_width = self.displayed_im.shape
        self.q_im = QImage(self.displayed_im, self.ar_width, self.ar_height, self.bytes_line, QImage.Format.Format_Grayscale8)
        self.im_preview.setPixmap(QPixmap.fromImage(self.q_im).scaled(self.im_preview.width(), self.im_preview.height(), Qt.AspectRatioMode.IgnoreAspectRatio))
                
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
        self.frame = self.seg_data.frame
        bmode = self.image_data.sc_bmode if self.image_data.sc_bmode is not None else self.image_data.bmode
        
        if self.image_data.bmode.ndim == 3:
            self.displayed_im = bmode[self.frame]
        else:
            self.displayed_im = bmode
        
        self.move_to_confirmation()
                
    def move_to_confirmation(self):
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.addItem(self.confirmation_layout)
        self.show_seg_confirmation_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.confirmation_layout, 10)
        
        self.segmentation_name_val.setText(self.seg_data.seg_name)
        self.confirmation_frame_label.setText(f"Frame: {self.frame}")
        
        if len(self.seg_data.splines) == 2:
            if self.image_data.sc_bmode is not None:
                self.spline_x = self.seg_data.sc_splines[0]
                self.spline_y = self.seg_data.sc_splines[1]
            else:
                self.spline_x = self.seg_data.splines[0]
                self.spline_y = self.seg_data.splines[1]
        
        self.plot_on_canvas()
            
    def plot_on_canvas(self):
        self.ax.clear()
        if self.image_data.sc_bmode is not None:
            width = self.displayed_im.shape[1]*self.image_data.sc_lateral_res
            height = self.displayed_im.shape[0]*self.image_data.sc_axial_res
        else:
            width = self.displayed_im.shape[1]*self.image_data.lateral_res
            height = self.displayed_im.shape[0]*self.image_data.axial_res
        aspect = width/height
        im = self.ax.imshow(self.displayed_im, cmap="gray")
        extent = im.get_extent()
        self.ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        self.figure.set_facecolor((0, 0, 0, 0))
        self.ax.axis("off")
        
        if hasattr(self, "spline_x") and hasattr(self, "spline_y"):
            self.spline = self.ax.plot(self.spline_x, self.spline_y, 
                                    color="cyan", zorder=1, linewidth=0.75)

        self.figure.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
        )
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        self.canvas.draw()  # Refresh canvas
            
        
    class DrawNewSeg2d:
        """Class to draw new segmentation on 2D image.
        """
        
        def __init__(self, image_data: UltrasoundRfImage, displayed_im: np.ndarray, seg_gui: "SegSelectionGUI"):
            self.image_data = image_data
            self.displayed_im = displayed_im
            self.seg_gui = seg_gui
            self.spline_x = np.array([]); self.spline_y = np.array([])
            self.points_plotted_x = []; self.points_plotted_y = []
            self.scattered_points = []; self.rect_coords = []
            
            self.selector = RectangleSelector(
                self.seg_gui.ax_draw,
                self.draw_rect,
                useblit=True,
                props=dict(linestyle="-", color="cyan", fill=False),
            )
            self.selector.set_active(False)
            
            self.seg_gui.draw_freehand_button.clicked.connect(self.start_draw_freehand)
            self.seg_gui.user_draw_rectangle_button.clicked.connect(self.start_draw_rect)
            self.seg_gui.draw_rectangle_button.clicked.connect(self.draw_rect_clicked)
            self.seg_gui.draw_roi_button.clicked.connect(self.draw_freehand_clicked)
            self.seg_gui.back_from_rectangle_button.clicked.connect(self.back_from_rect)
            self.seg_gui.undo_last_pt_button.clicked.connect(self.undo_last_pt)
            self.seg_gui.close_roi_button.clicked.connect(self.close_interpolation)
            self.seg_gui.redraw_roi_button.clicked.connect(self.undo_last_roi)
            self.seg_gui.back_from_freehand_button.clicked.connect(self.back_from_freehand)
            self.seg_gui.save_freehand_button.clicked.connect(self.save_roi_clicked)
            self.seg_gui.save_rect_button.clicked.connect(self.save_roi_clicked)
            self.seg_gui.back_from_save_button.clicked.connect(self.back_from_save)
            self.seg_gui.choose_save_folder_button.clicked.connect(self.select_save_folder)
            self.seg_gui.clear_save_folder_button.clicked.connect(self.seg_gui.save_folder_input.clear)
            self.seg_gui.save_roi_button.clicked.connect(self.save_segmentation)
            
            
        def draw_rect(self, event1, event2):
            self.rect_coords = [
                int(event1.xdata),
                int(event1.ydata),
                int(event2.xdata),
                int(event2.ydata),
            ]
            self.plot_patch()
            
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
