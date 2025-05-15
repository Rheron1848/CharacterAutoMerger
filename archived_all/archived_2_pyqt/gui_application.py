# gui_application.py

import sys
import os
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSizePolicy, QMessageBox, QListWidget, QListWidgetItem,
    QGraphicsPolygonItem, QGraphicsEllipseItem, QColorDialog, QScrollArea,
    QFormLayout, QLineEdit, QDoubleSpinBox, QCheckBox, QGroupBox, QToolBar,
    QSpacerItem, QFrame, QGraphicsSimpleTextItem, QGraphicsLineItem
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QTransform, QPolygonF,
    QAction, QIcon # Will need icons later
)
from PyQt6.QtCore import Qt, QSize, QPointF, QRectF, QLineF, pyqtSignal, QObject, QTimer

import cv2
import numpy as np
import traceback
import torch # Added for real model inference

# Attempt to import mmdet and mmpose related modules
# These are essential for the RealPoseDetector
# REMOVED try-except block for graceful degradation. Application will now fail fast if dependencies are missing.
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules
from mmengine.registry import DefaultScope
# from mmpose.structures import PoseDataSample # For type hints if needed
# from mmdet.structures import DetDataSample # For type hints if needed
# MMDET_MMPOS_AVAILABLE can be considered True if imports succeed, or removed if RealPoseDetector handles model init status.
# For simplicity, we'll remove it and let RealPoseDetector handle its state.
print("MMDetection, MMPose, and MMEngine components are expected to be loadable for full functionality.")


# --- Configuration (User might need to adjust these) ---
DWPOSE_DIR = r"D:\\drafts\\DWPose" 
DEFAULT_EXPORT_DIR = os.path.join(os.getcwd(), "CharacterAutoMerger_Exports_NewGUI")

# Model configurations, similar to CharacterAutoMerger.py's Config
MODEL_CONFIGS = {
    'det_config_filename': 'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py',
    'det_checkpoint_filename': 'weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
    'pose_config_filename': 'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
    'pose_checkpoint_filename': 'weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'det_confidence_threshold': 0.3, # Default person detection threshold for RTMDet
}

# Helper to get full model paths
def get_model_path(base_dir, file_path_suffix):
    return os.path.join(base_dir, file_path_suffix)


# Constants for interaction modes or steps
APP_MODE_UPLOAD = "upload"
APP_MODE_ADJUST_SKELETON = "adjust_skeleton"
APP_MODE_ADJUST_SEGMENTATION = "adjust_segmentation"
APP_MODE_EDIT_PARTS = "edit_parts"

IMAGE_TYPE_TEMPLATE = "template"
IMAGE_TYPE_HEAD = "head"
IMAGE_TYPE_BODY = "body"

# == Real Implementations for Core Logic ==

class RealPoseDetector:
    def __init__(self, dwpose_base_dir, model_configs_dict):
        self.dwpose_base_dir = dwpose_base_dir
        self.model_configs = model_configs_dict
        self.device = self.model_configs['device']
        self.detector = None
        self.pose_estimator = None
        # REMOVED: No longer checking MMDET_MMPOS_AVAILABLE. Always attempt to initialize.
        # Error handling is within _init_models and when detect_pose checks model readiness.
        self._init_models()
        # if MMDET_MMPOS_AVAILABLE:
        #     self._init_models()
        # else:
        #     print("RealPoseDetector: MMDetection/MMPose not available. Cannot initialize models.")

    def _init_models(self):
        try:
            # Register modules if not already done (idempotent)
            register_det_modules()
            register_pose_modules()

            det_config_path = get_model_path(self.dwpose_base_dir, self.model_configs['det_config_filename'])
            det_checkpoint_path = get_model_path(self.dwpose_base_dir, self.model_configs['det_checkpoint_filename'])
            pose_config_path = get_model_path(self.dwpose_base_dir, self.model_configs['pose_config_filename'])
            pose_checkpoint_path = get_model_path(self.dwpose_base_dir, self.model_configs['pose_checkpoint_filename'])

            for p in [det_config_path, det_checkpoint_path, pose_config_path, pose_checkpoint_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Required model file not found: {p}")

            self.detector = init_detector(det_config_path, det_checkpoint_path, device=self.device)
            self.pose_estimator = init_pose_estimator(pose_config_path, pose_checkpoint_path, device=self.device)
            print(f"RealPoseDetector: Models initialized successfully on device '{self.device}'.")
        except Exception as e:
            print(f"RealPoseDetector: Error initializing models: {e}")
            traceback.print_exc()
            self.detector = None # Ensure models are None if init fails
            self.pose_estimator = None
            # QMessageBox.critical(None, "Model Initialization Error", f"Failed to initialize pose detection models: {e}")


    def detect_pose(self, image_cv, image_type_str="unknown"):
        print(f"RealPoseDetector: Detecting pose for {image_type_str}...")
        h_orig, w_orig = image_cv.shape[:2]

        # MODIFIED: Simplified check, relies on models being None if _init_models failed.
        if self.detector is None or self.pose_estimator is None:
            print("RealPoseDetector: Models not available or not initialized. Returning empty keypoints.")
            # Return structure compatible with GUI's expectations
            return {'keypoints': [], 'source_image_size': (w_orig, h_orig)}
        
        try:
            # Person detection using RTMDet (expects BGR numpy array)
            # 替换旧的API调用
            # 使用检测器前注册检测相关模块
            register_det_modules()  # 在调用检测器前注册检测模块
            
            # 使用检测器获取人体边界框
            det_results_datasample = inference_detector(self.detector, image_cv) 
            
            pred_instances = det_results_datasample.pred_instances
            # Filter bboxes by score
            bboxes_with_scores = pred_instances.bboxes[pred_instances.scores > self.model_configs['det_confidence_threshold']]

            if bboxes_with_scores.shape[0] == 0:
                print(f"RealPoseDetector: No person detected with confidence > {self.model_configs['det_confidence_threshold']} for {image_type_str}.")
                return {'keypoints': [], 'source_image_size': (w_orig, h_orig)}

            # Convert bboxes tensor to numpy if it's not already for MMPose
            if isinstance(bboxes_with_scores, torch.Tensor):
                bboxes_np = bboxes_with_scores.cpu().numpy()
            else:
                bboxes_np = bboxes_with_scores 

            # 使用姿态估计器前注册姿态相关模块
            register_pose_modules()  # 在调用姿态估计器前注册姿态模块
            
            # Pose estimation using RTMPose (expects BGR numpy array and bboxes)
            pose_results_datasamples = inference_topdown(self.pose_estimator, image_cv, bboxes_np)
            
            processed_keypoints = []
            if pose_results_datasamples:
                # Assuming the first detected person is the target
                # MMPose returns a list of PoseDataSample objects
                main_person_pose_instances = pose_results_datasamples[0].pred_instances
                
                # keypoints tensor shape: (num_instances, num_keypoints, 2)
                # keypoint_scores tensor shape: (num_instances, num_keypoints)
                # For single person after bbox selection, num_instances should be 1
                keypoints_tensor = main_person_pose_instances.keypoints[0] 
                keypoint_scores_tensor = main_person_pose_instances.keypoint_scores[0]

                # 检查是否为PyTorch张量，如果是则转换为numpy数组，否则直接使用
                if isinstance(keypoints_tensor, torch.Tensor):
                    kps_np = keypoints_tensor.cpu().numpy()
                else:
                    kps_np = keypoints_tensor  # 假设已经是numpy数组
                
                if isinstance(keypoint_scores_tensor, torch.Tensor):
                    scores_np = keypoint_scores_tensor.cpu().numpy()
                else:
                    scores_np = keypoint_scores_tensor  # 假设已经是numpy数组

                for i in range(kps_np.shape[0]): # Iterate over number of keypoints
                    processed_keypoints.append([float(kps_np[i, 0]), float(kps_np[i, 1]), float(scores_np[i])])
            
            print(f"RealPoseDetector: Found {len(processed_keypoints)} keypoints for {image_type_str}.")
            return {
                'keypoints': processed_keypoints, # List of [x, y, confidence]
                'source_image_size': (w_orig, h_orig)
            }

        except Exception as e:
            print(f"RealPoseDetector: Error during pose detection for {image_type_str}: {e}")
            traceback.print_exc()
            try:
                # 确保在异常情况下也恢复原始scope
                default_scope = DefaultScope.get_instance()
                default_scope.scope_name = original_scope
            except:
                pass # Ignore any error if no scope to restore
            return {'keypoints': [], 'source_image_size': (w_orig, h_orig)}

class RealPartSegmenter:
    def __init__(self, config=None): # Config might hold thresholds if needed
        print("RealPartSegmenter initialized.")

    def pre_segment(self, image_cv, keypoints_data, part_definitions):
        print("RealPartSegmenter: Pre-segmenting parts based on keypoints...")
        # keypoints_data: {'keypoints': [[x,y,c], ...], 'source_image_size': (w,h)}
        # part_definitions: {'Head': [0,1,2,3,4], 'Torso': [5,6,11,12], ...}
        
        segmented_parts_data = {}
        all_keypoints = keypoints_data.get('keypoints', [])

        for part_name, kp_indices in part_definitions.items():
            relevant_qpoints = [] # QPointF for QPolygonF
            original_kps_for_part_data = [] # Store [x,y,c]

            for idx in kp_indices:
                if 0 <= idx < len(all_keypoints):
                    kp = all_keypoints[idx] # [x,y,c]
                    if kp[2] > 0.1: # Confidence threshold for including a keypoint in polygon definition
                        relevant_qpoints.append(QPointF(kp[0], kp[1]))
                        original_kps_for_part_data.append(kp)
            
            if not relevant_qpoints:
                print(f"No relevant keypoints with sufficient confidence for {part_name}, skipping polygon generation.")
                continue

            poly = QPolygonF()
            if len(relevant_qpoints) < 3: 
                # Fallback to bounding box if less than 3 points
                if relevant_qpoints: # Need at least one point for bbox
                    xs = [pt.x() for pt in relevant_qpoints]
                    ys = [pt.y() for pt in relevant_qpoints]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    poly.append(QPointF(min_x, min_y))
                    poly.append(QPointF(max_x, min_y))
                    poly.append(QPointF(max_x, max_y))
                    poly.append(QPointF(min_x, max_y))
                else: # Should not happen due to check above, but as a safeguard
                    continue
            else:
                # Create convex hull for a tighter fit
                np_points = np.array([[pt.x(), pt.y()] for pt in relevant_qpoints], dtype=np.float32)
                try:
                    hull_indices = cv2.convexHull(np_points, returnPoints=False)
                    if hull_indices is not None and len(hull_indices) >= 3:
                        hull_points = np_points[hull_indices.flatten()]
                        for pt_hull in hull_points:
                            poly.append(QPointF(float(pt_hull[0]), float(pt_hull[1])))
                    else: # Fallback if hull fails or gives too few points
                        raise ValueError("Convex hull resulted in <3 points or failed.")
                except Exception as e_hull: # Catch errors from convexHull (e.g. colinearity)
                    print(f"Convex hull failed for {part_name} ({e_hull}), using bounding box of relevant keypoints.")
                    xs = [pt.x() for pt in relevant_qpoints] # Use relevant_qpoints
                    ys = [pt.y() for pt in relevant_qpoints]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    poly.append(QPointF(min_x, min_y))
                    poly.append(QPointF(max_x, min_y))
                    poly.append(QPointF(max_x, max_y))
                    poly.append(QPointF(min_x, max_y))
            
            if poly.isEmpty():
                print(f"Polygon for {part_name} is empty after generation, skipping.")
                continue

            bounding_rect = poly.boundingRect()
            segmented_parts_data[part_name] = {
                'initial_polygon': poly,
                'original_keypoints_for_part': original_kps_for_part_data,
                # Mock compatible fields, might not be used if logic is self-contained
                'cropped_image_bgra': np.zeros((1,1,4), dtype=np.uint8), 
                'crop_rect': bounding_rect 
            }
        print(f"RealPartSegmenter: Generated {len(segmented_parts_data)} initial part polygons.")
        return segmented_parts_data

    def final_segment(self, original_image_cv, part_name, edited_polygon_qpolygonf):
        print(f"RealPartSegmenter: Final segmenting part '{part_name}' using edited polygon.")
        if edited_polygon_qpolygonf.isEmpty():
            print(f"Warning: Polygon for {part_name} is empty. Cannot segment.")
            return None, None 

        img_h, img_w = original_image_cv.shape[:2]
        bounding_rect_from_poly = edited_polygon_qpolygonf.boundingRect()
        
        # Calculate crop coordinates, ensuring they are within image bounds and valid
        x = max(0, int(bounding_rect_from_poly.x()))
        y = max(0, int(bounding_rect_from_poly.y()))
        # Calculate width and height based on clamped x,y and original image dimensions
        # Ensure right and bottom do not exceed image dimensions
        poly_br_x = bounding_rect_from_poly.x() + bounding_rect_from_poly.width()
        poly_br_y = bounding_rect_from_poly.y() + bounding_rect_from_poly.height()

        crop_w = min(img_w - x, int(max(0, poly_br_x - x)))
        crop_h = min(img_h - y, int(max(0, poly_br_y - y)))

        if crop_w <= 0 or crop_h <= 0:
            print(f"Warning: Invalid bounding box for {part_name} after clamping: x={x},y={y},w={crop_w},h={crop_h}. Cannot segment.")
            return None, None

        # Create a full-size mask first
        mask = np.zeros(original_image_cv.shape[:2], dtype=np.uint8)
        
        cv_contour_points = []
        for i in range(edited_polygon_qpolygonf.count()):
            pt = edited_polygon_qpolygonf.at(i)
            # Ensure points are within image bounds for fillPoly
            px = int(np.clip(pt.x(), 0, img_w -1))
            py = int(np.clip(pt.y(), 0, img_h -1))
            cv_contour_points.append([px, py])
        
        if not cv_contour_points or len(cv_contour_points) < 3:
            print(f"Warning: Not enough valid points in polygon for {part_name} to create mask. Points: {cv_contour_points}")
            return None, None

        cv_contour_np = np.array([cv_contour_points], dtype=np.int32)
        cv2.fillPoly(mask, cv_contour_np, 255)

        # Crop original image and the mask using calculated x,y,w,h
        cropped_original = original_image_cv[y : y + crop_h, x : x + crop_w]
        cropped_mask = mask[y : y + crop_h, x : x + crop_w]

        if cropped_original.size == 0 or cropped_mask.size == 0:
            print(f"Warning: Cropped image or mask is empty for {part_name}. Crop rect: x={x},y={y},w={crop_w},h={crop_h}")
            return None, None

        # Apply mask
        if cropped_original.ndim == 2: 
            bgra_part = cv2.cvtColor(cropped_original, cv2.COLOR_GRAY2BGRA)
        elif cropped_original.shape[2] == 3: 
            bgra_part = cv2.cvtColor(cropped_original, cv2.COLOR_BGR2BGRA)
        elif cropped_original.shape[2] == 4: 
            bgra_part = cropped_original.copy() # Assume it's already BGRA
        else:
            print(f"Warning: Unsupported number of channels {cropped_original.shape[2]} for part {part_name}.")
            return None, None
        
        if bgra_part.shape[2] == 4: # Ensure alpha channel exists
             bgra_part[:, :, 3] = cropped_mask
        else: 
            print(f"Warning: BGRA part does not have 4 channels for {part_name} after conversion.")
            return None, None

        position_info = {
            'crop_rect_qrectf': QRectF(float(x), float(y), float(crop_w), float(crop_h)), 
            'part_name': part_name,
        }
        print(f"RealPartSegmenter: Final segmentation for {part_name} done. Output BGRA shape: {bgra_part.shape}")
        return bgra_part, position_info


class RealWarper:
    def __init__(self, config=None):
        print("RealWarper initialized.")

    def calculate_initial_transform(self, part_name: str,
                                    part_definitions: dict, 
                                    template_kp_data: dict, 
                                    part_source_kp_data: dict 
                                   ):
        print(f"RealWarper: Calculating initial transform for '{part_name}'.")
        default_pos = QPointF(50.0, 50.0)
        default_scale = 1.0
        default_rotation = 0.0

        if not template_kp_data or not template_kp_data.get('keypoints') or \
           not part_source_kp_data or not part_source_kp_data.get('keypoints'):
            print(f"Warning (RealWarper): Missing keypoint data for template or part source ('{part_name}'). Using default transform.")
            return default_pos, default_scale, default_scale, default_rotation

        part_kp_indices = part_definitions.get(part_name)
        if not part_kp_indices:
            print(f"Warning (RealWarper): No keypoint indices defined for part '{part_name}'. Using default transform.")
            return default_pos, default_scale, default_scale, default_rotation

        template_all_kps = template_kp_data['keypoints']
        part_src_all_kps = part_source_kp_data['keypoints']

        relevant_template_pts_np = []
        for idx in part_kp_indices:
            if 0 <= idx < len(template_all_kps) and template_all_kps[idx][2] > 0.2: # Confidence threshold
                relevant_template_pts_np.append(np.array(template_all_kps[idx][:2])) # x,y

        relevant_part_src_pts_np = []
        for idx in part_kp_indices:
            if 0 <= idx < len(part_src_all_kps) and part_src_all_kps[idx][2] > 0.2: # Confidence threshold
                relevant_part_src_pts_np.append(np.array(part_src_all_kps[idx][:2])) # x,y
        
        if len(relevant_template_pts_np) < 1 or len(relevant_part_src_pts_np) < 1:
            print(f"Warning (RealWarper): Not enough valid keypoints for '{part_name}' in template ({len(relevant_template_pts_np)}) or source ({len(relevant_part_src_pts_np)}). Using default transform.")
            return default_pos, default_scale, default_scale, default_rotation

        # Calculate centroids
        centroid_template_np = np.mean(relevant_template_pts_np, axis=0)
        # centroid_part_src_np = np.mean(relevant_part_src_pts_np, axis=0) # Not directly used for position like this
        
        # Calculate scale based on span of keypoints
        current_scale = default_scale
        def get_max_span(points_np_list):
            if not points_np_list: return 1.0
            np_array = np.array(points_np_list)
            min_coords = np.min(np_array, axis=0)
            max_coords = np.max(np_array, axis=0)
            spans = max_coords - min_coords
            return np.max(spans) if spans.size > 0 else 1.0

        span_template = get_max_span(relevant_template_pts_np)
        span_part_src = get_max_span(relevant_part_src_pts_np)

        if span_part_src > 1e-3: # Avoid division by zero and very small spans
            current_scale = span_template / span_part_src
        current_scale = np.clip(current_scale, 0.2, 3.0) # Clamp scale to reasonable bounds

        # Position: Place the center of the part (as defined by InteractivePartItem's transformOrigin)
        # at the centroid of the template keypoints for that part.
        # The setPos() method of QGraphicsItem sets its top-left position in parent coordinates.
        # If transformOriginPoint is the center of the item, setPos(target_center - scaled_offset_to_center)
        # However, InteractivePartItem uses setTransformOriginPoint(self.local_rect.center()).
        # So, if we call item.setPos(X, Y), its center effectively lands at (X,Y) if origin is 0,0.
        # With origin at center, item.setPos(target_center_in_scene) should place the item's center at target_center_in_scene.

        final_pos = QPointF(float(centroid_template_np[0]), float(centroid_template_np[1]))
        
        print(f"RealWarper: part='{part_name}', pos=({final_pos.x():.2f}, {final_pos.y():.2f}), scale={current_scale:.3f}, rot={default_rotation:.2f}")
        return final_pos, float(current_scale), float(current_scale), float(default_rotation)

# == End Real Implementations ==


# == Custom QGraphicsItems ==
class EditablePointItem(QGraphicsEllipseItem):
    """ A QGraphicsItem for a draggable keypoint. """
    def __init__(self, x, y, radius, kp_index, label_text="", color=QColor("red"), parent=None):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, parent)
        self.setPos(x, y)
        self.kp_index = kp_index
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.initial_pos = QPointF(x,y)
        self.radius = radius # Store radius

        if label_text:
            self.label_item = QGraphicsSimpleTextItem(label_text, self)
            # Position the label, e.g., slightly above and to the right of the ellipse's top-right
            label_offset_x = self.radius * 0.8 # Adjust as needed
            label_offset_y = -self.radius * 2.2 # Adjust as needed, negative Y is up
            self.label_item.setPos(label_offset_x, label_offset_y)
            self.label_item.setBrush(QBrush(Qt.GlobalColor.black)) # Text color
            self.label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton) # Pass clicks through
            # from PyQt6.QtGui import QFont # Import QFont if not already available
            # self.label_item.setFont(QFont("Arial", 7)) # Optional: set font smaller

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(self.brush().color().lighter(130)))
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(self.brush().color().darker(130))) # Revert to original or slightly darker
        QApplication.restoreOverrideCursor()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            # Optional: Emit a signal here if the main window needs to know about the change immediately
            # self.parent_widget.keypoint_moved_signal.emit(self.kp_index, self.pos())
            pass
        return super().itemChange(change, value)

    def get_current_pos(self) -> QPointF:
        return self.pos()

    def reset_to_initial_pos(self):
        self.setPos(self.initial_pos)

# MODIFIED: Inherit from QObject first for signals
class EditablePolygonItem(QObject, QGraphicsPolygonItem):
    """
    A QGraphicsPolygonItem that allows its vertices to be moved.
    (Adapted from user's EditableSegmentItem with some modifications)
    """
    handle_size = 10.0
    handle_brush_default = QBrush(QColor(Qt.GlobalColor.darkCyan))
    handle_brush_hover = QBrush(QColor(Qt.GlobalColor.cyan))
    handle_pen = QPen(QColor(Qt.GlobalColor.black), 1)
    polygon_pen_default = QPen(QColor(Qt.GlobalColor.gray), 2, Qt.PenStyle.DashLine)
    polygon_pen_selected = QPen(QColor(Qt.GlobalColor.blue), 2, Qt.PenStyle.SolidLine)
    polygon_fill_color_default = QColor(128, 128, 128, 80)
    polygon_changed = pyqtSignal()

    def __init__(self, initial_polygon_qpolygonf: QPolygonF, part_name: str = "segment", parent=None):
        QObject.__init__(self)
        QGraphicsPolygonItem.__init__(self, parent)
        
        # --- Test with a known simple polygon FIRST ---
        simple_triangle = QPolygonF()
        simple_triangle.append(QPointF(10.0, 10.0))  # Using simple, clearly valid float coords
        simple_triangle.append(QPointF(100.0, 10.0))
        simple_triangle.append(QPointF(55.0, 100.0))
        
        print(f"    EditablePolygonItem '{part_name}': Before self.setPolygon() with HARDCODED SIMPLE TRIANGLE. Polygon to set: isEmpty={simple_triangle.isEmpty()}, count={simple_triangle.count()}")
        try:
            self.setPolygon(simple_triangle)
            print(f"    EditablePolygonItem '{part_name}': After self.setPolygon() with HARDCODED SIMPLE TRIANGLE. Item polygon: isEmpty={self.polygon().isEmpty()}, count={self.polygon().count()}")
        except Exception as e_simple_poly:
            print(f"    EditablePolygonItem '{part_name}': CRITICAL ERROR setting HARDCODED SIMPLE TRIANGLE: {e_simple_poly}")
            traceback.print_exc()
            # If this simple triangle itself fails, the problem is more fundamental with setPolygon in this context.
            # We should probably not proceed with the original polygon if this fails.

        # --- Then attempt with the original/copied polygon (currently commented out to isolate the simple triangle test) ---
        # print(f"    EditablePolygonItem '{part_name}': Before self.setPolygon(). Original Polygon to set: isEmpty={initial_polygon_qpolygonf.isEmpty()}, count={initial_polygon_qpolygonf.count()}")
        # polygon_to_set = QPolygonF(initial_polygon_qpolygonf)
        # print(f"    EditablePolygonItem '{part_name}': Created copy for setPolygon. Copied Polygon: isEmpty={polygon_to_set.isEmpty()}, count={polygon_to_set.count()}")
        # self.setPolygon(polygon_to_set) # Use the copy
        # print(f"    EditablePolygonItem '{part_name}': After self.setPolygon(). Current item polygon: isEmpty={self.polygon().isEmpty()}, count={self.polygon().count()}")
        
        self.part_name = part_name
        # self._polygon_points = QPolygonF(initial_polygon_qpolygonf) # Temporarily disable this too, to isolate further

        # --- Start of temporarily commented out section ---
        # print(f"EditablePolygonItem '{part_name}': About to set flags...")
        # self.setFlags(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable |
        #               QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable |
        #               QGraphicsPolygonItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        # print(f"EditablePolygonItem '{part_name}': Flags set. About to setAcceptHoverEvents...")
        # self.setAcceptHoverEvents(True)
        # print(f"EditablePolygonItem '{part_name}': HoverEvents set. About to setBrush...")
        # self.setBrush(QBrush(self.polygon_fill_color_default))
        # print(f"EditablePolygonItem '{part_name}': Brush set. About to setPen...")
        # self.setPen(self.polygon_pen_default)
        # print(f"EditablePolygonItem '{part_name}': Pen set.")

        # self.hovered_vertex_index = -1
        # self.dragged_vertex_index = -1
        # self.initial_mouse_item_pos = QPointF()
        # self.initial_vertex_pos = QPointF()
        # --- End of temporarily commented out section ---
        print(f"EditablePolygonItem '{part_name}': __init__ completed (mostly commented out, simple triangle test was run).")

    def get_handle_rect_at_vertex(self, index: int) -> QRectF:
        # If _polygon_points is disabled, this will fail. For now, this method won't be called if items are not interactable.
        if hasattr(self, '_polygon_points') and self._polygon_points and 0 <= index < self._polygon_points.count():
            vertex = self._polygon_points.at(index)
            half_handle = self.handle_size / 2.0
            return QRectF(vertex.x() - half_handle, vertex.y() - half_handle, self.handle_size, self.handle_size)
        return QRectF() # Return empty if _polygon_points is not available

    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget) # Draws the polygon based on self.polygon()
        
        # Temporarily disable handle painting if flags/selection are off
        # if self.isSelected() and hasattr(self, '_polygon_points') and self._polygon_points:
        #     self.setPen(self.polygon_pen_selected) # This might error if self.polygon_pen_selected is not defined due to commenting
        #     painter.setPen(self.handle_pen)
        #     for i in range(self._polygon_points.count()):
        #         handle_rect = self.get_handle_rect_at_vertex(i)
        #         if i == self.hovered_vertex_index:
        #             painter.setBrush(self.handle_brush_hover)
        #         else:
        #             painter.setBrush(self.handle_brush_default)
        #         painter.drawEllipse(handle_rect)
        # else:
        #     # self.setPen(self.polygon_pen_default) # This might error if self.polygon_pen_default is not defined
        #     pass # Keep original pen if not selected or interactive elements are off

    def hoverMoveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        # if not self.flags() & QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable: # If not selectable, no hover logic
        #     super().hoverMoveEvent(event)
        #     return
        # Temporarily simplified
        super().hoverMoveEvent(event) 

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        # if not self.flags() & QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable: # If not selectable, no press logic
        #    super().mousePressEvent(event)
        #    return
        # Temporarily simplified
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        # if not self.flags() & QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable: # If not movable, no move logic
        #    super().mouseMoveEvent(event)
        #    return
        # Temporarily simplified
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        # if not self.flags() & QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable: # If not selectable, no release logic
        #     super().mouseReleaseEvent(event)
        #     return
        # Temporarily simplified
        super().mouseReleaseEvent(event)

    def get_edited_qpolygonf(self) -> QPolygonF:
        if hasattr(self, '_polygon_points') and self._polygon_points:
            return QPolygonF(self._polygon_points)
        # Fallback or error if _polygon_points is not initialized due to commenting out
        print("Warning: get_edited_qpolygonf called but _polygon_points might not be initialized!")
        return self.polygon() # Return the base polygon if _polygon_points is unavailable

    def set_fill_color(self, color: QColor):
        self.setBrush(QBrush(color)) # This should still work
        self.update()

class InteractivePartItem(QObject, QGraphicsPixmapItem): # Also ensure QObject for signals here
    """
    A QGraphicsPixmapItem that can be moved, scaled, and rotated.
    (Adapted from user's InteractivePartItem, aiming for clearer transform management)
    """
    # Handle types (simplified for now, can be expanded)
    Handle_Rotate, Handle_Scale_BR = range(2) # Bottom-right scale, top rotate
    handle_size = 12.0
    rotation_handle_offset = 20.0 # Distance of rotation handle from top-center

    # Signals
    transform_changed = pyqtSignal(str) # part_name

    def __init__(self, pixmap: QPixmap, part_name: str,
                 original_cv_image_bgra: np.ndarray, # Store for potential re-processing or export
                 original_crop_info: dict, # Store crop x,y,w,h relative to its source image
                 parent=None):
        QObject.__init__(self)
        # MODIFIED: Call parent-only constructor for QGraphicsPixmapItem, then set pixmap
        QGraphicsPixmapItem.__init__(self, parent)
        self.setPixmap(pixmap)

        self.part_name = part_name
        self.original_cv_image_bgra = original_cv_image_bgra
        self.original_crop_info = original_crop_info # Store crop info for potential re-processing

        self.setFlags(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        # self.setCacheMode(self.CacheMode.DeviceCoordinateCache) # Can help with performance
        self.setOpacity(0.85) # Semi-transparent during editing

        # Store the original pixmap rectangle (in its local 0,0 coordinates)
        self.local_rect = QRectF(self.pixmap().rect())
        self.setTransformOriginPoint(self.local_rect.center()) # Default origin

        # Interaction state
        self.current_interaction_mode = None # e.g., Handle_Rotate
        self.mouse_press_item_pos = QPointF()   # Mouse pos in item coords at press
        self.mouse_press_scene_pos = QPointF()  # Mouse pos in scene coords at press
        self.item_press_pos = QPointF()         # Item pos in parent coords at press
        self.item_press_scale = self.scale()    # Item scale at press
        self.item_press_rotation = self.rotation() # Item rotation at press

        # Pens and brushes
        self.selection_pen = QPen(QColor(Qt.GlobalColor.darkBlue), 1.5, Qt.PenStyle.DashLine)
        self.handle_pen = QPen(QColor(Qt.GlobalColor.black), 1)
        self.handle_fill_brush = QBrush(QColor(200, 200, 200, 200)) # Semi-transparent light gray
        self.rotate_handle_fill_brush = QBrush(QColor(Qt.GlobalColor.cyan, 200))

    def get_handle_rect(self, handle_type: int, current_rect: QRectF = None) -> QRectF:
        """Returns the rect for a given handle in item's local coordinates."""
        if current_rect is None:
            current_rect = self.local_rect # Use original pixmap rect for handle positions

        half_h_size = self.handle_size / 2.0
        if handle_type == self.Handle_Scale_BR:
            return QRectF(current_rect.right() - half_h_size, current_rect.bottom() - half_h_size,
                          self.handle_size, self.handle_size)
        elif handle_type == self.Handle_Rotate:
            rot_center_x = current_rect.center().x()
            rot_center_y = current_rect.top() - self.rotation_handle_offset
            return QRectF(rot_center_x - half_h_size, rot_center_y - half_h_size,
                          self.handle_size, self.handle_size)
        return QRectF()

    def paint(self, painter: QPainter, option, widget=None):
        # Let QGraphicsPixmapItem draw the pixmap first
        super().paint(painter, option, widget)

        if self.isSelected():
            painter.save()
            # Draw selection border (in item's local coordinates)
            painter.setPen(self.selection_pen)
            painter.setBrush(Qt.GlobalColor.transparent)
            painter.drawRect(self.local_rect)

            # Draw handles
            painter.setPen(self.handle_pen)

            # Scale Handle (Bottom Right)
            painter.setBrush(self.handle_fill_brush)
            painter.drawEllipse(self.get_handle_rect(self.Handle_Scale_BR))

            # Rotation Handle (Top Center)
            painter.setBrush(self.rotate_handle_fill_brush)
            rot_handle_rect = self.get_handle_rect(self.Handle_Rotate)
            painter.drawEllipse(rot_handle_rect)
            # Line to rotation handle
            painter.drawLine(QPointF(self.local_rect.center().x(), self.local_rect.top()),
                             rot_handle_rect.center())
            painter.restore()

    def hoverMoveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        cursor = Qt.CursorShape.ArrowCursor
        self.current_interaction_mode = None # Reset mode assumption
        if self.isSelected():
            item_pos = event.pos() # Mouse position in item's local coordinates
            if self.get_handle_rect(self.Handle_Scale_BR).contains(item_pos):
                cursor = Qt.CursorShape.SizeFDiagCursor
                self.current_interaction_mode = self.Handle_Scale_BR # Tentative mode
            elif self.get_handle_rect(self.Handle_Rotate).contains(item_pos):
                cursor = Qt.CursorShape.CrossCursor # Or a specific rotate cursor
                self.current_interaction_mode = self.Handle_Rotate # Tentative mode
        self.setCursor(cursor)
        super().hoverMoveEvent(event) # Allow parent to also handle if needed

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            item_local_pos = event.pos()
            # Check if a handle was clicked (current_interaction_mode was set by hover)
            if self.get_handle_rect(self.Handle_Scale_BR).contains(item_local_pos):
                self.current_interaction_mode = self.Handle_Scale_BR
            elif self.get_handle_rect(self.Handle_Rotate).contains(item_local_pos):
                self.current_interaction_mode = self.Handle_Rotate
            else: # Clicked on the item itself, not a handle
                self.current_interaction_mode = None


            if self.current_interaction_mode is not None: # Clicked on a handle
                self.mouse_press_item_pos = item_local_pos
                self.mouse_press_scene_pos = event.scenePos()
                self.item_press_pos = self.pos()
                self.item_press_scale = self.scale()
                self.item_press_rotation = self.rotation()
                # Crucial: Store the transform origin point at the moment of press
                # This is in parent coordinates if origin is item's center
                self.press_transform_origin_scene = self.mapToScene(self.transformOriginPoint())

                event.accept()
                return
        super().mousePressEvent(event) # For moving the item itself

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.current_interaction_mode is None: # Not dragging a handle
            super().mouseMoveEvent(event) # Allow base class to handle item movement
            # If moving, ItemPositionHasChanged will be emitted by base.
            # We can catch that in itemChange if needed.
            return

        current_scene_pos = event.scenePos()
        current_item_pos = event.pos()

        if self.current_interaction_mode == self.Handle_Scale_BR:
            # Calculate scale factor based on distance from transform origin to mouse
            # This implements uniform scaling from the item's transform origin.
            initial_dist_vec = self.mouse_press_scene_pos - self.press_transform_origin_scene
            current_dist_vec = current_scene_pos - self.press_transform_origin_scene

            initial_dist = math.sqrt(initial_dist_vec.x()**2 + initial_dist_vec.y()**2)
            current_dist = math.sqrt(current_dist_vec.x()**2 + current_dist_vec.y()**2)

            if initial_dist > 1e-6: # Avoid division by zero
                scale_factor_change = current_dist / initial_dist
                new_scale = self.item_press_scale * scale_factor_change
                # Apply minimum scale to prevent inversion or too small items
                new_scale = max(0.05, new_scale)
                self.setScale(new_scale)

        elif self.current_interaction_mode == self.Handle_Rotate:
            # Calculate angle change around transform origin
            angle_initial_rad = math.atan2(self.mouse_press_scene_pos.y() - self.press_transform_origin_scene.y(),
                                           self.mouse_press_scene_pos.x() - self.press_transform_origin_scene.x())
            angle_current_rad = math.atan2(current_scene_pos.y() - self.press_transform_origin_scene.y(),
                                           current_scene_pos.x() - self.press_transform_origin_scene.x())
            delta_angle_deg = math.degrees(angle_current_rad - angle_initial_rad)
            self.setRotation(self.item_press_rotation + delta_angle_deg)

        self.prepareGeometryChange() # Important before changing transform properties
        # self.transform_changed.emit(self.part_name) # Emit after actual change
        event.accept()


    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.current_interaction_mode is not None:
            self.current_interaction_mode = None
            self.transform_changed.emit(self.part_name) # Emit on release
            event.accept()
            # Update cursor via hover
            self.hoverMoveEvent(event)
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsPixmapItem.GraphicsItemChange.ItemPositionHasChanged and self.isSelected():
             # Emitted when the item is moved by base class (not by handle drag)
             if self.current_interaction_mode is None: # Ensure it's a move, not a handle op finishing
                self.transform_changed.emit(self.part_name)
        # if change == QGraphicsPixmapItem.GraphicsItemChange.ItemTransformHasChanged:
            # This is generic, scale/rotation changes also trigger it.
            # self.transform_changed.emit(self.part_name)
            # pass
        return super().itemChange(change, value)

    def get_transform_data(self) -> dict:
        center = self.mapToScene(self.transformOriginPoint())
        return {
            'part_name': self.part_name,
            'pos_x_scene': self.scenePos().x(),
            'pos_y_scene': self.scenePos().y(),
            'scale': self.scale(), # Assuming uniform scale for now
            'rotation_deg': self.rotation(),
            'transform_origin_scene_x': center.x(),
            'transform_origin_scene_y': center.y(),
            'z_value': self.zValue()
        }

# == Main Window ==
class MainWindow(QMainWindow):
    # Signals for process steps (can be used for progress updates or inter-module communication)
    # images_uploaded_signal = pyqtSignal()
    # poses_detected_signal = pyqtSignal()
    # ...

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Character Auto Merger GUI (New)")
        self.setGeometry(50, 50, 1600, 900)

        # --- Image Data Storage ---
        self.image_paths = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None}
        self.image_cv_originals = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None}
        self.image_pixmaps_original_size = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None} # Store full-size pixmaps for display

        # --- Skeleton rendering helper ---
        self.current_skeleton_point_items = [] # Holds EditablePointItem instances for current view
        self.skeleton_bone_items = []      # Holds QGraphicsLineItem for bones

        # --- Processing Results Storage ---
        self.skeletons_data = { # Stores raw keypoints from detector
            IMAGE_TYPE_TEMPLATE: None, # {'keypoints': [[x,y,c], ...], 'source_image_size': (w,h)}
            IMAGE_TYPE_HEAD: None,
            IMAGE_TYPE_BODY: None
        }
        self.adjusted_skeletons_data = { # Stores user-adjusted keypoints
            IMAGE_TYPE_TEMPLATE: None,
            IMAGE_TYPE_HEAD: None,
            IMAGE_TYPE_BODY: None
        }
        self.pre_segmented_parts_polygons = { # Stores initial polygons for parts on head/body images
            IMAGE_TYPE_HEAD: {}, # { 'part_name': {'initial_polygon': QPolygonF, ...}, ... }
            IMAGE_TYPE_BODY: {}
        }
        self.final_extracted_parts = { # Stores final BGRA images of parts and their info
            # 'PartName': {'image_bgra': np.ndarray, 'position_info': dict, 'source_image_type': str}
        }
        self.template_display_item = None # QGraphicsPixmapItem for template bg in edit mode
        self.scene_interactive_parts = {} # part_name -> InteractivePartItem in edit mode

        # --- UI State ---
        self.current_app_mode = APP_MODE_UPLOAD
        self.current_image_type_for_adjustment = None # e.g. IMAGE_TYPE_HEAD or IMAGE_TYPE_BODY for skeleton/segmentation
        self.current_part_for_segment_adjustment = None # e.g. "Head" or "Torso"

        # --- Core Logic Instances (Mocked for now) ---
        # self.pose_detector = MockPoseDetector()
        # self.part_segmenter = MockPartSegmenter()
        # self.warper = MockWarper()
        # --- Replace with Real Implementations ---
        print(f"Attempting to initialize RealPoseDetector with DWPOSE_DIR: {DWPOSE_DIR}")
        try:
            self.pose_detector = RealPoseDetector(DWPOSE_DIR, MODEL_CONFIGS)
        except Exception as e_pd:
            QMessageBox.critical(self, "Critical Error", f"Failed to initialize RealPoseDetector: {e_pd}\\nApplication might not function correctly.")
            # Fallback to mock if real fails, to allow GUI to still load for other tests
            print(f"Falling back to MockPoseDetector due to error: {e_pd}")
            self.pose_detector = None # Or handle this more gracefully


        self.part_segmenter = RealPartSegmenter()
        self.warper = RealWarper()


        # --- Part Definitions (COCO 17 keypoints based, adjust as per actual model) ---
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow,
        # 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        self.PART_DEFINITIONS = {
            "Head": [0, 1, 2, 3, 4],
            "Torso": [5, 6, 12, 11], # Shoulders and Hips
            "LeftArm": [5, 7, 9],    # LShoulder, LElbow, LWrist
            "RightArm": [6, 8, 10],  # RShoulder, RElbow, RWrist
            "LeftLeg": [11, 13, 15], # LHip, LKnee, LAnkle
            "RightLeg": [12, 14, 16] # RHip, RKnee, RAnkle
            # Add more granular parts if needed, e.g., hands, feet, neck.
        }
        # Colors for different part polygons/skeletons
        self.part_colors = [
            QColor("red"), QColor("green"), QColor("blue"),
            QColor("yellow"), QColor("cyan"), QColor("magenta"),
            QColor("orange"), QColor("purple"), QColor("brown")
        ]

        # Define bone connections based on COCO 17 keypoints
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow,
        # 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        self.BONE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head: Nose-LEye, Nose-REye, LEye-LEar, REye-REar
            (5, 6),                          # Torso: LShoulder-RShoulder
            (5, 7), (7, 9),                  # Left Arm: LShoulder-LElbow, LElbow-LWrist
            (6, 8), (8, 10),                 # Right Arm: RShoulder-RElbow, RElbow-RWrist
            (11, 12),                        # Pelvis: LHip-RHip
            (5, 11),                         # Torso: LShoulder-LHip
            (6, 12),                         # Torso: RShoulder-RHip
            (11, 13), (13, 15),              # Left Leg: LHip-LKnee, LKnee-LAnkle
            (12, 14), (14, 16)               # Right Leg: RHip-RKnee, RKnee-RAnkle
        ]
        self.BONE_PEN = QPen(QColor(Qt.GlobalColor.darkGreen), 2) # Pen for drawing bones
        self.BONE_PEN.setCosmetic(True) # Ensures line width is pixel-width regardless of zoom

        self._init_ui()
        self._update_ui_for_mode()

        if not os.path.exists(DEFAULT_EXPORT_DIR):
            os.makedirs(DEFAULT_EXPORT_DIR)

    def _init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.overall_layout = QHBoxLayout(self.main_widget) # Main horizontal split

        # --- Left Panel (Controls, File Uploads, Step Info) ---
        self.left_panel = QWidget()
        self.left_panel.setObjectName("left_panel") # For easier identification in diagnostics
        self.left_panel.setFixedWidth(350)
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        
        upload_group = QGroupBox("1. Upload Images")
        upload_layout = QFormLayout()
        self.btn_upload_template = QPushButton("Upload Template")
        self.lbl_template_file = QLabel("None")
        self.btn_upload_head = QPushButton("Upload Head")
        self.lbl_head_file = QLabel("None")
        self.btn_upload_body = QPushButton("Upload Body")
        self.lbl_body_file = QLabel("None")
        upload_layout.addRow(self.btn_upload_template, self.lbl_template_file)
        upload_layout.addRow(self.btn_upload_head, self.lbl_head_file)
        upload_layout.addRow(self.btn_upload_body, self.lbl_body_file)
        upload_group.setLayout(upload_layout)
        self.left_panel_layout.addWidget(upload_group)

        preview_group = QGroupBox("Image Previews")
        preview_layout = QVBoxLayout()
        self.preview_labels = {}
        for img_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
            lbl = QLabel(f"{img_type.capitalize()} Preview (click to show in main view)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(300, 150)
            lbl.setStyleSheet("border: 1px solid lightgrey; color: grey;")
            lbl.mousePressEvent = lambda event, t=img_type: self._show_image_in_main_view(t)
            preview_layout.addWidget(lbl)
            self.preview_labels[img_type] = lbl
        preview_group.setLayout(preview_layout)
        self.left_panel_layout.addWidget(preview_group)

        self.step_controls_group = QGroupBox("Current Step Controls")
        self.step_controls_layout = QVBoxLayout()
        self.step_controls_group.setLayout(self.step_controls_layout)
        self.left_panel_layout.addWidget(self.step_controls_group)
        self.left_panel_layout.addStretch()
        self.overall_layout.addWidget(self.left_panel, 0) # Stretch factor 0 for left_panel

        # --- Center Panel (Main Graphics View) ---
        self.center_panel = QWidget()
        self.center_panel.setObjectName("center_panel")
        self.center_panel.setMinimumSize(400, 300) # Set a reasonable minimum size
        self.center_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.center_panel_layout = QVBoxLayout(self.center_panel)
        
        self.graphics_scene = QGraphicsScene(self) # Scene should be created before view uses it
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        zoom_toolbar = QToolBar("Zoom")
        self.action_zoom_in = QAction("Zoom In", self)
        self.action_zoom_out = QAction("Zoom Out", self)
        self.action_zoom_fit = QAction("Fit View", self)
        zoom_toolbar.addAction(self.action_zoom_in)
        zoom_toolbar.addAction(self.action_zoom_out)
        zoom_toolbar.addAction(self.action_zoom_fit)
        self.center_panel_layout.addWidget(zoom_toolbar)
        self.center_panel_layout.addWidget(self.graphics_view, 1) # graphics_view expands with stretch factor 1
        self.overall_layout.addWidget(self.center_panel, 1) # Stretch factor 1 for center_panel to expand

        # --- Right Panel (Layers, Properties - initially hidden or minimal) ---
        self.right_panel = QWidget()
        self.right_panel.setObjectName("right_panel")
        self.right_panel.setFixedWidth(300)
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        
        self.layers_list_widget = QListWidget()
        self.right_panel_layout.addWidget(QLabel("Layers / Parts:"))
        self.right_panel_layout.addWidget(self.layers_list_widget)

        self.properties_group = QGroupBox("Part Properties")
        self.properties_layout = QFormLayout()
        self.prop_pos_x = QLineEdit(); self.prop_pos_x.setReadOnly(True)
        self.prop_pos_y = QLineEdit(); self.prop_pos_y.setReadOnly(True)
        self.prop_scale = QLineEdit(); self.prop_scale.setReadOnly(True)
        self.prop_rotation = QLineEdit(); self.prop_rotation.setReadOnly(True)
        self.properties_layout.addRow("Pos X:", self.prop_pos_x)
        self.properties_layout.addRow("Pos Y:", self.prop_pos_y)
        self.properties_layout.addRow("Scale:", self.prop_scale)
        self.properties_layout.addRow("Rotation:", self.prop_rotation)
        self.properties_group.setLayout(self.properties_layout)
        self.right_panel_layout.addWidget(self.properties_group)
        self.right_panel_layout.addStretch()
        self.overall_layout.addWidget(self.right_panel, 0) # Stretch factor 0 for right_panel
        self.right_panel.setVisible(False) # Keep it initially hidden as per original logic

        # Ensure center_panel is visible (this was the fix from previous steps)
        self.center_panel.setVisible(True)

        # == Diagnostic additions (kept for now) ==
        # 1. Simplest render test: Add a fixed rectangle to the scene early on.
        test_rect_item = self.graphics_scene.addRect(0, 0, 200, 100, QPen(Qt.GlobalColor.magenta), QBrush(Qt.GlobalColor.yellow))
        test_rect_item.setZValue(1000) 
        print(f"MainWindow init: Added test_rect_item. sceneBoundingRect: {test_rect_item.sceneBoundingRect()}, isVisible: {test_rect_item.isVisible()}")
        print(f"MainWindow init: Scene items now: {self.graphics_scene.items()}")

        # 2. Ensure view background is transparent / set a style for debugging
        self.graphics_view.setStyleSheet("background-color: transparent; border: 2px solid red;")
        self.center_panel.setStyleSheet("background-color: lightblue;")

        # Deferred check of widget geometries
        QTimer.singleShot(200, self._deferred_ui_diagnostics) 
        # == End Diagnostic additions ==

        # --- Status Bar ---
        self.statusBar().showMessage("Ready. Please upload images.")

        # --- Connect Signals ---
        self.btn_upload_template.clicked.connect(lambda: self._upload_image_file(IMAGE_TYPE_TEMPLATE))
        self.btn_upload_head.clicked.connect(lambda: self._upload_image_file(IMAGE_TYPE_HEAD))
        self.btn_upload_body.clicked.connect(lambda: self._upload_image_file(IMAGE_TYPE_BODY))

        self.action_zoom_in.triggered.connect(lambda: self.graphics_view.scale(1.2, 1.2))
        self.action_zoom_out.triggered.connect(lambda: self.graphics_view.scale(1 / 1.2, 1 / 1.2))
        self.action_zoom_fit.triggered.connect(self._fit_view)

        self.graphics_scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.layers_list_widget.currentItemChanged.connect(self._on_layer_list_selection_changed)

    def _deferred_ui_diagnostics(self):
        print("--- Deferred UI Diagnostics (after window show attempt) ---")
        if hasattr(self, 'main_widget') and self.main_widget:
            print(f"main_widget: geometry={self.main_widget.geometry()}, visible={self.main_widget.isVisible()}, children={self.main_widget.children()}")
        if hasattr(self, 'overall_layout') and self.overall_layout:
            print(f"overall_layout: count={self.overall_layout.count()}")
            if self.overall_layout.itemAt(0) and self.overall_layout.itemAt(0).widget():
                 print(f"overall_layout item 0 (left_panel?): widget={self.overall_layout.itemAt(0).widget().objectName() or type(self.overall_layout.itemAt(0).widget())}, size={self.overall_layout.itemAt(0).widget().size()}, visible={self.overall_layout.itemAt(0).widget().isVisible()}")
            if self.overall_layout.itemAt(1) and self.overall_layout.itemAt(1).widget():
                 print(f"overall_layout item 1 (center_panel?): widget={self.overall_layout.itemAt(1).widget().objectName() or type(self.overall_layout.itemAt(1).widget())}, size={self.overall_layout.itemAt(1).widget().size()}, visible={self.overall_layout.itemAt(1).widget().isVisible()}")

        if hasattr(self, 'center_panel') and self.center_panel:
            print(f"center_panel: geometry={self.center_panel.geometry()}, visible={self.center_panel.isVisible()}, children={self.center_panel.children()}")
            # Check layout of center_panel
            if self.center_panel.layout() and isinstance(self.center_panel.layout(), QVBoxLayout):
                cp_layout = self.center_panel.layout()
                print(f"center_panel_layout: count={cp_layout.count()}")
                for i in range(cp_layout.count()):
                    item = cp_layout.itemAt(i)
                    widget = item.widget()
                    if widget:
                        print(f"  Item {i} in center_panel_layout (graphics_view?): widget={widget.objectName() or type(widget)}, size={widget.size()}, visible={widget.isVisible()}")
                    else:
                        print(f"  Item {i} in center_panel_layout is a spacer or sub-layout.")
            else:
                print("center_panel has no layout or layout is not QVBoxLayout")

        if hasattr(self, 'graphics_view') and self.graphics_view:
            print(f"graphics_view: geometry={self.graphics_view.geometry()}, size={self.graphics_view.size()}, visible={self.graphics_view.isVisible()}")
            print(f"graphics_view.scene(): {self.graphics_view.scene()}")
            if self.graphics_view.scene():
                print(f"graphics_view.scene().items(): {self.graphics_view.scene().items()}") # Should show test_rect_item
        print("--- End Deferred UI Diagnostics ---")


    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout)
                        sub_layout.deleteLater()


    def _update_ui_for_mode(self):
        """Dynamically updates the control panel based on the current app mode."""
        self._clear_layout(self.step_controls_layout) # Clear previous controls
        self.right_panel.setVisible(False) # Default hide right panel

        if self.current_app_mode == APP_MODE_UPLOAD:
            self.step_controls_group.setTitle("1. Upload Images")
            self.btn_proceed_to_skeleton = QPushButton("➡️ Process Poses / Skeletons")
            self.btn_proceed_to_skeleton.clicked.connect(self._action_start_pose_detection)
            self.btn_proceed_to_skeleton.setEnabled(
                all(self.image_paths.values()) # Enabled if all images are uploaded
            )
            self.step_controls_layout.addWidget(self.btn_proceed_to_skeleton)
            self.statusBar().showMessage("Upload template, head, and body images.")

        elif self.current_app_mode == APP_MODE_ADJUST_SKELETON:
            self.step_controls_group.setTitle(f"2. Adjust Skeleton: {self.current_image_type_for_adjustment.capitalize()}")
            # Radio buttons or buttons to switch between template, head, body for adjustment
            hbox = QHBoxLayout()
            for img_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                btn = QPushButton(img_type.capitalize())
                btn.setCheckable(True)
                btn.setChecked(img_type == self.current_image_type_for_adjustment)
                btn.clicked.connect(lambda checked, t=img_type: self._switch_skeleton_adjustment_view(t))
                hbox.addWidget(btn)
            self.step_controls_layout.addLayout(hbox)

            self.btn_reset_current_skeleton = QPushButton("Reset Current Skeleton")
            self.btn_reset_current_skeleton.clicked.connect(self._action_reset_current_skeleton_adj)
            self.step_controls_layout.addWidget(self.btn_reset_current_skeleton)

            self.btn_confirm_skeletons = QPushButton("✅ Confirm All Skeletons & Proceed to Segmentation")
            self.btn_confirm_skeletons.clicked.connect(self._action_confirm_all_skeleton_adjustments)
            # Enable if all skeletons have been (at least initially) processed
            self.btn_confirm_skeletons.setEnabled(all(self.skeletons_data.values()))
            self.step_controls_layout.addWidget(self.btn_confirm_skeletons)
            self.statusBar().showMessage(f"Adjust skeleton points for {self.current_image_type_for_adjustment}. Click image name to switch.")

        elif self.current_app_mode == APP_MODE_ADJUST_SEGMENTATION:
            self.step_controls_group.setTitle(f"3. Adjust Segmentation: {self.current_image_type_for_adjustment.capitalize()} ({self.current_part_for_segment_adjustment or 'No Part'})")
            # Buttons to switch between Head and Body images for segmentation
            hbox_img_type = QHBoxLayout()
            for img_type in [IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]: # Only head and body have parts to segment
                btn = QPushButton(f"Show {img_type.capitalize()} Segments")
                btn.setCheckable(True)
                btn.setChecked(img_type == self.current_image_type_for_adjustment)
                btn.clicked.connect(lambda checked, t=img_type: self._switch_segmentation_adjustment_view(t))
                hbox_img_type.addWidget(btn)
            self.step_controls_layout.addLayout(hbox_img_type)

            # List/Buttons to select which part's polygon to edit on the current image
            self.segment_part_list = QListWidget()
            self.segment_part_list.setFixedHeight(150)
            self.populate_segment_part_list() # Populate with parts from current image
            self.segment_part_list.currentItemChanged.connect(self._on_segment_part_selected_for_adj)
            self.step_controls_layout.addWidget(QLabel("Select part to adjust:"))
            self.step_controls_layout.addWidget(self.segment_part_list)

            self.btn_confirm_segmentation = QPushButton("✅ Confirm All Segments & Extract Parts")
            self.btn_confirm_segmentation.clicked.connect(self._action_confirm_all_segment_adjustments)
            # Enable if pre_segmented_parts_polygons has data for head and body
            self.btn_confirm_segmentation.setEnabled(
                bool(self.pre_segmented_parts_polygons.get(IMAGE_TYPE_HEAD)) and \
                bool(self.pre_segmented_parts_polygons.get(IMAGE_TYPE_BODY))
            )
            self.step_controls_layout.addWidget(self.btn_confirm_segmentation)
            self.statusBar().showMessage(f"Adjust segmentation for {self.current_part_for_segment_adjustment or 'part'} on {self.current_image_type_for_adjustment} image.")

        elif self.current_app_mode == APP_MODE_EDIT_PARTS:
            self.step_controls_group.setTitle("4. Edit Parts on Template")
            self.right_panel.setVisible(True) # Show layers and properties
            # Populate layers list in _action_place_parts_on_template
            
            # Add Z-order buttons
            z_order_layout = QHBoxLayout()
            btn_bring_forward = QPushButton("Bring Forward")
            btn_send_backward = QPushButton("Send Backward")
            btn_bring_to_front = QPushButton("Bring to Front")
            btn_send_to_back = QPushButton("Send to Back")
            z_order_layout.addWidget(btn_bring_forward)
            z_order_layout.addWidget(btn_send_backward)
            # z_order_layout.addWidget(btn_bring_to_front)
            # z_order_layout.addWidget(btn_send_to_back)
            self.step_controls_layout.addLayout(z_order_layout)
            btn_bring_forward.clicked.connect(lambda: self._change_z_order(1))
            btn_send_backward.clicked.connect(lambda: self._change_z_order(-1))
            # btn_bring_to_front.clicked.connect(lambda: self._change_z_order('front'))
            # btn_send_to_back.clicked.connect(lambda: self._change_z_order('back'))


            self.btn_export_final = QPushButton("💾 Export Final Image (PNG)")
            self.btn_export_final.clicked.connect(self._action_export_final_image)
            self.btn_export_final.setEnabled(bool(self.final_extracted_parts))
            self.step_controls_layout.addWidget(self.btn_export_final)
            self.statusBar().showMessage("Edit parts on the template. Select a part to transform. Use layer list.")

        self.step_controls_layout.addStretch() # Push controls to the top

    def _fit_view(self):
        # Before fitting, print the view's current size and viewport size
        print(f"Fit_view: View size: {self.graphics_view.size()}, Viewport size: {self.graphics_view.viewport().size()}")
        if self.graphics_view.size().isEmpty() or self.graphics_view.viewport().size().isEmpty():
            print("WARNING: Fit_view called when view or viewport size is empty!")
            # return # Optionally skip fitting if view has no size

        # Reset transform before fitting to ensure a clean state
        self.graphics_view.resetTransform()
        print(f"Fit_view: View transform AFTER resetTransform: {self.graphics_view.transform()}")

        if not self.graphics_scene.itemsBoundingRect().isEmpty():
            items_br = self.graphics_scene.itemsBoundingRect()
            print(f"Fit_view: Attempting to fit to itemsBoundingRect: {items_br}")
            self.graphics_view.fitInView(items_br, Qt.AspectRatioMode.KeepAspectRatio)
            print(f"Fit_view: View transform AFTER fitInView: {self.graphics_view.transform()}")
            print(f"Fit_view: Viewport visible scene rect (approx): {self.graphics_view.mapToScene(self.graphics_view.viewport().rect()).boundingRect()}")
        elif self.template_display_item:
            template_br = self.template_display_item.sceneBoundingRect() # Use sceneBoundingRect for accuracy
            print(f"Fit_view: Attempting to fit to template_display_item.sceneBoundingRect: {template_br}")
            self.graphics_view.fitInView(template_br, Qt.AspectRatioMode.KeepAspectRatio)
            print(f"Fit_view (template): View transform AFTER fitInView: {self.graphics_view.transform()}")
            print(f"Fit_view (template): Viewport visible scene rect (approx): {self.graphics_view.mapToScene(self.graphics_view.viewport().rect()).boundingRect()}")
        else:
            print("Fit_view: No itemsBoundingRect and no template_display_item to fit.")
        
        # It might be useful to see the scene rect the view is trying to map to its viewport
        print(f"Fit_view: Final sceneRect of the view (after potential fit): {self.graphics_view.sceneRect()}")
        self.graphics_view.viewport().update() # Force repaint after fitting

    def _upload_image_file(self, image_type_str: str):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {image_type_str.capitalize()} Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if not file_path: return

        try:
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                raise ValueError(f"OpenCV could not read image: {file_path}")

            self.image_paths[image_type_str] = file_path
            self.image_cv_originals[image_type_str] = img_cv
            
            # Store full-size pixmap for potential display without re-conversion
            q_pixmap = self._convert_cv_to_pixmap(img_cv)
            if q_pixmap.isNull():
                raise ValueError("Failed to convert CV image to QPixmap.")
            self.image_pixmaps_original_size[image_type_str] = q_pixmap


            # Update preview label
            preview_pixmap = q_pixmap.scaled(
                self.preview_labels[image_type_str].size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_labels[image_type_str].setPixmap(preview_pixmap)
            
            # Update file label
            if image_type_str == IMAGE_TYPE_TEMPLATE: self.lbl_template_file.setText(os.path.basename(file_path))
            elif image_type_str == IMAGE_TYPE_HEAD: self.lbl_head_file.setText(os.path.basename(file_path))
            elif image_type_str == IMAGE_TYPE_BODY: self.lbl_body_file.setText(os.path.basename(file_path))
            
            print(f"Uploaded {image_type_str}: {file_path}")
            self._show_image_in_main_view(image_type_str) # Show newly uploaded image
            self._update_ui_for_mode() # Check if "Proceed" button can be enabled

        except Exception as e:
            QMessageBox.critical(self, "Upload Error", f"Failed to upload or process {image_type_str} image:\\n{e}")
            self.image_paths[image_type_str] = None
            self.image_cv_originals[image_type_str] = None
            self.image_pixmaps_original_size[image_type_str] = None
            self.preview_labels[image_type_str].setText(f"{image_type_str.capitalize()} Preview (click to show)")
            self.preview_labels[image_type_str].setPixmap(QPixmap()) # Clear preview
            if image_type_str == IMAGE_TYPE_TEMPLATE: self.lbl_template_file.setText("None")
            elif image_type_str == IMAGE_TYPE_HEAD: self.lbl_head_file.setText("None")
            elif image_type_str == IMAGE_TYPE_BODY: self.lbl_body_file.setText("None")


    def _convert_cv_to_pixmap(self, cv_img: np.ndarray) -> QPixmap:
        if cv_img is None: return QPixmap()
        try:
            # Minimal check for obviously empty images before conversion
            if cv_img.size == 0:
                print("Warning: CV image is empty (size 0) before conversion.")
                return QPixmap()

            if cv_img.ndim == 3 and cv_img.shape[2] == 4: # BGRA
                # QImage needs RGBA
                rgba_image = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                h, w, ch = rgba_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgba_image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            elif cv_img.ndim == 3 and cv_img.shape[2] == 3: # BGR
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            elif cv_img.ndim == 2: # Grayscale
                h, w = cv_img.shape
                bytes_per_line = w
                q_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                print(f"Unsupported image format for QPixmap conversion: shape {cv_img.shape}")
                return QPixmap()

            if q_image.isNull():
                print("Warning: QImage conversion resulted in null QImage.")
                return QPixmap()
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"Error converting CV image to Pixmap: {e}")
            traceback.print_exc()
            return QPixmap()

    def _clear_scene(self):
        # Clear all items except potentially a background item if we manage it separately
        # For now, clear all
        print("_clear_scene: Clearing all items from graphics_scene.")
        # It's generally better to let Qt manage item deletion when they are removed from the scene.
        # Calling 'del item' might not be necessary or could be problematic if references exist elsewhere.
        # self.graphics_scene.clear() is the most straightforward way.
        
        # Store items before clearing for debugging (optional)
        # items_before_clear = list(self.graphics_scene.items())
        # print(f"_clear_scene: Items before clear: {items_before_clear}")

        self.graphics_scene.clear() # This removes all items and should be sufficient.

        # print(f"_clear_scene: Items after clear: {self.graphics_scene.items()}")


        self.template_display_item = None
        # Clear any specific item maps
        self.current_skeleton_point_items = []
        self.skeleton_bone_items = [] # Also clear bone items list
        self.current_segment_polygon_items = {} # part_name -> EditablePolygonItem
        self.scene_interactive_parts.clear()
        self.layers_list_widget.clear()

    def _show_image_in_main_view(self, image_type_str: str, fit_view=True):
        """Displays the specified original image in the main graphics view."""
        print(f"_show_image_in_main_view: Called for {image_type_str}")
        self._clear_scene()
        pixmap = self.image_pixmaps_original_size.get(image_type_str)
        if pixmap and not pixmap.isNull():
            print(f"[{image_type_str}] Pixmap is valid. Size: {pixmap.size()}, Rect: {pixmap.rect()}")
            bg_item = self.graphics_scene.addPixmap(pixmap)
            bg_item.setZValue(-100) # Ensure it's a background
            self.graphics_scene.setSceneRect(pixmap.rect().toRectF())
            # print(f"[{image_type_str}] bg_item added. Mapped to scene: {bg_item.mapToScene(bg_item.boundingRect())}")
            print(f"[{image_type_str}] bg_item added. sceneBoundingRect: {bg_item.sceneBoundingRect()}, isVisible: {bg_item.isVisible()}")
            print(f"[{image_type_str}] Scene items after adding bg: {self.graphics_scene.items()}")
            print(f"[{image_type_str}] Scene rect set to: {self.graphics_scene.sceneRect()}")

            if fit_view:
                print(f"[{image_type_str}] Calling _fit_view after adding background.")
                self._fit_view()
            self.current_displaying_image_type = image_type_str
            print(f"Displaying {image_type_str} in main view. (End of _show_image_in_main_view success path)")
        else:
            self.graphics_scene.setSceneRect(QRectF()) # Clear scene rect
            self.current_displaying_image_type = None
            print(f"No pixmap to display for {image_type_str}. Pixmap isNull: {pixmap.isNull() if pixmap else 'None'}")
        self.graphics_view.viewport().update() # Ensure viewport updates

    # --- Skeleton Adjustment Mode Methods ---
    def _display_skeleton_on_image(self, image_type_str: str, skeleton_data: dict):
        print(f"_display_skeleton_on_image: Called for {image_type_str}")
        self._show_image_in_main_view(image_type_str, fit_view=True) # Base image, also clears old points/bones via _clear_scene

        # self.current_skeleton_point_items and self.skeleton_bone_items are cleared in _clear_scene

        if skeleton_data and 'keypoints' in skeleton_data:
            keypoints = skeleton_data['keypoints'] # list of [x,y,conf]
            print(f"[{image_type_str}] Found {len(keypoints)} keypoints to display.")

            # Store points by index for easy lookup when drawing bones
            points_by_index = {}

            for i, kp in enumerate(keypoints):
                x, y, conf = kp[0], kp[1], kp[2]
                color = QColor("red") if conf > 0.5 else QColor("orange") if conf > 0.2 else QColor("yellow")
                point_item = EditablePointItem(x, y, radius=5, kp_index=i, label_text=str(i), color=color)
                point_item.setToolTip(f"KP {i}: ({x:.1f}, {y:.1f}), Conf: {conf:.2f}")
                point_item.setZValue(10) # Ensure points are above lines
                self.graphics_scene.addItem(point_item)
                self.current_skeleton_point_items.append(point_item)
                points_by_index[i] = point_item

            # Draw bones
            min_bone_confidence = 0.1 # Minimum confidence for a keypoint to be part of a bone
            for kp_idx1, kp_idx2 in self.BONE_CONNECTIONS:
                if kp_idx1 in points_by_index and kp_idx2 in points_by_index:
                    point1_data = keypoints[kp_idx1]
                    point2_data = keypoints[kp_idx2]
                    # Only draw bone if both points have minimal confidence
                    if point1_data[2] > min_bone_confidence and point2_data[2] > min_bone_confidence:
                        p1 = points_by_index[kp_idx1].pos() # Use current position of EditablePointItem
                        p2 = points_by_index[kp_idx2].pos()
                        line_item = QGraphicsLineItem(QLineF(p1, p2))
                        line_item.setPen(self.BONE_PEN)
                        line_item.setZValue(0) # Bones behind points
                        self.graphics_scene.addItem(line_item)
                        self.skeleton_bone_items.append(line_item)

            print(f"Displayed {len(self.current_skeleton_point_items)} skeleton points and {len(self.skeleton_bone_items)} bones for {image_type_str}.")
            print(f"[{image_type_str}] All points added. Scene items count: {len(self.graphics_scene.items())}")
            if self.current_skeleton_point_items:
                last_pt = self.current_skeleton_point_items[-1]
                print(f"[{image_type_str}] Last point added: pos={last_pt.pos()}, scenePos={last_pt.scenePos()}, sceneBoundingRect={last_pt.sceneBoundingRect()}, isVisible={last_pt.isVisible()}")
        else:
            print(f"[{image_type_str}] No keypoints data or empty keypoints list to display.")

        print(f"[{image_type_str}] Calling _fit_view after adding skeleton points.")
        self._fit_view() # Fit view after adding points
        self.graphics_view.viewport().update() # Extra force update
        QApplication.processEvents() # Process any pending events
        print(f"_display_skeleton_on_image: Finished for {image_type_str}")

    def _switch_skeleton_adjustment_view(self, image_type_str: str):
        # Save adjustments from the PREVIOUS view if it was valid, different, and in the correct mode.
        if self.current_image_type_for_adjustment is not None and \
           self.current_image_type_for_adjustment != image_type_str and \
           self.current_app_mode == APP_MODE_ADJUST_SKELETON:
            self._save_current_skeleton_adjustments()
        # If it's the same image_type_str being "re-selected", we skip saving
        # and proceed to re-display it. This helps if the initial display was incomplete.

        self.current_image_type_for_adjustment = image_type_str # Update to the new type

        # Attempt to load the skeleton data for the selected image type
        # Prioritize adjusted data, then fall back to original detected data.
        skeleton_to_display = self.adjusted_skeletons_data.get(image_type_str) or \
                              self.skeletons_data.get(image_type_str)

        if skeleton_to_display:
            # If skeleton data exists (even if keypoints list is empty), display it.
            self._display_skeleton_on_image(image_type_str, skeleton_to_display)
        else:
            # If no skeleton data at all, just show the image and inform the user.
            # This case might occur if pose detection was skipped or failed catastrophically for this image type.
            self._show_image_in_main_view(image_type_str) # Show base image
            QMessageBox.information(self, "No Skeleton Data", 
                                    f"No skeleton data found for {image_type_str}. "
                                    "Please ensure poses were processed for this image.")
        
        self._update_ui_for_mode() # Update button check states, status bar, etc.

    def _save_current_skeleton_adjustments(self):
        """Saves the positions of EditablePointItems to adjusted_skeletons_data."""
        if not self.current_image_type_for_adjustment or not self.current_skeleton_point_items:
            print("Save current skeleton: No current image type or no point items to save.") # Debug print
            return

        adj_kps = []
        original_kps_data = self.skeletons_data.get(self.current_image_type_for_adjustment)
        if not original_kps_data or not original_kps_data.get('keypoints'):
            print(f"Warning: No original keypoints data to base adjustments on for {self.current_image_type_for_adjustment}")
            return

        original_keypoints_list = original_kps_data['keypoints']

        for point_item in self.current_skeleton_point_items:
            kp_idx = point_item.kp_index
            new_pos = point_item.get_current_pos()
            original_conf = original_keypoints_list[kp_idx][2] if kp_idx < len(original_keypoints_list) else 0.1 # Keep original confidence
            adj_kps.append([new_pos.x(), new_pos.y(), original_conf])

        if not self.adjusted_skeletons_data.get(self.current_image_type_for_adjustment):
            self.adjusted_skeletons_data[self.current_image_type_for_adjustment] = {}
        
        # Keep original source_image_size
        self.adjusted_skeletons_data[self.current_image_type_for_adjustment]['keypoints'] = adj_kps
        self.adjusted_skeletons_data[self.current_image_type_for_adjustment]['source_image_size'] = original_kps_data['source_image_size']

        print(f"Saved {len(adj_kps)} adjusted skeleton points for {self.current_image_type_for_adjustment}.")

    # --- Segmentation Adjustment Mode Methods ---
    def _display_segmentation_polygons_on_image(self, image_type_str: str, polygons_data: dict):
        self._show_image_in_main_view(image_type_str, fit_view=True)

        for item in self.current_segment_polygon_items.values():
            if item.scene() == self.graphics_scene:
                self.graphics_scene.removeItem(item)
        self.current_segment_polygon_items = {}

        color_idx = 0
        if polygons_data:
            print(f"_display_segmentation_polygons: Processing {len(polygons_data)} polygons for {image_type_str}")
            for part_name, part_poly_info in polygons_data.items():
                initial_qpoly = part_poly_info.get('initial_polygon')
                
                print(f"  Part: {part_name}")
                if initial_qpoly:
                    print(f"    Polygon data before use: isEmpty={initial_qpoly.isEmpty()}, count={initial_qpoly.count()}, boundingRect={initial_qpoly.boundingRect()}")
                    if initial_qpoly.count() > 0:
                        for i_pt in range(initial_qpoly.count()):
                            print(f"      Point {i_pt}: {initial_qpoly.at(i_pt)}")
                else:
                    print(f"    Polygon data ('initial_polygon') is None for part {part_name}.")
                    continue 

                if not (initial_qpoly and not initial_qpoly.isEmpty()):
                    print(f"    Skipping part {part_name} due to empty or None initial_qpoly after preliminary check.")
                    continue
                
                if initial_qpoly.count() < 3:
                    print(f"    WARNING: Polygon for {part_name} has < 3 points ({initial_qpoly.count()}). May not render or cause issues. Skipping this part.")
                    continue

                # +++ TEST: Create a plain QGraphicsPolygonItem first +++
                test_plain_poly_item = None
                try:
                    print(f"      TEST: Attempting to create plain QGraphicsPolygonItem with initial_qpoly for part '{part_name}'...")
                    test_plain_poly_item = QGraphicsPolygonItem(initial_qpoly) # No parent initially
                    print(f"      TEST: Plain QGraphicsPolygonItem created successfully: {test_plain_poly_item}")
                    # Optionally, try adding it to scene and removing to test further
                    # print(f"      TEST: Attempting to add plain item to scene...")
                    # self.graphics_scene.addItem(test_plain_poly_item)
                    # print(f"      TEST: Plain item added to scene. BoundingRect: {test_plain_poly_item.boundingRect()}")
                    # self.graphics_scene.removeItem(test_plain_poly_item)
                    # print(f"      TEST: Plain item removed from scene.")
                except Exception as e_test_plain:
                    print(f"      TEST: CRITICAL ERROR creating/handling plain QGraphicsPolygonItem for part '{part_name}': {e_test_plain}")
                    traceback.print_exc()
                    # If this test fails, we might want to skip trying to create EditablePolygonItem
                    # continue # or raise an error to stop further processing for this part
                finally:
                    # Clean up the test item if it was created and not added, or if added and removed
                    if test_plain_poly_item and test_plain_poly_item.scene() is None: # Ensure it's not in a scene if we didn't manage to remove it
                        del test_plain_poly_item # Help GC if not parented or in scene
                # +++ END TEST +++

                print(f"    Proceeding to create EditablePolygonItem for {part_name}...") #This was the previous crash point log
                poly_item = EditablePolygonItem(initial_qpoly, part_name=part_name)
                print(f"    EditablePolygonItem for {part_name} created. Instance: {poly_item}")
                
                fill_color = self.part_colors[color_idx % len(self.part_colors)].lighter(150)
                fill_color.setAlpha(80)
                print(f"    Attempting to set_fill_color for {part_name}...")
                poly_item.set_fill_color(fill_color)
                print(f"    set_fill_color for {part_name} done.")

                print(f"    Attempting to setPen for {part_name}...")
                poly_item.setPen(QPen(self.part_colors[color_idx % len(self.part_colors)], 2))
                print(f"    setPen for {part_name} done.")

                print(f"    Attempting to setToolTip for {part_name}...")
                poly_item.setToolTip(f"Segment: {part_name}")
                print(f"    setToolTip for {part_name} done.")

                print(f"    Attempting to connect polygon_changed signal for {part_name}...")
                poly_item.polygon_changed.connect(self._on_segment_polygon_edited)
                print(f"    polygon_changed signal for {part_name} connected.")
                
                print(f"    Attempting to add poly_item for {part_name} to scene...")
                self.graphics_scene.addItem(poly_item)
                print(f"    poly_item for {part_name} ADDED to scene successfully.")
                
                self.current_segment_polygon_items[part_name] = poly_item
                color_idx += 1

            print(f"Displayed {len(self.current_segment_polygon_items)} segmentation polygons for {image_type_str}.")
        else:
            print(f"_display_segmentation_polygons: No polygons_data provided for {image_type_str}.")
        self._fit_view()

    def _switch_segmentation_adjustment_view(self, image_type_str: str):
        if self.current_image_type_for_adjustment == image_type_str and self.graphics_scene.items() and self.current_app_mode == APP_MODE_ADJUST_SEGMENTATION :
             # If polygons are already shown for this image type
            if any(isinstance(item, EditablePolygonItem) for item in self.graphics_scene.items()):
                return 

        if self.current_image_type_for_adjustment and self.current_app_mode == APP_MODE_ADJUST_SEGMENTATION:
            self._save_current_segmentation_adjustments() # Save polygons from previous view

        self.current_image_type_for_adjustment = image_type_str
        self.current_part_for_segment_adjustment = None # Reset selected part
        
        polygons_to_display = self.pre_segmented_parts_polygons.get(image_type_str, {})
        self._display_segmentation_polygons_on_image(image_type_str, polygons_to_display)
        
        self.populate_segment_part_list() # Update the list of parts for the new image
        self._update_ui_for_mode()

    def populate_segment_part_list(self):
        self.segment_part_list.clear()
        if self.current_image_type_for_adjustment and self.current_app_mode == APP_MODE_ADJUST_SEGMENTATION:
            parts_on_current_image = self.pre_segmented_parts_polygons.get(self.current_image_type_for_adjustment, {}).keys()
            for part_name in parts_on_current_image:
                self.segment_part_list.addItem(part_name)
            if self.segment_part_list.count() > 0:
                self.segment_part_list.setCurrentRow(0) # Select the first one by default


    def _on_segment_part_selected_for_adj(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        if not current_item:
            self.current_part_for_segment_adjustment = None
            # Deselect all polygons in scene? Or rely on scene selection?
            self.graphics_scene.clearSelection()
            self._update_ui_for_mode()
            return

        part_name = current_item.text()
        self.current_part_for_segment_adjustment = part_name
        
        # Highlight the corresponding polygon in the scene
        for p_name, poly_item in self.current_segment_polygon_items.items():
            poly_item.setSelected(p_name == part_name)
            if p_name == part_name:
                poly_item.setZValue(10) # Bring to front
                self.graphics_view.ensureVisible(poly_item)
            else:
                poly_item.setZValue(0)
        self._update_ui_for_mode() # Update status bar or titles

    def _on_segment_polygon_edited(self):
        # This is a slot that can be connected to EditablePolygonItem's polygon_changed signal.
        # It can be used for real-time feedback or to mark data as "dirty".
        # For now, actual saving happens when switching views or confirming.
        # print(f"Polygon for part {sender_item.part_name if sender_item else 'unknown'} edited.")
        pass


    def _save_current_segmentation_adjustments(self):
        """Saves the state of EditablePolygonItems for the current image type."""
        if not self.current_image_type_for_adjustment or not self.current_segment_polygon_items:
            return
        
        # Ensure the entry for the image type exists
        if not self.pre_segmented_parts_polygons.get(self.current_image_type_for_adjustment):
            self.pre_segmented_parts_polygons[self.current_image_type_for_adjustment] = {}

        updated_count = 0
        for part_name, poly_item in self.current_segment_polygon_items.items():
            # Ensure the entry for the part exists
            if not self.pre_segmented_parts_polygons[self.current_image_type_for_adjustment].get(part_name):
                 self.pre_segmented_parts_polygons[self.current_image_type_for_adjustment][part_name] = {}

            # Update the polygon. Note: 'initial_polygon' might be a misnomer here if we are saving edits.
            # It's better to have 'current_polygon' or similar.
            self.pre_segmented_parts_polygons[self.current_image_type_for_adjustment][part_name]['initial_polygon'] = poly_item.get_edited_qpolygonf()
            updated_count +=1
        print(f"Saved {updated_count} adjusted segmentation polygons for {self.current_image_type_for_adjustment}.")

    # --- Edit Parts on Template Mode Methods ---
    def _display_parts_on_template(self):
        self._clear_scene()
        self.layers_list_widget.clear()
        self.scene_interactive_parts.clear()

        # 1. Display template image as background
        template_pixmap = self.image_pixmaps_original_size.get(IMAGE_TYPE_TEMPLATE)
        if template_pixmap and not template_pixmap.isNull():
            self.template_display_item = self.graphics_scene.addPixmap(template_pixmap)
            self.template_display_item.setZValue(-100)
            self.graphics_scene.setSceneRect(template_pixmap.rect().toRectF())
        else:
            QMessageBox.warning(self, "Error", "Template image not available for editing mode.")
            self.current_app_mode = APP_MODE_UPLOAD # Revert
            self._update_ui_for_mode()
            return
        
        # 2. Add extracted parts as InteractivePartItems
        # self.final_extracted_parts = { 'PartName': {'image_bgra': np.ndarray, 'position_info': dict, 'source_image_type': str} }
        z_val_counter = 0
        template_skeleton_data = self.adjusted_skeletons_data.get(IMAGE_TYPE_TEMPLATE) or self.skeletons_data.get(IMAGE_TYPE_TEMPLATE)

        for part_name, part_info_dict in self.final_extracted_parts.items():
            image_bgra = part_info_dict.get('image_bgra')
            position_info = part_info_dict.get('position_info') # This is crop info from its original source
            source_img_type = part_info_dict.get('source_image_type')

            if image_bgra is None or image_bgra.size == 0:
                print(f"Skipping part {part_name} as its image is empty.")
                continue
            
            part_pixmap = self._convert_cv_to_pixmap(image_bgra)
            if part_pixmap.isNull():
                print(f"Could not create pixmap for part {part_name}")
                continue

            interactive_item = InteractivePartItem(
                part_pixmap,
                part_name,
                original_cv_image_bgra=image_bgra,
                original_crop_info=position_info # Store how it was cropped from its source
            )
            interactive_item.setZValue(z_val_counter)
            interactive_item.transform_changed.connect(self._on_interactive_part_transformed)

            # TODO: Calculate initial placement using MockWarper
            # Need part's original keypoints (relative to its own image_bgra, or map original source keypoints)
            # For now, place at random/default
            part_original_skeleton = self.adjusted_skeletons_data.get(source_img_type) or self.skeletons_data.get(source_img_type)
            
            # We need to get keypoints that were originally inside this part's crop_rect from source_img_type
            # And transform them to be local to the cropped part_image_bgra
            # This is complex and MockPartSegmenter doesn't provide this refined local keypoint data yet.
            # For the mock, we'll just call warper without detailed keypoints.
            
            initial_pos_pt, initial_scale_x, initial_scale_y, initial_rot = \
                self.warper.calculate_initial_transform(
                    part_name, # Pass part_name
                    self.PART_DEFINITIONS, # Pass all part definitions (for indexing)
                    template_skeleton_data, # Template skeleton
                    part_original_skeleton # Skeleton from the part's original source image
                )

            interactive_item.setPos(initial_pos_pt)
            # Assuming uniform scale from warper for now. If separate sx, sy, need to adjust.
            interactive_item.setScale(initial_scale_x) # setScale handles uniform scale
            interactive_item.setRotation(initial_rot)

            self.graphics_scene.addItem(interactive_item)
            self.scene_interactive_parts[part_name] = interactive_item
            self.layers_list_widget.addItem(part_name)
            z_val_counter +=1

        if self.layers_list_widget.count() > 0:
            self.layers_list_widget.setCurrentRow(0) # Select first part

        self._fit_view()
        print(f"Displayed {len(self.scene_interactive_parts)} parts on template for editing.")


    def _on_scene_selection_changed(self):
        if self.current_app_mode != APP_MODE_EDIT_PARTS:
            return
        
        selected_items_in_scene = self.graphics_scene.selectedItems()
        # self.layers_list_widget.clearSelection() # This causes signals loop sometimes
        self.layers_list_widget.blockSignals(True)

        if not selected_items_in_scene:
            self.layers_list_widget.setCurrentItem(None) # Deselect in list
            self._update_part_properties_display(None)
        elif isinstance(selected_items_in_scene[0], InteractivePartItem):
            selected_part_item = selected_items_in_scene[0]
            part_name = selected_part_item.part_name
            list_items = self.layers_list_widget.findItems(part_name, Qt.MatchFlag.MatchExactly)
            if list_items:
                self.layers_list_widget.setCurrentItem(list_items[0])
            self._update_part_properties_display(selected_part_item)
        else: # Something else selected, or no interactive part
             self.layers_list_widget.setCurrentItem(None)
             self._update_part_properties_display(None)

        self.layers_list_widget.blockSignals(False)


    def _on_layer_list_selection_changed(self, current_qlist_item: QListWidgetItem, previous_qlist_item: QListWidgetItem):
        if self.current_app_mode != APP_MODE_EDIT_PARTS or not current_qlist_item:
            self.graphics_scene.clearSelection() # Deselect in scene if list is cleared
            self._update_part_properties_display(None)
            return

        part_name = current_qlist_item.text()
        scene_item = self.scene_interactive_parts.get(part_name)

        self.graphics_scene.blockSignals(True) # Avoid recursive signals
        self.graphics_scene.clearSelection()
        if scene_item:
            scene_item.setSelected(True)
            self.graphics_view.ensureVisible(scene_item)
            self._update_part_properties_display(scene_item)
        else:
            self._update_part_properties_display(None)
        self.graphics_scene.blockSignals(False)
        
    def _on_interactive_part_transformed(self, part_name:str):
        """Called when an InteractivePartItem's transform changes."""
        if self.current_app_mode == APP_MODE_EDIT_PARTS:
            item = self.scene_interactive_parts.get(part_name)
            if item and item.isSelected(): # Update properties only if it's the selected one
                self._update_part_properties_display(item)

    def _update_part_properties_display(self, item: InteractivePartItem = None):
        if item and isinstance(item, InteractivePartItem):
            transform_data = item.get_transform_data()
            self.prop_pos_x.setText(f"{transform_data['pos_x_scene']:.2f}")
            self.prop_pos_y.setText(f"{transform_data['pos_y_scene']:.2f}")
            self.prop_scale.setText(f"{transform_data['scale']:.3f}")
            self.prop_rotation.setText(f"{transform_data['rotation_deg']:.2f}")
        else:
            self.prop_pos_x.clear()
            self.prop_pos_y.clear()
            self.prop_scale.clear()
            self.prop_rotation.clear()

    def _change_z_order(self, direction): # direction: 1 for forward, -1 for backward, 'front', 'back'
        selected_list_items = self.layers_list_widget.selectedItems()
        if not selected_list_items: return
        current_list_item = selected_list_items[0]
        part_name = current_list_item.text()
        item_to_move = self.scene_interactive_parts.get(part_name)
        if not item_to_move: return

        current_z = item_to_move.zValue()
        all_parts_sorted_by_z = sorted(self.scene_interactive_parts.values(), key=lambda x: x.zValue())
        
        # Re-assign Z values to be contiguous to simplify logic if they are not
        # This is important for simple forward/backward if Zs are sparse
        current_max_z = -1
        if all_parts_sorted_by_z:
            current_max_z = all_parts_sorted_by_z[-1].zValue()

        if direction == 1: # Bring Forward
            # Find next item with higher Z, swap Z with it
            # This is simpler if Z values are dense (0, 1, 2...).
            # For now, just increment Z, then re-sort and re-assign all Zs to maintain order.
            item_to_move.setZValue(current_z + 1.5) # Move it slightly above
        elif direction == -1: # Send Backward
            item_to_move.setZValue(current_z - 1.5) # Move it slightly below
        # elif direction == 'front':
        #     item_to_move.setZValue(current_max_z + 1)
        # elif direction == 'back':
        #      item_to_move.setZValue(all_parts_sorted_by_z[0].zValue() -1 if all_parts_sorted_by_z else 0)


        # Re-normalize Z values for all interactive parts to maintain drawing order
        # and update list widget order
        updated_all_parts_sorted = sorted(self.scene_interactive_parts.values(), key=lambda x: x.zValue())
        
        self.layers_list_widget.blockSignals(True)
        self.layers_list_widget.clear()
        for i, part_item in enumerate(updated_all_parts_sorted):
            part_item.setZValue(i) # Normalize Z from 0 upwards
            self.layers_list_widget.addItem(part_item.part_name)
            if part_item == item_to_move:
                self.layers_list_widget.setCurrentRow(i) # Keep selection on moved item
        self.layers_list_widget.blockSignals(False)
        
        self.graphics_scene.update() # Force repaint
        print(f"Adjusted Z-order for {part_name}.")


    # --- Action Methods (Button Clicks) ---
    def _action_start_pose_detection(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Processing poses...")
        try:
            # Check if pose detector is functional (models loaded)
            if self.pose_detector is None or \
               not isinstance(self.pose_detector, RealPoseDetector) or \
               self.pose_detector.detector is None or \
               self.pose_detector.pose_estimator is None:
                 QMessageBox.warning(self, "Pose Detection Error",
                                    "Real pose detection models are not loaded. Please check DWPose path and model files, or previous console errors. Application may have fallen back to a non-functional detector.")
                 QApplication.restoreOverrideCursor()
                 self.statusBar().showMessage("Pose detection models not loaded. Cannot proceed.")
                 return


            for img_type, img_cv in self.image_cv_originals.items():
                if img_cv is not None:
                    print(f"Detecting pose for {img_type}...")
                    # Pass the string identifier for the image type
                    detected_data = self.pose_detector.detect_pose(img_cv, img_type)
                    self.skeletons_data[img_type] = detected_data
                    # Initialize adjusted data with original data
                    if self.skeletons_data[img_type] and self.skeletons_data[img_type].get('keypoints'):
                        self.adjusted_skeletons_data[img_type] = {
                            'keypoints': list(self.skeletons_data[img_type]['keypoints']), # Deep copy list of lists
                            'source_image_size': self.skeletons_data[img_type]['source_image_size']
                        }
                    else: # if pose detection failed or returned empty
                         self.adjusted_skeletons_data[img_type] = {
                             'keypoints': [], 
                             'source_image_size': (img_cv.shape[1], img_cv.shape[0]) if img_cv is not None else (0,0)
                        }


                else:
                    QMessageBox.warning(self, "Missing Image", f"{img_type.capitalize()} image not uploaded. Cannot detect pose.")
                    QApplication.restoreOverrideCursor() # Restore cursor before early return
                    self.statusBar().showMessage(f"Missing {img_type} image.")
                    return # Important: stop processing if an image is missing
            
            self.current_app_mode = APP_MODE_ADJUST_SKELETON
            # Default to showing template skeleton first, or head if no template processing
            self.current_image_type_for_adjustment = IMAGE_TYPE_TEMPLATE 
            self._switch_skeleton_adjustment_view(self.current_image_type_for_adjustment) # Display the first one

        except Exception as e:
            QMessageBox.critical(self, "Pose Detection Error", f"Error during pose detection: {e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_for_mode()

    def _action_reset_current_skeleton_adj(self):
        if self.current_image_type_for_adjustment and self.skeletons_data.get(self.current_image_type_for_adjustment):
            # Reset adjusted data to original detected data
            original_data = self.skeletons_data[self.current_image_type_for_adjustment]
            self.adjusted_skeletons_data[self.current_image_type_for_adjustment] = {
                'keypoints': list(original_data['keypoints']),
                'source_image_size': original_data['source_image_size']
            }
            # Re-display
            self._display_skeleton_on_image(self.current_image_type_for_adjustment, original_data)
            print(f"Skeleton for {self.current_image_type_for_adjustment} reset to original.")
        else:
            QMessageBox.information(self, "Info", "No skeleton data to reset for the current view.")

    def _action_confirm_all_skeleton_adjustments(self):
        # Save adjustments for the currently viewed skeleton first
        self._save_current_skeleton_adjustments()

        # Check if all necessary skeletons are there (at least head and body)
        if not (self.adjusted_skeletons_data.get(IMAGE_TYPE_HEAD) and \
                self.adjusted_skeletons_data.get(IMAGE_TYPE_BODY)):
            QMessageBox.warning(self, "Data Missing", "Skeleton data for Head or Body is missing after adjustment. Cannot proceed.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Pre-segmenting parts...")
        try:
            # Perform pre-segmentation for Head and Body images
            for img_type in [IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                current_skeleton = self.adjusted_skeletons_data.get(img_type)
                original_image = self.image_cv_originals.get(img_type)
                if current_skeleton and original_image is not None:
                    # MockPartSegmenter expects part_definitions
                    # For head image, only segment "Head". For body, segment all other parts.
                    defs_for_img = {}
                    if img_type == IMAGE_TYPE_HEAD:
                        if "Head" in self.PART_DEFINITIONS:
                            defs_for_img["Head"] = self.PART_DEFINITIONS["Head"]
                    elif img_type == IMAGE_TYPE_BODY:
                        for pname, pindices in self.PART_DEFINITIONS.items():
                            if pname != "Head": # Don't segment head from body image, unless explicitly designed
                                defs_for_img[pname] = pindices
                    
                    if defs_for_img:
                         self.pre_segmented_parts_polygons[img_type] = \
                            self.part_segmenter.pre_segment(original_image, current_skeleton, defs_for_img)
                    else:
                         self.pre_segmented_parts_polygons[img_type] = {} # No parts to define for this image type

            self.current_app_mode = APP_MODE_ADJUST_SEGMENTATION
            # Default to showing Head image for segmentation adjustment
            self.current_image_type_for_adjustment = IMAGE_TYPE_HEAD
            self._switch_segmentation_adjustment_view(self.current_image_type_for_adjustment)

        except Exception as e:
            QMessageBox.critical(self, "Pre-segmentation Error", f"Error during part pre-segmentation: {e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_for_mode()

    def _action_confirm_all_segment_adjustments(self):
        self._save_current_segmentation_adjustments() # Save current view's polygons
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Extracting final parts...")
        self.final_extracted_parts.clear()
        try:
            total_parts_extracted = 0
            for img_type, parts_dict in self.pre_segmented_parts_polygons.items():
                # img_type is IMAGE_TYPE_HEAD or IMAGE_TYPE_BODY
                original_image_cv = self.image_cv_originals.get(img_type)
                if original_image_cv is None:
                    print(f"Error: Original image for {img_type} not found for final segmentation.")
                    continue

                for part_name, poly_data in parts_dict.items():
                    # poly_data should be {'initial_polygon': QPolygonF, ...}
                    # 'initial_polygon' now holds the (potentially) edited polygon
                    edited_qpolygon = poly_data.get('initial_polygon')
                    if edited_qpolygon and not edited_qpolygon.isEmpty():
                        part_image_bgra, part_pos_info = \
                            self.part_segmenter.final_segment(original_image_cv, part_name, edited_qpolygon)
                        
                        if part_image_bgra is not None and part_image_bgra.size > 0:
                            self.final_extracted_parts[part_name] = {
                                'image_bgra': part_image_bgra,
                                'position_info': part_pos_info, # Crop info from original source
                                'source_image_type': img_type
                            }
                            total_parts_extracted += 1
                        else:
                            print(f"Warning: Final segmentation for {part_name} from {img_type} resulted in an empty image.")
                    else:
                        print(f"Warning: No polygon data to segment {part_name} from {img_type}.")
            
            if total_parts_extracted == 0:
                QMessageBox.warning(self, "No Parts Extracted", "No valid parts were extracted after segmentation. Cannot proceed to editing.")
                QApplication.restoreOverrideCursor() # Restore cursor
                self.statusBar().showMessage("No parts extracted.")
                return # Stay in segmentation adjustment mode or revert

            print(f"Total {total_parts_extracted} parts extracted. Proceeding to edit mode.")
            self.current_app_mode = APP_MODE_EDIT_PARTS
            self._display_parts_on_template() # This will also populate the layer list

        except Exception as e:
            QMessageBox.critical(self, "Final Segmentation Error", f"Error during final part extraction: {e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_for_mode()


    def _action_export_final_image(self):
        if not self.template_display_item:
            QMessageBox.warning(self, "Export Error", "Template image not loaded in scene.")
            return
        if not self.scene_interactive_parts:
            QMessageBox.warning(self, "Export Error", "No parts are on the template to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Final Image", DEFAULT_EXPORT_DIR, "PNG Image (*.png)"
        )
        if not file_path: return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Exporting image...")
        try:
            # Scene rect should be based on the template
            target_rect = self.template_display_item.boundingRect()
            export_qimage = QImage(target_rect.size().toSize(), QImage.Format.Format_ARGB32_Premultiplied)
            export_qimage.fill(Qt.GlobalColor.transparent)

            painter = QPainter(export_qimage)
            # Ensure high quality rendering for export
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            # Important: render items in their Z-order by sorting them.
            # QGraphicsScene.render() itself respects Z-order if items are correctly set.
            # For safety, can fetch items and sort by Z before manually painting if issues.
            # However, self.scene.render() should be sufficient.
            
            # Deselect items before rendering to avoid selection boxes in export
            current_selection = self.graphics_scene.selectedItems()
            self.graphics_scene.clearSelection()

            self.graphics_scene.render(painter, QRectF(export_qimage.rect()), target_rect)
            painter.end()

            if export_qimage.save(file_path):
                QMessageBox.information(self, "Export Successful", f"Image saved to:\\n{file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to save the image.")

            # Restore selection if any
            for item in current_selection: item.setSelected(True)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export: {e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Export complete. Ready for new task or further edits.")
            # Optionally, reset to upload mode or offer to start new
            # self.current_app_mode = APP_MODE_UPLOAD
            # self._reset_all_data()
            # self._update_ui_for_mode()


    def _reset_all_data(self): # Call when starting new or after export if desired
        self.image_paths = {img_type: None for img_type in self.image_paths}
        self.image_cv_originals = {img_type: None for img_type in self.image_cv_originals}
        self.image_pixmaps_original_size = {img_type: None for img_type in self.image_pixmaps_original_size}
        
        for lbl_type, lbl in self.preview_labels.items():
            lbl.setText(f"{lbl_type.capitalize()} Preview")
            lbl.setPixmap(QPixmap())
        self.lbl_template_file.setText("None")
        self.lbl_head_file.setText("None")
        self.lbl_body_file.setText("None")

        self.skeletons_data = {img_type: None for img_type in self.skeletons_data}
        self.adjusted_skeletons_data = {img_type: None for img_type in self.adjusted_skeletons_data}
        self.pre_segmented_parts_polygons = {img_type: {} for img_type in self.pre_segmented_parts_polygons}
        self.final_extracted_parts.clear()
        
        self._clear_scene()
        self.current_app_mode = APP_MODE_UPLOAD
        self.current_image_type_for_adjustment = None
        self.current_part_for_segment_adjustment = None
        self.statusBar().showMessage("Data reset. Please upload new images.")
        self._update_ui_for_mode()


    def closeEvent(self, event):
        # Clean up any resources if necessary
        # For example, stop any running threads from core logic
        print("Closing application...")
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Potentially set application style here, e.g., app.setStyle('Fusion')
    # Apply a stylesheet for better look and feel (optional)
    # try:
    #     with open("stylesheet.qss", "r") as f:
    #         app.setStyleSheet(f.read())
    # except FileNotFoundError:
    #     print("Stylesheet not found, using default.")

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec()) 