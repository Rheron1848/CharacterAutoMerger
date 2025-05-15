from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle, Ellipse, PushMatrix, PopMatrix, Translate, Scale as KivyScale, Mesh, Rotate
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, ListProperty, StringProperty, NumericProperty, BooleanProperty, DictProperty
from kivy.utils import get_color_from_hex
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path

import os
import math
import time
import traceback
import numpy as np
import cv2
import torch
from PIL import Image as PILImage, ImageDraw as PILImageDraw
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

try:
    from plyer import filechooser
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    # Define a dummy filechooser for graceful degradation if plyer is not found.
    class filechooser_dummy:
        def open_file(self, **kwargs):
            print("PLYER NOT FOUND: Cannot open native file dialog.")
            # Simulate no file selected
            return None
    filechooser = filechooser_dummy()
    print("Plyer module not found. Native file chooser will be unavailable. App will use a dummy fallback.")

# --- MMDetection and MMPose Imports ---
# Ensure these are importable in your environment
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model as init_pose_estimator
    from mmdet.utils import register_all_modules as register_det_modules
    from mmpose.utils import register_all_modules as register_pose_modules
    MMDET_MMPOS_AVAILABLE = True
    print("MMDetection, MMPose, and MMEngine components seem loadable.")
except ImportError as e:
    MMDET_MMPOS_AVAILABLE = False
    print(f"MMDetection/MMPose Import Error: {e}. Pose detection functionality will be disabled.")
    # Define dummy functions if imports fail, so the app can still run (with errors for pose detection)
    def init_detector(*args, **kwargs):
        raise ImportError("MMDet init_detector not available")
    def inference_detector(*args, **kwargs):
        raise ImportError("MMDet inference_detector not available")
    def init_pose_estimator(*args, **kwargs):
        raise ImportError("MMPose init_pose_estimator not available")
    def inference_topdown(*args, **kwargs):
        raise ImportError("MMPose inference_topdown not available")
    def register_det_modules(): pass
    def register_pose_modules(): pass


# --- Configuration (User might need to adjust these) ---
DWPOSE_DIR = r"D:\drafts\DWPose" # Make sure this path is correct for your system
DEFAULT_EXPORT_DIR = os.path.join(os.getcwd(), "CharacterAutoMerger_Exports_Kivy")

# Model configurations (copied for now, adjust if RealPoseDetector is fully integrated)
MODEL_CONFIGS = {
    'det_config_filename': 'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py',
    'det_checkpoint_filename': 'weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
    'pose_config_filename': 'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
    'pose_checkpoint_filename': 'weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'det_confidence_threshold': 0.3,
}

# Constants
IMAGE_TYPE_TEMPLATE = "template"
IMAGE_TYPE_HEAD = "head"
IMAGE_TYPE_BODY = "body"

APP_MODE_UPLOAD = "upload"
APP_MODE_ADJUST_SKELETON = "adjust_skeleton"
APP_MODE_ADJUST_SEGMENTATION = "adjust_segmentation"
APP_MODE_EDIT_PARTS = "edit_parts"

# Part Definitions (COCO 17 keypoints based, adjusted with user feedback)
PART_DEFINITIONS = {
    "Head": [0, 1, 2, 3, 4],        # Nose, LEye, REye, LEar, REar
    "Torso": [5, 6, 11, 12],        # LShoulder, RShoulder, LHip, RHip
    "WaistHip": [11, 12],         # LHip, RHip - More focused hip/waist area
    "LeftLeg": [11, 13, 15],       # LHip, LKnee, LAnkle
    "RightLeg": [12, 14, 16],      # RHip, RKnee, RAnkle
    "LeftArm": [5, 7, 9],          # LShoulder, LElbow, LWrist
    "RightArm": [6, 8, 10]         # RShoulder, RElbow, RWrist
}

# 部位颜色定义
PART_COLORS = {
    "Head": [0.2, 0.7, 1.0, 0.5],
    "Torso": [0.7, 0.2, 1.0, 0.5],
    "LeftArm": [0.2, 1.0, 0.3, 0.5],
    "RightArm": [1.0, 0.5, 0.2, 0.5],
    "LeftLeg": [1.0, 0.8, 0.2, 0.5],
    "RightLeg": [0.5, 0.8, 1.0, 0.5],
    "WaistHip": [0.8, 0.2, 0.5, 0.5]
}

# Bone Connections (COCO 17 keypoints)
BONE_CONNECTIONS = [
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

# Helper to get full model paths (already exists, ensure it's identical or compatible)
def get_model_path(base_dir, file_path_suffix):
    return os.path.join(base_dir, file_path_suffix)

# Function to show Kivy Popup for errors (replaces QMessageBox)
def show_error_popup(title, message):
    content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
    content.add_widget(Label(text=message, size_hint_y=None, height=dp(80)))
    btn_ok = Button(text="OK", size_hint_y=None, height=dp(44))
    content.add_widget(btn_ok)
    popup = Popup(title=title, content=content, size_hint=(0.75, 0.25), auto_dismiss=False)
    btn_ok.bind(on_release=popup.dismiss)
    popup.open()

class RealPoseDetector:
    def __init__(self, dwpose_base_dir, model_configs_dict):
        self.dwpose_base_dir = dwpose_base_dir
        self.model_configs = model_configs_dict
        self.device = self.model_configs['device']
        self.detector = None
        self.pose_estimator = None
        if MMDET_MMPOS_AVAILABLE:
            self._init_models()
        else:
            msg = "RealPoseDetector: MMDetection/MMPose not available. Cannot initialize models."
            print(msg)
            # show_error_popup("Model Init Warning", msg) # Avoid popup on init, just print

    def _init_models(self):
        try:
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
            msg = f"RealPoseDetector: Error initializing models: {e}"
            print(msg)
            traceback.print_exc()
            self.detector = None
            self.pose_estimator = None
            # show_error_popup("Model Initialization Error", msg) # Avoid popup on init, just print

    def detect_pose(self, image_cv, image_type_str="unknown"):
        print(f"RealPoseDetector: Detecting pose for {image_type_str}...")
        if not MMDET_MMPOS_AVAILABLE or self.detector is None or self.pose_estimator is None:
            msg = "RealPoseDetector: Models not available or not initialized. Returning empty keypoints."
            print(msg)
            # show_error_popup("Pose Detection Error", msg) # Can be noisy, perhaps return error code
            return {'keypoints': [], 'source_image_size': (image_cv.shape[1], image_cv.shape[0]) if image_cv is not None else (0,0)}
        
        h_orig, w_orig = image_cv.shape[:2]
        try:
            # 检查图像是否有透明通道（第4通道）
            if len(image_cv.shape) == 3 and image_cv.shape[2] == 4:
                print(f"检测到透明通道图像，为姿势检测添加灰色背景: {image_type_str}")
                # 创建灰色背景图像
                gray_bg = np.ones((h_orig, w_orig, 3), dtype=np.uint8) * 200  # 浅灰色背景
                
                # 提取alpha通道
                alpha = image_cv[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                
                # 提取RGB通道 (转换BGRA到BGR)
                rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)
                
                # 合成图像：RGB * alpha + 灰色背景 * (1-alpha)
                detection_image = (rgb * alpha + gray_bg * (1 - alpha)).astype(np.uint8)
                
                print(f"已为透明图像添加灰色背景用于骨架检测")
            else:
                detection_image = image_cv.copy()
            
            register_det_modules()
            det_results_datasample = inference_detector(self.detector, detection_image)
            pred_instances = det_results_datasample.pred_instances
            bboxes_with_scores = pred_instances.bboxes[pred_instances.scores > self.model_configs['det_confidence_threshold']]

            if bboxes_with_scores.shape[0] == 0:
                print(f"RealPoseDetector: No person detected with confidence > {self.model_configs['det_confidence_threshold']} for {image_type_str}.")
                return {'keypoints': [], 'source_image_size': (w_orig, h_orig)}

            bboxes_np = bboxes_with_scores.cpu().numpy() if isinstance(bboxes_with_scores, torch.Tensor) else bboxes_with_scores

            register_pose_modules()
            pose_results_datasamples = inference_topdown(self.pose_estimator, detection_image, bboxes_np)
            
            processed_keypoints = []
            if pose_results_datasamples:
                main_person_pose_instances = pose_results_datasamples[0].pred_instances
                keypoints_tensor = main_person_pose_instances.keypoints[0] 
                keypoint_scores_tensor = main_person_pose_instances.keypoint_scores[0]

                kps_np = keypoints_tensor.cpu().numpy() if isinstance(keypoints_tensor, torch.Tensor) else keypoints_tensor
                scores_np = keypoint_scores_tensor.cpu().numpy() if isinstance(keypoint_scores_tensor, torch.Tensor) else keypoint_scores_tensor

                for i in range(kps_np.shape[0]):
                    processed_keypoints.append([float(kps_np[i, 0]), float(kps_np[i, 1]), float(scores_np[i])])
            
            print(f"RealPoseDetector: Found {len(processed_keypoints)} keypoints for {image_type_str}.")
            return {'keypoints': processed_keypoints, 'source_image_size': (w_orig, h_orig)}
        except Exception as e:
            msg = f"RealPoseDetector: Error during pose detection for {image_type_str}: {e}"
            print(msg)
            traceback.print_exc()
            # show_error_popup("Pose Detection Runtime Error", msg)
            return {'keypoints': [], 'source_image_size': (w_orig, h_orig)}


# --- Kivy Drawing Widget ---
class DrawingCanvas(Widget):
    texture = ObjectProperty(None)
    image_original_size = ListProperty([0, 0]) # Store original W, H of the image for coord mapping
    skeleton_data = ListProperty([]) # List of [x,y,conf] scaled to original image size
    polygons_data = ListProperty([]) # 多边形数据: [{'part_name': 'PartName', 'points': [[x,y],...], 'color': [r,g,b,a]}]
    # interactive_parts = ListProperty([]) # For later

    # Properties for scaling and offset to draw image centered and fit
    display_scale = NumericProperty(1.0)
    display_offset_x = NumericProperty(0.0)
    display_offset_y = NumericProperty(0.0)
    
    # Dragging state for keypoints
    selected_keypoint_idx = NumericProperty(-1)  # -1 means no keypoint selected
    highlight_radius = NumericProperty(10.0)  # Highlight radius for selection
    
    # 分割多边形编辑状态
    selected_polygon_idx = NumericProperty(-1)  # 当前选中的多边形索引
    selected_polygon_point_idx = NumericProperty(-1)  # 当前选中的多边形顶点索引
    edit_mode = StringProperty('skeleton')  # 'skeleton' 或 'polygon'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Bind primary properties that affect display to _redraw
        # _redraw will internally call _update_display_params
        self.bind(size=self._redraw, pos=self._redraw,
                  texture=self._redraw, image_original_size=self._redraw,
                  skeleton_data=self._redraw, # For skeleton visibility changes
                  polygons_data=self._redraw, # 绑定多边形数据以触发重绘
                  # Also bind to internal display params if they change from elsewhere (though not typical for this setup)
                  display_scale=self._redraw, 
                  display_offset_x=self._redraw,
                  display_offset_y=self._redraw)
                  
        # Register touch events for keypoint dragging
        self.register_event_type('on_keypoint_moved')
        self.register_event_type('on_keypoint_selected')
        
        # 注册多边形编辑事件
        self.register_event_type('on_polygon_point_moved')
        self.register_event_type('on_polygon_selected')
        self.register_event_type('on_polygon_point_added')
        self.register_event_type('on_polygon_point_deleted')
        
        # 用于跟踪双击事件
        self.last_touch_pos = None
        self.last_touch_time = 0
        self.double_click_timeout = 0.3  # 双击超时时间（秒）
        
        print("DrawingCanvas已初始化，准备进行关节点和分割区域编辑")

    def _update_display_params(self, *args): # Pass *args to accept kivy property change event args
        # print(f"DrawingCanvas ({self.id if hasattr(self, 'id') else 'no_id'}): _update_display_params triggered by: {args}")
        if self.texture and self.image_original_size[0] > 0 and self.image_original_size[1] > 0 and self.width > 0 and self.height > 0:
            img_w, img_h = self.image_original_size
            
            scale_w = self.width / img_w
            scale_h = self.height / img_h
            new_display_scale = min(scale_w, scale_h) * 0.9  # 添加0.9系数使图像显示有一些边距
            
            scaled_img_w = img_w * new_display_scale
            scaled_img_h = img_h * new_display_scale
            new_display_offset_x = (self.width - scaled_img_w) / 2
            new_display_offset_y = (self.height - scaled_img_h) / 2
            
            # Update properties only if they actually change to avoid redraw loops if not careful
            if abs(self.display_scale - new_display_scale) > 1e-6 : self.display_scale = new_display_scale
            if abs(self.display_offset_x - new_display_offset_x) > 1e-6 : self.display_offset_x = new_display_offset_x
            if abs(self.display_offset_y - new_display_offset_y) > 1e-6 : self.display_offset_y = new_display_offset_y
            
            print(f"更新显示参数: 比例: {self.display_scale:.3f}, 偏移: ({self.display_offset_x:.1f}, {self.display_offset_y:.1f})")

        else:
            # Reset if no texture or invalid sizes
            if self.display_scale != 1.0: self.display_scale = 1.0
            if self.display_offset_x != 0.0: self.display_offset_x = 0.0
            if self.display_offset_y != 0.0: self.display_offset_y = 0.0
            print("重置显示参数: 无效的纹理或尺寸")
        # No canvas.ask_update() here, property changes will trigger _redraw if bound

    def on_keypoint_moved(self, keypoint_idx, new_x, new_y):
        if 0 <= keypoint_idx < len(self.skeleton_data):
            old_conf = self.skeleton_data[keypoint_idx][2]
            # Update the skeleton data with new coordinates, preserve confidence
            self.skeleton_data[keypoint_idx] = [new_x, new_y, old_conf]
            print(f"关节点 {keypoint_idx} 已更新位置到 ({new_x:.1f}, {new_y:.1f})")
            return True
        return False
    
    def on_keypoint_selected(self, keypoint_idx):
        # This is just a placeholder event - could be used to update UI or show specific keypoint info
        print(f"关节点选中事件: 关节点 {keypoint_idx}")
        pass

    def on_skeleton_data(self, instance, value):
        # This is a Kivy property observer, already bound to _redraw in __init__
        # self.canvas.ask_update() # Not strictly needed if skeleton_data is bound to _redraw
        pass # _redraw will handle it

    def screen_to_image_coords(self, x, y):
        """Convert screen coordinates to image coordinates"""
        if self.image_original_size[0] == 0 or self.image_original_size[1] == 0:
            return None
            
        # 首先将屏幕坐标转换为控件内坐标
        x_local = x - self.pos[0]
        y_local = y - self.pos[1]
            
        # 调整为图像偏移和缩放
        img_x = (x_local - self.display_offset_x) / self.display_scale
        # Y坐标在Kivy中是从底部向上，需要为图像坐标系调整
        img_y = (y_local - self.display_offset_y) / self.display_scale
        
        # 图像坐标系转换：Kivy的Y坐标是从底部向上，需要转换为从顶部向下
        img_y = self.image_original_size[1] - img_y
        
        # 检查点是否在图像边界内
        if 0 <= img_x <= self.image_original_size[0] and 0 <= img_y <= self.image_original_size[1]:
            return (img_x, img_y)
        return None
    
    def find_keypoint_at(self, touch_x, touch_y, search_radius=30):  # 增大搜索半径，使关节点更容易选中
        """Find a keypoint near the touch position within search radius"""
        if not self.skeleton_data:
            return -1
            
        # Get original image coordinates
        img_coords = self.screen_to_image_coords(touch_x, touch_y)
        if not img_coords:
            return -1
            
        # Scale search radius based on display scale
        scaled_search_radius = search_radius / self.display_scale
        
        # Check each keypoint
        closest_kp_idx = -1
        closest_distance = float('inf')
        
        for i, kp in enumerate(self.skeleton_data):
            if kp[2] < 0.1:  # Skip low confidence keypoints
                continue
                
            # Calculate distance in image coordinates
            kp_x, kp_y, _ = kp
            dist = ((kp_x - img_coords[0]) ** 2 + (kp_y - img_coords[1]) ** 2) ** 0.5
            
            if dist < scaled_search_radius and dist < closest_distance:
                closest_distance = dist
                closest_kp_idx = i
                
        return closest_kp_idx

    def on_polygon_point_moved(self, polygon_idx, point_idx, new_x, new_y):
        """当多边形点被移动时触发"""
        if 0 <= polygon_idx < len(self.polygons_data) and 0 <= point_idx < len(self.polygons_data[polygon_idx]['points']):
            # 更新多边形点的位置
            self.polygons_data[polygon_idx]['points'][point_idx] = [new_x, new_y]
            print(f"多边形 {self.polygons_data[polygon_idx]['part_name']} 的点 {point_idx} 已移动到 ({new_x:.1f}, {new_y:.1f})")
            return True
        return False
    
    def on_polygon_selected(self, polygon_idx):
        """当多边形被选中时触发"""
        print(f"多边形选中事件: {polygon_idx}")
        pass

    def find_polygon_point_at(self, touch_x, touch_y, search_radius=15):
        """查找触摸位置附近的多边形顶点"""
        if not self.polygons_data:
            return -1, -1
            
        # 获取原始图像坐标
        img_coords = self.screen_to_image_coords(touch_x, touch_y)
        if not img_coords:
            return -1, -1
            
        # 根据显示比例缩放搜索半径
        scaled_search_radius = search_radius / self.display_scale
        
        # 检查每个多边形的每个点
        closest_poly_idx = -1
        closest_point_idx = -1
        closest_distance = float('inf')
        
        for poly_idx, poly_data in enumerate(self.polygons_data):
            for point_idx, point in enumerate(poly_data['points']):
                # 计算距离
                dist = ((point[0] - img_coords[0]) ** 2 + (point[1] - img_coords[1]) ** 2) ** 0.5
                
                if dist < scaled_search_radius and dist < closest_distance:
                    closest_distance = dist
                    closest_poly_idx = poly_idx
                    closest_point_idx = point_idx
                    
        return closest_poly_idx, closest_point_idx
    
    def on_touch_down(self, touch):
        """处理触摸按下事件以选择关键点或多边形顶点"""
        if super(DrawingCanvas, self).on_touch_down(touch):
            return True
            
        # 只处理此小部件内的触摸
        if not self.collide_point(touch.x, touch.y):
            return False
            
        # 处理双击逻辑
        current_time = time.time()
        if self.last_touch_pos and current_time - self.last_touch_time < self.double_click_timeout:
            # 计算与上次触摸位置的距离，确认是双击而非拖动
            distance = ((touch.x - self.last_touch_pos[0]) ** 2 + (touch.y - self.last_touch_pos[1]) ** 2) ** 0.5
            if distance < 20:  # 如果距离小于20像素，视为双击
                self._handle_double_click(touch)
                self.last_touch_pos = None  # 重置位置，避免连续双击
                return True
                
        # 保存当前触摸信息，用于检测下一次可能的双击
        self.last_touch_pos = (touch.x, touch.y)
        self.last_touch_time = current_time
            
        if self.edit_mode == 'skeleton':
            # 尝试选择关键点
            kp_idx = self.find_keypoint_at(touch.x, touch.y)
            self.selected_keypoint_idx = kp_idx
            
            if kp_idx >= 0:
                print(f"关节点选中: {kp_idx}")
                self.dispatch('on_keypoint_selected', kp_idx)
                touch.grab(self)
                return True
        
        elif self.edit_mode == 'polygon':
            # 尝试选择多边形顶点
            poly_idx, point_idx = self.find_polygon_point_at(touch.x, touch.y)
            
            if poly_idx >= 0 and point_idx >= 0:
                print(f"多边形点选中: 多边形 {poly_idx}, 点 {point_idx}")
                self.selected_polygon_idx = poly_idx
                self.selected_polygon_point_idx = point_idx
                self.dispatch('on_polygon_selected', poly_idx)
                touch.grab(self)
                return True
                
        return False
        
    def on_touch_move(self, touch):
        """处理触摸移动事件以拖动关键点或多边形顶点"""
        if super(DrawingCanvas, self).on_touch_move(touch):
            return True
            
        # 只有在我们抓取了这个触摸并且有选中的点时才处理
        if touch.grab_current is not self:
            return False
            
        # 转换触摸位置为图像坐标
        img_coords = self.screen_to_image_coords(touch.x, touch.y)
        if not img_coords:
            return False
            
        img_x, img_y = img_coords
        
        if self.edit_mode == 'skeleton' and self.selected_keypoint_idx >= 0:
            # 更新关键点位置并分发事件
            print(f"关节点移动: {self.selected_keypoint_idx} 到 ({img_x:.1f}, {img_y:.1f})")
            self.dispatch('on_keypoint_moved', self.selected_keypoint_idx, img_x, img_y)
            return True
            
        elif self.edit_mode == 'polygon' and self.selected_polygon_idx >= 0 and self.selected_polygon_point_idx >= 0:
            # 确保多边形数据存在
            if self.selected_polygon_idx < len(self.polygons_data):
                # 更新多边形顶点位置并分发事件
                print(f"多边形点移动: 多边形 {self.selected_polygon_idx}, 点 {self.selected_polygon_point_idx} 到 ({img_x:.1f}, {img_y:.1f})")
                self.dispatch('on_polygon_point_moved', self.selected_polygon_idx, self.selected_polygon_point_idx, img_x, img_y)
                self._redraw()  # 强制重绘
                return True
            
        return False
    
    def on_touch_up(self, touch):
        """处理触摸释放事件"""
        if super(DrawingCanvas, self).on_touch_up(touch):
            return True
            
        # 如果我们抓取了触摸，释放它
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
            
        return False

    def draw_skeleton(self):
        if not self.skeleton_data: return

        # Get original image height for Y-coordinate transformation
        img_w_orig, img_h_orig = self.image_original_size
        if img_h_orig == 0: 
            # print("Warning: DrawingCanvas.draw_skeleton - image_original_size height is 0. Cannot draw skeleton correctly.")
            return # Avoid issues if original image height is unknown

        with self.canvas.after: # Ensure drawing happens *after* background image
            PushMatrix()
            Translate(self.pos[0] + self.display_offset_x, self.pos[1] + self.display_offset_y) 
            KivyScale(self.display_scale, self.display_scale, 1)

            # Draw bones
            Color(0, 1, 0, 0.8) # Green for bones
            for kp_idx1, kp_idx2 in BONE_CONNECTIONS:
                if 0 <= kp_idx1 < len(self.skeleton_data) and 0 <= kp_idx2 < len(self.skeleton_data):
                    p1_data = self.skeleton_data[kp_idx1]
                    p2_data = self.skeleton_data[kp_idx2]
                    if p1_data[2] > 0.1 and p2_data[2] > 0.1:
                        # Apply Y-coordinate transformation from top-left origin to bottom-left origin
                        y1_kivy = img_h_orig - p1_data[1]
                        y2_kivy = img_h_orig - p2_data[1]
                        Line(points=[p1_data[0], y1_kivy, p2_data[0], y2_kivy], width=1.5)
            
            # Draw keypoints
            for i, kp in enumerate(self.skeleton_data):
                x, y_original, conf = kp[0], kp[1], kp[2]
                if conf > 0.1: 
                    # Highlight selected keypoint
                    if i == self.selected_keypoint_idx:
                        Color(0, 0.8, 1, 0.9)  # Cyan for selected
                        radius = 8  # 增大选中关节点的半径，更容易看到
                    else:
                        if conf > 0.5: Color(1,0,0,0.9) # Red for high confidence
                        elif conf > 0.2: Color(1,1,0,0.8) # Yellow for medium
                        else: Color(0,1,1,0.7) # Cyan for low
                        radius = 5  # 增大普通关节点的半径，更容易点击
                    
                    # Apply Y-coordinate transformation
                    y_kivy = img_h_orig - y_original
                    # Ellipse pos is its bottom-left corner. Adjust for centering.
                    Ellipse(pos=(x - radius, y_kivy - radius), size=(radius * 2, radius * 2))
            PopMatrix()

    def draw_polygons(self):
        """绘制分割多边形，使用优化算法减少性能开销"""
        if not self.polygons_data:
            return
            
        # 获取原始图像高度用于Y坐标转换
        img_w_orig, img_h_orig = self.image_original_size
        if img_h_orig == 0:
            return
            
        with self.canvas.after:
            PushMatrix()
            Translate(self.pos[0] + self.display_offset_x, self.pos[1] + self.display_offset_y)
            KivyScale(self.display_scale, self.display_scale, 1)
            
            # 绘制每个多边形
            for poly_idx, poly_data in enumerate(self.polygons_data):
                part_name = poly_data['part_name']
                points = poly_data['points']
                color = poly_data.get('color', PART_COLORS.get(part_name, [0.5, 0.5, 0.5, 0.5]))
                
                if len(points) < 3:  # 至少需要3个点
                    continue
                
                # 转换点坐标为Kivy坐标系
                kivy_points = []
                for point in points:
                    x, y_orig = point
                    y_kivy = img_h_orig - y_orig  # 转换Y坐标
                    kivy_points.extend([x, y_kivy])
                
                # 绘制填充多边形
                if len(kivy_points) >= 6:  # 至少需要3个点（每个点2个坐标值）
                    # 使用半透明颜色填充多边形
                    fill_color = color.copy()  # 创建颜色副本
                    # 确保透明度值在0.2到0.4之间，减少叠加效果的视觉干扰
                    fill_color[3] = max(0.2, min(0.4, fill_color[3]))
                    Color(*fill_color)
                    
                    try:
                        # 创建顶点数据（x, y, u, v格式）
                        num_points = len(kivy_points) // 2
                        vertices = []
                        indices = []
                        
                        # 计算多边形中心点
                        center_x = sum(kivy_points[i] for i in range(0, len(kivy_points), 2)) / num_points
                        center_y = sum(kivy_points[i+1] for i in range(0, len(kivy_points), 2)) / num_points
                        
                        # 中心点作为第一个顶点
                        vertices.extend([center_x, center_y, 0, 0])
                        
                        # 添加多边形的所有顶点
                        for i in range(0, len(kivy_points), 2):
                            vertices.extend([kivy_points[i], kivy_points[i+1], 0, 0])
                        
                        # 创建三角形扇形的索引
                        for i in range(1, num_points + 1):
                            indices.extend([0, i, i+1 if i < num_points else 1])
                        
                        # 使用Mesh绘制三角形
                        Mesh(vertices=vertices, indices=indices, mode='triangles')
                    except Exception as e:
                        print(f"绘制填充多边形时出错: {e}")
                else:
                    print(f"  点数不足，无法绘制填充多边形: {len(kivy_points)//2}个点")
                
                # 绘制多边形轮廓 - 按照顺序连接顶点
                if len(kivy_points) >= 4:  # 至少需要2个点
                    print(f"  绘制多边形轮廓")
                    Color(0, 0, 0, 0.8)
                    # 确保轮廓线正确闭合，不要穿过多边形内部
                    Line(points=kivy_points + kivy_points[:2], width=1.5)  # 闭合线条
                
                # 绘制多边形顶点
                for i in range(0, len(kivy_points), 2):
                    x, y = kivy_points[i], kivy_points[i+1]
                    point_idx = i // 2
                    
                    # 高亮显示选中的点
                    if poly_idx == self.selected_polygon_idx and point_idx == self.selected_polygon_point_idx:
                        Color(0, 0.8, 1, 0.9)  # 青色为选中状态
                        radius = 8
                    else:
                        Color(1, 0.5, 0, 0.8)  # 橙色为普通状态
                        radius = 5
                        
                    Ellipse(pos=(x - radius, y - radius), size=(radius * 2, radius * 2))
                
            PopMatrix()
    
    def debug_add_sample_polygon(self):
        """添加一个示例多边形用于测试"""
        # 获取图像尺寸
        img_w, img_h = self.image_original_size
        if img_w <= 0 or img_h <= 0:
            print("图像尺寸无效，无法添加示例多边形")
            return
            
        # 创建一个居中的示例多边形
        center_x = img_w / 2
        center_y = img_h / 2
        size = min(img_w, img_h) / 4
        
        sample_polygon = {
            'part_name': 'Test',
            'points': [
                [center_x - size, center_y - size],  # 左下
                [center_x + size, center_y - size],  # 右下
                [center_x + size, center_y + size],  # 右上
                [center_x - size, center_y + size]   # 左上
            ],
            'color': [1, 0, 0, 0.5]  # 红色半透明
        }
        
        self.polygons_data.append(sample_polygon)
        print(f"添加了示例多边形，当前多边形数量: {len(self.polygons_data)}")
        self.edit_mode = 'polygon'
        self._redraw()

    def _redraw(self, *args):
        # 使用节流机制避免频繁重绘
        if not hasattr(self, '_last_redraw_time'):
            self._last_redraw_time = 0
            self._redraw_threshold = 1/30  # 限制为每秒最多30次重绘
        
        # 检查是否需要节流重绘请求
        current_time = time.time()
        if current_time - self._last_redraw_time < self._redraw_threshold:
            # 如果已经有一个待处理的重绘任务，则不再添加新的
            if not hasattr(self, '_pending_redraw') or not self._pending_redraw:
                self._pending_redraw = True
                Clock.schedule_once(self._deferred_redraw, self._redraw_threshold)
            return
        
        self._last_redraw_time = current_time
        self._pending_redraw = False
        
        # 执行实际重绘
        self._perform_redraw()
    
    def _deferred_redraw(self, dt):
        """延迟执行的重绘操作"""
        self._pending_redraw = False
        self._last_redraw_time = time.time()
        self._perform_redraw()
    
    def _perform_redraw(self):
        """实际执行重绘的方法"""
        # 确保显示参数总是最新的
        self._update_display_params()

        self.canvas.before.clear()
        self.canvas.after.clear()
        with self.canvas.before:
            PushMatrix()  # 添加矩阵栈操作，确保坐标系相对于当前控件
            Translate(self.pos[0], self.pos[1])  # 平移到控件自身的坐标系
            
            if self.texture:
                Color(1,1,1,1) 
                img_w_orig, img_h_orig = self.image_original_size
                if img_w_orig > 0 and img_h_orig > 0:
                    scaled_w = img_w_orig * self.display_scale
                    scaled_h = img_h_orig * self.display_scale
                    Rectangle(texture=self.texture, 
                              pos=(self.display_offset_x, self.display_offset_y), # 现在是相对于控件左下角
                              size=(scaled_w, scaled_h))
            
            PopMatrix()  # 恢复之前的坐标系
        
        # 根据编辑模式决定绘制什么
        if self.edit_mode == 'skeleton':
            self.draw_skeleton()
        elif self.edit_mode == 'polygon':
            self.draw_polygons()
        elif self.edit_mode == 'part_edit':
            self.draw_editable_parts()
            
    def draw_editable_parts(self):
        """绘制可编辑的部件，用于直接在画布上编辑"""
        # 获取原始图像高度用于Y坐标转换
        img_w_orig, img_h_orig = self.image_original_size
        if img_h_orig == 0:
            return
            
        # 初始化部件编辑器字典（如果不存在）
        if not hasattr(self, 'part_editors'):
            self.part_editors = {}
            
        app = App.get_running_app()
        if not hasattr(app, 'editable_parts') or not app.editable_parts:
            # 如果没有可编辑部件，清空现有编辑器
            for editor in list(self.part_editors.values()):
                self.remove_widget(editor)
            self.part_editors.clear()
            return
            
        # 检查哪些部件编辑器需要添加、更新或删除
        current_parts = set(app.editable_parts.keys())
        existing_parts = set(self.part_editors.keys())
        
        # 需要删除的部件
        parts_to_remove = existing_parts - current_parts
        # 需要添加的部件
        parts_to_add = current_parts - existing_parts
        # 需要更新的部件
        parts_to_update = current_parts.intersection(existing_parts)
        
        # 删除不再需要的编辑器
        for part_name in parts_to_remove:
            editor = self.part_editors.pop(part_name)
            self.remove_widget(editor)
        
        # 创建新的部件编辑器
        for part_name in parts_to_add:
            part = app.editable_parts[part_name]
            is_selected = app.current_part_for_edit == part_name
            
            # 计算编辑器的大小（基于目标多边形的大小）
            target_poly = np.array(part.target_polygon)
            min_x, min_y = np.min(target_poly, axis=0)
            max_x, max_y = np.max(target_poly, axis=0)
            width = max(50, (max_x - min_x) * 1.05)  # 减小倍数为1.05，且最小宽度为50
            height = max(50, (max_y - min_y) * 1.05)  # 减小倍数为1.05，且最小高度为50
            
            # 转换图像坐标到控件坐标
            widget_pos = self.convert_image_to_widget_coords(*part.position)
            
            # 创建编辑器实例
            editor = PartEditor(
                part_name=part_name,
                position=widget_pos,
                scale=part.scale,
                rotation=part.rotation,
                size=(width * part.scale, height * part.scale)  # 应用缩放到大小
            )
            
            # 使用实例方法而不是lambda来避免闭包问题
            editor.bind(on_part_edited=self._on_part_editor_event)
            editor.target_part_name = part_name  # 添加标识符以便回调函数知道是哪个部件
            
            # 添加到画布
            self.add_widget(editor)
            self.part_editors[part_name] = editor
        
        # 更新现有的部件编辑器
        for part_name in parts_to_update:
            part = app.editable_parts[part_name]
            editor = self.part_editors[part_name]
            is_selected = app.current_part_for_edit == part_name
            
            # 只有在实际发生变化时才更新编辑器属性
            # 检查是否需要更新位置
            widget_pos = self.convert_image_to_widget_coords(*part.position)
            if abs(editor.position[0] - widget_pos[0]) > 1 or abs(editor.position[1] - widget_pos[1]) > 1:
                editor.position = widget_pos
                
            # 检查是否需要更新缩放
            if abs(editor.scale - part.scale) > 0.01:
                editor.scale = part.scale
                
            # 检查是否需要更新旋转
            if abs(editor.rotation - part.rotation) > 0.5:
                editor.rotation = part.rotation
            
            # 更新视觉状态
            editor.update_visual()
            
        # 确保选中的部件在最上层显示
        if hasattr(app, 'current_part_for_edit') and app.current_part_for_edit in self.part_editors:
            selected_editor = self.part_editors[app.current_part_for_edit]
            # 将选中的编辑器移到最前
            self.remove_widget(selected_editor)
            self.add_widget(selected_editor)

    def _on_part_editor_event(self, editor, *args):
        """处理部件编辑器的事件，避免lambda闭包问题"""
        # 添加更新频率限制
        if not hasattr(self, '_last_editor_event_time'):
            self._last_editor_event_time = 0
            self._editor_event_threshold = 0.1  # 限制为每100毫秒最多更新一次UI和预览
        
        current_time = time.time()
        update_preview = current_time - self._last_editor_event_time >= self._editor_event_threshold
        
        if hasattr(editor, 'target_part_name'):
            try:
                # 确保我们能获取到正确的部件
                app = App.get_running_app()
                part_name = editor.target_part_name
                if part_name in app.editable_parts:
                    # 获取编辑器中的属性
                    img_x, img_y = self.convert_widget_to_image_coords(*editor.position)
                    part = app.editable_parts[part_name]
                    
                    # 更新部件属性（这个应该总是执行，确保数据一致性）
                    part.update_position(img_x, img_y)
                    part.update_scale(editor.scale)
                    part.update_rotation(editor.rotation)
                    
                    # 只有在满足更新频率限制时才更新UI和预览
                    if update_preview:
                        self._last_editor_event_time = current_time
                        
                        # 更新UI显示
                        app.main_layout_widget.ids.prop_pos_x.text = str(int(img_x))
                        app.main_layout_widget.ids.prop_pos_y.text = str(int(img_y))
                        app.main_layout_widget.ids.prop_scale.text = str(round(editor.scale, 2))
                        app.main_layout_widget.ids.prop_rotation.text = str(int(editor.rotation))
                        
                        # 延迟更新合成预览
                        if hasattr(app, '_update_composite_preview_delayed'):
                            Clock.unschedule(app._update_composite_preview_delayed)
                            Clock.schedule_once(app._update_composite_preview_delayed, 0.2)
                        else:
                            # 如果没有延迟方法，直接更新，但这里应该确保应用里有这个方法
                            print("使用直接更新合成预览方法")
                            app._update_composite_preview()
                        
                        print(f"已更新部件 {part_name}: 位置({img_x:.1f}, {img_y:.1f}), 缩放={editor.scale:.2f}, 旋转={editor.rotation:.1f}°")
            except Exception as e:
                print(f"部件编辑器事件处理错误: {e}")
                traceback.print_exc()
        else:
            print("错误: 部件编辑器没有target_part_name属性")

    def convert_image_to_widget_coords(self, img_x, img_y):
        """将图像坐标转换为控件坐标"""
        # 获取原始图像尺寸
        img_w, img_h = self.image_original_size
        if img_w == 0 or img_h == 0:
            return (0, 0)
            
        # 调整Y坐标（从顶部向下到从底部向上）
        img_y_kivy = img_h - img_y
        
        # 缩放和偏移 - 相对于画布左下角
        widget_x = img_x * self.display_scale + self.display_offset_x
        widget_y = img_y_kivy * self.display_scale + self.display_offset_y
        
        # 加上画布位置，得到屏幕坐标
        screen_x = widget_x + self.pos[0]
        screen_y = widget_y + self.pos[1]
        
        return (screen_x, screen_y)
        
    def convert_widget_to_image_coords(self, widget_x, widget_y):
        """将控件坐标转换为图像坐标"""
        # 获取原始图像尺寸
        img_w, img_h = self.image_original_size
        if img_w == 0 or img_h == 0:
            return (0, 0)
            
        # 先转换为相对于画布的坐标
        local_x = widget_x - self.pos[0]
        local_y = widget_y - self.pos[1]
        
        # 反向缩放和偏移
        img_x_kivy = (local_x - self.display_offset_x) / self.display_scale
        img_y_kivy = (local_y - self.display_offset_y) / self.display_scale
        
        # 调整Y坐标（从底部向上到从顶部向下）
        img_y = img_h - img_y_kivy
        
        # 确保坐标在图像范围内
        img_x = max(0, min(img_w, img_x_kivy))
        img_y = max(0, min(img_h, img_y))
        
        return (img_x, img_y)
        
    def _on_part_edited(self, part_name, editor):
        """处理部件编辑事件"""
        app = App.get_running_app()
        if not hasattr(app, 'editable_parts') or part_name not in app.editable_parts:
            return
            
        # 获取编辑器中的属性
        img_x, img_y = self.convert_widget_to_image_coords(*editor.position)
        part = app.editable_parts[part_name]
        
        # 更新部件属性
        part.update_position(img_x, img_y)
        part.update_scale(editor.scale)
        part.update_rotation(editor.rotation)
        
        # 更新UI显示
        app.main_layout_widget.ids.prop_pos_x.text = str(int(img_x))
        app.main_layout_widget.ids.prop_pos_y.text = str(int(img_y))
        app.main_layout_widget.ids.prop_scale.text = str(round(editor.scale, 2))
        app.main_layout_widget.ids.prop_rotation.text = str(int(editor.rotation))
        
        # 更新合成预览
        app._update_composite_preview()

    def on_polygon_point_added(self, polygon_idx, new_point_pos, segment_index):
        """当多边形添加点时触发"""
        print(f"多边形点添加事件: 多边形 {polygon_idx}, 位置 {new_point_pos}, 线段索引 {segment_index}")
        pass
        
    def on_polygon_point_deleted(self, polygon_idx, point_idx):
        """当多边形删除点时触发"""
        print(f"多边形点删除事件: 多边形 {polygon_idx}, 点索引 {point_idx}")
        pass

    def _handle_double_click(self, touch):
        """处理双击事件，用于添加或删除多边形点"""
        if self.edit_mode != 'polygon':
            return False
            
        # 获取图像坐标
        img_coords = self.screen_to_image_coords(touch.x, touch.y)
        if not img_coords:
            return False
            
        # 首先检查是否双击了某个点（删除点）
        poly_idx, point_idx = self.find_polygon_point_at(touch.x, touch.y)
        if poly_idx >= 0 and point_idx >= 0:
            # 发送删除点事件
            self.dispatch('on_polygon_point_deleted', poly_idx, point_idx)
            print(f"双击删除点: 多边形 {poly_idx}, 点 {point_idx}")
            return True
            
        # 如果没有双击点，检查是否双击了线段（添加点）
        poly_idx, segment_idx, point_on_line = self.find_line_segment_at(touch.x, touch.y)
        if poly_idx >= 0 and segment_idx >= 0 and point_on_line:
            # 发送添加点事件
            self.dispatch('on_polygon_point_added', poly_idx, point_on_line, segment_idx)
            print(f"双击添加点: 多边形 {poly_idx}, 位置 {point_on_line}, 线段 {segment_idx}")
            return True
            
        return False
        
    def find_line_segment_at(self, touch_x, touch_y, max_distance=10):
        """查找触摸位置附近的多边形线段"""
        if not self.polygons_data:
            return -1, -1, None
            
        # 转换为图像坐标
        img_coords = self.screen_to_image_coords(touch_x, touch_y)
        if not img_coords:
            return -1, -1, None
            
        touch_point = np.array(img_coords)
        closest_poly_idx = -1
        closest_segment_idx = -1
        closest_distance = float('inf')
        closest_point_on_line = None
        
        # 检查每个多边形的每条边
        for poly_idx, poly_data in enumerate(self.polygons_data):
            points = poly_data['points']
            if len(points) < 3:
                continue
                
            # 对多边形的每条边进行检查
            for i in range(len(points)):
                # 取当前点和下一个点（如果是最后一个点，则连接到第一个点）
                p1 = np.array(points[i])
                p2 = np.array(points[(i+1) % len(points)])
                
                # 计算点到线段的距离
                line_vec = p2 - p1
                line_length = np.linalg.norm(line_vec)
                if line_length == 0:
                    continue
                    
                # 单位向量
                line_unit_vec = line_vec / line_length
                
                # 点到p1的向量
                point_vec = touch_point - p1
                
                # 点在线上的投影长度
                projection_length = np.dot(point_vec, line_unit_vec)
                
                # 如果投影在线段外，取端点距离
                if projection_length < 0:
                    closest_point = p1
                    distance = np.linalg.norm(touch_point - p1)
                elif projection_length > line_length:
                    closest_point = p2
                    distance = np.linalg.norm(touch_point - p2)
                else:
                    # 计算投影点坐标
                    closest_point = p1 + projection_length * line_unit_vec
                    # 计算点到线的垂直距离
                    distance = np.linalg.norm(touch_point - closest_point)
                
                # 根据缩放比例调整距离阈值
                scaled_max_distance = max_distance / self.display_scale
                
                if distance < scaled_max_distance and distance < closest_distance:
                    closest_distance = distance
                    closest_poly_idx = poly_idx
                    closest_segment_idx = i
                    closest_point_on_line = [float(closest_point[0]), float(closest_point[1])]
        
        return closest_poly_idx, closest_segment_idx, closest_point_on_line


class MainLayout(BoxLayout):
    # These object properties will be automatically linked to the ids in the .kv file
    # Left Panel
    lbl_template_file = ObjectProperty(None)
    lbl_head_file = ObjectProperty(None)
    lbl_body_file = ObjectProperty(None)
    preview_template = ObjectProperty(None)
    preview_head = ObjectProperty(None)
    preview_body = ObjectProperty(None)
    step_controls_area = ObjectProperty(None)
    status_bar = ObjectProperty(None)
    
    # Center Panel
    drawing_canvas = ObjectProperty(None) # Replaces main_display_image for drawing canvas
    center_panel_container = ObjectProperty(None) # Will hold the main drawing widget

    # Right Panel
    prop_pos_x = ObjectProperty(None)
    prop_pos_y = ObjectProperty(None)
    prop_scale = ObjectProperty(None)
    prop_rotation = ObjectProperty(None)
    
    # 分割点编辑相关控件
    polygon_points_list = ObjectProperty(None)
    btn_add_point = ObjectProperty(None)
    btn_delete_point = ObjectProperty(None)

class CharacterAutoMergerApp(App):
    # Store the image_type string temporarily when file chooser is opened
    _current_image_type_for_chooser = StringProperty(None)
    pose_detector = ObjectProperty(None) # To hold RealPoseDetector instance

    def build(self):
        # 注册中文字体
        self._register_chinese_font()
        
        self.title = "Character Auto Merger (Kivy)"
        Window.size = (1600, 900)
        # Window.clearcolor = get_color_from_hex("#222222") # Optional: set a global background color
        
        # Initialize data storage
        self.image_paths = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None}
        self.image_cv_originals = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None}
        self.image_textures = {IMAGE_TYPE_TEMPLATE: None, IMAGE_TYPE_HEAD: None, IMAGE_TYPE_BODY: None}

        self.skeletons_data = {
            IMAGE_TYPE_TEMPLATE: None,
            IMAGE_TYPE_HEAD: None,
            IMAGE_TYPE_BODY: None
        }
        self.adjusted_skeletons_data = {
            IMAGE_TYPE_TEMPLATE: None,
            IMAGE_TYPE_HEAD: None,
            IMAGE_TYPE_BODY: None
        }
        self.pre_segmented_parts_polygons = {
            IMAGE_TYPE_TEMPLATE: {},
            IMAGE_TYPE_HEAD: {},
            IMAGE_TYPE_BODY: {}
        }
        self.final_extracted_parts = {} 

        self.current_app_mode = APP_MODE_UPLOAD
        self.current_image_type_for_adjustment = None
        self.current_part_for_segment_adjustment = None
        
        # --- Instantiate Core Logic ---
        if MMDET_MMPOS_AVAILABLE:
            try:
                self.pose_detector = RealPoseDetector(DWPOSE_DIR, MODEL_CONFIGS)
                if self.pose_detector.detector is None or self.pose_detector.pose_estimator is None:
                    show_error_popup("模型加载失败", "姿态检测模型初始化失败。请检查控制台和DWPose路径。姿态检测功能将不可用。")
                    self.pose_detector = None # Ensure it's None if failed
            except Exception as e_init_pd:
                show_error_popup("检测器初始化错误", f"创建RealPoseDetector时出错: {e_init_pd}")
                self.pose_detector = None # Ensure it's None if failed
        else:
            show_error_popup("缺少MMDetection/MMPose", "未找到MMDetection或MMPose库。姿态检测功能将被禁用。")
            self.pose_detector = None

        # 初始化部位分割器
        self.part_segmenter = RealPartSegmenter()
        
        # self.warper = RealWarper() # Will add later

        if not os.path.exists(DEFAULT_EXPORT_DIR):
            os.makedirs(DEFAULT_EXPORT_DIR)
        
        self.main_layout_widget = MainLayout()
        Clock.schedule_once(self._post_build_init) # For things that need widget IDs to be resolved
        return self.main_layout_widget

    def _post_build_init(self, dt):
        self._update_ui_for_mode() # Initialize UI for the default mode
        if not PLYER_AVAILABLE:
            # Show a one-time startup warning if plyer is missing and we are using the dummy
            if isinstance(filechooser, filechooser_dummy):
                 show_error_popup("Dependency Missing", "Plyer library not found. Native file dialogs will not work. Please install plyer.")

        if self.main_layout_widget.ids.drawing_canvas:
            print("DrawingCanvas found in ids.")
        else:
            print("WARNING: DrawingCanvas NOT found in ids. Check kv file.")
        print("Kivy App Initialized. Main layout IDs should be available.")
        # Example: print(self.main_layout_widget.ids.lbl_template_file)

    def open_file_chooser(self, image_type_str):
        self._current_image_type_for_chooser = image_type_str

        if not PLYER_AVAILABLE: 
            # This block executes if plyer failed to import, meaning 'filechooser' is our dummy.
            show_error_popup("Feature Unavailable", 
                             "Plyer library is missing or failed to load. Cannot open native file dialog.\n"
                             "Please install plyer: pip install plyer")
            # Call the dummy version, which will print to console and return None
            # This maintains a similar flow for the calling code, expecting a None for no selection.
            filechooser.open_file() # This is filechooser_dummy().open_file()
            # Update status to reflect that no dialog was truly shown if Plyer is missing.
            self.main_layout_widget.ids.status_bar.text = f"Native file dialog unavailable for {image_type_str} (Plyer missing)."
            return

        # If we reach here, PLYER_AVAILABLE is True, and 'filechooser' is the real plyer.filechooser module.
        try:
            selection = filechooser.open_file(
                title=f"Select {image_type_str.capitalize()} Image",
                filters=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")] 
            )

            if selection and isinstance(selection, list) and len(selection) > 0:
                selected_path = selection[0] # Take the first selected file
                self._load_selected_image(selected_path, self._current_image_type_for_chooser)
            else:
                print("File selection cancelled or no file selected.")
                current_status = self.main_layout_widget.ids.status_bar.text
                self.main_layout_widget.ids.status_bar.text = f"File selection cancelled for {image_type_str}. ({current_status})"

        except Exception as e:
            msg = f"Error using Plyer file dialog for {image_type_str}: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("File Dialog Error", msg)
            self.main_layout_widget.ids.status_bar.text = f"Error opening dialog for {image_type_str}."

    def _load_selected_image(self, file_path, image_type_str):
        try:
            # 使用IMREAD_UNCHANGED来支持透明通道
            img_cv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise ValueError(f"OpenCV could not read image: {file_path}")

            # 打印图像信息以便调试
            print(f"图像 {file_path} 信息: 形状={img_cv.shape}, 数据类型={img_cv.dtype}")

            self.image_paths[image_type_str] = file_path
            
            # 处理可能的透明通道
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                # 普通RGB图像
                self.image_cv_originals[image_type_str] = img_cv
                # 转换为Kivy纹理
                buf = cv2.flip(img_cv, 0).tostring()
                texture = Texture.create(size=(img_cv.shape[1], img_cv.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            elif len(img_cv.shape) == 4 and img_cv.shape[2] == 4:
                # 带透明通道的RGBA图像 - 确保正确处理BGRA顺序
                print(f"检测到透明通道图像: {file_path}")
                self.image_cv_originals[image_type_str] = img_cv
                # 确保数据是BGRA格式，OpenCV读取的格式是BGRA
                buf = cv2.flip(img_cv, 0).tostring()
                texture = Texture.create(size=(img_cv.shape[1], img_cv.shape[0]), colorfmt='bgra')
                texture.blit_buffer(buf, colorfmt='bgra', bufferfmt='ubyte')
            elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                # 带透明通道的标准RGBA/BGRA图像
                print(f"检测到透明通道图像: {file_path}")
                self.image_cv_originals[image_type_str] = img_cv
                buf = cv2.flip(img_cv, 0).tostring()
                texture = Texture.create(size=(img_cv.shape[1], img_cv.shape[0]), colorfmt='bgra')
                texture.blit_buffer(buf, colorfmt='bgra', bufferfmt='ubyte')
            else:
                # 单通道图像或其他类型，转换为RGB
                print(f"警告: 图像 {file_path} 格式不是标准RGB或RGBA，尝试转换")
                print(f"图像形状: {img_cv.shape}, 通道数: {img_cv.shape[2] if len(img_cv.shape) > 2 else 1}")
                if len(img_cv.shape) == 2:  # 灰度图像
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                self.image_cv_originals[image_type_str] = img_cv
                buf = cv2.flip(img_cv, 0).tostring()
                texture = Texture.create(size=(img_cv.shape[1], img_cv.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                
            self.image_textures[image_type_str] = texture

            # Update UI elements (assuming ids are accessible via self.main_layout_widget.ids)
            filename = os.path.basename(file_path)
            if image_type_str == IMAGE_TYPE_TEMPLATE:
                self.main_layout_widget.ids.lbl_template_file.text = f"Template: {filename}"
                self.main_layout_widget.ids.preview_template.texture = texture
            elif image_type_str == IMAGE_TYPE_HEAD:
                self.main_layout_widget.ids.lbl_head_file.text = f"Head: {filename}"
                self.main_layout_widget.ids.preview_head.texture = texture
            elif image_type_str == IMAGE_TYPE_BODY:
                self.main_layout_widget.ids.lbl_body_file.text = f"Body: {filename}"
                self.main_layout_widget.ids.preview_body.texture = texture

            # Show the newly uploaded image in the main display area
            self.main_layout_widget.ids.drawing_canvas.texture = texture
            self.main_layout_widget.ids.drawing_canvas.image_original_size = [img_cv.shape[1], img_cv.shape[0]]
            self.main_layout_widget.ids.drawing_canvas.skeleton_data = [] # Clear old skeleton
            self.main_layout_widget.ids.status_bar.text = f"{image_type_str.capitalize()} loaded: {filename}"
            
            self._update_ui_for_mode() # Check if proceed button can be enabled

        except Exception as e:
            print(f"Error loading image {file_path} for {image_type_str}: {e}")
            traceback.print_exc()
            self.main_layout_widget.ids.status_bar.text = f"Error loading {image_type_str}"
            show_error_popup("图像加载错误", f"无法加载图像 {file_path}：{e}")

    def _update_ui_for_mode(self):
        """Dynamically updates the control panel based on the current app mode."""
        controls_layout = self.main_layout_widget.ids.step_controls_area
        if not controls_layout: 
            print("Warning: step_controls_area not found in _update_ui_for_mode. UI might not update correctly.")
            return
        controls_layout.clear_widgets()

        status_bar = self.main_layout_widget.ids.status_bar
        right_panel = self.main_layout_widget.ids.right_panel
        right_panel.disabled = True # Default hide/disable right panel
        # right_panel.opacity = 0 # Another way to hide
        
        # 获取右侧面板的子组件
        polygon_edit_panel = None
        part_properties_panel = None
        
        if hasattr(self.main_layout_widget.ids, 'polygon_edit_panel'):
            polygon_edit_panel = self.main_layout_widget.ids.polygon_edit_panel
            polygon_edit_panel.opacity = 0
            polygon_edit_panel.disabled = True
        
        if hasattr(self.main_layout_widget.ids, 'part_properties_panel'):
            part_properties_panel = self.main_layout_widget.ids.part_properties_panel
            part_properties_panel.opacity = 0
            part_properties_panel.disabled = True

        if self.current_app_mode == APP_MODE_UPLOAD:
            title_label = Label(text="1. 上传与处理", size_hint_y=None, height=dp(30), font_size='16sp')
            controls_layout.add_widget(title_label)
            btn_proceed = Button(text="处理姿势/骨架", size_hint_y=None, height=dp(44))
            btn_proceed.bind(on_release=lambda x: self._action_start_pose_detection()) 
            
            all_paths_uploaded = all(self.image_paths.values())
            # More robust check for pose_detector readiness
            pose_detector_ready = bool(self.pose_detector and \
                                     hasattr(self.pose_detector, 'detector') and self.pose_detector.detector is not None and \
                                     hasattr(self.pose_detector, 'pose_estimator') and self.pose_detector.pose_estimator is not None)

            print(f"APP_MODE_UPLOAD: All paths uploaded: {all_paths_uploaded}, Pose detector truly ready: {pose_detector_ready}")

            btn_proceed.disabled = not (all_paths_uploaded and pose_detector_ready)
            if btn_proceed.disabled:
                reason = []
                if not all_paths_uploaded: reason.append("not all images uploaded")
                if not pose_detector_ready: reason.append("pose detector not (fully) ready")
                print(f"Process Poses button is DISABLED. Reasons: {', '.join(reason) if reason else 'unknown (check logs)'}")
            else:
                print("Process Poses button is ENABLED.")

            controls_layout.add_widget(btn_proceed)
            if status_bar: status_bar.text = "上传模板、头部和身体图像。"

        elif self.current_app_mode == APP_MODE_ADJUST_SKELETON:
            title_label = Label(text=f"2. 调整骨架: {self.current_image_type_for_adjustment.capitalize() if self.current_image_type_for_adjustment else 'N/A'}", 
                                size_hint_y=None, height=dp(30), font_size='16sp')
            controls_layout.add_widget(title_label)
            
            # Add instructions for dragging keypoints
            instructions_label = Label(
                text="点击并拖动关键点进行调整。选中的关键点将被高亮显示。",
                size_hint_y=None, height=dp(40), font_size='14sp'
            )
            controls_layout.add_widget(instructions_label)
            
            img_type_selection_layout = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(5))
            for img_type_iter in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                btn = Button(text=img_type_iter.capitalize(), 
                             background_color=(0.2,0.6,1,1) if img_type_iter == self.current_image_type_for_adjustment else (0.5,0.5,0.5,1))
                btn.bind(on_release=lambda instance, type_str=img_type_iter: self._switch_skeleton_adjustment_view(type_str))
                img_type_selection_layout.add_widget(btn)
            controls_layout.add_widget(img_type_selection_layout)

            btn_confirm_skeletons = Button(text="确认骨架并继续", size_hint_y=None, height=dp(44))
            btn_confirm_skeletons.bind(on_release=lambda x: self._action_confirm_all_skeleton_adjustments())
            # Enable if all skeletons have (at least initially) processed and have data
            btn_confirm_skeletons.disabled = not all(
                self.adjusted_skeletons_data.get(it) and self.adjusted_skeletons_data[it].get('keypoints') is not None 
                for it in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]
            )
            controls_layout.add_widget(btn_confirm_skeletons)
            if status_bar: status_bar.text = f"调整骨架: {self.current_image_type_for_adjustment.capitalize() if self.current_image_type_for_adjustment else 'N/A'}。点击图像名称切换。"
 
        elif self.current_app_mode == APP_MODE_ADJUST_SEGMENTATION:
            title_label = Label(text=f"3. 调整分割: {self.current_part_for_segment_adjustment}", 
                               size_hint_y=None, height=dp(30), font_size='16sp')
            controls_layout.add_widget(title_label)
            
            # 添加分割调整说明
            instructions_label = Label(
                text="查看自动分割结果。使用右侧面板编辑分割区域。\n双击线段添加点，双击顶点删除点。",
                size_hint_y=None, height=dp(50), font_size='14sp'
            )
            controls_layout.add_widget(instructions_label)
            
            # 图像类型选择
            img_type_selection_layout = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(5))
            for img_type_iter in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                btn = Button(text=img_type_iter.capitalize(), 
                            background_color=(0.2,0.6,1,1) if img_type_iter == self.current_image_type_for_adjustment else (0.5,0.5,0.5,1))
                btn.bind(on_release=lambda instance, type_str=img_type_iter: self._switch_segmentation_image(type_str))
                img_type_selection_layout.add_widget(btn)
            controls_layout.add_widget(img_type_selection_layout)
            
            # 部位选择
            part_selection_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(180), spacing=dp(5))
            part_selection_layout.add_widget(Label(text="选择部位:", size_hint_y=None, height=dp(20)))
            
            parts_grid = GridLayout(cols=2, spacing=dp(5), size_hint_y=None, height=dp(150))
            for part_name in PART_DEFINITIONS.keys():
                btn = Button(text=part_name, 
                            background_color=(0.2,0.6,1,1) if part_name == self.current_part_for_segment_adjustment else (0.5,0.5,0.5,1))
                btn.bind(on_release=lambda instance, part=part_name: self._switch_segmentation_part(part))
                parts_grid.add_widget(btn)
            part_selection_layout.add_widget(parts_grid)
            controls_layout.add_widget(part_selection_layout)
            
            # 显示所有部位按钮
            btn_show_all = Button(text="切换显示模式", size_hint_y=None, height=dp(44))
            btn_show_all.bind(on_release=lambda x: self._show_all_part_polygons(self.current_image_type_for_adjustment))
            controls_layout.add_widget(btn_show_all)
            
            # 确认分割按钮
            btn_confirm_segmentation = Button(text="确认分割并继续", size_hint_y=None, height=dp(44))
            btn_confirm_segmentation.bind(on_release=lambda x: self._action_confirm_all_segmentations())
            controls_layout.add_widget(btn_confirm_segmentation)
            
            if status_bar: status_bar.text = f"调整{self.current_image_type_for_adjustment}的{self.current_part_for_segment_adjustment}分割"
            
            # 启用右侧面板用于分割点编辑
            right_panel.disabled = False
            
            # 显示多边形编辑面板，隐藏属性面板
            if polygon_edit_panel and part_properties_panel:
                polygon_edit_panel.opacity = 1
                polygon_edit_panel.disabled = False
                part_properties_panel.opacity = 0
                part_properties_panel.disabled = True
                
                # 更新点列表
                if self.current_part_for_segment_adjustment and self.current_image_type_for_adjustment:
                    if self.current_image_type_for_adjustment in self.pre_segmented_parts_polygons:
                        part_polygons = self.pre_segmented_parts_polygons[self.current_image_type_for_adjustment]
                        if self.current_part_for_segment_adjustment in part_polygons:
                            points = part_polygons[self.current_part_for_segment_adjustment]
                            self._update_polygon_points_list(points)
                            
                # 注册双击事件处理
                canvas = self.main_layout_widget.ids.drawing_canvas
                if canvas:
                    canvas.bind(on_polygon_point_added=self._on_polygon_point_added)
                    canvas.bind(on_polygon_point_deleted=self._on_polygon_point_deleted)
        
        elif self.current_app_mode == APP_MODE_EDIT_PARTS:
            title_label = Label(text="4. 编辑部件", 
                              size_hint_y=None, height=dp(30), font_size='16sp')
            controls_layout.add_widget(title_label)
            
            # 添加部件编辑说明
            instructions_label = Label(
                text="移动、缩放和旋转部件。在右侧面板中调整属性。",
                size_hint_y=None, height=dp(40), font_size='14sp'
            )
            controls_layout.add_widget(instructions_label)
            
            # 部件选择
            part_selection_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(180), spacing=dp(5))
            part_selection_layout.add_widget(Label(text="选择部件:", size_hint_y=None, height=dp(20)))
            
            parts_grid = GridLayout(cols=2, spacing=dp(5), size_hint_y=None, height=dp(150))
            # 如果有编辑部件列表，则显示部件按钮
            if hasattr(self, 'editable_parts') and self.editable_parts:
                for part_name in self.editable_parts.keys():
                    btn = Button(text=part_name, 
                               background_color=(0.2,0.6,1,1) if part_name == self.current_part_for_edit else (0.5,0.5,0.5,1))
                    # 修复Lambda引用，避免闭包问题
                    part_name_copy = part_name  # 创建显式副本
                    app_ref = self  # 保存应用实例引用
                    btn.bind(on_release=lambda instance, pn=part_name_copy: app_ref._select_part_for_edit(pn))
                    parts_grid.add_widget(btn)
            else:
                parts_grid.add_widget(Label(text="没有可编辑的部件", size_hint_y=None, height=dp(30)))
            part_selection_layout.add_widget(parts_grid)
            controls_layout.add_widget(part_selection_layout)
            
            # 导出按钮
            btn_export = Button(text="导出最终图像", size_hint_y=None, height=dp(44))
            btn_export.bind(on_release=lambda x: self._action_export_final_image())
            controls_layout.add_widget(btn_export)
            
            if status_bar: status_bar.text = "在模板上编辑部件。使用右侧面板调整选中部件的属性。"
            
            # 启用右侧面板用于编辑部件属性
            right_panel.disabled = False
            
            # 显示属性面板，隐藏多边形编辑面板
            if polygon_edit_panel and part_properties_panel:
                polygon_edit_panel.opacity = 0
                polygon_edit_panel.disabled = True
                part_properties_panel.opacity = 1
                part_properties_panel.disabled = False
                
            # 关闭双击事件处理
            canvas = self.main_layout_widget.ids.drawing_canvas
            if canvas:
                canvas.unbind(on_polygon_point_added=self._on_polygon_point_added)
                canvas.unbind(on_polygon_point_deleted=self._on_polygon_point_deleted)
    
    def _on_polygon_point_added(self, canvas, polygon_idx, new_point_pos, segment_index):
        """处理添加多边形点的事件"""
        if polygon_idx < 0 or polygon_idx >= len(canvas.polygons_data):
            return
            
        # 获取当前多边形
        polygon = canvas.polygons_data[polygon_idx]
        points = polygon['points']
        
        # 计算新点应当插入的位置（在segment_index之后）
        insert_index = (segment_index + 1) % len(points)
        
        # 插入新点
        points.insert(insert_index, new_point_pos)
        
        # 更新选中点
        canvas.selected_polygon_idx = polygon_idx
        canvas.selected_polygon_point_idx = insert_index
        
        # 更新画布和点列表
        canvas._redraw()
        self._update_polygon_points_list(points)
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"在线段 {segment_index} 添加了新点"
        
        # 保存更改
        self._save_current_segmentation_edits()
        
    def _on_polygon_point_deleted(self, canvas, polygon_idx, point_idx):
        """处理删除多边形点的事件"""
        if polygon_idx < 0 or polygon_idx >= len(canvas.polygons_data):
            return
            
        # 获取当前多边形
        polygon = canvas.polygons_data[polygon_idx]
        points = polygon['points']
        
        # 检查是否可以删除（至少需要保留3个点）
        if len(points) <= 3:
            self.main_layout_widget.ids.status_bar.text = "多边形至少需要3个点，无法删除"
            return
            
        # 删除点
        del points[point_idx]
        
        # 重置选中点
        canvas.selected_polygon_idx = polygon_idx
        canvas.selected_polygon_point_idx = -1
        
        # 更新画布和点列表
        canvas._redraw()
        self._update_polygon_points_list(points)
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"删除了点 {point_idx}"
        
        # 保存更改
        self._save_current_segmentation_edits()

    def _switch_segmentation_image(self, image_type):
        """切换分割调整视图中的图像类型"""
        # 保存当前分割编辑内容（如有）
        self._save_current_segmentation_edits()
        
        # 更新当前图像类型
        self.current_image_type_for_adjustment = image_type
        
        # 显示此图像类型的当前部位分割
        self._show_part_segmentation(image_type, self.current_part_for_segment_adjustment)
        
        # 更新UI状态
        self._update_ui_for_mode()

    def _switch_segmentation_part(self, part_name):
        """切换分割调整视图中的部位"""
        # 保存当前分割编辑内容（如有）
        self._save_current_segmentation_edits()
        
        # 更新当前部位
        print(f"切换分割部位从 {self.current_part_for_segment_adjustment} 到 {part_name}")
        self.current_part_for_segment_adjustment = part_name
        
        # 显示当前图像类型的此部位分割
        self._show_part_segmentation(self.current_image_type_for_adjustment, part_name)
        
        # 更新UI状态
        self._update_ui_for_mode()

    def _save_current_segmentation_edits(self):
        """保存当前分割编辑内容"""
        if not self.current_image_type_for_adjustment or not self.current_part_for_segment_adjustment:
            print("无法保存分割编辑：当前图像类型或部位未设置")
            return
        
        canvas = self.main_layout_widget.ids.drawing_canvas
        if not canvas:
            print("无法保存分割编辑：画布未找到")
            return
            
        print(f"尝试保存 {self.current_image_type_for_adjustment} 的 {self.current_part_for_segment_adjustment} 分割编辑")
        print(f"当前画布多边形数量: {len(canvas.polygons_data)}")
        
        # 从画布获取编辑后的多边形数据
        if canvas.polygons_data:
            img_type = self.current_image_type_for_adjustment
            part_name = self.current_part_for_segment_adjustment
            
            # 查找当前部位的多边形
            for poly_data in canvas.polygons_data:
                if poly_data['part_name'] == part_name:
                    points = poly_data['points']
                    if points and len(points) >= 3:  # 确保有效多边形
                        # 确保多边形字典已初始化
                        if img_type not in self.pre_segmented_parts_polygons:
                            self.pre_segmented_parts_polygons[img_type] = {}
                            
                        # 保存多边形点数据
                        self.pre_segmented_parts_polygons[img_type][part_name] = points
                        print(f"已保存 {img_type} 的 {part_name} 多边形，点数: {len(points)}")
                        return
            
            # 如果未找到匹配的多边形
            print(f"未找到 {img_type} 的 {part_name} 多边形数据")
        else:
            print("画布上没有多边形可保存")

    def _action_confirm_all_segmentations(self):
        """确认所有分割并进入部件编辑模式"""
        # 保存当前编辑
        self._save_current_segmentation_edits()
        
        # 打印调试信息，检查方法属性
        print("\n----- 调试信息：方法检查 -----")
        print("当前类属性:")
        for attr_name in dir(self):
            if not attr_name.startswith('__'):
                print(f"  - {attr_name}")
        
        print("\n可编辑部件方法检查:")
        print(f"  - hasattr(_select_part_for_edit): {hasattr(self, '_select_part_for_edit')}")
        print(f"  - 方法ID (如存在): {id(getattr(self, '_select_part_for_edit', None))}")
        print("----- 调试信息结束 -----\n")
        
        # 验证分割结果
        valid_segmentations = all(
            img_type in self.pre_segmented_parts_polygons and len(self.pre_segmented_parts_polygons[img_type]) > 0
            for img_type in [IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]
        )
        
        if not valid_segmentations:
            show_error_popup("分割不完整", "头部或身体图像的分割不完整。请重新处理。")
            return
        
        # 初始化变形器（如果尚未初始化）
        if not hasattr(self, 'warper') or self.warper is None:
            self.warper = RealWarper()
        
        # 创建可编辑部件
        try:
            print("创建可编辑部件...")
            self.main_layout_widget.ids.status_bar.text = "正在创建可编辑部件..."
            
            # 得到模板图像尺寸，用于目标坐标系
            template_img = self.image_cv_originals[IMAGE_TYPE_TEMPLATE]
            template_h, template_w = template_img.shape[:2]
            
            # 创建可编辑部件字典
            self.editable_parts = {}
            
            # 从头部图像创建部件
            head_img = self.image_cv_originals[IMAGE_TYPE_HEAD]
            head_polygons = self.pre_segmented_parts_polygons[IMAGE_TYPE_HEAD]
            
            for part_name, polygon in head_polygons.items():
                if part_name == "Head" and polygon and len(polygon) >= 3:
                    # 为头部创建初始目标多边形（使用模板上的头部位置）
                    template_head_poly = self.pre_segmented_parts_polygons[IMAGE_TYPE_TEMPLATE].get("Head")
                    
                    if template_head_poly and len(template_head_poly) >= 3:
                        # 计算中心点
                        src_center_x = sum(p[0] for p in polygon) / len(polygon)
                        src_center_y = sum(p[1] for p in polygon) / len(polygon)
                        
                        dst_center_x = sum(p[0] for p in template_head_poly) / len(template_head_poly)
                        dst_center_y = sum(p[1] for p in template_head_poly) / len(template_head_poly)
                        
                        # 创建头部部件 - 使用目标中心点作为初始位置
                        head_part = Part(
                            part_name="Head",
                            source_image=head_img,
                            source_polygon=polygon,
                            target_polygon=template_head_poly,
                            position=(dst_center_x, dst_center_y),
                            scale=1.0,
                            rotation=0.0
                        )
                        
                        self.editable_parts["Head"] = head_part
            
            # 从身体图像创建部件
            body_img = self.image_cv_originals[IMAGE_TYPE_BODY]
            body_polygons = self.pre_segmented_parts_polygons[IMAGE_TYPE_BODY]
            
            for part_name, polygon in body_polygons.items():
                if part_name in ["Torso", "LeftArm", "RightArm", "LeftLeg", "RightLeg", "WaistHip"] and polygon and len(polygon) >= 3:
                    # 为部件创建初始目标多边形（使用模板上的相应部位位置）
                    template_part_poly = self.pre_segmented_parts_polygons[IMAGE_TYPE_TEMPLATE].get(part_name)
                    
                    if template_part_poly and len(template_part_poly) >= 3:
                        # 计算中心点
                        src_center_x = sum(p[0] for p in polygon) / len(polygon)
                        src_center_y = sum(p[1] for p in polygon) / len(polygon)
                        
                        dst_center_x = sum(p[0] for p in template_part_poly) / len(template_part_poly)
                        dst_center_y = sum(p[1] for p in template_part_poly) / len(template_part_poly)
                        
                        # 创建部件 - 使用目标中心点作为初始位置
                        body_part = Part(
                            part_name=part_name,
                            source_image=body_img,
                            source_polygon=polygon,
                            target_polygon=template_part_poly,
                            position=(dst_center_x, dst_center_y),
                            scale=1.0,
                            rotation=0.0
                        )
                        
                        self.editable_parts[part_name] = body_part
            
            # 设置当前编辑部件（默认先编辑头部）
            self.current_part_for_edit = "Head" if "Head" in self.editable_parts else None
            
            # 生成并显示初始合成预览
            self._update_composite_preview()
            
            # 更新UI
            self.current_app_mode = APP_MODE_EDIT_PARTS
            self._update_ui_for_mode()
            
            # 确保DrawingCanvas在编辑模式下
            canvas = self.main_layout_widget.ids.drawing_canvas
            canvas.edit_mode = 'part_edit'
            canvas._redraw()  # 强制重绘显示编辑界面
            
            self.main_layout_widget.ids.status_bar.text = "部件创建完成。拖拽移动位置，拖动顶部蓝色圆点旋转，拖动右下角红色方块缩放（按Shift键等比例缩放）。"
            
        except Exception as e:
            msg = f"创建部件时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("部件创建错误", msg)
            self.main_layout_widget.ids.status_bar.text = "创建部件失败。"
            
    def _update_composite_preview(self):
        """更新并显示合成预览"""
        if not hasattr(self, 'editable_parts') or not self.editable_parts:
            return
            
        try:
            # 获取模板图像
            template_img = self.image_cv_originals[IMAGE_TYPE_TEMPLATE].copy()
            h, w = template_img.shape[:2]
            
            # 为每个部件创建变形图像和掩码
            warped_parts = {}
            
            for part_name, part in self.editable_parts.items():
                # 源图像和多边形
                src_img = part.source_image
                src_poly = np.array(part.source_polygon, dtype=np.float32)
                
                # 变换后的目标多边形
                dst_poly = np.array(part.target_polygon, dtype=np.float32)
                
                # 变形部件
                warped_img, mask = self.warper.warp_part(src_img, src_poly, dst_poly, (w, h))
                
                warped_parts[part_name] = {
                    'warped_image': warped_img,
                    'mask': mask
                }
            
            # 合成最终图像
            composite = self.warper.composite_parts(template_img, warped_parts)
            
            # 转换为Kivy纹理
            buf = cv2.flip(composite, 0).tostring()
            texture = Texture.create(size=(w, h), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            # 更新画布显示
            canvas = self.main_layout_widget.ids.drawing_canvas
            canvas.texture = texture
            canvas.image_original_size = [w, h]
            
            # 确保画布处于正确的编辑模式
            if self.current_app_mode == APP_MODE_EDIT_PARTS:
                canvas.edit_mode = 'part_edit'
            
            # 存储合成结果用于导出
            self.composite_result = composite
            
            # 更新状态栏
            self.main_layout_widget.ids.status_bar.text = "预览已更新"
            
        except Exception as e:
            msg = f"更新合成预览时出错: {e}"
            print(msg)
            traceback.print_exc()
            self.main_layout_widget.ids.status_bar.text = "预览更新失败"

    def _action_export_final_image(self):
        """导出最终合成图像"""
        if not hasattr(self, 'composite_result') or self.composite_result is None:
            show_error_popup("导出错误", "没有可用的合成结果。请先完成部件编辑。")
            return
            
        try:
            # 确保导出目录存在
            if not os.path.exists(DEFAULT_EXPORT_DIR):
                os.makedirs(DEFAULT_EXPORT_DIR)
                
            # 生成唯一文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(DEFAULT_EXPORT_DIR, f"character_merged_{timestamp}.png")
            
            # 保存图像
            cv2.imwrite(export_path, self.composite_result)
            
            self.main_layout_widget.ids.status_bar.text = f"已导出图像到: {export_path}"
            show_error_popup("导出成功", f"合成图像已保存到:\n{export_path}")
            
        except Exception as e:
            msg = f"导出图像时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("导出错误", msg)

    def dp(self, value):
        """Helper to convert dp to pixels if needed, though Kivy handles dp in kv lang."""
        return value # Kivy's metrics handle dp directly in kv. For python, use kivy.metrics.dp

    def _select_part_for_edit(self, part_name):
        """选择要编辑的部件"""
        if not hasattr(self, 'editable_parts') or part_name not in self.editable_parts:
            show_error_popup("部件不可用", f"部件 {part_name} 不可用或未创建。")
            return
            
        self.current_part_for_edit = part_name
        part = self.editable_parts[part_name]
        
        # 更新右侧面板属性
        self.main_layout_widget.ids.prop_pos_x.text = str(int(part.position[0]))
        self.main_layout_widget.ids.prop_pos_y.text = str(int(part.position[1]))
        self.main_layout_widget.ids.prop_scale.text = str(round(part.scale, 2))
        self.main_layout_widget.ids.prop_rotation.text = str(int(part.rotation))
        
        # 更新左侧控制面板按钮
        controls_layout = self.main_layout_widget.ids.step_controls_area
        parts_grid = None
        
        # 查找部件选择网格
        for child in controls_layout.children:
            if isinstance(child, BoxLayout) and child.orientation == 'vertical':
                for subchild in child.children:
                    if isinstance(subchild, GridLayout) and len(subchild.children) > 0:
                        # 找到了包含按钮的网格布局
                        parts_grid = subchild
                        break
                if parts_grid:
                    break
                    
        # 更新按钮高亮状态
        if parts_grid:
            print(f"找到部件网格，有 {len(parts_grid.children)} 个子控件")
            for btn in parts_grid.children:
                if isinstance(btn, Button):
                    if btn.text == part_name:
                        print(f"设置按钮 '{btn.text}' 高亮")
                        btn.background_color = (0.2, 0.6, 1, 1)  # 高亮显示
                    else:
                        btn.background_color = (0.5, 0.5, 0.5, 1)  # 普通显示
        else:
            print("未找到部件选择网格布局")
        
        # 更新DrawingCanvas上的部件选择状态
        canvas = self.main_layout_widget.ids.drawing_canvas
        if canvas.edit_mode != 'part_edit':
            canvas.edit_mode = 'part_edit'
        canvas._redraw()  # 重绘画布以更新所有部件
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"正在编辑 {part_name} 部件。拖拽移动位置，拖动顶部蓝色圆点旋转，拖动右下角红色方块缩放（按Shift键等比例缩放）。"

    def _register_chinese_font(self):
        """注册中文字体支持"""
        try:
            # 尝试添加系统常见中文字体路径
            # Windows 系统字体路径
            resource_add_path('C:/Windows/Fonts')
            
            # 尝试几种常见的中文字体
            font_alternatives = [
                ('simhei.ttf', 'SimHei'),       # 黑体
                ('simkai.ttf', 'SimKai'),       # 楷体
                ('simsun.ttc', 'SimSun'),       # 宋体
                ('msyh.ttc', 'Microsoft YaHei') # 微软雅黑
            ]
            
            for font_file, font_name in font_alternatives:
                try:
                    LabelBase.register(DEFAULT_FONT, font_file)
                    print(f"成功注册中文字体: {font_name}")
                    return
                except Exception as e:
                    print(f"尝试加载字体 {font_name} 失败: {e}")
            
            # 如果以上字体都加载失败，尝试使用Kivy内置的字体
            print("所有中文字体加载失败，将使用默认字体，可能导致中文显示为方框")
            
        except Exception as e:
            print(f"注册中文字体时出错: {e}")
            traceback.print_exc()

    def _action_start_pose_detection(self):
        """处理所有图像的姿态检测并转到骨架调整模式"""
        if not self.pose_detector:
            show_error_popup("检测器未初始化", "姿态检测器未能正确初始化，无法进行检测。")
            return
        
        try:
            self.main_layout_widget.ids.status_bar.text = "正在进行姿态检测..."
            
            # 处理每个图像的姿态检测
            for image_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                if image_type not in self.image_cv_originals or self.image_cv_originals[image_type] is None:
                    show_error_popup("图像缺失", f"缺少{image_type}图像，无法进行检测。")
                    return
                    
                # 执行姿态检测
                self.skeletons_data[image_type] = self.pose_detector.detect_pose(self.image_cv_originals[image_type], image_type)
                
                # 初始化调整后的骨架数据（初始时与检测结果相同）
                self.adjusted_skeletons_data[image_type] = self.skeletons_data[image_type].copy()
                
            # 切换到骨架调整模式并显示第一个图像的骨架
            self.current_app_mode = APP_MODE_ADJUST_SKELETON
            self.current_image_type_for_adjustment = IMAGE_TYPE_TEMPLATE
            
            # 显示第一个图像的骨架
            self._switch_skeleton_adjustment_view(IMAGE_TYPE_TEMPLATE)
            
            # 更新UI
            self._update_ui_for_mode()
            self.main_layout_widget.ids.status_bar.text = "姿态检测完成，现在可以调整骨架。"
            
        except Exception as e:
            msg = f"姿态检测时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("检测错误", msg)
            self.main_layout_widget.ids.status_bar.text = "姿态检测失败。"

    def _switch_skeleton_adjustment_view(self, image_type):
        """切换到指定图像的骨架调整视图"""
        if image_type not in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
            return
        
        # 在切换前保存当前骨架数据
        if self.current_image_type_for_adjustment:
            current_canvas = self.main_layout_widget.ids.drawing_canvas
            if current_canvas and current_canvas.skeleton_data:
                self.adjusted_skeletons_data[self.current_image_type_for_adjustment]['keypoints'] = current_canvas.skeleton_data
        
        # 更新当前正在调整的图像类型
        self.current_image_type_for_adjustment = image_type
        
        # 更新画布显示
        canvas = self.main_layout_widget.ids.drawing_canvas
        canvas.texture = self.image_textures[image_type]
        canvas.image_original_size = [self.image_cv_originals[image_type].shape[1], self.image_cv_originals[image_type].shape[0]]
        
        # 设置骨架数据
        if self.adjusted_skeletons_data.get(image_type) and 'keypoints' in self.adjusted_skeletons_data[image_type]:
            canvas.skeleton_data = self.adjusted_skeletons_data[image_type]['keypoints']
        else:
            canvas.skeleton_data = []
        
        # 设置编辑模式为骨架
        canvas.edit_mode = 'skeleton'
        
        # 清空多边形数据（如果有）
        canvas.polygons_data = []
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"正在调整{image_type}图像的骨架"
        
        # 更新UI, 确保按钮高亮状态更新
        self._update_ui_for_mode()

    def _action_confirm_all_skeleton_adjustments(self):
        """确认所有骨架调整并进入分割调整模式"""
        # 先保存当前视图的骨架数据
        if self.current_image_type_for_adjustment:
            current_canvas = self.main_layout_widget.ids.drawing_canvas
            if current_canvas and current_canvas.skeleton_data:
                self.adjusted_skeletons_data[self.current_image_type_for_adjustment]['keypoints'] = current_canvas.skeleton_data
        
        # 验证是否所有骨架都有数据
        all_skeletons_valid = True
        for img_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
            if not self.adjusted_skeletons_data.get(img_type) or not self.adjusted_skeletons_data[img_type].get('keypoints'):
                all_skeletons_valid = False
                show_error_popup("骨架数据不完整", f"缺少{img_type}的骨架数据，请重新处理。")
                break
            
            # 检查是否有足够的关键点
            keypoints = self.adjusted_skeletons_data[img_type]['keypoints']
            if len(keypoints) < 10:  # 假设至少需要10个有效关键点
                all_skeletons_valid = False
                show_error_popup("骨架数据不充分", f"{img_type}的骨架数据关键点不足，请重新处理。")
                break
        
        if not all_skeletons_valid:
            return
        
        # 使用边缘检测方法生成分割
        self.main_layout_widget.ids.status_bar.text = "正在使用边缘检测生成分割..."
        
        try:
            for img_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                # 使用新的边缘检测方法生成分割
                self.generate_edge_based_segmentation(img_type)
                
            # 转到分割调整模式
            self.current_app_mode = APP_MODE_ADJUST_SEGMENTATION
            self.current_image_type_for_adjustment = IMAGE_TYPE_TEMPLATE
            self.current_part_for_segment_adjustment = "Head"  # 默认从头部开始
            
            # 显示所有分割预览（不只是头部）
            self._show_all_part_polygons(IMAGE_TYPE_TEMPLATE)
            
            # 更新头部的点列表，因为它是默认选中的部位
            if "Head" in self.pre_segmented_parts_polygons.get(IMAGE_TYPE_TEMPLATE, {}):
                head_points = self.pre_segmented_parts_polygons[IMAGE_TYPE_TEMPLATE]["Head"]
                self._update_polygon_points_list(head_points)
            
            # 更新UI
            self._update_ui_for_mode()
            self.main_layout_widget.ids.status_bar.text = "骨架调整完成，现在可以调整分割。使用右侧面板编辑多边形顶点。"
            
        except Exception as e:
            msg = f"生成分割时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("分割生成错误", msg)
            
            # 回退到基本的骨架分割方法
            print("使用基本分割方法作为备选...")
            for img_type in [IMAGE_TYPE_TEMPLATE, IMAGE_TYPE_HEAD, IMAGE_TYPE_BODY]:
                self._generate_basic_segmentation_from_skeleton(img_type)
                
            # 仍然进入分割模式，但显示一条警告
            self.current_app_mode = APP_MODE_ADJUST_SEGMENTATION
            self.current_image_type_for_adjustment = IMAGE_TYPE_TEMPLATE
            self.current_part_for_segment_adjustment = "Head"
            
            # 显示分割预览
            self._show_all_part_polygons(IMAGE_TYPE_TEMPLATE)
            
            # 更新UI
            self._update_ui_for_mode()
            self.main_layout_widget.ids.status_bar.text = "使用基本分割。边缘检测分割失败，请手动调整分割。"

    def _show_part_segmentation(self, image_type, part_name):
        """显示特定图像和部位的分割预览"""
        if image_type not in self.pre_segmented_parts_polygons:
            print(f"警告: 未找到{image_type}的分割数据")
            self.main_layout_widget.ids.status_bar.text = f"无法显示{image_type}的分割预览，数据不存在"
            return
        
        # 获取部位的多边形数据
        part_polygons = self.pre_segmented_parts_polygons[image_type]
        print(f"显示{image_type}的分割预览, 可用部位: {list(part_polygons.keys())}")
        
        if part_name not in part_polygons:
            print(f"警告: 未找到{image_type}的{part_name}分割数据")
            self.main_layout_widget.ids.status_bar.text = f"无法显示{image_type}的{part_name}分割预览"
            return
        
        # 更新画布
        canvas = self.main_layout_widget.ids.drawing_canvas
        canvas.texture = self.image_textures[image_type]
        canvas.image_original_size = [self.image_cv_originals[image_type].shape[1], self.image_cv_originals[image_type].shape[0]]
        
        # 清空之前的数据
        canvas.skeleton_data = []
        canvas.polygons_data = []
        
        # 获取当前部位的多边形点
        selected_polygon_points = part_polygons[part_name]
        
        # 只显示当前选中的部位
        poly_data = {
            'part_name': part_name,
            'points': selected_polygon_points.copy(),  # 创建点数据的副本
            'color': PART_COLORS.get(part_name, [0.5, 0.5, 0.5, 0.7])  # 使用较高透明度
        }
        canvas.polygons_data.append(poly_data)
        
        # 更新点列表显示
        self._update_polygon_points_list(selected_polygon_points)
        
        # 设置编辑模式为多边形
        canvas.edit_mode = 'polygon'
        
        # 强制重绘
        canvas._redraw()
        
        # 设置为单部位显示模式
        self.showing_all_parts = False
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"正在编辑{image_type}的{part_name}分割"
        
    # 添加显示所有部位/切换显示的变量
    showing_all_parts = False
        
    def _show_all_part_polygons(self, image_type):
        """显示所有部位的多边形或切换回单部位显示"""
        if self.showing_all_parts:
            # 如果当前正在显示所有部位，切换回只显示当前部位
            self._show_part_segmentation(image_type, self.current_part_for_segment_adjustment)
            self.main_layout_widget.ids.status_bar.text = f"只显示 {self.current_part_for_segment_adjustment} 部位"
            return
            
        if image_type not in self.pre_segmented_parts_polygons:
            print(f"警告: 未找到{image_type}的分割数据")
            self.main_layout_widget.ids.status_bar.text = f"无法显示{image_type}的分割预览，数据不存在"
            return
        
        # 获取多边形数据
        part_polygons = self.pre_segmented_parts_polygons[image_type]
        if not part_polygons:
            print(f"警告: {image_type}没有分割数据")
            return
            
        # 更新画布
        canvas = self.main_layout_widget.ids.drawing_canvas
        canvas.texture = self.image_textures[image_type]
        canvas.image_original_size = [self.image_cv_originals[image_type].shape[1], self.image_cv_originals[image_type].shape[0]]
        
        # 清空之前的数据
        canvas.skeleton_data = []
        canvas.polygons_data = []
        
        # 添加所有部位的多边形
        for part_name, polygon_points in part_polygons.items():
            if len(polygon_points) >= 3:
                # 为当前选中的部位使用更高透明度
                alpha = 0.7 if part_name == self.current_part_for_segment_adjustment else 0.3
                color = PART_COLORS.get(part_name, [0.5, 0.5, 0.5, 0.5])
                color[3] = alpha  # 修改透明度
                
                poly_data = {
                    'part_name': part_name,
                    'points': polygon_points.copy(),
                    'color': color
                }
                canvas.polygons_data.append(poly_data)
                
        # 设置编辑模式为多边形
        canvas.edit_mode = 'polygon'
        
        # 强制重绘
        canvas._redraw()
        
        # 设置为显示所有部位模式
        self.showing_all_parts = True
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"显示{image_type}的所有分割区域，高亮显示 {self.current_part_for_segment_adjustment}"
        
        # 如果当前有选中的部位，更新点列表
        if self.current_part_for_segment_adjustment in part_polygons:
            points = part_polygons[self.current_part_for_segment_adjustment]
            self._update_polygon_points_list(points)
            
    def _update_polygon_points_list(self, points):
        """更新多边形点列表显示"""
        points_list = self.main_layout_widget.ids.polygon_points_list
        if not points_list:
            return
            
        # 清空当前列表
        points_list.clear_widgets()
        
        # 添加每个点到列表
        for i, point in enumerate(points):
            x, y = point
            point_item = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(30),
                spacing=dp(5)
            )
            
            # 点序号标签
            idx_label = Label(
                text=f"点 {i+1}:",
                size_hint_x=0.2
            )
            
            # 坐标标签
            coord_label = Label(
                text=f"({x:.1f}, {y:.1f})",
                size_hint_x=0.8,
                halign='left'
            )
            
            point_item.add_widget(idx_label)
            point_item.add_widget(coord_label)
            
            # 使整个行可点击
            btn = Button(opacity=0, size_hint=(1, 1))
            btn.point_index = i  # 存储点索引
            btn.bind(on_release=lambda instance: self._select_polygon_point_from_list(instance.point_index))
            point_item.add_widget(btn)
            
            # 添加到列表
            points_list.add_widget(point_item)
            
    def _select_polygon_point_from_list(self, point_index):
        """从列表中选择多边形点"""
        canvas = self.main_layout_widget.ids.drawing_canvas
        
        # 找到当前部位的多边形索引
        poly_idx = -1
        for i, poly in enumerate(canvas.polygons_data):
            if poly['part_name'] == self.current_part_for_segment_adjustment:
                poly_idx = i
                break
                
        if poly_idx >= 0:
            # 更新选中的多边形和点
            canvas.selected_polygon_idx = poly_idx
            canvas.selected_polygon_point_idx = point_index
            
            # 更新视图
            canvas._redraw()
            self.main_layout_widget.ids.status_bar.text = f"已选中多边形点 {point_index+1}"

    # 添加边缘检测和蒙版生成相关方法
    def generate_edge_based_segmentation(self, image_type):
        """使用边缘检测和骨架信息生成更精确的分割区域"""
        if not self.adjusted_skeletons_data.get(image_type) or not self.adjusted_skeletons_data[image_type].get('keypoints'):
            print(f"警告: {image_type}没有有效的骨架数据，无法生成基于边缘的分割")
            return
            
        img = self.image_cv_originals[image_type]
        if img is None:
            print(f"警告: {image_type}没有有效的图像数据")
            return
            
        # 生成主体蒙版
        mask = self._generate_foreground_mask(img)
        
        # 结合骨架信息进行分割
        keypoints = self.adjusted_skeletons_data[image_type]['keypoints']
        part_polygons = self._segment_with_mask_and_skeleton(img, mask, keypoints)
        
        # 更新分割数据
        self.pre_segmented_parts_polygons[image_type] = part_polygons
        print(f"已为{image_type}生成基于边缘的分割，共{len(part_polygons)}个部位")
        
        return part_polygons
        
    def _generate_foreground_mask(self, img):
        """生成前景蒙版"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用自适应阈值分割
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作，去除噪点
            kernel = np.ones((5,5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 寻找最大轮廓（假设主体是最大对象）
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # 如果找不到轮廓，返回全白（全前景）蒙版
                return np.ones(img.shape[:2], dtype=np.uint8) * 255
                
            # 找到最大面积的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 创建蒙版
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [max_contour], 0, 255, -1)
            
            # 膨胀蒙版确保覆盖完整
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            return mask
            
        except Exception as e:
            print(f"生成前景蒙版时出错: {e}")
            # 发生错误时返回全白蒙版
            return np.ones(img.shape[:2], dtype=np.uint8) * 255
            
    def _segment_with_mask_and_skeleton(self, img, mask, keypoints):
        """结合蒙版和骨架信息进行分割"""
        h, w = img.shape[:2]
        part_polygons = {}
        
        # 为每个部位生成多边形
        for part_name, kp_indices in PART_DEFINITIONS.items():
            valid_keypoints = []
            for idx in kp_indices:
                if idx < len(keypoints) and keypoints[idx][2] > 0.1:  # 检查置信度
                    valid_keypoints.append((keypoints[idx][0], keypoints[idx][1]))
            
            print(f"{part_name}: 有效关键点 {len(valid_keypoints)}/{len(kp_indices)}")
            
            # 根据部位类型选择不同的分割策略
            if part_name == "Head" and len(valid_keypoints) >= 1:
                # 头部用固定六边形
                center_x, center_y = valid_keypoints[0] if valid_keypoints else (w/2, h/4)
                size = min(w, h) * 0.15  # 头部大小为图像较小尺寸的15%
                
                # 生成六边形
                points = []
                for i in range(6):
                    angle = 2 * np.pi * i / 6 + np.pi/6  # 旋转30度，使六边形更美观
                    px = center_x + size * np.cos(angle)
                    py = center_y + size * np.sin(angle)
                    points.append([px, py])
                
                part_polygons[part_name] = points
                
            elif len(valid_keypoints) >= 2:
                # 使用骨架点定义中心区域，然后向外膨胀到边缘
                try:
                    # 1. 创建初始骨架蒙版（中心区域）
                    kp_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 将关键点连成线条或填充多边形
                    pts = np.array(valid_keypoints, dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.fillConvexPoly(kp_mask, pts, 255)
                    elif len(pts) == 2:
                        cv2.line(kp_mask, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), 255, 5)
                    
                    # 2. 计算骨架区域的中心点
                    center_x = sum(p[0] for p in valid_keypoints) / len(valid_keypoints)
                    center_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                    center_point = (int(center_x), int(center_y))
                    
                    # 3. 从小到大逐步膨胀骨架，直到接触到边缘或达到最大膨胀值
                    # 最初的膨胀尺寸较小，确保从骨架内部开始
                    initial_kernel_size = int(min(w, h) * 0.01)
                    initial_kernel = np.ones((initial_kernel_size, initial_kernel_size), np.uint8)
                    dilated_kp_mask = cv2.dilate(kp_mask, initial_kernel, iterations=1)
                    
                    # 创建一系列逐渐膨胀的蒙版
                    max_iterations = 10  # 最大膨胀次数
                    
                    # 保存最终的部位蒙版
                    final_part_mask = dilated_kp_mask.copy()
                    
                    for i in range(max_iterations):
                        # 更大的膨胀核心，用于后续膨胀
                        kernel_size = int(min(w, h) * (0.01 + 0.01 * i))
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        
                        # 使用新核心膨胀
                        next_dilated_mask = cv2.dilate(dilated_kp_mask, kernel, iterations=1)
                        
                        # 将膨胀的骨架与全局前景蒙版相交，确保不超出人物边界
                        constrained_mask = cv2.bitwise_and(mask, next_dilated_mask)
                        
                        # 检查是否达到边缘边界
                        contours, _ = cv2.findContours(constrained_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # 到达边缘，保存当前结果
                            final_part_mask = constrained_mask
                            break
                        else:
                            # 继续膨胀
                            dilated_kp_mask = next_dilated_mask
                            final_part_mask = constrained_mask
                    
                    # 4. 在最终蒙版上找到轮廓
                    contours, _ = cv2.findContours(final_part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # 选择最大轮廓
                        main_contour = max(contours, key=cv2.contourArea)
                        
                        # 简化轮廓到合理的点数
                        epsilon = 0.01 * cv2.arcLength(main_contour, True)
                        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                        
                        # 限制最大点数为12，如果超过则进一步简化
                        if len(approx_contour) > 12:
                            epsilon = 0.02 * cv2.arcLength(main_contour, True)
                            approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                        
                        # 提取多边形点
                        poly_points = []
                        for pt in approx_contour:
                            poly_points.append([float(pt[0][0]), float(pt[0][1])])
                        
                        # 如果点数太少，则使用六边形代替
                        if len(poly_points) < 6:
                            # 生成6个点的均匀多边形
                            size = min(w, h) * 0.2  # 部位大小
                            
                            # 生成六边形
                            points = []
                            for i in range(6):
                                angle = 2 * np.pi * i / 6
                                px = center_x + size * np.cos(angle)
                                py = center_y + size * np.sin(angle)
                                points.append([px, py])
                            
                            poly_points = points
                        
                        part_polygons[part_name] = poly_points
                    else:
                        # 没有找到轮廓，使用默认六边形
                        size = min(w, h) * 0.15
                        
                        # 生成六边形
                        points = []
                        for i in range(6):
                            angle = 2 * np.pi * i / 6
                            px = center_x + size * np.cos(angle)
                            py = center_y + size * np.sin(angle)
                            points.append([px, py])
                        
                        part_polygons[part_name] = points
                
                except Exception as e:
                    print(f"为{part_name}创建分割时出错: {e}")
                    # 错误时使用默认六边形
                    if valid_keypoints:
                        center_x = sum(p[0] for p in valid_keypoints) / len(valid_keypoints)
                        center_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                        size = min(w, h) * 0.1
                        
                        # 生成六边形
                        points = []
                        for i in range(6):
                            angle = 2 * np.pi * i / 6
                            px = center_x + size * np.cos(angle)
                            py = center_y + size * np.sin(angle)
                            points.append([px, py])
                        
                        part_polygons[part_name] = points
            
        return part_polygons
        
    def add_polygon_point(self):
        """添加多边形点"""
        if self.current_app_mode != APP_MODE_ADJUST_SEGMENTATION:
            return
            
        if not self.current_image_type_for_adjustment or not self.current_part_for_segment_adjustment:
            show_error_popup("无法添加点", "请先选择要编辑的图像和部位")
            return
            
        canvas = self.main_layout_widget.ids.drawing_canvas
        if not canvas or not canvas.polygons_data:
            show_error_popup("无法添加点", "当前没有可编辑的多边形")
            return
            
        # 找到要编辑的多边形
        poly_idx = -1
        for i, poly in enumerate(canvas.polygons_data):
            if poly['part_name'] == self.current_part_for_segment_adjustment:
                poly_idx = i
                break
                
        if poly_idx == -1:
            show_error_popup("无法添加点", f"未找到{self.current_part_for_segment_adjustment}的多边形")
            return
            
        # 获取多边形点
        points = canvas.polygons_data[poly_idx]['points']
        
        # 如果没有点，添加一个中心点
        if not points:
            w, h = canvas.image_original_size
            new_point = [w/2, h/2]
            canvas.polygons_data[poly_idx]['points'] = [new_point]
            self._update_polygon_points_list(points)
            canvas._redraw()
            return
            
        # 计算新点位置（在最后一个点和第一个点之间）
        if len(points) >= 1:
            last_point = points[-1]
            first_point = points[0]
            new_x = (last_point[0] + first_point[0]) / 2
            new_y = (last_point[1] + first_point[1]) / 2
            new_point = [new_x, new_y]
            
            # 添加新点
            points.append(new_point)
            canvas._redraw()
            
            # 更新点列表显示
            self._update_polygon_points_list(points)
            
            print(f"已添加新点 ({new_x:.1f}, {new_y:.1f}), 当前点数: {len(points)}")
            
    def delete_polygon_point(self):
        """删除选中的多边形点"""
        if self.current_app_mode != APP_MODE_ADJUST_SEGMENTATION:
            return
            
        if not self.current_image_type_for_adjustment or not self.current_part_for_segment_adjustment:
            show_error_popup("无法删除点", "请先选择要编辑的图像和部位")
            return
            
        canvas = self.main_layout_widget.ids.drawing_canvas
        if not canvas or not canvas.polygons_data:
            show_error_popup("无法删除点", "当前没有可编辑的多边形")
            return
            
        # 找到要编辑的多边形
        poly_idx = -1
        for i, poly in enumerate(canvas.polygons_data):
            if poly['part_name'] == self.current_part_for_segment_adjustment:
                poly_idx = i
                break
                
        if poly_idx == -1:
            show_error_popup("无法删除点", f"未找到{self.current_part_for_segment_adjustment}的多边形")
            return
            
        # 获取选中的点
        if canvas.selected_polygon_idx == poly_idx and canvas.selected_polygon_point_idx >= 0:
            points = canvas.polygons_data[poly_idx]['points']
            if len(points) <= 3:
                show_error_popup("无法删除点", "多边形至少需要3个点")
                return
                
            # 删除选中的点
            del points[canvas.selected_polygon_point_idx]
            
            # 重置选中状态
            canvas.selected_polygon_idx = -1
            canvas.selected_polygon_point_idx = -1
            
            # 更新显示
            canvas._redraw()
            
            # 更新点列表显示
            self._update_polygon_points_list(points)
            
            print(f"已删除点，当前点数: {len(points)}")
        else:
            show_error_popup("无法删除点", "请先选择要删除的点")
            
    def _generate_basic_segmentation_from_skeleton(self, image_type):
        """从骨架数据生成基本分割多边形（作为边缘检测失败的备选方案）"""
        if image_type not in self.adjusted_skeletons_data or not self.adjusted_skeletons_data[image_type].get('keypoints'):
            print(f"警告: {image_type}没有有效的骨架数据")
            return
        
        keypoints = self.adjusted_skeletons_data[image_type]['keypoints']
        # 直接从图像获取宽高
        img_width = self.image_cv_originals[image_type].shape[1]
        img_height = self.image_cv_originals[image_type].shape[0]
        
        print(f"为{image_type}生成基本分割多边形, 关键点数量: {len(keypoints)}")
        
        # 初始化分割多边形字典
        part_polygons = {}
        
        # 为每个部位生成基本分割多边形
        for part_name, kp_indices in PART_DEFINITIONS.items():
            valid_keypoints = []
            for idx in kp_indices:
                if idx < len(keypoints) and keypoints[idx][2] > 0.1:  # 检查置信度
                    valid_keypoints.append((keypoints[idx][0], keypoints[idx][1]))
            
            print(f"{part_name}: 有效关键点 {len(valid_keypoints)}/{len(kp_indices)}")
            
            # 根据不同部位使用不同的多边形生成策略
            if part_name == "Head" and len(valid_keypoints) >= 1:
                # 头部：使用六边形（而不是圆形）以保持与增强版一致
                center_x, center_y = valid_keypoints[0] if len(valid_keypoints) >= 1 else (img_width/2, img_height/4)
                
                # 确定头部大小，可以从其他关键点估计或使用固定比例
                head_size = min(img_width, img_height) * 0.15  # 头部大小为图像较小尺寸的15%
                
                # 生成六边形，逆时针顺序
                points = []
                for i in range(6):
                    angle = 2 * np.pi * i / 6 + np.pi/6  # 旋转30度，使六边形更美观
                    px = center_x + head_size * np.cos(angle)
                    py = center_y + head_size * np.sin(angle)
                    points.append([px, py])
                
                part_polygons[part_name] = points
                print(f"  - 为头部创建了六边形多边形，点数: {len(points)}")
                
            elif (part_name in ["LeftArm", "RightArm", "LeftLeg", "RightLeg"]) and len(valid_keypoints) >= 2:
                # 四肢：创建沿着骨骼线的加宽区域
                # 对关键点进行排序，确保它们沿一个方向
                sorted_keypoints = []
                
                # 对手臂和腿部使用特定的排序方式
                if part_name in ["LeftArm", "RightArm"]:
                    # 手臂的关键点通常从肩膀到手腕
                    # 简单地按照y坐标排序：肩膀在上方，所以y较小
                    sorted_keypoints = sorted(valid_keypoints, key=lambda p: p[1])
                elif part_name in ["LeftLeg", "RightLeg"]:
                    # 腿部的关键点通常从臀部到脚踝
                    # 简单地按照y坐标排序：臀部在上方，所以y较小
                    sorted_keypoints = sorted(valid_keypoints, key=lambda p: p[1])
                else:
                    sorted_keypoints = valid_keypoints
                
                # 创建沿骨骼线的多边形
                if len(sorted_keypoints) >= 2:
                    # 计算骨骼线宽度
                    limb_width = min(img_width, img_height) * 0.1  # 宽度为图像较小尺寸的10%
                    
                    # 创建多边形点
                    poly_points = []
                    
                    # 先添加一侧的点（从上到下）
                    for i in range(len(sorted_keypoints) - 1):
                        p1 = sorted_keypoints[i]
                        p2 = sorted_keypoints[i+1]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        length = np.sqrt(dx*dx + dy*dy)
                        if length > 0:
                            nx = -dy / length * limb_width/2
                            ny = dx / length * limb_width/2
                            poly_points.append([p1[0] + nx, p1[1] + ny])
                    
                    # 添加底部点
                    bottom = sorted_keypoints[-1]
                    dx = bottom[0] - sorted_keypoints[-2][0]
                    dy = bottom[1] - sorted_keypoints[-2][1]
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        nx = -dy / length * limb_width/2
                        ny = dx / length * limb_width/2
                        poly_points.append([bottom[0] + nx, bottom[1] + ny])
                    
                    # 然后添加另一侧的点（从下到上，形成闭环）
                    for i in range(len(sorted_keypoints) - 1, 0, -1):
                        p1 = sorted_keypoints[i]
                        p2 = sorted_keypoints[i-1]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        length = np.sqrt(dx*dx + dy*dy)
                        if length > 0:
                            nx = -dy / length * limb_width/2
                            ny = dx / length * limb_width/2
                            poly_points.append([p1[0] - nx, p1[1] - ny])
                    
                    # 添加顶部点完成闭环
                    top = sorted_keypoints[0]
                    dx = top[0] - sorted_keypoints[1][0]
                    dy = top[1] - sorted_keypoints[1][1]
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        nx = -dy / length * limb_width/2
                        ny = dx / length * limb_width/2
                        poly_points.append([top[0] - nx, top[1] - ny])
                    
                    part_polygons[part_name] = poly_points
                    print(f"  - 为{part_name}创建了轮廓多边形，点数: {len(poly_points)}")
                
            elif part_name == "Torso" and len(valid_keypoints) >= 4:
                # 躯干：使用肩膀和臀部关键点构建梯形
                try:
                    # 按y坐标排序关键点
                    top_points = []  # 肩膀
                    bottom_points = []  # 臀部
                    
                    # 简单地按y坐标分类
                    mean_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                    for p in valid_keypoints:
                        if p[1] < mean_y:
                            top_points.append(p)
                        else:
                            bottom_points.append(p)
                    
                    # 确保每组至少有两个点
                    if len(top_points) < 2 or len(bottom_points) < 2:
                        raise ValueError("没有足够的点来构建躯干")
                    
                    # 在每组中，按x坐标排序
                    top_points = sorted(top_points, key=lambda p: p[0])
                    bottom_points = sorted(bottom_points, key=lambda p: p[0])
                    
                    # 创建梯形（逆时针顺序）
                    torso_points = []
                    torso_points.append([top_points[0][0], top_points[0][1]])  # 左上
                    torso_points.append([top_points[-1][0], top_points[-1][1]])  # 右上
                    torso_points.append([bottom_points[-1][0], bottom_points[-1][1]])  # 右下
                    torso_points.append([bottom_points[0][0], bottom_points[0][1]])  # 左下
                    
                    part_polygons[part_name] = torso_points
                    print(f"  - 为躯干创建了梯形多边形，点数: {len(torso_points)}")
                except Exception as e:
                    print(f"  - 创建躯干多边形时出错: {e}")
                    if len(valid_keypoints) >= 3:
                        # 尝试使用六边形代替
                        center_x = sum(p[0] for p in valid_keypoints) / len(valid_keypoints)
                        center_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                        size = min(img_width, img_height) * 0.2  # 躯干通常较大
                        
                        # 生成六边形
                        points = []
                        for i in range(6):
                            angle = 2 * np.pi * i / 6
                            px = center_x + size * np.cos(angle)
                            py = center_y + size * np.sin(angle)
                            points.append([px, py])
                        
                        part_polygons[part_name] = points
                        print(f"  - 为躯干创建了六边形多边形（备选），点数: {len(points)}")
                
            elif len(valid_keypoints) >= 3:
                # 创建六边形（而非凸包）以保持一致性
                center_x = sum(p[0] for p in valid_keypoints) / len(valid_keypoints)
                center_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                
                # 计算适当大小
                max_dist = 0
                for kp in valid_keypoints:
                    dist = np.sqrt((kp[0] - center_x)**2 + (kp[1] - center_y)**2)
                    max_dist = max(max_dist, dist)
                
                size = max_dist * 2.0  # 足够覆盖所有关键点
                
                # 生成六边形
                points = []
                for i in range(6):
                    angle = 2 * np.pi * i / 6
                    px = center_x + size * np.cos(angle)
                    py = center_y + size * np.sin(angle)
                    points.append([px, py])
                
                part_polygons[part_name] = points
                print(f"  - 为{part_name}创建了六边形多边形，点数: {len(points)}")
                
            elif len(valid_keypoints) == 2:
                print(f"警告: {image_type}的{part_name}只有2个有效关键点，使用六边形代替")
                # 对于只有2个点的情况，创建六边形以保持一致性
                center_x = sum(p[0] for p in valid_keypoints) / len(valid_keypoints)
                center_y = sum(p[1] for p in valid_keypoints) / len(valid_keypoints)
                size = min(img_width, img_height) * 0.1
                
                # 生成六边形
                points = []
                for i in range(6):
                    angle = 2 * np.pi * i / 6
                    px = center_x + size * np.cos(angle)
                    py = center_y + size * np.sin(angle)
                    points.append([px, py])
                
                part_polygons[part_name] = points
        
        # 存储分割多边形
        self.pre_segmented_parts_polygons[image_type] = part_polygons
        print(f"为{image_type}生成了{len(part_polygons)}个分割多边形")
            
    def _show_all_part_polygons(self, image_type):
        """显示所有部位的多边形或切换回单部位显示"""
        if self.showing_all_parts:
            # 如果当前正在显示所有部位，切换回只显示当前部位
            self._show_part_segmentation(image_type, self.current_part_for_segment_adjustment)
            self.main_layout_widget.ids.status_bar.text = f"只显示 {self.current_part_for_segment_adjustment} 部位"
            return
            
        if image_type not in self.pre_segmented_parts_polygons:
            print(f"警告: 未找到{image_type}的分割数据")
            self.main_layout_widget.ids.status_bar.text = f"无法显示{image_type}的分割预览，数据不存在"
            return
        
        # 获取多边形数据
        part_polygons = self.pre_segmented_parts_polygons[image_type]
        if not part_polygons:
            print(f"警告: {image_type}没有分割数据")
            return
            
        # 更新画布
        canvas = self.main_layout_widget.ids.drawing_canvas
        canvas.texture = self.image_textures[image_type]
        canvas.image_original_size = [self.image_cv_originals[image_type].shape[1], self.image_cv_originals[image_type].shape[0]]
        
        # 清空之前的数据
        canvas.skeleton_data = []
        canvas.polygons_data = []
        
        # 添加所有部位的多边形
        for part_name, polygon_points in part_polygons.items():
            if len(polygon_points) >= 3:
                # 为当前选中的部位使用更高透明度
                alpha = 0.7 if part_name == self.current_part_for_segment_adjustment else 0.3
                color = PART_COLORS.get(part_name, [0.5, 0.5, 0.5, 0.5])
                color[3] = alpha  # 修改透明度
                
                poly_data = {
                    'part_name': part_name,
                    'points': polygon_points.copy(),
                    'color': color
                }
                canvas.polygons_data.append(poly_data)
                
        # 设置编辑模式为多边形
        canvas.edit_mode = 'polygon'
        
        # 强制重绘
        canvas._redraw()
        
        # 设置为显示所有部位模式
        self.showing_all_parts = True
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"显示{image_type}的所有分割区域，高亮显示 {self.current_part_for_segment_adjustment}"
        
        # 如果当前有选中的部位，更新点列表
        if self.current_part_for_segment_adjustment in part_polygons:
            points = part_polygons[self.current_part_for_segment_adjustment]
            self._update_polygon_points_list(points)
            
    def _update_polygon_points_list(self, points):
        """更新多边形点列表显示"""
        points_list = self.main_layout_widget.ids.polygon_points_list
        if not points_list:
            return
            
        # 清空当前列表
        points_list.clear_widgets()
        
        # 添加每个点到列表
        for i, point in enumerate(points):
            x, y = point
            point_item = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(30),
                spacing=dp(5)
            )
            
            # 点序号标签
            idx_label = Label(
                text=f"点 {i+1}:",
                size_hint_x=0.2
            )
            
            # 坐标标签
            coord_label = Label(
                text=f"({x:.1f}, {y:.1f})",
                size_hint_x=0.8,
                halign='left'
            )
            
            point_item.add_widget(idx_label)
            point_item.add_widget(coord_label)
            
            # 使整个行可点击
            btn = Button(opacity=0, size_hint=(1, 1))
            btn.point_index = i  # 存储点索引
            btn.bind(on_release=lambda instance: self._select_polygon_point_from_list(instance.point_index))
            point_item.add_widget(btn)
            
            # 添加到列表
            points_list.add_widget(point_item)
            
    def _select_polygon_point_from_list(self, point_index):
        """从列表中选择多边形点"""
        canvas = self.main_layout_widget.ids.drawing_canvas
        
        # 找到当前部位的多边形索引
        poly_idx = -1
        for i, poly in enumerate(canvas.polygons_data):
            if poly['part_name'] == self.current_part_for_segment_adjustment:
                poly_idx = i
                break
                
        if poly_idx >= 0:
            # 更新选中的多边形和点
            canvas.selected_polygon_idx = poly_idx
            canvas.selected_polygon_point_idx = point_index
            
            # 更新视图
            canvas._redraw()
            self.main_layout_widget.ids.status_bar.text = f"已选中多边形点 {point_index+1}"

# 在RealPartSegmenter类之后，添加RealWarper类和Part类
class RealWarper:
    """处理图像变形和合成"""
    
    def __init__(self):
        self.warped_parts = {}  # 存储变形后的部件
        self.warp_cache = {}    # 缓存变形结果
        
    def warp_part(self, src_image, src_polygon, dst_polygon, output_size):
        """将源图像中的一部分区域变形到目标区域"""
        out_w, out_h = output_size
        
        # 创建缓存键
        cache_key = (
            id(src_image),
            tuple(map(tuple, src_polygon.tolist())) if isinstance(src_polygon, np.ndarray) else tuple(map(tuple, src_polygon)),
            tuple(map(tuple, dst_polygon.tolist())) if isinstance(dst_polygon, np.ndarray) else tuple(map(tuple, dst_polygon)),
            output_size
        )
        
        # 检查缓存
        if cache_key in self.warp_cache:
            return self.warp_cache[cache_key]
        
        try:
            # 将多边形点转换为numpy数组
            src_points = np.array(src_polygon, dtype=np.float32)
            dst_points = np.array(dst_polygon, dtype=np.float32)
            
            # 打印调试信息
            print(f"源多边形点数: {len(src_points)}, 目标多边形点数: {len(dst_points)}")
            print(f"源多边形坐标: {src_points[:3]}...")
            print(f"目标多边形坐标: {dst_points[:3]}...")
            
            # 确保点数一致，取两个多边形点数的最小值
            min_points = min(len(src_points), len(dst_points))
            if min_points < 3:
                print(f"警告: 多边形点数不足，无法进行变形: 源={len(src_points)}点, 目标={len(dst_points)}点")
                # 返回一个空图像
                warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                mask = np.zeros((out_h, out_w), dtype=np.uint8)
                result = (warped, mask)
                self.warp_cache[cache_key] = result
                return result
            
            # 截取相同数量的点
            src_points = src_points[:min_points]
            dst_points = dst_points[:min_points]
            
            # 检查点是否有效 (非NaN和无穷大)
            if np.any(np.isnan(src_points)) or np.any(np.isnan(dst_points)) or \
               np.any(np.isinf(src_points)) or np.any(np.isinf(dst_points)):
                print("警告: 检测到无效的点(NaN或无穷大)")
                # 尝试修复点
                src_points = np.nan_to_num(src_points, nan=0.0, posinf=src_image.shape[1], neginf=0.0)
                dst_points = np.nan_to_num(dst_points, nan=0.0, posinf=out_w, neginf=0.0)
            
            # 检查点是否在有效范围内
            src_points[:, 0] = np.clip(src_points[:, 0], 0, src_image.shape[1] - 1)
            src_points[:, 1] = np.clip(src_points[:, 1], 0, src_image.shape[0] - 1)
            dst_points[:, 0] = np.clip(dst_points[:, 0], 0, out_w - 1)
            dst_points[:, 1] = np.clip(dst_points[:, 1], 0, out_h - 1)
            
            # 计算源多边形的边界框，确保在图像范围内
            x_min = max(0, int(np.min(src_points[:, 0])))
            y_min = max(0, int(np.min(src_points[:, 1])))
            x_max = min(src_image.shape[1], int(np.max(src_points[:, 0])))
            y_max = min(src_image.shape[0], int(np.max(src_points[:, 1])))
            
            # 确保边界框有效
            if x_min >= x_max or y_min >= y_max:
                print("警告: 无效的边界框，无法进行变形")
                warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                mask = np.zeros((out_h, out_w), dtype=np.uint8)
                result = (warped, mask)
                self.warp_cache[cache_key] = result
                return result
            
            # 处理透明通道（如果有）
            has_alpha = len(src_image.shape) == 3 and src_image.shape[2] == 4
            
            # 创建原始图像的掩码
            src_mask = np.zeros(src_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(src_mask, [src_points.astype(np.int32)], 255)
            
            # 如果有透明通道，将它与掩码结合
            if has_alpha:
                alpha_channel = src_image[:, :, 3]
                # 将原始alpha通道与多边形掩码结合
                src_mask = cv2.bitwise_and(src_mask, alpha_channel)
                
                # 获取BGR部分
                src_image_bgr = src_image[:, :, :3].copy()
            else:
                # 如果没有alpha通道，使用原图
                src_image_bgr = src_image.copy()
            
            # 查找透视变换矩阵
            try:
                # 对于复杂变形，使用findHomography - 添加参数控制以避免异常
                M, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                
                if M is None or np.any(np.isnan(M)) or np.any(np.isinf(M)):
                    raise ValueError("透视变换矩阵包含无效值")
                    
                # 检查变换矩阵是否会导致变形过大
                det = np.abs(np.linalg.det(M[:2, :2]))
                if det > 10.0 or det < 0.1:
                    print(f"警告: 透视变换可能导致过度变形，行列式={det}，改用仿射变换")
                    raise ValueError("变形过大")
                
                # 应用透视变换到RGB图像 - 使用更快的INTER_LINEAR插值
                warped = cv2.warpPerspective(src_image_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                mask = cv2.warpPerspective(src_mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                
            except Exception as e:
                print(f"透视变换失败: {e}")
                print("尝试使用仿射变换...")
                
                # 如果透视变换失败，尝试使用仿射变换（只需要3个点）
                try:
                    # 确保有3个有效点
                    if len(src_points) >= 3 and len(dst_points) >= 3:
                        src_points_aff = src_points[:3]
                        dst_points_aff = dst_points[:3]
                        
                        M_aff = cv2.getAffineTransform(src_points_aff, dst_points_aff)
                        
                        if M_aff is None or np.any(np.isnan(M_aff)) or np.any(np.isinf(M_aff)):
                            raise ValueError("仿射变换矩阵包含无效值")
                            
                        warped = cv2.warpAffine(src_image_bgr, M_aff, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        mask = cv2.warpAffine(src_mask, M_aff, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    else:
                        raise ValueError("没有足够的点进行仿射变换")
                except Exception as e_aff:
                    print(f"仿射变换也失败: {e_aff}")
                    print("返回空结果")
                    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    mask = np.zeros((out_h, out_w), dtype=np.uint8)
            
            # 确保掩码是8位单通道
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # 缓存结果
            result = (warped, mask)
            self.warp_cache[cache_key] = result
            
            # 限制缓存大小，避免内存泄漏
            if len(self.warp_cache) > 20:  # 只保留最近20个变形结果
                self.warp_cache.pop(next(iter(self.warp_cache)))
                
            return result
            
        except Exception as e:
            print(f"变形错误: {e}")
            traceback.print_exc()
            # 创建空结果
            warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            mask = np.zeros((out_h, out_w), dtype=np.uint8)
            result = (warped, mask)
            self.warp_cache[cache_key] = result
            return result

    def composite_parts(self, template_image, parts_dict):
        """将所有部件合成到模板图像上"""
        result = template_image.copy()
        
        # 按照部件类型的顺序合成
        part_order = ["RightLeg", "LeftLeg", "WaistHip", "Torso", "RightArm", "LeftArm", "Head"]
        
        # 优化：使用浮点数计算，避免多次转换
        result_float = result.astype(np.float32)
        
        for part_name in part_order:
            if part_name in parts_dict:
                part_info = parts_dict[part_name]
                warped_img = part_info['warped_image']
                mask = part_info['mask']
                
                # 应用掩码
                if mask.shape[:2] != warped_img.shape[:2]:
                    mask = cv2.resize(mask, (warped_img.shape[1], warped_img.shape[0]))
                
                # 将掩码转换为浮点数，避免每次除以255
                mask_float = mask.astype(np.float32) / 255.0
                
                # 使用广播运算加速计算
                for c in range(3):  # 处理BGR三个通道
                    result_float[:,:,c] = result_float[:,:,c] * (1.0 - mask_float) + warped_img[:,:,c] * mask_float
        
        return result_float.astype(np.uint8)

# 添加Part类用于模板上的部件编辑
class Part:
    """表示一个可编辑的部件"""
    
    def __init__(self, part_name, source_image, source_polygon, target_polygon=None, 
                position=(0, 0), scale=1.0, rotation=0.0):
        self.part_name = part_name
        self.source_image = source_image  # 源图像 (CV格式)
        self.source_polygon = source_polygon  # 源图像上的多边形
        self.target_polygon = target_polygon or source_polygon.copy()  # 目标图像上的多边形
        
        # 计算初始中心点（如果没有指定位置）
        if position == (0, 0):
            self.position = (
                sum(p[0] for p in self.target_polygon) / len(self.target_polygon),
                sum(p[1] for p in self.target_polygon) / len(self.target_polygon)
            )
        else:
            self.position = position
            
        self.scale = scale  # 缩放因子
        self.rotation = rotation  # 旋转角度
        
        # 部件中心点
        self.center_x = self.position[0]
        self.center_y = self.position[1]
        
        # 应用初始变换
        self.apply_transform()
        
    def apply_transform(self):
        """应用当前变换到目标多边形"""
        # 复制原始多边形
        transformed_poly = np.array(self.source_polygon, dtype=np.float32)
        
        # 计算原始多边形中心
        src_center_x = np.mean(transformed_poly[:, 0])
        src_center_y = np.mean(transformed_poly[:, 1])
        
        # 平移到原点
        transformed_poly[:, 0] -= src_center_x
        transformed_poly[:, 1] -= src_center_y
        
        # 应用缩放
        transformed_poly *= self.scale
        
        # 应用旋转 (角度转弧度)
        theta = self.rotation * np.pi / 180
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 创建一个新数组存储旋转结果，避免在循环中修改原始值
        rotated_poly = np.zeros_like(transformed_poly)
        
        for i in range(len(transformed_poly)):
            x = transformed_poly[i, 0]
            y = transformed_poly[i, 1]
            rotated_poly[i, 0] = x * cos_theta - y * sin_theta
            rotated_poly[i, 1] = x * sin_theta + y * cos_theta
        
        # 使用旋转后的结果
        transformed_poly = rotated_poly
        
        # 平移到目标位置
        transformed_poly[:, 0] += self.position[0]
        transformed_poly[:, 1] += self.position[1]
        
        self.target_polygon = transformed_poly.tolist()
        
        # 更新中心点
        self.center_x = self.position[0]
        self.center_y = self.position[1]
        
        return self.target_polygon
        
    def update_position(self, x, y):
        """更新部件位置"""
        self.position = (x, y)
        return self.apply_transform()
        
    def update_scale(self, scale):
        """更新部件缩放"""
        self.scale = scale
        return self.apply_transform()
        
    def update_rotation(self, rotation):
        """更新部件旋转"""
        self.rotation = rotation
        return self.apply_transform()

# 添加RealPartSegmenter类
class RealPartSegmenter:
    """处理图像分割"""
    
    def __init__(self):
        self.segmented_parts = {}
    
    def segment_parts(self, image, keypoints, part_definitions):
        """根据骨架关键点分割图像部位"""
        # 这是一个简化的实现，实际应用中可能需要更复杂的算法
        segmented_parts = {}
        
        for part_name, kp_indices in part_definitions.items():
            valid_keypoints = []
            for idx in kp_indices:
                if idx < len(keypoints) and keypoints[idx][2] > 0.1:  # 检查置信度
                    valid_keypoints.append((keypoints[idx][0], keypoints[idx][1]))
            
            if len(valid_keypoints) >= 3:  # 至少需要3个点形成多边形
                # 创建凸包
                points = np.array(valid_keypoints)
                try:
                    hull = ConvexHull(points)
                    polygon_points = []
                    for vertex in hull.vertices:
                        polygon_points.append([points[vertex, 0], points[vertex, 1]])
                    
                    segmented_parts[part_name] = polygon_points
                except Exception as e:
                    print(f"为{part_name}创建凸包时出错: {e}")
                    # 如果凸包创建失败，使用原始点
                    segmented_parts[part_name] = [[p[0], p[1]] for p in valid_keypoints]
        
        return segmented_parts
    
    def render_segmentation_preview(self, image, part_polygons):
        """生成分割预览图像"""
        preview_img = image.copy()
        h, w = preview_img.shape[:2]
        
        # 创建叠加层
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个部位绘制多边形
        for part_name, points in part_polygons.items():
            if len(points) < 3:
                continue
                
            # 将颜色从[0-1]转换为[0-255]
            color = PART_COLORS.get(part_name, [0.5, 0.5, 0.5, 0.5])
            bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            # 绘制填充多边形
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], bgr_color)
            
            # 绘制轮廓
            cv2.polylines(overlay, [pts], True, (0, 0, 0), 2)
            
            # 添加部位名称标签
            center_x = int(sum(p[0] for p in points) / len(points))
            center_y = int(sum(p[1] for p in points) / len(points))
            cv2.putText(overlay, part_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加叠加层到原始图像
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, preview_img, 1 - alpha, 0, preview_img)
        
        return preview_img

# 添加部件编辑模式下的直接操作功能
class PartEditor(Widget):
    """用于在画布上直接编辑部件的小部件"""
    part_name = StringProperty("")
    position = ListProperty([0.0, 0.0])
    scale = NumericProperty(1.0)
    rotation = NumericProperty(0.0)
    
    def __init__(self, **kwargs):
        # 先初始化重要变量
        self.dragging = False
        self.scaling = False
        self.rotating = False
        self.drag_start_pos = None
        self.drag_current_pos = None
        self.prev_touch_pos = None
        self.scale_rotation_ref = None  # 用于缩放和旋转的参考点
        self.double_tap_time = 0
        self.force_dispatch = False
        self._last_update_time = 0  # 用于限制更新频率
        self._update_threshold = 1/30  # 限制为每秒30次更新
        
        # 安全处理位置和大小
        if 'position' in kwargs:
            pos = kwargs['position']
            try:
                kwargs['position'] = [float(pos[0]), float(pos[1])]
            except Exception as e:
                print(f"警告：无效的position值 {pos}: {e}")
                kwargs['position'] = [0.0, 0.0]
        
        if 'size' in kwargs:
            size = kwargs['size']
            try:
                kwargs['size'] = (float(size[0]), float(size[1]))
            except Exception as e:
                print(f"警告：无效的size值 {size}: {e}")
                kwargs['size'] = (150.0, 150.0)
        else:
            kwargs['size'] = (150.0, 150.0)
        
        self.register_event_type('on_part_edited')
        super(PartEditor, self).__init__(**kwargs)
        
        # 添加控制点 - 用于UI交互
        self.scale_handle_size = dp(24)  # 增大控制点大小，更容易触摸
        self.rotation_handle_size = dp(24)  # 增大控制点大小，更容易触摸
        self.handle_touch_area = dp(40)  # 增大触摸区域，让用户更容易操作
        
        # 添加视觉状态
        self.active_control = None  # 当前活动控制点
        
        # 延迟初始绘制，避免初始化期间的性能问题
        Clock.schedule_once(self.draw, 0.1)
    
    def on_part_edited(self, *args):
        """部件被编辑时触发的事件"""
        pass
    
    def on_position(self, instance, value):
        """当位置改变时更新小部件位置"""
        try:
            # 确保值是合法的浮点数
            x = float(value[0])
            y = float(value[1])
            width = float(self.width)
            height = float(self.height)
            # 计算控件左下角位置
            new_pos = (x - width/2, y - height/2)
            # 使用直接赋值而不是属性赋值，避免循环调用
            self.x = new_pos[0]
            self.y = new_pos[1]
        except Exception as e:
            print(f"设置position时出错: {e} - 值: {value}")
        
    def on_scale(self, instance, value):
        """当缩放改变时更新小部件大小"""
        try:
            # 保持中心位置不变的情况下调整大小
            center_x = float(self.position[0])
            center_y = float(self.position[1])
            new_size = (150.0 * value, 150.0 * value)
            # 直接赋值避免触发属性绑定
            self.width = new_size[0]
            self.height = new_size[1]
            # 更新位置
            self.x = center_x - self.width/2
            self.y = center_y - self.height/2
        except Exception as e:
            print(f"设置scale时出错: {e} - 值: {value}")
        
    def on_pos(self, instance, pos):
        """当位置改变时更新position属性"""
        try:
            if not hasattr(self, 'avoid_recursion'):
                self.avoid_recursion = False
                
            if self.avoid_recursion:
                return
                
            self.avoid_recursion = True
            center_x = float(pos[0]) + float(self.width)/2
            center_y = float(pos[1]) + float(self.height)/2
            # 更新position但不触发on_position
            self.position[0] = center_x
            self.position[1] = center_y
            self.avoid_recursion = False
        except Exception as e:
            print(f"设置pos时出错: {e} - 值: {pos}")
            self.avoid_recursion = False
        
    def on_size(self, instance, size):
        """当大小改变时更新位置以保持中心点不变"""
        try:
            if not hasattr(self, 'avoid_size_recursion'):
                self.avoid_size_recursion = False
                
            if self.avoid_size_recursion:
                return
                
            self.avoid_size_recursion = True
            # 更新位置以保持中心点不变
            center_x = self.position[0]
            center_y = self.position[1]
            self.x = center_x - size[0]/2
            self.y = center_y - size[1]/2
            self.avoid_size_recursion = False
        except Exception as e:
            print(f"设置size时出错: {e} - 值: {size}")
            self.avoid_size_recursion = False

    def on_touch_down(self, touch):
        """处理触摸事件"""
        # 检查点击是否在部件内
        if not self.collide_point(*touch.pos):
            return False
            
        # 计算触摸点到控制点的距离
        # 旋转控制点 - 顶部中心
        rotation_control_x = self.x + self.width/2
        rotation_control_y = self.y + self.height + self.rotation_handle_size/2
        dist_to_rotation = ((touch.x - rotation_control_x)**2 + (touch.y - rotation_control_y)**2)**0.5
        
        # 缩放控制点 - 右下角
        scale_control_x = self.x + self.width - self.scale_handle_size/2
        scale_control_y = self.y + self.scale_handle_size/2
        dist_to_scale = ((touch.x - scale_control_x)**2 + (touch.y - scale_control_y)**2)**0.5
        
        # 增大控制区域的检测范围，提高用户操作体验
        control_detection_radius = 30
        
        if dist_to_rotation < control_detection_radius:
            # 旋转操作
            self.rotating = True
            self.active_control = 'rotate'
            # 记录旋转参考点（部件中心）
            self.scale_rotation_ref = (self.position[0], self.position[1])
            self.prev_touch_pos = touch.pos
            # 将touch设为已处理
            touch.grab(self)
            # 提供视觉反馈
            self.draw()
            return True
        elif dist_to_scale < control_detection_radius:
            # 缩放操作
            self.scaling = True
            self.active_control = 'scale'
            # 记录缩放参考点（部件中心）
            self.scale_rotation_ref = (self.position[0], self.position[1])
            self.prev_touch_pos = touch.pos
            # 将touch设为已处理
            touch.grab(self)
            # 提供视觉反馈
            self.draw()
            return True
        else:
            # 开始拖动
            self.dragging = True
            self.active_control = 'drag'
            self.drag_start_pos = touch.pos
            self.drag_current_pos = touch.pos
            # 将touch设为已处理
            touch.grab(self)
            # 提供视觉反馈
            self.draw()
            
        # 检查是否是双击（用于切换选中状态）
        if touch.is_double_tap:
            # 触发双击事件
            app = App.get_running_app()
            if hasattr(app, '_select_part_for_edit') and callable(getattr(app, '_select_part_for_edit')):
                app._select_part_for_edit(self.part_name)
            
        # 无论点击在哪里，都将部件选中
        app = App.get_running_app()
        if hasattr(app, '_select_part_for_edit') and callable(getattr(app, '_select_part_for_edit')):
            app._select_part_for_edit(self.part_name)
            
        return True
    
    def on_touch_move(self, touch):
        """处理移动事件"""
        if touch.grab_current is not self:
            return False
        
        # 检查更新频率限制
        current_time = time.time()
        if current_time - self._last_update_time < self._update_threshold:
            # 如果更新太快，延迟处理
            return True
        
        self._last_update_time = current_time
        
        # 根据激活的控制操作类型处理
        if self.active_control == 'rotate' and self.scale_rotation_ref: # self.scale_rotation_ref is set in on_touch_down as the initial center
            # 旋转操作
            center_x, center_y = self.scale_rotation_ref # 使用手势开始时固定的中心点作为旋转轴
            
            prev_vector_x = self.prev_touch_pos[0] - center_x
            prev_vector_y = self.prev_touch_pos[1] - center_y
            current_vector_x = touch.x - center_x
            current_vector_y = touch.y - center_y
            
            prev_angle = math.atan2(prev_vector_y, prev_vector_x)
            current_angle = math.atan2(current_vector_y, current_vector_x)
            
            angle_diff = (current_angle - prev_angle) * 180 / math.pi
            
            new_rotation = (self.rotation + angle_diff) % 360
            self.rotation = new_rotation
            
            self.prev_touch_pos = touch.pos
            self.dispatch('on_part_edited')
            self.draw()
            print(f"旋转: PartEditor '{self.part_name}' Angle={self.rotation:.1f}°, Diff={angle_diff:.1f}°")
            return True
            
        elif self.active_control == 'scale' and self.scale_rotation_ref:
            equal_scale = False
            try:
                if 'shift' in Window.modifiers:
                    equal_scale = True
            except Exception as e:
                print(f"检查Shift键状态时出错 (Window.modifiers): {e}")

            original_scale = self.scale
            center_x, center_y = self.position # 部件中心点

            if not equal_scale:
                # Shift 未按下：尝试根据光标相对于中心的X,Y独立移动计算缩放因子，然后平均
                # 这仍然会通过单一的self.scale应用，所以是等比的，但计算方式不同
                prev_dist_to_center_x = abs(self.prev_touch_pos[0] - center_x)
                prev_dist_to_center_y = abs(self.prev_touch_pos[1] - center_y)
                current_dist_to_center_x = abs(touch.x - center_x)
                current_dist_to_center_y = abs(touch.y - center_y)

                scale_factor_x = current_dist_to_center_x / prev_dist_to_center_x if prev_dist_to_center_x > 1 else 1.0 # Avoid div by zero or small numbers
                scale_factor_y = current_dist_to_center_y / prev_dist_to_center_y if prev_dist_to_center_y > 1 else 1.0
                
                # 使用两个因子中变化较大的那个，或者平均值，来影响整体缩放
                # Averaging might feel more natural for a single corner drag for proportional scaling
                scale_factor = (scale_factor_x + scale_factor_y) / 2.0
                # Alternatively, one could use a geometric mean or prioritize the larger change if non-proportional feel is desired
                # scale_factor = math.sqrt(scale_factor_x * scale_factor_y)

                new_scale = self.scale * scale_factor
                if 0.1 <= new_scale <= 10.0: 
                    self.scale = new_scale
                print(f"无Shift缩放 (等比，但因子计算不同): PartEditor '{self.part_name}' Scale={self.scale:.2f}, Factor={scale_factor:.2f} (X:{scale_factor_x:.2f}, Y:{scale_factor_y:.2f})")

            else: # equal_scale is True (Shift pressed)
                # Shift 按下：等比例缩放，基于光标到部件中心的径向距离变化
                prev_radial_dist = math.sqrt((self.prev_touch_pos[0] - center_x)**2 + (self.prev_touch_pos[1] - center_y)**2)
                current_radial_dist = math.sqrt((touch.x - center_x)**2 + (touch.y - center_y)**2)
                
                if prev_radial_dist > 1: # Avoid div by zero or small numbers
                    scale_factor = current_radial_dist / prev_radial_dist
                    new_scale = self.scale * scale_factor
                    if 0.1 <= new_scale <= 10.0:
                        self.scale = new_scale
                    print(f"Shift缩放 (等比): PartEditor '{self.part_name}' Scale={self.scale:.2f}, Factor={scale_factor:.2f}")
            
            self.prev_touch_pos = touch.pos
            
        elif self.active_control == 'drag' and self.dragging and self.drag_current_pos:
            # 拖拽操作
            # 计算拖动偏移
            dx = touch.pos[0] - self.drag_current_pos[0]
            dy = touch.pos[1] - self.drag_current_pos[1]
            
            # 更新位置
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            self.position[0] = new_x
            self.position[1] = new_y
            
            # 更新拖动参考点
            self.drag_current_pos = touch.pos
            
            # 分发事件
            self.dispatch('on_part_edited')
            
            # 重绘视觉反馈
            self.draw()
            
            return True
            
        return False
        
    def on_touch_up(self, touch):
        """处理触摸释放事件"""
        if touch.grab_current is not self:
            return False
            
        # 释放touch
        touch.ungrab(self)
            
        if self.dragging or self.scaling or self.rotating:
            # 重置状态
            was_active = self.dragging or self.scaling or self.rotating
            self.dragging = False
            self.scaling = False
            self.rotating = False
            self.drag_start_pos = None
            self.drag_current_pos = None
            self.prev_touch_pos = None
            self.scale_rotation_ref = None
            self.active_control = None
            
            # 重绘以更新视觉效果
            self.draw()
            
            # 只有在之前确实执行了操作时才触发编辑事件
            if was_active:
                # 最后触发一次部件编辑事件，确保所有的更改都被处理
                self.dispatch('on_part_edited')
            
            return True
            
        return False
        
    def draw(self, *args):
        """绘制部件编辑器，使用优化的绘制策略"""
        with self.canvas:
            # 清除先前的绘制
            self.canvas.clear()
            
            # 选中状态显示边框
            app = App.get_running_app()
            is_selected = hasattr(app, 'current_part_for_edit') and app.current_part_for_edit == self.part_name
            
            # 绘制边框 - 所有部件都显示边框，选中的用不同颜色
            if is_selected:
                Color(0.2, 0.6, 1.0, 0.8)  # 蓝色边框表示选中
            else:
                Color(0.5, 0.5, 0.5, 0.5)  # 灰色边框表示未选中
            
            Line(rectangle=(self.x, self.y, self.width, self.height), width=2)
                
            if is_selected:
                # 绘制控制点 - 旋转控制点（顶部中心）
                if self.active_control == 'rotate':
                    # 高亮显示激活的旋转控制点
                    Color(0.2, 1.0, 1.0, 1.0)  # 更亮的青色
                    Ellipse(
                        pos=(self.x + self.width/2 - self.rotation_handle_size/2, 
                             self.y + self.height), 
                        size=(self.rotation_handle_size, self.rotation_handle_size)
                    )
                else:
                    # 普通状态的旋转控制点
                    Color(0.2, 0.8, 0.8, 1.0)
                    Ellipse(
                        pos=(self.x + self.width/2 - self.rotation_handle_size/2, 
                             self.y + self.height), 
                        size=(self.rotation_handle_size, self.rotation_handle_size)
                    )
                
                # 缩放控制点（右下角）
                if self.active_control == 'scale':
                    # 高亮显示激活的缩放控制点
                    Color(1.0, 0.2, 0.2, 1.0)  # 更亮的红色
                    Rectangle(
                        pos=(self.x + self.width - self.scale_handle_size, 
                             self.y), 
                        size=(self.scale_handle_size, self.scale_handle_size)
                    )
                else:
                    # 普通状态的缩放控制点
                    Color(0.8, 0.2, 0.2, 1.0)
                    Rectangle(
                        pos=(self.x + self.width - self.scale_handle_size, 
                             self.y), 
                        size=(self.scale_handle_size, self.scale_handle_size)
                    )
                
                # 如果正在拖动，显示拖动状态
                if self.active_control == 'drag':
                    # 添加整体半透明高亮效果
                    Color(0.4, 0.7, 1.0, 0.2)
                    Rectangle(
                        pos=(self.x, self.y),
                        size=(self.width, self.height)
                    )
    
    def update_visual(self):
        """更新视觉显示，使用节流方式减少不必要的重绘"""
        # 使用延迟执行，避免在同一帧内多次更新
        Clock.unschedule(self.draw)  # 取消任何待处理的绘制任务
        Clock.schedule_once(self.draw, 0)  # 在下一帧执行绘制

    def _on_part_edited(self, part_name, editor):
        """处理部件编辑事件"""
        app = App.get_running_app()
        if not hasattr(app, 'editable_parts') or part_name not in app.editable_parts:
            return
            
        # 获取编辑器中的属性
        img_x, img_y = self.convert_widget_to_image_coords(*editor.position)
        part = app.editable_parts[part_name]
        
        # 更新部件属性
        part.update_position(img_x, img_y)
        part.update_scale(editor.scale)
        part.update_rotation(editor.rotation)
        
        # 更新UI显示
        app.main_layout_widget.ids.prop_pos_x.text = str(int(img_x))
        app.main_layout_widget.ids.prop_pos_y.text = str(int(img_y))
        app.main_layout_widget.ids.prop_scale.text = str(round(editor.scale, 2))
        app.main_layout_widget.ids.prop_rotation.text = str(int(editor.rotation))
        
        # 更新合成预览
        if hasattr(app, '_update_composite_preview_delayed'):
            Clock.unschedule(app._update_composite_preview_delayed)
            Clock.schedule_once(app._update_composite_preview_delayed, 0.2)
        else:
            # 如果没有延迟方法，直接更新
            app._update_composite_preview()

    def _select_part_for_edit(self, part_name):
        """选择要编辑的部件"""
        if not hasattr(self, 'editable_parts') or part_name not in self.editable_parts:
            show_error_popup("部件不可用", f"部件 {part_name} 不可用或未创建。")
            return
            
        self.current_part_for_edit = part_name
        part = self.editable_parts[part_name]
        
        # 更新右侧面板属性
        self.main_layout_widget.ids.prop_pos_x.text = str(int(part.position[0]))
        self.main_layout_widget.ids.prop_pos_y.text = str(int(part.position[1]))
        self.main_layout_widget.ids.prop_scale.text = str(round(part.scale, 2))
        self.main_layout_widget.ids.prop_rotation.text = str(int(part.rotation))
        
        # 更新左侧控制面板按钮
        controls_layout = self.main_layout_widget.ids.step_controls_area
        parts_grid = None
        
        # 查找部件选择网格
        for child in controls_layout.children:
            if isinstance(child, BoxLayout) and child.orientation == 'vertical':
                for subchild in child.children:
                    if isinstance(subchild, GridLayout) and len(subchild.children) > 0:
                        # 找到了包含按钮的网格布局
                        parts_grid = subchild
                        break
                if parts_grid:
                    break
                    
        # 更新按钮高亮状态
        if parts_grid:
            print(f"找到部件网格，有 {len(parts_grid.children)} 个子控件")
            for btn in parts_grid.children:
                if isinstance(btn, Button):
                    if btn.text == part_name:
                        print(f"设置按钮 '{btn.text}' 高亮")
                        btn.background_color = (0.2, 0.6, 1, 1)  # 高亮显示
                    else:
                        btn.background_color = (0.5, 0.5, 0.5, 1)  # 普通显示
        else:
            print("未找到部件选择网格布局")
        
        # 更新DrawingCanvas上的部件选择状态
        canvas = self.main_layout_widget.ids.drawing_canvas
        if canvas.edit_mode != 'part_edit':
            canvas.edit_mode = 'part_edit'
        canvas._redraw()  # 重绘画布以更新所有部件
        
        # 更新状态栏
        self.main_layout_widget.ids.status_bar.text = f"正在编辑 {part_name} 部件。拖拽移动位置，拖动顶部蓝色圆点旋转，拖动右下角红色方块缩放（按Shift键等比例缩放）。"

    def apply_part_properties(self):
        """应用右侧面板中设置的部件属性"""
        if not hasattr(self, 'current_part_for_edit') or not self.current_part_for_edit or not hasattr(self, 'editable_parts'):
            return
            
        try:
            # 获取属性值
            pos_x = float(self.main_layout_widget.ids.prop_pos_x.text)
            pos_y = float(self.main_layout_widget.ids.prop_pos_y.text)
            scale = float(self.main_layout_widget.ids.prop_scale.text)
            rotation = float(self.main_layout_widget.ids.prop_rotation.text)
            
            # 更新部件属性
            part = self.editable_parts[self.current_part_for_edit]
            part.update_position(pos_x, pos_y)
            part.update_scale(scale)
            part.update_rotation(rotation)
            
            # 更新编辑器位置和大小
            canvas = self.main_layout_widget.ids.drawing_canvas
            if canvas.edit_mode == 'part_edit' and hasattr(canvas, 'part_editors') and self.current_part_for_edit in canvas.part_editors:
                editor = canvas.part_editors[self.current_part_for_edit]
                # 更新编辑器属性
                widget_pos = canvas.convert_image_to_widget_coords(pos_x, pos_y)
                editor.position = widget_pos
                editor.scale = scale
                editor.rotation = rotation
                editor.draw()  # 更新视觉效果
            
            # 更新合成预览
            self._update_composite_preview()
            
            self.main_layout_widget.ids.status_bar.text = f"已更新 {self.current_part_for_edit} 部件属性。"
            
        except ValueError as e:
            show_error_popup("输入错误", "请输入有效的数值。")
        except Exception as e:
            msg = f"应用部件属性时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("应用属性错误", msg)

    def reset_part_properties(self):
        """重置当前部件的属性为默认值"""
        if not hasattr(self, 'current_part_for_edit') or not self.current_part_for_edit or not hasattr(self, 'editable_parts'):
            return
            
        part = self.editable_parts[self.current_part_for_edit]
        
        try:
            # 获取对应的模板多边形中心
            template_poly = self.pre_segmented_parts_polygons[IMAGE_TYPE_TEMPLATE].get(self.current_part_for_edit, [])
            if template_poly and len(template_poly) >= 3:
                # 重置为模板多边形的中心位置
                center_x = sum(p[0] for p in template_poly) / len(template_poly)
                center_y = sum(p[1] for p in template_poly) / len(template_poly)
                # 更新部件属性
                part.update_position(center_x, center_y)
            else:
                # 没有模板多边形时，使用初始位置
                part.update_position(part.position[0], part.position[1])
                
            # 重置缩放和旋转
            part.update_scale(1.0)
            part.update_rotation(0.0)
            
            # 更新UI显示
            self.main_layout_widget.ids.prop_pos_x.text = str(int(part.position[0]))
            self.main_layout_widget.ids.prop_pos_y.text = str(int(part.position[1]))
            self.main_layout_widget.ids.prop_scale.text = "1.0"
            self.main_layout_widget.ids.prop_rotation.text = "0.0"
            
            # 更新编辑器位置和大小
            canvas = self.main_layout_widget.ids.drawing_canvas
            if canvas.edit_mode == 'part_edit' and hasattr(canvas, 'part_editors') and self.current_part_for_edit in canvas.part_editors:
                editor = canvas.part_editors[self.current_part_for_edit]
                # 更新编辑器属性
                widget_pos = canvas.convert_image_to_widget_coords(part.position[0], part.position[1])
                editor.position = widget_pos
                editor.scale = 1.0
                editor.rotation = 0.0
                editor.draw()  # 更新视觉效果
            
            # 更新合成预览
            self._update_composite_preview()
            
            self.main_layout_widget.ids.status_bar.text = f"已重置 {self.current_part_for_edit} 部件属性。"
            
        except Exception as e:
            msg = f"重置部件属性时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("重置属性错误", msg)

    def _action_export_final_image(self):
        """导出最终合成图像"""
        if not hasattr(self, 'composite_result') or self.composite_result is None:
            show_error_popup("导出错误", "没有可用的合成结果。请先完成部件编辑。")
            return
            
        try:
            # 确保导出目录存在
            if not os.path.exists(DEFAULT_EXPORT_DIR):
                os.makedirs(DEFAULT_EXPORT_DIR)
                
            # 生成唯一文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(DEFAULT_EXPORT_DIR, f"character_merged_{timestamp}.png")
            
            # 保存图像
            cv2.imwrite(export_path, self.composite_result)
            
            self.main_layout_widget.ids.status_bar.text = f"已导出图像到: {export_path}"
            show_error_popup("导出成功", f"合成图像已保存到:\n{export_path}")
            
        except Exception as e:
            msg = f"导出图像时出错: {e}"
            print(msg)
            traceback.print_exc()
            show_error_popup("导出错误", msg)

    def dp(self, value):
        """Helper to convert dp to pixels if needed, though Kivy handles dp in kv lang."""
        return value # Kivy's metrics handle dp directly in kv. For python, use kivy.metrics.dp

    def _update_composite_preview_delayed(self, dt):
        """延迟执行的合成预览更新，避免在拖动时频繁更新"""
        self._update_composite_preview()
        
    def _update_composite_preview(self):
        """更新并显示合成预览"""
        if not hasattr(self, 'editable_parts') or not self.editable_parts:
            return
            
        try:
            # 获取模板图像
            template_img = self.image_cv_originals[IMAGE_TYPE_TEMPLATE].copy()
            h, w = template_img.shape[:2]
            
            # 为每个部件创建变形图像和掩码
            warped_parts = {}
            
            # 添加进度显示
            self.main_layout_widget.ids.status_bar.text = "正在计算合成预览..."
            
            for part_name, part in self.editable_parts.items():
                # 源图像和多边形
                src_img = part.source_image
                src_poly = np.array(part.source_polygon, dtype=np.float32)
                
                # 变换后的目标多边形
                dst_poly = np.array(part.target_polygon, dtype=np.float32)
                
                # 变形部件
                warped_img, mask = self.warper.warp_part(src_img, src_poly, dst_poly, (w, h))
                
                warped_parts[part_name] = {
                    'warped_image': warped_img,
                    'mask': mask
                }
            
            # 合成最终图像
            composite = self.warper.composite_parts(template_img, warped_parts)
            
            # 转换为Kivy纹理
            buf = cv2.flip(composite, 0).tostring()
            texture = Texture.create(size=(w, h), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            # 更新画布显示
            canvas = self.main_layout_widget.ids.drawing_canvas
            canvas.texture = texture
            canvas.image_original_size = [w, h]
            
            # 确保画布处于正确的编辑模式
            if self.current_app_mode == APP_MODE_EDIT_PARTS:
                canvas.edit_mode = 'part_edit'
            
            # 存储合成结果用于导出
            self.composite_result = composite
            
            # 更新状态栏
            self.main_layout_widget.ids.status_bar.text = "预览已更新"
            
        except Exception as e:
            msg = f"更新合成预览时出错: {e}"
            print(msg)
            traceback.print_exc()
            self.main_layout_widget.ids.status_bar.text = "预览更新失败"

if __name__ == '__main__':
    CharacterAutoMergerApp().run() 