import sys
import os
import math # 添加 math 模块
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QSizePolicy, QMessageBox, QListWidget, QListWidgetItem,
                             QGraphicsPolygonItem) # Add QGraphicsPolygonItem
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QTransform, QPolygonF # 添加 QTransform, QPolygonF
from PyQt6.QtCore import Qt, QSize, QPointF, QRectF, QLineF # 添加 QRectF, QLineF
import cv2
import numpy as np
import traceback

# 假设 CharacterAutoMerger.py 在同一目录下或 Python 路径中
try:
    # 导入需要的类
    from CharacterAutoMerger import (CharacterAutoMerger, Config, PoseDetector,
                                     BodyPartExtractor, EnvironmentManager, PositionCalculator)
except ImportError as e:
    print(f"错误: 无法导入 CharacterAutoMerger 模块: {e}")
    print("请确保 CharacterAutoMerger.py 文件在当前目录或 Python 路径中。")
    # 在实际应用中，这里可能需要显示一个错误对话框并退出
    sys.exit(1)

# --- 配置 --- (需要用户根据实际情况修改)
DWPOSE_DIR = r"D:\drafts\DWPose" # <--- 修改为你本地的 DWPOSE 目录
DEFAULT_EXPORT_DIR = os.path.join(os.getcwd(), "CharacterAutoMerger_Exports_GUI") # 默认导出到运行目录下的子文件夹

# --- Editable Segment Item for adjusting segmentation --- 
class EditableSegmentItem(QGraphicsPolygonItem):
    Handle_None, Handle_Vertex = range(2)
    handle_size = 8.0
    handle_brush = QBrush(QColor(Qt.GlobalColor.darkCyan))
    handle_pen = QPen(QColor(Qt.GlobalColor.black), 1)
    polygon_pen_selected = QPen(QColor(Qt.GlobalColor.blue), 2, Qt.PenStyle.SolidLine)
    polygon_pen_default = QPen(QColor(Qt.GlobalColor.gray), 1, Qt.PenStyle.DashLine)

    def __init__(self, initial_polygon: QPolygonF, part_name: str = "segment", parent=None):
        super().__init__(initial_polygon, parent)
        self.part_name = part_name
        self.setFlags(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsPolygonItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(Qt.GlobalColor.transparent)) # Make it transparent by default, color can be set from outside
        self.setPen(self.polygon_pen_default)

        self._polygon = initial_polygon # Keep a mutable copy
        self.current_interaction_mode = self.Handle_None
        self.hovered_vertex_index = -1
        self.dragged_vertex_index = -1
        self.initial_mouse_pos = QPointF()
        self.initial_vertex_pos = QPointF()

    def get_handle_rect_at_vertex(self, index: int) -> QRectF:
        if 0 <= index < self._polygon.count():
            vertex = self._polygon.at(index)
            half_handle = self.handle_size / 2.0
            return QRectF(vertex.x() - half_handle, vertex.y() - half_handle, 
                          self.handle_size, self.handle_size)
        return QRectF()

    def paint(self, painter: QPainter, option, widget=None):
        # Let the parent class draw the polygon first based on self.polygon()
        super().paint(painter, option, widget)

        if self.isSelected():
            self.setPen(self.polygon_pen_selected) # Change pen when selected
            painter.setBrush(self.handle_brush)
            painter.setPen(self.handle_pen)
            for i in range(self.polygon().count()): # Use self.polygon() which is the current shape
                handle_rect = self.get_handle_rect_at_vertex(i)
                painter.drawEllipse(handle_rect)
        else:
            self.setPen(self.polygon_pen_default) # Revert pen when deselected

    def hoverMoveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        cursor = Qt.CursorShape.ArrowCursor
        self.hovered_vertex_index = -1
        if self.isSelected():
            pos = event.pos() # Item's local coordinates
            for i in range(self._polygon.count()):
                if self.get_handle_rect_at_vertex(i).contains(pos):
                    cursor = Qt.CursorShape.CrossCursor
                    self.hovered_vertex_index = i
                    break
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected() and self.hovered_vertex_index != -1:
            self.current_interaction_mode = self.Handle_Vertex
            self.dragged_vertex_index = self.hovered_vertex_index
            self.initial_mouse_pos = event.pos()
            self.initial_vertex_pos = self._polygon.at(self.dragged_vertex_index)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.current_interaction_mode == self.Handle_Vertex and self.dragged_vertex_index != -1:
            delta = event.pos() - self.initial_mouse_pos
            new_vertex_pos = self.initial_vertex_pos + delta
            
            # Update the polygon
            self.prepareGeometryChange()
            self._polygon.replace(self.dragged_vertex_index, new_vertex_pos)
            self.setPolygon(self._polygon) # This triggers an update
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.current_interaction_mode == self.Handle_Vertex:
            self.current_interaction_mode = self.Handle_None
            self.dragged_vertex_index = -1
            # Re-evaluate hover for cursor update
            self.hoverMoveEvent(event) # Pass a QGraphicsSceneHoverEvent like object or create one if needed
                                       # Or simply let the next actual hover event fix the cursor.
            event.accept()
            return
        super().mouseReleaseEvent(event)
    
    def get_edited_polygon(self) -> QPolygonF:
        return self._polygon


# 定义可交互的身体部位图元
class InteractivePartItem(QGraphicsPixmapItem):
    # 扩展控制柄类型枚举
    (Handle_None, 
     Handle_Scale_TL, Handle_Scale_TR, Handle_Scale_BL, Handle_Scale_BR, 
     Handle_Scale_T, Handle_Scale_R, Handle_Scale_B, Handle_Scale_L, 
     Handle_Rotate) = range(10)
    handle_size = 10.0

    def __init__(self, pixmap: QPixmap, original_cv_image: np.ndarray, # 存储原始CV图像用于导出
                 original_keypoints: list, 
                 crop_offset_x: int, crop_offset_y: int, 
                 part_name: str = "part", parent=None):
        super().__init__(pixmap, parent)
        self.part_name = part_name
        self.original_cv_image = original_cv_image # BGR 或 BGRA
        self.original_keypoints = original_keypoints
        self.crop_offset_x = crop_offset_x
        self.crop_offset_y = crop_offset_y

        self.setFlags(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True) # 需要悬停事件来改变光标
        self.setCacheMode(self.CacheMode.DeviceCoordinateCache) # 尝试解决拖影问题
        self.setOpacity(0.7) # 半透明
        self.setTransformOriginPoint(QPointF(self.pixmap().rect().center()))

        # 内部管理缩放因子和旋转
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._rotation = 0.0 # 度
        self.initial_pixmap_rect = self.pixmap().rect() # 记录原始 pixmap 尺寸
        self._update_internal_transform() # 初始化变换

        # 用于绘制关键点的画笔和颜色
        self.keypoint_pen = QPen(QColor("red"), 1)
        self.keypoint_radius = 2 # 调整关键点半径

        self.selection_pen = QPen(QColor(Qt.GlobalColor.darkBlue), 1, Qt.PenStyle.DashLine)
        self.handle_pen = QPen(QColor(Qt.GlobalColor.black), 1)
        self.handle_brush = QBrush(QColor(Qt.GlobalColor.lightGray))
        self.rotation_handle_brush = QBrush(QColor(Qt.GlobalColor.cyan))

        self.current_interaction_mode = self.Handle_None
        self.initial_mouse_scene_pos = QPointF()
        # 存储按下时的状态
        self.press_scale_x = 1.0
        self.press_scale_y = 1.0
        self.press_rotation = 0.0
        self.press_pos = QPointF()
        self.press_transform_origin = QPointF()

    def _update_internal_transform(self):
        """根据内部 _scale_x, _scale_y, _rotation 更新图元变换"""
        self.prepareGeometryChange() # 通知即将发生几何变化
        transform = QTransform()
        # 物体中心的变换原点
        origin = self.transformOriginPoint()
        # 1. 移动到原点
        transform.translate(origin.x(), origin.y())
        # 2. 旋转
        transform.rotate(self._rotation)
        # 3. 缩放
        transform.scale(self._scale_x, self._scale_y)
        # 4. 移回原位
        transform.translate(-origin.x(), -origin.y())
        self.setTransform(transform)

    def setItemScale(self, sx: float, sy: float):
        self._scale_x = max(0.05, sx) # 最小缩放值调整
        self._scale_y = max(0.05, sy)
        self._update_internal_transform()

    def setItemRotation(self, angle_degrees: float):
        self._rotation = angle_degrees
        self._update_internal_transform()

    # 重写 setScale 和 setRotation 以使用内部管理
    def setScale(self, scale: float): # QGraphicsItem 原有方法
        self.setItemScale(scale, scale)

    def setRotation(self, angle: float): # QGraphicsItem 原有方法
        self.setItemRotation(angle)

    def scale(self) -> float: # QGraphicsItem 原有方法, 返回平均或X轴缩放
        return self._scale_x 
        
    def rotation(self) -> float: # QGraphicsItem 原有方法
        return self._rotation

    def boundingRect(self) -> QRectF: # 需要根据变换后的尺寸动态计算
        # self.transform().mapRect(QRect) 返回 QRect, 需要转换为 QRectF
        return QRectF(self.transform().mapRect(self.initial_pixmap_rect))

    def get_handle_rect(self, handle_type: int) -> QRectF:
        # 控制柄现在基于 initial_pixmap_rect，因为它们在变换前绘制
        rect = self.initial_pixmap_rect
        half_handle = self.handle_size / 2.0
        center_x, center_y = rect.center().x(), rect.center().y()

        if handle_type == self.Handle_Scale_TL: return QRectF(rect.left() - half_handle, rect.top() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_TR: return QRectF(rect.right() - half_handle, rect.top() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_BL: return QRectF(rect.left() - half_handle, rect.bottom() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_BR: return QRectF(rect.right() - half_handle, rect.bottom() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_T: return QRectF(center_x - half_handle, rect.top() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_R: return QRectF(rect.right() - half_handle, center_y - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_B: return QRectF(center_x - half_handle, rect.bottom() - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Scale_L: return QRectF(rect.left() - half_handle, center_y - half_handle, self.handle_size, self.handle_size)
        if handle_type == self.Handle_Rotate:
            rot_center = QPointF(center_x, rect.top() - self.handle_size * 1.5)
            return QRectF(rot_center.x() - half_handle, rot_center.y() - half_handle, self.handle_size, self.handle_size)
        return QRectF()

    def get_rotation_handle_center(self) -> QPointF:
        rect = self.initial_pixmap_rect
        return QPointF(rect.center().x(), rect.top() - self.handle_size * 1.5)

    def paint(self, painter: QPainter, option, widget=None):
        # painter.save()
        # 注意：由于我们自己管理 transform，父类的 paint 可能不需要直接调用
        # super().paint(painter, option, widget) # 这会应用 QGraphicsItem 的 transform
        # 我们需要在未变换的坐标系中绘制 pixmap，然后由 QGraphicsView 应用总的 transform
        painter.drawPixmap(self.initial_pixmap_rect.topLeft(), self.pixmap())

        painter.setPen(self.keypoint_pen)
        
        if self.original_keypoints:
            for kp in self.original_keypoints:
                # 检查关键点格式 (带置信度或不带)
                has_confidence = len(kp) >= 3
                confidence = kp[2] if has_confidence else 1.0 # 默认置信度为1

                if confidence > 0.1: # 检查置信度 (可调整)
                    original_x, original_y = kp[0], kp[1]
                    local_x = original_x - self.crop_offset_x
                    local_y = original_y - self.crop_offset_y
                    
                    # 绘制一个小圆点
                    painter.drawEllipse(QPointF(local_x, local_y), 
                                        self.keypoint_radius, 
                                        self.keypoint_radius)
        
        if self.isSelected():
            painter.setPen(self.selection_pen)
            painter.drawRect(self.initial_pixmap_rect) # 绘制原始尺寸的包围框

            # 绘制所有角点缩放控制柄
            painter.setPen(self.handle_pen)
            painter.setBrush(self.handle_brush)
            for ht in range(self.Handle_Scale_TL, self.Handle_Scale_L + 1): # 所有缩放手柄
                painter.drawRect(self.get_handle_rect(ht))

            # 绘制旋转控制柄 (顶部中心上方)
            painter.setBrush(self.rotation_handle_brush)
            rot_center = self.get_rotation_handle_center()
            painter.drawLine(QPointF(self.initial_pixmap_rect.center().x(), self.initial_pixmap_rect.top()), rot_center)
            painter.drawEllipse(rot_center, self.handle_size / 2, self.handle_size / 2)
        # painter.restore()
    
    def hoverMoveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        self.mouse_over_handle = self.Handle_None
        cursor = Qt.CursorShape.ArrowCursor
        if self.isSelected():
            pos = event.pos() # 这是图元局部坐标下的鼠标位置
            # 需要将控制柄的局部矩形与局部鼠标位置比较
            if self.get_handle_rect(self.Handle_Scale_TL).contains(pos) or \
               self.get_handle_rect(self.Handle_Scale_BR).contains(pos):
                cursor = Qt.CursorShape.SizeFDiagCursor
                self.mouse_over_handle = self.Handle_Scale_TL if self.get_handle_rect(self.Handle_Scale_TL).contains(pos) else self.Handle_Scale_BR
            elif self.get_handle_rect(self.Handle_Scale_TR).contains(pos) or \
                 self.get_handle_rect(self.Handle_Scale_BL).contains(pos):
                cursor = Qt.CursorShape.SizeBDiagCursor
                self.mouse_over_handle = self.Handle_Scale_TR if self.get_handle_rect(self.Handle_Scale_TR).contains(pos) else self.Handle_Scale_BL
            elif self.get_handle_rect(self.Handle_Scale_T).contains(pos) or \
                 self.get_handle_rect(self.Handle_Scale_B).contains(pos):
                cursor = Qt.CursorShape.SizeVerCursor
                self.mouse_over_handle = self.Handle_Scale_T if self.get_handle_rect(self.Handle_Scale_T).contains(pos) else self.Handle_Scale_B
            elif self.get_handle_rect(self.Handle_Scale_L).contains(pos) or \
                 self.get_handle_rect(self.Handle_Scale_R).contains(pos):
                cursor = Qt.CursorShape.SizeHorCursor
                self.mouse_over_handle = self.Handle_Scale_L if self.get_handle_rect(self.Handle_Scale_L).contains(pos) else self.Handle_Scale_R
            elif self.get_handle_rect(self.Handle_Rotate).contains(pos):
                cursor = Qt.CursorShape.CrossCursor
                self.mouse_over_handle = self.Handle_Rotate
        self.setCursor(cursor)
        # super().hoverMoveEvent(event) # 不需要，因为我们自己处理光标

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected() and self.mouse_over_handle != self.Handle_None:
            self.current_interaction_mode = self.mouse_over_handle
            self.initial_mouse_scene_pos = event.scenePos()
            self.press_scale_x = self._scale_x
            self.press_scale_y = self._scale_y
            self.press_rotation = self._rotation
            self.press_pos = self.pos()
            self.press_transform_origin = self.transformOriginPoint() # 记录按下时的变换原点
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        mode = self.current_interaction_mode
        if mode == self.Handle_None: 
            super().mouseMoveEvent(event)
            return
        
        current_mouse_scene_pos = event.scenePos()
        # 对于以中心为原点的统一缩放 (角点控制柄)
        if mode in [self.Handle_Scale_TL, self.Handle_Scale_TR, self.Handle_Scale_BL, self.Handle_Scale_BR]:
            origin_scene_pos = self.mapToScene(self.press_transform_origin)
            dist_initial = QLineF(origin_scene_pos, self.initial_mouse_scene_pos).length()
            dist_current = QLineF(origin_scene_pos, current_mouse_scene_pos).length()
            if dist_initial > 1e-6:
                scale_factor_change = dist_current / dist_initial
                new_sx = self.press_scale_x * scale_factor_change
                new_sy = self.press_scale_y * scale_factor_change
                self.setItemScale(new_sx, new_sy)
        
        # 非均匀缩放 (边缘控制柄)
        elif mode in [self.Handle_Scale_L, self.Handle_Scale_R, self.Handle_Scale_T, self.Handle_Scale_B]:
            # 将鼠标位移从场景坐标转换到图元的旋转坐标系下的局部坐标
            # 变换：场景 -> 图元父级 -> 图元（无缩放，只有旋转和平移）
            delta_mouse_scene = current_mouse_scene_pos - self.initial_mouse_scene_pos
            
            # 构建一个只包含旋转的变换，用于将场景位移转换到图元旋转后的局部坐标系
            rotation_transform = QTransform().rotate(self.press_rotation)
            # 场景位移在图元旋转坐标系下的投影
            delta_mouse_local_rotated = rotation_transform.inverted()[0].map(delta_mouse_scene) 
            # delta_mouse_local_rotated 现在是 (dx, dy) 在图元旋转后的坐标轴上的变化

            new_sx, new_sy = self.press_scale_x, self.press_scale_y
            pos_offset_local = QPointF(0,0) # 因尺寸变化导致的位置补偿（局部坐标）

            orig_width_no_scale = self.initial_pixmap_rect.width()
            orig_height_no_scale = self.initial_pixmap_rect.height()

            if mode == self.Handle_Scale_L:
                new_width_visual = orig_width_no_scale * self.press_scale_x - delta_mouse_local_rotated.x()
                new_sx = new_width_visual / orig_width_no_scale if orig_width_no_scale else 1.0
                pos_offset_local.setX(delta_mouse_local_rotated.x() / 2.0)
            elif mode == self.Handle_Scale_R:
                new_width_visual = orig_width_no_scale * self.press_scale_x + delta_mouse_local_rotated.x()
                new_sx = new_width_visual / orig_width_no_scale if orig_width_no_scale else 1.0
                pos_offset_local.setX(delta_mouse_local_rotated.x() / 2.0)
            elif mode == self.Handle_Scale_T:
                new_height_visual = orig_height_no_scale * self.press_scale_y - delta_mouse_local_rotated.y()
                new_sy = new_height_visual / orig_height_no_scale if orig_height_no_scale else 1.0
                pos_offset_local.setY(delta_mouse_local_rotated.y() / 2.0)
            elif mode == self.Handle_Scale_B:
                new_height_visual = orig_height_no_scale * self.press_scale_y + delta_mouse_local_rotated.y()
                new_sy = new_height_visual / orig_height_no_scale if orig_height_no_scale else 1.0
                pos_offset_local.setY(delta_mouse_local_rotated.y() / 2.0)
            
            self.setItemScale(new_sx, new_sy)
            # 将局部位置补偿转换回场景坐标并应用
            pos_offset_scene = rotation_transform.map(pos_offset_local)
            self.setPos(self.press_pos + pos_offset_scene)

        elif mode == self.Handle_Rotate:
            origin_scene_pos = self.mapToScene(self.press_transform_origin)
            angle_initial_rad = math.atan2(self.initial_mouse_scene_pos.y() - origin_scene_pos.y(), 
                                         self.initial_mouse_scene_pos.x() - origin_scene_pos.x())
            angle_current_rad = math.atan2(current_mouse_scene_pos.y() - origin_scene_pos.y(), 
                                         current_mouse_scene_pos.x() - origin_scene_pos.x())
            delta_angle_deg = math.degrees(angle_current_rad - angle_initial_rad)
            self.setItemRotation(self.press_rotation + delta_angle_deg)
        
        event.accept()

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.current_interaction_mode != self.Handle_None:
            self.current_interaction_mode = self.Handle_None
            # hoverMoveEvent 将会根据鼠标位置重置光标
            # self.setCursor(Qt.CursorShape.ArrowCursor) 
            event.accept()
            return
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("角色自动合成器")
        self.setGeometry(100, 100, 1400, 800) # 增加了宽度以容纳右侧面板

        # --- 文件路径存储 ---
        self.template_path = None
        self.head_path = None
        self.body_path = None
        self.head_image_cv_original = None # Store original head CV2 image
        self.body_image_cv_original = None # Store original body CV2 image

        # --- 预览图尺寸 ---
        self.preview_size = QSize(250, 250) # 预览图的最大尺寸

        # --- 处理结果存储 ---
        self.template_landmarks = None
        self.head_landmarks = None
        self.body_landmarks = None
        self.extracted_head_part = None # 从 head_image 提取的头部 numpy 数组
        self.extracted_head_position = None # 对应的位置信息
        self.extracted_body_parts = None # 身体部位字典 {name: {'image': np.array, 'position': dict}}
        self.scene_items_map = {} # 部位名称 -> QGraphicsItem (InteractivePartItem or EditableSegmentItem)
        self.template_cv_image_for_export = None # Store loaded template for export
        self.segmentation_data_for_adjustment = {} # Stores contours and source image info

        # --- UI 状态 ---
        self.current_mode = "upload" # "upload", "adjust_segmentation", "edit_parts"

        # --- 初始化核心逻辑 --- (确保 DWPOSE_DIR 正确)
        try:
            if not os.path.exists(DWPOSE_DIR):
                 raise FileNotFoundError(f"DWPose 目录不存在: {DWPOSE_DIR}")
            if not os.path.exists(DEFAULT_EXPORT_DIR): os.makedirs(DEFAULT_EXPORT_DIR)
            # 可以在这里添加环境检查和设置，如果需要的话
            # env_manager = EnvironmentManager(DWPOSE_DIR)
            # env_manager.setup() # 注意：这可能会安装依赖并下载模型
            print(f"使用 DWPose 目录: {DWPOSE_DIR}")
            self.merger = CharacterAutoMerger(DWPOSE_DIR, DEFAULT_EXPORT_DIR)
            # 可以设置阈值
            self.merger.set_keypoint_thresholds(head_threshold=0.4, body_threshold=0.2)
            print("CharacterAutoMerger 初始化成功。")
        except Exception as e:
            print(f"初始化 CharacterAutoMerger 失败: {e}")
            QMessageBox.critical(self, "初始化错误", f"无法初始化核心处理模块:\n{e}\n\n请检查 DWPOSE_DIR 设置和依赖项。")
            # 禁用所有处理相关按钮
            # sys.exit(1) # 或者禁用按钮等
            self.merger = None # 标记为不可用

        # --- 主布局 (改为垂直布局) ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget) # 主布局改为垂直

        # --- 上方面板布局 (左侧+中心) ---
        top_panel_layout = QHBoxLayout()

        # --- 左侧控制面板 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(280) # 略微调整宽度

        # 文件上传区域
        upload_group = QVBoxLayout()

        # --- 模板上传 ---
        self.btn_upload_template = QPushButton("上传模板图像")
        self.lbl_template_path = QLabel("未选择文件")
        self.lbl_template_path.setWordWrap(True)
        self.lbl_template_preview = QLabel("模板预览")
        self.lbl_template_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_template_preview.setFixedSize(QSize(240,240)) # 略微调整预览大小
        self.lbl_template_preview.setStyleSheet("border: 1px solid lightgrey; color: grey;")
        self.btn_upload_template.clicked.connect(lambda: self.upload_file("template"))
        upload_group.addWidget(self.btn_upload_template)
        upload_group.addWidget(self.lbl_template_path)
        upload_group.addWidget(self.lbl_template_preview)
        upload_group.addSpacing(10)

        # --- 头部上传 ---
        self.btn_upload_head = QPushButton("上传头部图像")
        self.lbl_head_path = QLabel("未选择文件")
        self.lbl_head_path.setWordWrap(True)
        self.lbl_head_preview = QLabel("头部预览")
        self.lbl_head_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_head_preview.setFixedSize(QSize(240,240))
        self.lbl_head_preview.setStyleSheet("border: 1px solid lightgrey; color: grey;")
        self.btn_upload_head.clicked.connect(lambda: self.upload_file("head"))
        upload_group.addWidget(self.btn_upload_head)
        upload_group.addWidget(self.lbl_head_path)
        upload_group.addWidget(self.lbl_head_preview)
        upload_group.addSpacing(10)

        # --- 身体上传 ---
        self.btn_upload_body = QPushButton("上传身体图像")
        self.lbl_body_path = QLabel("未选择文件")
        self.lbl_body_path.setWordWrap(True)
        self.lbl_body_preview = QLabel("身体预览")
        self.lbl_body_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_body_preview.setFixedSize(QSize(240,240))
        self.lbl_body_preview.setStyleSheet("border: 1px solid lightgrey; color: grey;")
        self.btn_upload_body.clicked.connect(lambda: self.upload_file("body"))
        upload_group.addWidget(self.btn_upload_body)
        upload_group.addWidget(self.lbl_body_path)
        upload_group.addWidget(self.lbl_body_preview)

        left_layout.addLayout(upload_group)
        left_layout.addStretch() # 将上传部分推到顶部

        # --- 中心图像显示区域 (使用 QGraphicsView) ---
        center_area = QWidget()
        center_layout = QVBoxLayout(center_area)
        
        self.scene = QGraphicsScene(self)
        # self.scene.setBackgroundBrush(QColor("lightgrey")) # 可选：设置场景背景色
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing) # 抗锯齿
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform) # 平滑缩放
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate) # 减少拖影
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setStyleSheet("border: 1px solid grey;")
        
        center_layout.addWidget(self.view)

        # --- 右侧部位列表面板 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMinimumWidth(200)
        right_panel.setMaximumWidth(250)
        
        lbl_parts_list_title = QLabel("场景部位列表")
        lbl_parts_list_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parts_list_widget = QListWidget()
        self.parts_list_widget.currentItemChanged.connect(self.on_part_selected_in_list)
        # 连接场景选择变化到列表 (实现双向同步)
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)

        right_layout.addWidget(lbl_parts_list_title)
        right_layout.addWidget(self.parts_list_widget)
        top_panel_layout.addWidget(left_panel)
        top_panel_layout.addWidget(center_area, 1)
        top_panel_layout.addWidget(right_panel)

        # --- 底部控制按钮 ---
        bottom_button_layout = QHBoxLayout()
        self.btn_start_processing = QPushButton("1. 开始处理")
        self.btn_confirm_segmentation = QPushButton("2. 确认分割") 
        self.btn_edit_parts = QPushButton("3. 编辑部位") 
        self.btn_export = QPushButton("4. 导出结果")

        # 根据 merger 是否成功初始化来设置按钮状态
        initial_enable_state = self.merger is not None
        self.btn_start_processing.setEnabled(False) # 初始禁用，等待文件上传
        self.btn_confirm_segmentation.setEnabled(False)
        self.btn_edit_parts.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_upload_template.setEnabled(initial_enable_state)
        self.btn_upload_head.setEnabled(initial_enable_state)
        self.btn_upload_body.setEnabled(initial_enable_state)

        # 连接信号到槽函数
        self.btn_start_processing.clicked.connect(self.start_processing)
        self.btn_confirm_segmentation.clicked.connect(self.confirm_segmentation_and_proceed)
        self.btn_edit_parts.clicked.connect(self.go_to_edit_parts_step) # Renamed from go_to_next_step
        self.btn_export.clicked.connect(self.export_results)

        bottom_button_layout.addStretch()
        bottom_button_layout.addWidget(self.btn_start_processing)
        bottom_button_layout.addWidget(self.btn_confirm_segmentation)
        bottom_button_layout.addWidget(self.btn_edit_parts)
        bottom_button_layout.addWidget(self.btn_export)
        bottom_button_layout.addStretch()

        # --- 主布局整合 ---
        main_layout.addLayout(top_panel_layout, 1)
        main_layout.addLayout(bottom_button_layout)

        # TODO: 添加右侧面板（部位预览、图层控制）

    def upload_file(self, file_type):
        """处理文件上传逻辑"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {file_type} 图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
        )

        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "文件错误", f"无法加载图像文件:\n{file_path}")
                return

            try:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                if q_image.isNull(): raise ValueError("创建 QImage 失败")
                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull(): raise ValueError("从 QImage 创建 QPixmap 失败")

                scaled_pixmap = pixmap.scaled(self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

                if file_type == "template":
                    self.template_path = file_path
                    self.lbl_template_path.setText(os.path.basename(file_path))
                    self.lbl_template_preview.setPixmap(scaled_pixmap)
                    self.template_cv_image_for_export = img.copy()
                    self.display_template_in_view() # Display initially if desired
                elif file_type == "head":
                    self.head_path = file_path
                    self.lbl_head_path.setText(os.path.basename(file_path))
                    self.lbl_head_preview.setPixmap(scaled_pixmap)
                    self.head_image_cv_original = img.copy() # Store original head image
                elif file_type == "body":
                    self.body_path = file_path
                    self.lbl_body_path.setText(os.path.basename(file_path))
                    self.lbl_body_preview.setPixmap(scaled_pixmap)
                    self.body_image_cv_original = img.copy() # Store original body image

            except Exception as e:
                error_msg = f"处理图像预览时出错 ({os.path.basename(file_path)}):\n{e}"
                print(error_msg)
                QMessageBox.warning(self, "预览错误", error_msg)
                if file_type == "template": self.lbl_template_preview.setText("预览失败")
                elif file_type == "head": self.lbl_head_preview.setText("预览失败")
                elif file_type == "body": self.lbl_body_preview.setText("预览失败")
                return

            self.check_all_files_uploaded()

    def check_all_files_uploaded(self):
        """检查是否所有必需的文件都已上传 (并且 merger 可用)"""
        if self.merger and self.template_path and self.head_path and self.body_path:
            self.btn_start_processing.setEnabled(True)
            print("所有文件已上传，可以开始处理。")
        else:
            self.btn_start_processing.setEnabled(False)
            self.btn_confirm_segmentation.setEnabled(False)
            self.btn_edit_parts.setEnabled(False)

    # --- 按钮槽函数 --- 
    def start_processing(self):
        if not self.merger:
            QMessageBox.critical(self, "错误", "核心处理模块未初始化，无法处理。")
            return

        if not all([self.template_path, self.head_path, self.body_path]):
            QMessageBox.warning(self, "文件不完整", "请确保已上传模板、头部和身体图像。")
            return

        print("开始进行姿态检测和部位提取...")
        self.set_ui_state_processing(True) # 进入处理状态，禁用相关UI

        processing_outcome_can_edit = False # 标记处理结果是否允许进入编辑步骤
        try:
            # 加载图像 (再次加载以确保获取最新文件，虽然预览时已加载)
            template_img = cv2.imread(self.template_path)
            head_img = cv2.imread(self.head_path)
            body_img = cv2.imread(self.body_path)

            if template_img is None or head_img is None or body_img is None:
                 raise ValueError("无法加载一个或多个图像文件进行处理。")

            # 1. 检测姿态
            print("检测模板姿态...")
            self.template_landmarks = self.merger.pose_detector.detect_pose(template_img, "模板")
            print("检测头部姿态...")
            self.head_landmarks = self.merger.pose_detector.detect_pose(head_img, "头部")
            print("检测身体姿态...")
            self.body_landmarks = self.merger.pose_detector.detect_pose(body_img, "身体")

            # 2. 提取部位
            print("提取头部...")
            # 注意：我们是从 head_img 中提取头部用于后续放置
            self.extracted_head_part, self.extracted_head_position = \
                self.merger.body_part_extractor.extract_head(head_img, self.head_landmarks)

            print("提取身体部位...")
            self.extracted_body_parts = \
                self.merger.body_part_extractor.extract_body_parts(body_img, self.body_landmarks)

            # 检查提取结果是否有效 (简单检查)
            # 只有当实际提取到可用的图像数据时，才认为可以进入编辑步骤
            if (self.extracted_head_part is not None and self.extracted_head_part.size > 0) or \
               (self.extracted_body_parts and any(data.get('image') is not None and data['image'].size > 0 
                                                 for data in self.extracted_body_parts.values())):
                processing_outcome_can_edit = True
            
            if not processing_outcome_can_edit:
                print("警告：未能有效提取任何头部或身体图像部分用于编辑。")

            print(f"提取完成：头部 + {len(self.extracted_body_parts or {})} 个身体部位。")

            # 5. 启用 "下一步" 按钮
            if processing_outcome_can_edit:
                # Transition to segmentation adjustment phase
                self.current_mode = "adjust_segmentation"
                self.populate_segmentation_data() # New method to prepare data for adjustment
                self.display_for_segmentation_adjustment() # New method to show UI for adjustment
                QMessageBox.information(self, "处理完成", "姿态检测和部位提取完成！\n请调整分割区域，然后点击 '确认分割' 继续。")
            else:
                QMessageBox.warning(self, "处理完成", "处理已完成，但未能提取到有效的图像部位进行编辑。")


        except Exception as e:
            # processing_outcome_can_edit 保持 False
            error_msg = f"处理过程中发生错误:\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "处理失败", error_msg)
            # 处理失败，重置状态
            # self.btn_next_step.setEnabled(False) # 将在 finally 块中设置
            self.reset_extracted_data() # 清除可能部分成功的数据
        finally:
            # 无论成功失败，恢复UI状态(例如上传按钮)
            self.set_ui_state_processing(False)

            # 根据处理结果精确设置按钮的状态
            if processing_outcome_can_edit and self.current_mode == "adjust_segmentation":
                self.btn_confirm_segmentation.setEnabled(True)
                self.btn_start_processing.setEnabled(False) 
                self.btn_edit_parts.setEnabled(False)
            else:
                self.btn_confirm_segmentation.setEnabled(False)
                self.btn_edit_parts.setEnabled(False)
                # 如果处理失败或无有效结果，"开始处理"按钮的状态应由文件是否齐全决定
                self.check_all_files_uploaded()
            
            # self.btn_export 的状态由 go_to_edit_parts_step 控制

    def populate_segmentation_data(self):
        self.segmentation_data_for_adjustment.clear()
        if self.extracted_head_part is not None and self.extracted_head_position:
            self.segmentation_data_for_adjustment["Head"] = {
                'contours': self.extracted_head_position.get('segmentation_contours', []),
                'source_image_ref_name': 'head',
                'original_position_info': self.extracted_head_position
            }
        
        if self.extracted_body_parts:
            for part_name, part_data in self.extracted_body_parts.items():
                if part_data and part_data.get('image') is not None and part_data.get('position'):
                    self.segmentation_data_for_adjustment[part_name] = {
                        'contours': part_data['position'].get('segmentation_contours', []),
                        'source_image_ref_name': 'body',
                        'original_position_info': part_data['position']
                    }
    
    def display_for_segmentation_adjustment(self):
        self.scene.clear()
        self.parts_list_widget.clear()
        self.scene_items_map.clear()

        # For now, focus on adjusting body parts on the body image
        # Later, we can add a switcher or separate view for head segmentation adjustment if needed
        source_image_to_display_cv = self.body_image_cv_original
        current_source_ref_name = 'body' # Default to body image segments

        # Determine which image to display based on selected part or default to body
        # This part might need more complex logic if we allow switching between head/body view here
        # For now, always show body image and its parts for segmentation adjustment.
        # If head parts exist and body_image_cv_original is None, it might indicate an issue or a head-only adjust mode is needed.

        if self.segmentation_data_for_adjustment:
            # Check if there are any body parts to display, if not, and head parts exist, switch to head
            has_body_parts_to_adjust = any(data.get('source_image_ref_name') == 'body' for data in self.segmentation_data_for_adjustment.values())
            if not has_body_parts_to_adjust and "Head" in self.segmentation_data_for_adjustment:
                if self.head_image_cv_original is not None:
                    source_image_to_display_cv = self.head_image_cv_original
                    current_source_ref_name = 'head'
                    print("Switching segmentation view to HEAD image as no body parts found for adjustment.")
                else:
                    QMessageBox.warning(self, "图像缺失", "检测到头部待调整但头部原始图像未加载。")
                    return # Cannot proceed without the image
            elif source_image_to_display_cv is None and "Head" in self.segmentation_data_for_adjustment and self.head_image_cv_original is not None:
                 # If body image was none, but head exists and has an image
                 source_image_to_display_cv = self.head_image_cv_original
                 current_source_ref_name = 'head'
                 print("Body image not available for segmentation, switching to HEAD image.")

        if source_image_to_display_cv is None:
            QMessageBox.warning(self, "无图像", f"无法加载用于分割调整的源图像 ({current_source_ref_name})。")
            return

        source_pixmap = self.convert_cv_to_pixmap(source_image_to_display_cv)
        if source_pixmap.isNull():
            QMessageBox.warning(self, "转换错误", "无法将源图像转换为Pixmap。")
            return

        bg_item = self.scene.addPixmap(source_pixmap)
        bg_item.setZValue(-10) # Ensure background is behind polygons
        self.view.setSceneRect(source_pixmap.rect().toRectF())
        self.view.fitInView(source_pixmap.rect().toRectF(), Qt.AspectRatioMode.KeepAspectRatio)

        part_colors = [
            QColor(255, 0, 0, 100),    # Red
            QColor(0, 255, 0, 100),    # Green
            QColor(0, 0, 255, 100),    # Blue
            QColor(255, 255, 0, 100),  # Yellow
            QColor(0, 255, 255, 100),  # Cyan
            QColor(255, 0, 255, 100),  # Magenta
            QColor(128, 0, 128, 100),  # Purple
            QColor(255, 165, 0, 100), # Orange
            QColor(0, 128, 0, 100),   # Dark Green
        ]
        color_index = 0
        parts_displayed_count = 0

        for part_name, data in self.segmentation_data_for_adjustment.items():
            if data.get('source_image_ref_name') == current_source_ref_name:
                cv_contours = data.get('contours', [])
                if not cv_contours or cv_contours[0] is None: 
                    print(f"警告: 部位 '{part_name}' 没有有效的轮廓数据，跳过显示。")
                    continue

                main_contour_cv = cv_contours[0]
                q_polygon = QPolygonF()
                for point_array in main_contour_cv: 
                    q_polygon.append(QPointF(point_array[0][0], point_array[0][1]))
                
                # Use EditableSegmentItem instead of QGraphicsPolygonItem
                segment_item = EditableSegmentItem(q_polygon, part_name=part_name)
                
                brush_color = part_colors[color_index % len(part_colors)]
                # EditableSegmentItem sets its own brush to transparent by default, 
                # so we set the brush color directly if we want initial fill.
                # However, for adjustment, a transparent fill with distinct border is better.
                # Let's use the item's default pen for deselected state and set a specific brush for clarity.
                brush_color.setAlpha(100) # Semi-transparent
                segment_item.setBrush(brush_color)
                segment_item.setPen(QPen(brush_color.darker(150), 2))
                
                self.scene.addItem(segment_item)
                self.scene_items_map[part_name] = segment_item
                self.parts_list_widget.addItem(part_name)
                color_index += 1
        print(f"分割调整模式：在源图像 '{current_source_ref_name}' 上显示了 {color_index} 个部位的轮廓。")

    def confirm_segmentation_and_proceed(self):
        print("点击了 '确认分割' 按钮")
        
        new_extracted_head_part = None
        new_extracted_head_position = None
        new_extracted_body_parts = {}

        processing_error_occurred = False

        for part_name, scene_item in self.scene_items_map.items():
            if not isinstance(scene_item, EditableSegmentItem):
                # This can happen if other items (like background) are in scene_items_map
                # or if we are in a different mode and scene_items_map wasn't cleared properly.
                # For adjust_segmentation mode, we only expect EditableSegmentItem.
                if self.current_mode == "adjust_segmentation":
                    print(f"警告: scene_items_map 包含非 EditableSegmentItem 对象 '{part_name}', 跳过.")
                continue

            edited_qpolygon = scene_item.get_edited_polygon()
            if edited_qpolygon.isEmpty():
                print(f"警告: 部位 '{part_name}' 的编辑后多边形为空，跳过。")
                continue

            # 获取原始数据
            original_segment_data = self.segmentation_data_for_adjustment.get(part_name)
            if not original_segment_data:
                print(f"错误: 无法找到部位 '{part_name}' 的原始分割数据，跳过。")
                processing_error_occurred = True
                continue
            
            source_image_ref = original_segment_data.get('source_image_ref_name')
            original_cv_image = None
            if source_image_ref == 'head':
                original_cv_image = self.head_image_cv_original
            elif source_image_ref == 'body':
                original_cv_image = self.body_image_cv_original
            
            if original_cv_image is None:
                print(f"错误: 无法加载部位 '{part_name}' 的源图像 ('{source_image_ref}')，跳过。")
                processing_error_occurred = True
                continue

            original_pos_info = original_segment_data.get('original_position_info', {})
            original_keypoints_global = original_pos_info.get('keypoints', []) # Keypoints in global coords of original image

            # --- 在这里将进行裁剪和蒙版操作 --- 
            try:
                # 步骤 3a: QPolygonF 转 CV 轮廓
                cv_contour_points = []
                for i in range(edited_qpolygon.count()):
                    pt = edited_qpolygon.at(i)
                    cv_contour_points.append([int(pt.x()), int(pt.y())])
                
                if not cv_contour_points:
                    print(f"警告: 部位 '{part_name}' 转换后的CV轮廓点为空，跳过。")
                    continue
                
                cv_contour = np.array(cv_contour_points, dtype=np.int32).reshape((-1, 1, 2))

                # 步骤 3b: 创建蒙版
                img_h, img_w = original_cv_image.shape[:2]
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [cv_contour], 255)

                # 步骤 3c: 获取边界框 (基于编辑后的轮廓)
                x, y, w, h = cv2.boundingRect(cv_contour) # x,y,w,h of the edited polygon in original image space
                if w == 0 or h == 0:
                    print(f"警告: 部位 '{part_name}' 的边界框无效 (w={w}, h={h})，跳过。")
                    continue
                
                new_crop_offset_x = x
                new_crop_offset_y = y

                # 步骤 3d: 裁剪图像和蒙版
                cropped_part_cv = original_cv_image[y : y + h, x : x + w]
                cropped_mask_roi = mask[y : y + h, x : x + w]

                # 步骤 3e: 应用蒙版使背景透明
                if cropped_part_cv.shape[:2] != cropped_mask_roi.shape[:2]:
                    print(f"错误: 部位 '{part_name}' 裁剪的图像和蒙版尺寸不匹配。Img: {cropped_part_cv.shape}, Mask: {cropped_mask_roi.shape}")
                    processing_error_occurred = True
                    continue
                
                bgra_part_image = cv2.cvtColor(cropped_part_cv, cv2.COLOR_BGR2BGRA if cropped_part_cv.ndim == 3 and cropped_part_cv.shape[2] == 3 else cv2.COLOR_GRAY2BGRA if cropped_part_cv.ndim == 2 else cv2.COLOR_BGRA2BGRA)
                bgra_part_image[:, :, 3] = cropped_mask_roi

                # 步骤 3f: 更新关键点坐标
                updated_keypoints_local = []
                if original_keypoints_global:
                    for kp_global in original_keypoints_global:
                        # kp_global is [x_orig_img, y_orig_img, confidence]
                        global_x, global_y = kp_global[0], kp_global[1]
                        confidence = kp_global[2] if len(kp_global) > 2 else 1.0

                        # Check if the keypoint is within the new bounding box
                        if new_crop_offset_x <= global_x < new_crop_offset_x + w and \
                           new_crop_offset_y <= global_y < new_crop_offset_y + h:
                            local_x = global_x - new_crop_offset_x
                            local_y = global_y - new_crop_offset_y
                            updated_keypoints_local.append([local_x, local_y, confidence])
                
                # 更新位置信息字典
                new_position_info = {
                    'top': new_crop_offset_y, # Relative to original full image
                    'left': new_crop_offset_x, # Relative to original full image
                    'bottom': new_crop_offset_y + h,
                    'right': new_crop_offset_x + w,
                    'center_x': new_crop_offset_x + w // 2,
                    'center_y': new_crop_offset_y + h // 2,
                    'width': w, # Actual width of the cropped image
                    'height': h, # Actual height of the cropped image
                    'crop_offset_x': new_crop_offset_x, # This IS the new top-left in original image terms
                    'crop_offset_y': new_crop_offset_y,
                    'keypoints': updated_keypoints_local, # Keypoints are now local to the cropped image
                    'segmentation_contours': [cv_contour], # Store the edited contour (in original image coordinates)
                    'part_name': part_name
                }

                if part_name == "Head":
                    new_extracted_head_part = bgra_part_image
                    new_extracted_head_position = new_position_info
                else:
                    new_extracted_body_parts[part_name] = {
                        'image': bgra_part_image,
                        'position': new_position_info
                    }
                print(f"成功重新裁剪部位: {part_name}")
            except Exception as crop_exc:
                print(f"错误: 在为部位 '{part_name}' 应用裁剪和蒙版时发生异常: {crop_exc}")
                traceback.print_exc()
                processing_error_occurred = True
                continue # Skip to next part if one fails

        # --- 更新提取的数据 --- 
        self.extracted_head_part = new_extracted_head_part
        self.extracted_head_position = new_extracted_head_position
        self.extracted_body_parts = new_extracted_body_parts
        
        if processing_error_occurred:
            QMessageBox.warning(self, "裁剪错误", "在重新裁剪一个或多个部位时发生错误。请检查控制台输出。")
            # Decide if we should stay in adjust_segmentation mode or not
            # self.btn_confirm_segmentation.setEnabled(True) # Allow re-try?
            # return # Might be better to not proceed if errors occurred

        if self.extracted_head_part is None and not self.extracted_body_parts:
            QMessageBox.warning(self, "无数据", "确认分割后没有有效的部位数据。无法进入编辑模式。")
            self.btn_confirm_segmentation.setEnabled(True) # Allow re-try or re-process
            self.btn_edit_parts.setEnabled(False)
            return

        self.current_mode = "edit_parts"
        self.go_to_edit_parts_step() # Proceed to the actual editing step
        
        self.btn_confirm_segmentation.setEnabled(False)
        self.btn_edit_parts.setEnabled(True)
        self.btn_export.setEnabled(False) # Export is enabled after parts are on scene

    def go_to_edit_parts_step(self):
        print("点击了 '编辑部位' 按钮 / 或从确认分割进入")
        if self.extracted_head_part is None and not self.extracted_body_parts:
             QMessageBox.warning(self, "无数据", "没有有效的提取部位可以进行编辑。请先成功完成处理和分割步骤。")
             # Potentially revert to adjust_segmentation mode or allow re-processing
             self.current_mode = "adjust_segmentation" # Or check_all_files_uploaded to enable start_processing
             self.btn_edit_parts.setEnabled(False)
             self.btn_confirm_segmentation.setEnabled(True if self.segmentation_data_for_adjustment else False)
             return

        self.scene.clear() 
        self.parts_list_widget.clear()
        self.scene_items_map.clear()

        # 1. 加载模板背景
        if self.template_path:
            if self.template_cv_image_for_export is not None:
                template_pixmap = self.convert_cv_to_pixmap(self.template_cv_image_for_export)
                if not template_pixmap.isNull():
                    template_bg_item = QGraphicsPixmapItem(template_pixmap)
                    template_bg_item.setZValue(-1) # 确保在最底层
                    self.scene.addItem(template_bg_item)
                    self.view.setSceneRect(template_pixmap.rect().toRectF())
                    self.view.fitInView(template_pixmap.rect().toRectF(), Qt.AspectRatioMode.KeepAspectRatio)
                else:
                    QMessageBox.warning(self, "错误", "无法将模板图像转换为Pixmap用于编辑场景背景。")
            else:
                QMessageBox.warning(self, "错误", "未加载模板CV图像，无法设置编辑场景背景。")
        else:
            QMessageBox.warning(self, "无模板", "未找到模板图像路径，无法设置编辑场景背景。")
        
        # 2. 加载提取的头部
        if self.extracted_head_part is not None and self.extracted_head_position is not None:
            head_pixmap = self.convert_cv_to_pixmap(self.extracted_head_part)
            if not head_pixmap.isNull():
                head_item = InteractivePartItem(
                    pixmap=head_pixmap,
                    original_cv_image=self.extracted_head_part,
                    original_keypoints=self.extracted_head_position.get('keypoints', []),
                    crop_offset_x=self.extracted_head_position.get('crop_offset_x', 0),
                    crop_offset_y=self.extracted_head_position.get('crop_offset_y', 0),
                    part_name="Head"
                )
                try:
                    if (self.merger and self.merger.position_calculator and 
                        self.template_landmarks and self.head_landmarks):
                        # 调用 CharacterAutoMerger.py 中实际存在的方法
                        # 注意：CharacterAutoMerger.py 的 PositionCalculator.calculate_placement_position 
                        # 目前只返回 x, y，不包含 scale 和 rotation。
                        placement_info = self.merger.position_calculator.calculate_placement_position(
                            source_position=self.extracted_head_position, 
                            target_landmarks=self.template_landmarks, 
                            part_name="Head"
                        )
                        if placement_info:
                            head_item.setPos(placement_info.get('x', 0), placement_info.get('y', 0))
                            # 由于 CharacterAutoMerger.py 的 PositionCalculator 不提供 scale/rotation, 我们先使用默认值
                            head_item.setItemScale(1.0, 1.0) # 默认缩放
                            head_item.setItemRotation(0.0)  # 默认旋转
                    else:
                        print("警告: PositionCalculator 或相关 landmarks 未初始化，无法自动定位头部。")
                        head_item.setItemScale(1.0, 1.0)
                        head_item.setItemRotation(0.0)
                except Exception as e:
                    print(f"计算头部位置或应用变换时出错: {e}")
                    # 出错也设置默认值
                    head_item.setItemScale(1.0, 1.0)
                    head_item.setItemRotation(0.0)
                
                self.scene.addItem(head_item)
                self.scene_items_map["Head"] = head_item
                self.parts_list_widget.addItem("Head")
            else:
                print("警告: 无法为头部创建Pixmap")

        # 3. 加载提取的身体部位
        if self.extracted_body_parts:
            for part_name, part_data in self.extracted_body_parts.items():
                if part_data and part_data.get('image') is not None and part_data.get('position') is not None:
                    part_cv_image = part_data['image']
                    part_position_info = part_data['position']
                    part_pixmap = self.convert_cv_to_pixmap(part_cv_image)

                    if not part_pixmap.isNull():
                        body_part_item = InteractivePartItem(
                            pixmap=part_pixmap,
                            original_cv_image=part_cv_image,
                            original_keypoints=part_position_info.get('keypoints', []),
                            crop_offset_x=part_position_info.get('crop_offset_x', 0),
                            crop_offset_y=part_position_info.get('crop_offset_y', 0),
                            part_name=part_name
                        )
                        try:
                            if (self.merger and self.merger.position_calculator and 
                                self.template_landmarks and self.body_landmarks):
                                # 调用 CharacterAutoMerger.py 中实际存在的方法
                                # 注意：CharacterAutoMerger.py 的 PositionCalculator.calculate_placement_position 
                                # 目前只返回 x, y，不包含 scale 和 rotation。
                                placement_info = self.merger.position_calculator.calculate_placement_position(
                                    source_position=part_position_info, 
                                    target_landmarks=self.template_landmarks, 
                                    part_name=part_name
                                )
                                if placement_info:
                                    body_part_item.setPos(placement_info.get('x', 0), placement_info.get('y', 0))
                                    # 由于 CharacterAutoMerger.py 的 PositionCalculator 不提供 scale/rotation, 我们先使用默认值
                                    body_part_item.setItemScale(1.0, 1.0) # 默认缩放
                                    body_part_item.setItemRotation(0.0)  # 默认旋转
                            else:
                                print(f"警告: PositionCalculator 或相关 landmarks 未初始化，无法自动定位部位 '{part_name}'。")
                                body_part_item.setItemScale(1.0, 1.0)
                                body_part_item.setItemRotation(0.0)
                        except Exception as e:
                            print(f"计算部位 '{part_name}' 位置或应用变换时出错: {e}")
                            body_part_item.setItemScale(1.0, 1.0)
                            body_part_item.setItemRotation(0.0)

                        self.scene.addItem(body_part_item)
                        self.scene_items_map[part_name] = body_part_item
                        self.parts_list_widget.addItem(part_name)
                    else:
                        print(f"警告: 无法为部位 '{part_name}' 创建Pixmap")
                else:
                    print(f"警告: 部位 '{part_name}' 数据不完整，跳过加载。")

        self.btn_export.setEnabled(True)
        self.btn_edit_parts.setEnabled(False) # Done with this step, disable until re-triggered by mode change
        # self.btn_start_processing.setEnabled(False) # Should be handled by its own logic
        # self.btn_confirm_segmentation.setEnabled(False) # Done with this step
        print("进入可交互编辑模式，模板和提取的部位已加载。")

    def export_results(self):
        print("点击了 '导出结果' 按钮")
        # TODO: 实现导出逻辑
        if self.scene.itemsBoundingRect().isEmpty():
            QMessageBox.warning(self, "导出错误", "场景为空，无法导出。")
            return

        # 1. 获取保存路径
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            os.path.join(DEFAULT_EXPORT_DIR, "exported_character.png"), # 默认文件名和路径
            "PNG 文件 (*.png);;JPEG 文件 (*.jpg *.jpeg);;所有文件 (*)"
        )

        if not file_path:
            print("导出操作已取消。")
            return

        # 2. 创建 QImage
        # 使用 sceneRect() 来获取预设的场景大小，itemsBoundingRect() 可能只包含物体实际占据的区域
        scene_rect = self.scene.sceneRect() 
        if scene_rect.isEmpty() and self.template_cv_image_for_export is not None:
            # 如果 sceneRect 未设置（例如，在仅显示模板后），则基于模板图像大小
            h, w, _ = self.template_cv_image_for_export.shape
            target_rect = QRectF(0, 0, w, h)
        elif not scene_rect.isEmpty():
            target_rect = scene_rect
        else:
            # 如果两者都不可用，则使用物体边界
            target_rect = self.scene.itemsBoundingRect()
            if target_rect.isEmpty(): # 再次检查，防止 itemsBoundingRect 也为空
                 QMessageBox.warning(self, "导出错误", "无法确定导出图像的尺寸。")
                 return
        
        # 确保尺寸是整数
        image_size = QSize(int(target_rect.width()), int(target_rect.height()))
        if image_size.width() <= 0 or image_size.height() <= 0:
            QMessageBox.warning(self, "导出错误", f"无效的导出图像尺寸: {image_size.width()}x{image_size.height()}")
            return
            
        export_image = QImage(image_size, QImage.Format.Format_ARGB32_Premultiplied)
        export_image.fill(Qt.GlobalColor.transparent) # 默认透明背景

        # 3. 创建 QPainter
        painter = QPainter(export_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 4. 渲染场景
        # render 方法的 target 参数是 QRectF，表示源场景中要渲染的区域
        # 如果要渲染整个场景到图像，则 source rect 就是 scene.sceneRect() 或我们计算的 target_rect
        # painter 会将这个源区域绘制到整个 QImage 上 (QImage 的 rect)
        try:
            print(f"开始渲染场景到图像。场景区域: {target_rect}, 图像尺寸: {image_size}")
            self.scene.render(painter, QRectF(export_image.rect()), target_rect)
            print("场景渲染完成。")
        except Exception as e:
            QMessageBox.critical(self, "渲染错误", f"渲染场景到图像时发生错误: {e}")
            painter.end() # 确保 painter 结束
            return
        finally:
            painter.end() # 确保 painter 结束

        # 5. 保存 QImage
        if export_image.save(file_path):
            QMessageBox.information(self, "导出成功", f"结果已成功导出到:\n{file_path}")
            print(f"图像已保存到: {file_path}")
        else:
            QMessageBox.critical(self, "导出失败", f"无法保存图像到:\n{file_path}")
            print(f"保存图像失败: {file_path}")

        self.reset_extracted_data()
        self.template_cv_image_for_export = None # Also reset this as it depends on upload
        self.head_image_cv_original = None
        self.body_image_cv_original = None
        self.segmentation_data_for_adjustment.clear()
        self.current_mode = "upload"
        # Reset button states to initial upload phase
        self.check_all_files_uploaded() # This will typically disable start_processing
        self.btn_upload_template.setEnabled(self.merger is not None)
        self.btn_upload_head.setEnabled(self.merger is not None)
        self.btn_upload_body.setEnabled(self.merger is not None)

    def reset_extracted_data(self):
        """清除处理后的数据"""
        self.template_landmarks = None
        self.head_landmarks = None
        self.body_landmarks = None
        self.extracted_head_part = None
        self.extracted_head_position = None
        self.extracted_body_parts = None
        self.scene_items_map.clear()
        self.parts_list_widget.clear()
        self.template_cv_image_for_export = None

    def set_ui_state_processing(self, processing: bool):
        """根据处理状态设置UI控件的启用/禁用"""
        # More granular control is now in start_processing's finally block and confirm_segmentation
        # This can be simplified or removed if specific state changes handle all cases.
        # For now, keep it for the wait cursor part.
        # self.btn_start_processing.setEnabled(not processing and self.merger is not None and self.template_path is not None and self.head_path is not None and self.body_path is not None)
        # self.btn_confirm_segmentation.setEnabled(False) 
        # self.btn_edit_parts.setEnabled(False) 
        # self.btn_export.setEnabled(False) 

        upload_buttons_enabled = not processing and (self.merger is not None)
        self.btn_upload_template.setEnabled(upload_buttons_enabled)
        self.btn_upload_head.setEnabled(upload_buttons_enabled)
        self.btn_upload_body.setEnabled(upload_buttons_enabled)

        # Start processing is enabled based on file uploads and not processing
        if not processing:
            self.check_all_files_uploaded() # This will set btn_start_processing
        else:
            self.btn_start_processing.setEnabled(False)
        
        # Other buttons are managed by specific workflow steps
        if not processing:
            if self.current_mode == "adjust_segmentation":
                self.btn_confirm_segmentation.setEnabled(True)
                self.btn_edit_parts.setEnabled(False)
            elif self.current_mode == "edit_parts":
                self.btn_confirm_segmentation.setEnabled(False)
                self.btn_edit_parts.setEnabled(True) # Or false if it auto-disables after entering
                self.btn_export.setEnabled(True if self.scene.items() else False)
            else: # upload mode or after export/reset
                self.btn_confirm_segmentation.setEnabled(False)
                self.btn_edit_parts.setEnabled(False)
                self.btn_export.setEnabled(False)

        if processing:
             QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
             print("UI已禁用，正在处理...")
        else:
             QApplication.restoreOverrideCursor()
             print("UI已恢复。")

    def convert_cv_to_pixmap(self, cv_img: np.ndarray) -> QPixmap:
        """将 OpenCV 图像 (numpy array) 转换为 QPixmap."""
        if cv_img is None: return QPixmap() # 返回空 QPixmap
        
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width

        if channels == 4: # BGRA (通常OpenCV读取带alpha的PNG为此格式)
            # QImage 需要 RGBA 格式
            rgba_image = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
            q_image = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        elif channels == 3: # BGR
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            print(f"警告: 不支持的图像通道数 {channels} in convert_cv_to_pixmap")
            return QPixmap()

        if q_image.isNull():
            print("警告: convert_cv_to_pixmap 中的 QImage 为空")
            return QPixmap()
        pixmap = QPixmap.fromImage(q_image)
        if pixmap.isNull():
            print("警告: convert_cv_to_pixmap 中的 QPixmap 为空")
        return pixmap

    def display_template_in_view(self):
        if not self.template_path or self.template_cv_image_for_export is None: return
        pixmap = self.convert_cv_to_pixmap(self.template_cv_image_for_export)
        if pixmap.isNull(): return
        self.scene.clear(); self.parts_list_widget.clear(); self.scene_items_map.clear()
        bg = self.scene.addPixmap(pixmap); bg.setZValue(-10)
        self.view.setSceneRect(pixmap.rect().toRectF()); self.view.fitInView(pixmap.rect().toRectF(), Qt.AspectRatioMode.KeepAspectRatio)

    def on_part_selected_in_list(self, current_qlist_item: QListWidgetItem, previous_qlist_item: QListWidgetItem):
        if not current_qlist_item: return
        part_name = current_qlist_item.text()
        scene_item = self.scene_items_map.get(part_name)

        if scene_item:
            # For EditableSegmentItem, we might want to bring it to front or highlight it
            if self.current_mode == "adjust_segmentation":
                # scene_item.setSelected(True) # This might not be needed if list drives selection
                # Deselect all other items in scene manually if not using SingleSelection
                for item in self.scene.items():
                    if isinstance(item, QGraphicsPolygonItem): # Or EditableSegmentItem
                        item.setSelected(item == scene_item)
                if isinstance(scene_item, QGraphicsPolygonItem):
                    scene_item.setZValue(1) # Bring to front for better visibility
                    # Reset Z-value of previously selected item if any
                    if previous_qlist_item:
                        prev_scene_item = self.scene_items_map.get(previous_qlist_item.text())
                        if prev_scene_item and prev_scene_item != scene_item:
                            prev_scene_item.setZValue(0)
            elif self.current_mode == "edit_parts" and isinstance(scene_item, InteractivePartItem):
                self.scene.clearSelection()
                scene_item.setSelected(True)
            
            self.view.ensureVisible(scene_item)
            print(f"List selected: '{part_name}'. Mode: {self.current_mode}")

    def on_scene_selection_changed(self):
        selected_scene_items = self.scene.selectedItems()
        self.parts_list_widget.selectionModel().clear() # 清除列表选择
        if selected_scene_items:
            # 假设一次只选择一个 (QGraphicsView.SingleSelection)
            # 如果允许多选，逻辑会更复杂
            selected_part_item = selected_scene_items[0]
            if isinstance(selected_part_item, InteractivePartItem):
                part_name = selected_part_item.part_name
                # 查找列表中的对应项并选中它
                list_items = self.parts_list_widget.findItems(part_name, Qt.MatchFlag.MatchExactly)
                if list_items:
                    self.parts_list_widget.blockSignals(True)
                    self.parts_list_widget.setCurrentItem(list_items[0])
                    self.parts_list_widget.blockSignals(False)
                    print(f"Scene selected (InteractivePartItem): '{part_name}'. List item: {list_items[0].text()}")
            elif isinstance(selected_part_item, QGraphicsPolygonItem): # Later EditableSegmentItem
                # Find part_name associated with this polygon item
                found_part_name = None
                for name, item in self.scene_items_map.items():
                    if item == selected_part_item:
                        found_part_name = name
                        break
                if found_part_name:
                    list_items = self.parts_list_widget.findItems(found_part_name, Qt.MatchFlag.MatchExactly)
                    if list_items:
                        self.parts_list_widget.blockSignals(True)
                        self.parts_list_widget.setCurrentItem(list_items[0])
                        self.parts_list_widget.blockSignals(False)
                        print(f"Scene selected (Polygon): '{found_part_name}'. List item: {list_items[0].text()}")

        else:
            # 没有场景项被选中，列表也清空选择
            # self.parts_list_widget.clearSelection() # Cleared at the beginning
            print("Scene selection cleared.")

# TODO for CharacterAutoMerger GUI (未来增强功能):
#
# 1. 高级变形与交互:
#    - 真正的自由变换 (通过拖动角点进行扭曲/透视):
#      - 需要一个自定义的 QGraphicsItem，可以通过独立操控其角点来进行变形。
#      - 这可能涉及到管理一个单应性变换 (3x3 矩阵)。
#      - 该图元需要处理其自身纹理到变形后四边形的映射。
#      - QGraphicsPixmapItem 可能不够用; 可能需要 QGraphicsObject 并自定义绘制和形状。
#    - 网格/弯曲变形 (类似 Photoshop 的弯曲工具):
#      - 涉及覆盖一个网格，并允许操控网格交点或贝塞尔手柄。
#      - 非常复杂，很可能需要专门的二维网格变形库或复杂的自定义渲染。
#    - 缩放/旋转时的约束键:
#      - Shift + 拖动角点: 在进行角点缩放时约束宽高比 (即使角点通常允许非均匀缩放，也可以实现此功能)。
#      - Shift + 拖动旋转手柄: 将旋转吸附到15或45度的增量。
#      - 数值变换输入: 允许用户为选定的部位输入精确的 X/Y 位置、缩放 (X/Y) 和旋转值。
#      - 中心轴点操控: 允许用户可视化地移动一个部位的 transformOriginPoint。
#
# 2. UI/UX 增强:
#    - 图层顺序控制:
#      - 允许在右侧的 QListWidget 中重新排序部位 (例如，拖放或上/下按钮)。
#      - 这个顺序应影响 QGraphicsScene 中图元的 Z 值以及最终导出时的绘制顺序。
#    - 可见性切换: 在 QListWidget 中每个部位旁边添加复选框，以在 QGraphicsScene 中显示/隐藏它们 (如果隐藏则从导出中排除)。
#    - 透明度滑块: (可能在右侧面板) 用于动态更改所选 InteractivePartItem 的透明度。
#    - QGraphicsView 的缩放/平移工具:
#      - 更明确的工具栏按钮，用于放大、缩小、适应视图、平移工具 (如果鼠标滚轮/中键对所有用户来说不够直观)。
#      - 显示当前缩放级别。
#    - 状态栏: 显示信息，如选定的工具/模式、当前鼠标坐标 (场景/视图)、选定部位的尺寸/旋转。
#    - 撤销/重做堆栈 (QAction, QUndoStack):
#      - 用于变换 (移动、缩放、旋转)、图元添加/删除、透明度更改等操作。
#      - 这是一个重要的功能，需要仔细实现命令模式。
#    - 保存/加载项目状态:
#      - 将所有相关信息保存到一个自定义项目文件 (例如 JSON 或 XML): 源图像路径、每个部位的当前变换、图层顺序等。
#      - 允许重新加载此项目文件以继续编辑。
#    - 改进的错误处理和用户反馈:
#      - 更具体的错误消息和指引。
#      - 对可能耗时较长的操作提供进度指示器或忙碌光标 (已部分通过 WaitCursor 实现)。
#    - 最终导出预览: 在保存到磁盘之前，快速渲染一个不透明的最终合成图 (可能在对话框中显示)。
#    - 可自定义的控制柄颜色/大小: 用户对交互元素的偏好设置。
#    - 右键上下文菜单: 在场景中的部位或列表中的部位上右击，可执行快速操作 (删除、复制、设置透明度、置于顶层/底层)。
#
# 3. 导出选项:
#    - 最终合成图的背景选择: 透明、纯色选择器，或使用原始模板。
#    - 控制导出分辨率/尺寸: 针对场景渲染 (预览) 以及最终合成图 (如果尺寸不同于模板)。
#    - PSD 导出 (高级):
#      - 重建一个分层的 PSD 文件。每个 InteractivePartItem 成为一个图层。
#      - 如果 PSD 库支持，变换需要被烘焙或作为智能对象保留。
#      - 这非常复杂，需要一个强大的第三方 PSD 库 (例如，用于读取的 psd-tools，可能还需要其他库来写入复杂的 PSD)。
#    - 导出单独变换后的图层: 选项，用于将每个部位保存为单独的 PNG 文件，应用其最终变换并具有透明背景。
#
# 4. 性能:
#    - 图像处理的线程化:
#      - 将 `start_processing` (姿态检测、通过 CharacterAutoMerger 进行的部位提取) 移至一个单独的 QThread。
#      - 这将防止在这些可能耗时的操作期间 GUI 卡顿。
#      - 使用信号/槽将进度和结果传递回主 GUI 线程。
#    - QGraphicsScene 优化: 如果场景中有许多图元或非常大的图像导致性能下降 (例如，使用 QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)。
#    - 缓存: 如果图像在没有更改的情况下被频繁重新加载/重绘，则缓存 QPixmap 的转换结果。
#
# 5. 代码优化与健壮性:
#    -进一步模块化: 如果 `InteractivePartItem` 或 `MainWindow` 变得过于庞大，则进行拆分。
#    - 更严格的错误检查: 特别是围绕文件 I/O、图像处理和外部库调用。
#    - 全面的文档字符串和注释: 针对所有类和复杂方法。
#    - 单元测试: 针对关键逻辑 (例如，变换计算、如果部位提取被分离出来)。
#    - 依赖管理:清晰的 requirements.txt 或类似文件，用于设置环境。
#
# 6. 特定功能增强:
#    - 关键点编辑: 如果姿态检测不完美，允许用户手动添加、删除或移动图像上的关键点。
#      然后，这些编辑过的关键点可以反馈给 `BodyPartExtractor` 或 `PositionCalculator`。
#    - 对称/镜像工具: 针对手臂/腿等部位，如果只提供了一侧。
#    - 颜色调整: 针对单个部位的基本亮度、对比度、色彩平衡调整。
#
# 优先级和实现顺序将取决于用户需求和开发资源。


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    if window.merger is None: sys.exit(1)
    window.show()
    sys.exit(app.exec()) 