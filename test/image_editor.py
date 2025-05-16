import os
import math
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.image import Image
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.window import Window
from kivy.properties import BooleanProperty, ObjectProperty, NumericProperty, StringProperty
from kivy.metrics import dp
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.lang import Builder

try:
    from plyer import filechooser
except ImportError:
    print("Warning: plyer not installed, file chooser will not be available")
    filechooser = None

class ImageLayerWidget(ScatterLayout):
    """表示一个可交互的图像图层，支持移动、旋转和形变"""
    
    source = StringProperty(None)
    is_selected = BooleanProperty(False)
    visible = BooleanProperty(True)
    
    def __init__(self, **kwargs):
        super(ImageLayerWidget, self).__init__(**kwargs)
        self._drag_mode = None
        self._initial_touch_data = {}
        self.do_rotation = False  # 禁用ScatterLayout默认的旋转
        self.do_scale = False     # 禁用ScatterLayout默认的缩放
        self.do_translation = True  # 启用移动功能
        self.auto_bring_to_front = True  # 确保拖动时图层位于最上层
        self.image = None
        self.handle_size = dp(30)  # 进一步增大控制点大小，更容易点击
        self.bind(is_selected=self.update_canvas)
        self.bind(visible=self.on_visibility_changed)
        print("初始化ImageLayerWidget: do_translation=", self.do_translation)
        
    def load_image(self, source):
        """加载图像并设置初始大小"""
        self.clear_widgets()
        self.source = source
        
        # 使用新的推荐属性而不是弃用的属性
        if hasattr(Image, 'fit_mode'):  # 检查新版本Kivy是否支持fit_mode
            self.image = Image(source=source, fit_mode='contain')
        else:
            # 向后兼容旧版本
            self.image = Image(source=source, allow_stretch=True, keep_ratio=True)
        
        # 确保图像已加载纹理
        if self.image.texture:
            self.size = self.image.texture_size
            
        self.add_widget(self.image)
        self.image.size = self.size
        self.update_canvas()
        
    def _on_texture_loaded(self, instance, texture):
        """当图像纹理加载完成时调用"""
        if texture:
            print(f"纹理延迟加载完成，大小: {texture.size}")
            self.size = texture.size
            self.image.size = self.size
            self.update_canvas()
        
    def update_canvas(self, *args):
        """根据选中状态更新控制点和边界框的绘制"""
        self.canvas.after.clear()
        if not self.is_selected or not self.visible:
            return
            
        with self.canvas.after:
            # 确保图像已加载并且有纹理
            if self.image and self.image.texture:
                # 获取图像的实际显示尺寸和位置
                # 直接使用图像纹理的尺寸，而不是图像或图层尺寸
                texture_size = self.image.texture.size
                
                # 根据图像的fit_mode或keep_ratio属性，计算实际显示尺寸
                if hasattr(self.image, 'fit_mode') and self.image.fit_mode == 'contain':
                    # 新版本Kivy使用fit_mode
                    img_width, img_height = self._calculate_fitted_size(texture_size, self.image.size)
                elif hasattr(self.image, 'keep_ratio') and self.image.keep_ratio:
                    # 旧版本Kivy使用keep_ratio
                    img_width, img_height = self._calculate_fitted_size(texture_size, self.image.size)
                else:
                    # 如果没有使用比例保持，则使用当前图像尺寸
                    img_width, img_height = self.image.size
                
                # print(f"图像纹理尺寸: {texture_size}, 当前显示尺寸: {(img_width, img_height)}")
                
                # 计算图像在图层内的中心偏移量
                img_x = (self.width - img_width) / 2
                img_y = (self.height - img_height) / 2
                
                # 绘制边界框 - 使用绿色，贴合图像实际尺寸
                Color(0, 1, 0, 1)
                Line(rectangle=(img_x, img_y, img_width, img_height), width=1.5)
                
                # 绘制中心旋转点 - 使用红色
                Color(1, 0, 0, 1)
                center_x, center_y = img_x + img_width / 2, img_y + img_height / 2
                handle_radius = self.handle_size / 2
                Ellipse(
                    pos=(center_x - handle_radius, center_y - handle_radius), 
                    size=(self.handle_size, self.handle_size)
                )
                
                # 绘制四个角的形变点 - 使用蓝色
                Color(0, 0, 1, 1)
                
                # 绘制左下角控制点
                Ellipse(
                    pos=(img_x - handle_radius, img_y - handle_radius), 
                    size=(self.handle_size, self.handle_size)
                )
                
                # 绘制右下角控制点
                Ellipse(
                    pos=(img_x + img_width - handle_radius, img_y - handle_radius), 
                    size=(self.handle_size, self.handle_size)
                )
                
                # 绘制左上角控制点
                Ellipse(
                    pos=(img_x - handle_radius, img_y + img_height - handle_radius), 
                    size=(self.handle_size, self.handle_size)
                )
                
                # 绘制右上角控制点
                Ellipse(
                    pos=(img_x + img_width - handle_radius, img_y + img_height - handle_radius), 
                    size=(self.handle_size, self.handle_size)
                )
                
                # 添加文字标签标识每个控制点
                Color(1, 1, 1, 1)  # 白色文字
                from kivy.core.text import Label as CoreLabel
                for point, label in [
                    ((img_x, img_y), 'BL'), 
                    ((img_x + img_width, img_y), 'BR'),
                    ((img_x, img_y + img_height), 'TL'),
                    ((img_x + img_width, img_y + img_height), 'TR')
                ]:
                    text = CoreLabel(text=label, font_size=14)
                    text.refresh()
                    texture = text.texture
                    Rectangle(
                        pos=(point[0] - texture.width / 2, point[1] - texture.height / 2),
                        size=texture.size, 
                        texture=texture
                    )
    
    def _calculate_fitted_size(self, texture_size, container_size):
        """计算保持宽高比的图像实际显示尺寸"""
        if not texture_size or not container_size:
            return container_size
            
        tex_width, tex_height = texture_size
        cont_width, cont_height = container_size
        
        # 如果纹理尺寸为零，避免除零错误
        if tex_width == 0 or tex_height == 0:
            return container_size
            
        # 计算宽高比
        tex_ratio = tex_width / float(tex_height)
        cont_ratio = cont_width / float(cont_height)
        
        # 根据宽高比决定如何适配
        if tex_ratio > cont_ratio:  # 纹理更宽
            return cont_width, cont_width / tex_ratio
        else:  # 纹理更高
            return cont_height * tex_ratio, cont_height
    
    def on_visibility_changed(self, instance, value):
        """处理可见性改变"""
        self.opacity = 1.0 if value else 0.0
    
    def on_size(self, *args):
        """当大小改变时更新控制点"""
        if self.image:
            self.image.size = self.size
        self.update_canvas()
    
    def on_touch_down(self, touch):
        if not self.visible:
            return False
            
        if self.collide_point(*touch.pos):
            # 如果图层被选中，检查是否点击控制点
            if self.is_selected:
                # 转换触摸位置到局部坐标
                local_touch = self.to_local(*touch.pos)
                print(f"触摸点击: 全局={touch.pos}, 局部={local_touch}, 图层位置={self.pos}, 图层大小={self.size}")
                
                # 获取图像的实际尺寸和位置，与update_canvas方法一致
                img_x = img_y = 0
                img_width = img_height = 0
                
                if self.image and self.image.texture:
                    # 获取图像纹理尺寸
                    texture_size = self.image.texture.size
                    
                    # 使用相同的方法计算图像实际显示尺寸
                    if hasattr(self.image, 'fit_mode') and self.image.fit_mode == 'contain':
                        img_width, img_height = self._calculate_fitted_size(texture_size, self.image.size)
                    elif hasattr(self.image, 'keep_ratio') and self.image.keep_ratio:
                        img_width, img_height = self._calculate_fitted_size(texture_size, self.image.size)
                    else:
                        img_width, img_height = self.image.size
                    
                    # 计算图像在图层内的中心偏移量
                    img_x = (self.width - img_width) / 2
                    img_y = (self.height - img_height) / 2
                
                # 检查是否触摸旋转点
                center_x, center_y = img_x + img_width / 2, img_y + img_height / 2
                rotation_handle_dist = math.sqrt(
                    (local_touch[0] - center_x) ** 2 + 
                    (local_touch[1] - center_y) ** 2
                )
                
                if rotation_handle_dist <= self.handle_size:
                    self._drag_mode = 'rotate'
                    self._initial_touch_data = {
                        'center': self.center,
                        'rotation': self.rotation,
                        'touch_pos': touch.pos
                    }
                    print(f"检测到旋转操作: 中心点距离={rotation_handle_dist}, 开始旋转")
                    touch.grab(self)
                    return True
            
            # 如果未检测到控制点操作，则执行移动操作
            print("未检测到控制点操作，准备执行移动")
            touch.grab(self)
            return True
        
        return super(ImageLayerWidget, self).on_touch_down(touch)
        
    def on_touch_move(self, touch):
        if touch.grab_current is self:
            # 减少日志输出，只在拖动开始时输出一次
            if not hasattr(self, '_logging_started') or not self._logging_started:
                print(f"拖动开始: 模式={self._drag_mode}, 初始位置={touch.pos}")
                self._logging_started = True
            
            if self._drag_mode == 'rotate':
                # 处理旋转
                cx, cy = self.center
                px, py = touch.pos
                
                # 计算从中心点到触摸点的角度
                angle = math.degrees(math.atan2(py - cy, px - cx))
                
                # 初始角度
                initial_angle = math.degrees(math.atan2(
                    self._initial_touch_data['touch_pos'][1] - self._initial_touch_data['center'][1],
                    self._initial_touch_data['touch_pos'][0] - self._initial_touch_data['center'][0]
                ))
                
                # 应用旋转变化
                new_rotation = self._initial_touch_data['rotation'] + angle - initial_angle
                # 减少日志输出
                # print(f"旋转: 初始角度={initial_angle:.2f}, 当前角度={angle:.2f}, 新旋转值={new_rotation:.2f}")
                self.rotation = new_rotation
                return True
            
            # 如果没有特定的拖拽模式，执行默认的移动
            if self._drag_mode is None:
                # 直接移动图层
                self.pos = (self.x + touch.dx, self.y + touch.dy)
                # 减少日志输出
                # print(f"默认移动: dx={touch.dx}, dy={touch.dy}, 新位置={self.pos}")
                return True
        
        return super(ImageLayerWidget, self).on_touch_move(touch)
    
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            # 仅在拖动结束时输出最终信息
            if hasattr(self, '_logging_started') and self._logging_started:
                print(f"拖动结束: 模式={self._drag_mode}, 最终位置={touch.pos}, 图层大小={self.size}, 图层位置={self.pos}")
                self._logging_started = False
            
            touch.ungrab(self)
            self._drag_mode = None
            self._initial_touch_data = {}
            return True
        
        return super(ImageLayerWidget, self).on_touch_up(touch)

    def transform_with_touch(self, touch):
        """重写ScatterLayout的transform_with_touch方法，添加日志来跟踪移动行为"""
        # 减少日志输出
        # print(f"transform_with_touch被调用: x={touch.x}, y={touch.y}, dx={touch.dx}, dy={touch.dy}")
        
        # 如果已经被控制点处理，则不执行移动
        if self._drag_mode is not None:
            # 减少日志输出
            # print("transform_with_touch: 由于有激活的拖拽模式，跳过")
            return False
            
        # 确保只处理已经被on_touch_down判定为应当移动的触摸
        if touch.grab_current is self:
            changed = False
            
            # 移动图层 - 直接修改pos属性而不是应用变换矩阵
            new_x = self.x + touch.dx
            new_y = self.y + touch.dy
            self.pos = (new_x, new_y)
            # 减少日志输出
            # print(f"直接移动图层: dx={touch.dx}, dy={touch.dy}, 新位置={self.pos}")
            changed = True
                
            return changed
        
        return False

    def transform_size(self, base_width, base_height, scale_x=None, scale_y=None, lock_aspect_ratio=False):
        """统一处理图层形变的函数
        
        参数:
            base_width: X轴的基准宽度
            base_height: Y轴的基准高度
            scale_x: X轴的缩放比例，None表示不改变
            scale_y: Y轴的缩放比例，None表示不改变
            lock_aspect_ratio: 是否锁定宽高比
        
        返回:
            tuple: (new_width, new_height) 形变后的新尺寸
            bool: 操作是否成功
        """
        if base_width <= 0 or base_height <= 0:
            print("无法进行形变：基准尺寸无效")
            return None, False
            
        old_width, old_height = self.size
        old_center_x = self.x + old_width / 2
        old_center_y = self.y + old_height / 2
        
        # 计算新尺寸
        new_width = old_width
        new_height = old_height
        
        # 如果提供了X缩放比例
        if scale_x is not None:
            new_width = base_width * scale_x
            
            # 检查是否需要保持宽高比
            if lock_aspect_ratio and base_height > 0:
                aspect_ratio = base_width / base_height
                if aspect_ratio > 0:  # 避免除以零
                    new_height = new_width / aspect_ratio
        
        # 如果提供了Y缩放比例
        if scale_y is not None:
            new_height = base_height * scale_y
            
            # 检查是否需要保持宽高比
            if lock_aspect_ratio and base_width > 0:
                aspect_ratio = base_width / base_height
                if aspect_ratio > 0:  # 避免除以零
                    new_width = new_height * aspect_ratio
        
        # 确保尺寸不小于最小值
        min_dimension = 10
        new_width = max(new_width, min_dimension)
        new_height = max(new_height, min_dimension)
        
        # 应用新尺寸，保持图层中心点不变
        self.size = (new_width, new_height)
        self.pos = (old_center_x - new_width / 2, old_center_y - new_height / 2)
        
        print(f"应用新尺寸: 宽度={new_width}, 高度={new_height}, 位置={self.pos}")
        return (new_width, new_height), True


class EditorCanvasWidget(RelativeLayout):
    """编辑器画布，作为所有图层的容器"""
    
    selected_layer = ObjectProperty(None, allownone=True)
    # 定义固定的画布大小
    CANVAS_WIDTH = 1920
    CANVAS_HEIGHT = 1080
    
    def __init__(self, **kwargs):
        super(EditorCanvasWidget, self).__init__(**kwargs)
        self.layers = []
        self._skip_next_resize = False  # 添加标记，用于协调窗口大小变化时的更新
        
        # 添加画布边界视觉提示
        with self.canvas.before:
            Color(0.1, 0.1, 0.1, 1)  # 画布背景色
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
            
            # 添加画布边界指示
            Color(0.3, 0.3, 0.3, 1)  # 边界颜色
            self._border_rect = Rectangle(
                pos=(0, 0), 
                size=(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
            )
        
        # 绑定大小变化事件，以便在调整窗口大小时更新画布边界
        self.bind(size=self._update_canvas_display, pos=self._update_canvas_display)
    
    def _update_canvas_display(self, *args):
        """更新画布显示，确保边界矩形始终可见，并相应更新所有图层位置"""
        if hasattr(self, '_bg_rect'):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size
        
        if hasattr(self, '_border_rect'):
            # 记录旧的画布位置和大小
            old_pos = getattr(self, '_last_border_pos', (0, 0))
            old_size = getattr(self, '_last_border_size', (0, 0))
            
            # 计算缩放比例
            scale_x = self.width / self.CANVAS_WIDTH
            scale_y = self.height / self.CANVAS_HEIGHT
            scale = min(scale_x, scale_y)
            
            # 计算新的画布尺寸和位置
            display_width = self.CANVAS_WIDTH * scale
            display_height = self.CANVAS_HEIGHT * scale
            offset_x = (self.width - display_width) / 2
            offset_y = (self.height - display_height) / 2
            
            # 更新画布边界显示
            self._border_rect.pos = (offset_x, offset_y)
            self._border_rect.size = (display_width, display_height)
            
            # 记录新的位置和大小
            self._last_border_pos = (offset_x, offset_y)
            self._last_border_size = (display_width, display_height)
            
            # 计算位置偏移量
            dx = offset_x - old_pos[0]
            dy = offset_y - old_pos[1]
            
            # 更新所有图层的位置，保持相对位置不变
            if dx != 0 or dy != 0:
                for layer in self.layers:
                    layer.pos = (layer.x + dx, layer.y + dy)
    
    def add_image_as_layer(self, image_path):
        """添加新图层并加载图像"""
        try:
            print(f"尝试加载图像: {image_path}")
            layer = ImageLayerWidget()
            layer.load_image(image_path)
            
            # 适配固定画布大小，而不是窗口大小
            self._scale_image_to_fit_canvas(layer)
            
            # 计算居中位置，确保图层位于画布中央
            # 注意：这里需要考虑画布可能被缩放显示的情况
            if hasattr(self, '_border_rect'):
                # 获取当前显示的画布边界
                canvas_x, canvas_y = self._border_rect.pos
                canvas_width, canvas_height = self._border_rect.size
                
                # 保证图层完全在画布内
                max_x = canvas_x + canvas_width - layer.width
                max_y = canvas_y + canvas_height - layer.height
                
                # 计算图层在画布中居中的位置，确保不会超出边界
                layer_x = max(canvas_x, min(max_x, canvas_x + (canvas_width - layer.width) / 2))
                layer_y = max(canvas_y, min(max_y, canvas_y + (canvas_height - layer.height) / 2))
                layer.pos = (layer_x, layer_y)
                
                print(f"图层位置: {layer.pos}, 画布区域: 位置={self._border_rect.pos}, 大小={self._border_rect.size}")
            else:
                # 如果画布边界尚未初始化，使用默认居中计算
                layer_x = max(0, (self.width - layer.width) / 2)
                layer_y = max(0, (self.height - layer.height) / 2)
                layer.pos = (layer_x, layer_y)
            
            print(f"画布尺寸: {self.CANVAS_WIDTH}x{self.CANVAS_HEIGHT}, 图层位置: {layer.pos}, 图层大小: {layer.size}")
            
            self.add_widget(layer)
            self.layers.append(layer)
            self.select_layer(layer)
            
            # 为图层添加窗口大小变化事件的回调
            if not hasattr(self, '_resize_callbacks'):
                self._resize_callbacks = []
                Window.bind(on_resize=self._on_window_resize)
            
            return layer
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _scale_image_to_fit_canvas(self, layer):
        """缩放图层以适应固定画布大小，保持宽高比"""
        if not layer or not layer.image or not layer.image.texture:
            return
            
        # 获取图像原始尺寸
        orig_width, orig_height = layer.size
        
        # 计算可用空间（考虑边距，使用固定画布大小）
        available_width = self.CANVAS_WIDTH * 0.9  # 留出10%边距
        available_height = self.CANVAS_HEIGHT * 0.9  # 留出10%边距
        
        # 计算缩放比例
        scale_x = available_width / orig_width
        scale_y = available_height / orig_height
        scale = min(scale_x, scale_y, 1.0)  # 不超过原始大小
        
        # 应用缩放
        if scale < 1.0:  # 只有当需要缩小时才缩放
            new_width = orig_width * scale
            new_height = orig_height * scale
            layer.size = (new_width, new_height)
            print(f"图像已缩放: 原始大小={orig_width}x{orig_height}, 缩放后={new_width}x{new_height}, 缩放比例={scale}")
        else:
            print(f"图像尺寸保持不变: {orig_width}x{orig_height}")
    
    def _on_window_resize(self, instance, width, height):
        """当窗口大小改变时，更新画布显示和所有图层的位置"""
        # 检查是否应该跳过本次更新
        if hasattr(self, '_skip_next_resize') and self._skip_next_resize:
            self._skip_next_resize = False
            return
            
        # 设置标记，避免_update_layout再次触发更新
        self._skip_next_resize = True
            
        # 更新画布边界显示
        self._update_canvas_display()
        
        # 更新每个图层的位置，保持居中
        if hasattr(self, 'layers') and hasattr(self, '_border_rect'):
            canvas_x, canvas_y = self._border_rect.pos
            canvas_width, canvas_height = self._border_rect.size
            
            for layer in self.layers:
                # 只更新当前选中的图层，以避免移动用户已经手动定位的图层
                if layer == self.selected_layer:
                    # 重新计算图层位置，使其居中在画布中
                    layer_x = canvas_x + (canvas_width - layer.width) / 2
                    layer_y = canvas_y + (canvas_height - layer.height) / 2
                    layer.pos = (layer_x, layer_y)
                    print(f"窗口大小变化: {width}x{height}, 更新图层位置: {layer.pos}, 图层大小: {layer.size}")
    
    def select_layer(self, layer=None):
        """选择一个图层"""
        # 取消之前选中图层的选中状态
        if self.selected_layer:
            self.selected_layer.is_selected = False
        
        # 选中新图层
        self.selected_layer = layer
        
        # 获取App实例以访问滑块控件
        app = App.get_running_app()
        
        if layer:
            layer.is_selected = True
            
            # 确保选中的图层在最上层
            self.remove_widget(layer)
            self.add_widget(layer)  # 添加到最顶层
            
            # 更新滑块基准尺寸
            app._base_width_for_slider_scale = max(layer.width, 1.0)
            app._base_height_for_slider_scale = max(layer.height, 1.0)
            
            # 重置滑块值，阻止触发回调
            app._lock_slider_event = True
            if hasattr(app, 'slider_x') and app.slider_x:
                app.slider_x.value = 1.0
                app.slider_x.disabled = False
            if hasattr(app, 'slider_y') and app.slider_y:
                app.slider_y.value = 1.0
                app.slider_y.disabled = False
            if hasattr(app, 'lock_aspect_ratio_checkbox') and app.lock_aspect_ratio_checkbox:
                app.lock_aspect_ratio_checkbox.disabled = False
            app._lock_slider_event = False
            
            print(f"已选择图层，基准尺寸: 宽={app._base_width_for_slider_scale}, 高={app._base_height_for_slider_scale}")
        else:
            # 禁用滑块
            if hasattr(app, 'slider_x') and app.slider_x:
                app.slider_x.disabled = True
            if hasattr(app, 'slider_y') and app.slider_y:
                app.slider_y.disabled = True
            if hasattr(app, 'lock_aspect_ratio_checkbox') and app.lock_aspect_ratio_checkbox:
                app.lock_aspect_ratio_checkbox.disabled = True
            
            # 重置基准尺寸
            app._base_width_for_slider_scale = 0.0
            app._base_height_for_slider_scale = 0.0
            print("已取消选择图层，滑块已禁用")
    
    def delete_selected_layer(self):
        """删除当前选中的图层"""
        if self.selected_layer:
            layer = self.selected_layer
            self.layers.remove(layer)
            self.remove_widget(layer)
            self.select_layer(None if not self.layers else self.layers[-1])
    
    def toggle_selected_layer_visibility(self):
        """切换当前选中图层的可见性"""
        if self.selected_layer:
            self.selected_layer.visible = not self.selected_layer.visible
    
    def move_selected_layer_up(self):
        """将当前选中图层上移一层"""
        if not self.selected_layer or len(self.layers) <= 1:
            return
            
        current_index = self.layers.index(self.selected_layer)
        if current_index < len(self.layers) - 1:
            # 交换列表中的位置
            self.layers[current_index], self.layers[current_index + 1] = \
                self.layers[current_index + 1], self.layers[current_index]
            
            # 重新排列子控件顺序
            self.reorder_layers()
    
    def move_selected_layer_down(self):
        """将当前选中图层下移一层"""
        if not self.selected_layer or len(self.layers) <= 1:
            return
            
        current_index = self.layers.index(self.selected_layer)
        if current_index > 0:
            # 交换列表中的位置
            self.layers[current_index], self.layers[current_index - 1] = \
                self.layers[current_index - 1], self.layers[current_index]
            
            # 重新排列子控件顺序
            self.reorder_layers()
    
    def reorder_layers(self):
        """根据layers列表重新排列子控件顺序"""
        for layer in self.layers:
            self.remove_widget(layer)
        
        # 按照列表顺序添加图层（索引小的在底层）
        for layer in self.layers:
            self.add_widget(layer)
        
        # 确保选中的图层在最上层显示
        if self.selected_layer:
            self.remove_widget(self.selected_layer)
            self.add_widget(self.selected_layer)
    
    def on_touch_down(self, touch):
        # 从上到下检查是否点击了某个图层
        for layer in reversed(self.children):
            if isinstance(layer, ImageLayerWidget) and layer.visible and layer.collide_point(*touch.pos):
                if layer != self.selected_layer:
                    self.select_layer(layer)
                return super(EditorCanvasWidget, self).on_touch_down(touch)
        
        # 如果点击了空白区域，取消当前选择
        self.select_layer(None)
        return super(EditorCanvasWidget, self).on_touch_down(touch)


class ImageEditorApp(App):
    """主应用程序"""
    
    # 新增App级别属性用于滑块控制
    _base_width_for_slider_scale = NumericProperty(0.0)
    _base_height_for_slider_scale = NumericProperty(0.0)
    _lock_slider_event = BooleanProperty(False) # 用于防止滑块事件的递归触发
    
    # UI控件的ObjectProperty引用
    slider_x = ObjectProperty(None)
    slider_y = ObjectProperty(None)
    lock_aspect_ratio_checkbox = ObjectProperty(None)
    editor_canvas = ObjectProperty(None)

    def build(self):
        # 加载kv文件
        from kivy.lang import Builder
        
        # 创建根控件 - 使用正确的方法
        root = Builder.load_file('image_editor.kv')
        
        # 绑定窗口大小变化事件，以便在窗口调整大小时更新布局
        Window.bind(on_resize=self._update_layout)
        
        # 手动获取对组件的引用
        self.editor_canvas = root.ids.editor_canvas
        self.slider_x = root.ids.slider_x
        self.slider_y = root.ids.slider_y
        self.lock_aspect_ratio_checkbox = root.ids.lock_aspect_ratio_checkbox
        
        # 初始化滑块状态
        self._base_width_for_slider_scale = 0.0
        self._base_height_for_slider_scale = 0.0
        self.slider_x.disabled = True
        self.slider_y.disabled = True
        self.lock_aspect_ratio_checkbox.disabled = True
        
        # 初始化当前没有选中的图层
        Clock.schedule_once(lambda dt: self.editor_canvas.select_layer(None), 0.1)
        
        print("应用初始化完成，缩放控件已设置为禁用状态")
        
        return root
    
    def _update_layout(self, instance, width, height):
        """当窗口大小改变时更新布局"""
        # 更新画布高度
        canvas_height = height - dp(30) - dp(60)  # 标题30dp, 工具栏60dp
        self.editor_canvas.height = canvas_height
        self.editor_canvas.pos_hint = {'x': 0, 'top': 1 - dp(30)/height}
    
    def open_file_dialog(self, instance):
        """打开文件选择对话框"""
        if filechooser:
            filechooser.open_file(
                on_selection=self.handle_selection,
                filters=[["Image Files", "*.png", "*.jpg", "*.jpeg", "*.bmp"]]
            )
        else:
            print("Warning: filechooser is not available, cannot open file dialog")
            # 为了测试，提供一个默认图像，如果存在的话
            test_images = ['test.png', 'test.jpg']
            for img in test_images:
                if os.path.exists(img):
                    self.handle_selection([img])
                    break
    
    def handle_selection(self, selection):
        """处理文件选择结果"""
        if selection and len(selection) > 0:
            for file_path in selection:
                self.editor_canvas.add_image_as_layer(file_path)
    
    def on_slider_x_value(self, instance, value):
        if self._lock_slider_event or not self.editor_canvas.selected_layer or self._base_width_for_slider_scale <= 0:
            return

        # 添加调试输出
        print(f"X轴滑块值改变: {value}, 基准宽度: {self._base_width_for_slider_scale}")
        
        # 调用统一形变函数
        self.transform_layer_size(scale_x=value)

    def on_slider_y_value(self, instance, value):
        if self._lock_slider_event or not self.editor_canvas.selected_layer or self._base_height_for_slider_scale <= 0:
            return

        # 添加调试输出
        print(f"Y轴滑块值改变: {value}, 基准高度: {self._base_height_for_slider_scale}")
        
        # 调用统一形变函数
        self.transform_layer_size(scale_y=value)

    def on_lock_aspect_ratio_change(self, checkbox, value):
        """处理锁定宽高比复选框状态变化"""
        print(f"锁定宽高比状态变化: {value}")
        
        # 如果勾选了锁定，且有选中图层，可以将当前宽高比设置为基准宽高比
        if value and self.editor_canvas.selected_layer:
            if self._base_width_for_slider_scale > 0 and self._base_height_for_slider_scale > 0:
                # 当前宽高比已经记录，不需要额外操作
                print("已锁定宽高比，使用当前基准比例")
            else:
                # 初始化基准宽高比
                layer = self.editor_canvas.selected_layer
                self._base_width_for_slider_scale = max(layer.width, 1.0)
                self._base_height_for_slider_scale = max(layer.height, 1.0)
                print(f"已初始化宽高比: {self._base_width_for_slider_scale}:{self._base_height_for_slider_scale}")
        
        # 如果取消锁定，允许独立缩放
        if not value:
            print("已解除宽高比锁定，X和Y可以独立缩放")
    
    def _update_text_size(self, instance, size):
        """更新Label的text_size以适应其大小"""
        instance.text_size = (instance.width, None)

    def transform_layer_size(self, scale_x=None, scale_y=None):
        """统一处理图层形变的函数
        
        参数:
            scale_x: X轴的缩放比例，None表示不改变
            scale_y: Y轴的缩放比例，None表示不改变
        """
        if not self.editor_canvas.selected_layer or self._base_width_for_slider_scale <= 0 or self._base_height_for_slider_scale <= 0:
            print("无法进行形变：没有选中图层或基准尺寸无效")
            return False
            
        layer = self.editor_canvas.selected_layer
        # 调用图层的transform_size方法
        new_size, success = layer.transform_size(
            self._base_width_for_slider_scale,
            self._base_height_for_slider_scale,
            scale_x,
            scale_y,
            self.lock_aspect_ratio_checkbox and self.lock_aspect_ratio_checkbox.active
        )
        
        if success and new_size:
            new_width, new_height = new_size
            
            # 根据计算结果更新滑杆值（不触发事件）
            if scale_x is not None and self.lock_aspect_ratio_checkbox and self.lock_aspect_ratio_checkbox.active:
                self._lock_slider_event = True
                self.slider_y.value = new_height / self._base_height_for_slider_scale if self._base_height_for_slider_scale > 0 else 1.0
                self._lock_slider_event = False
                print(f"锁定宽高比: 根据X轴更新Y轴值为 {self.slider_y.value}")
                
            if scale_y is not None and self.lock_aspect_ratio_checkbox and self.lock_aspect_ratio_checkbox.active:
                self._lock_slider_event = True
                self.slider_x.value = new_width / self._base_width_for_slider_scale if self._base_width_for_slider_scale > 0 else 1.0
                self._lock_slider_event = False
                print(f"锁定宽高比: 根据Y轴更新X轴值为 {self.slider_x.value}")
                
        return success


if __name__ == '__main__':
    ImageEditorApp().run() 