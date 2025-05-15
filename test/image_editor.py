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
            print(f"图像已加载: {source}，大小: {self.size}")
        else:
            print(f"警告: 图像纹理未加载: {source}")
            # 设置一个默认大小
            self.size = (300, 300)
            
            # 绑定纹理加载完成事件
            self.image.bind(texture=self._on_texture_loaded)
            
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
                    
                # 检查是否触摸形变点 - 使用更宽松的距离检测
                # 定义四个角点，使用实际图像边界
                handle_size = self.handle_size * 1.5  # 增大检测范围
                handle_points = [
                    (img_x, img_y, 'BL'),                           # 左下
                    (img_x + img_width, img_y, 'BR'),               # 右下
                    (img_x, img_y + img_height, 'TL'),              # 左上
                    (img_x + img_width, img_y + img_height, 'TR')   # 右上
                ]
                
                # 找到距离最近的控制点
                closest_corner = None
                min_dist = float('inf')
                
                for x, y, corner in handle_points:
                    dist = math.sqrt((local_touch[0] - x) ** 2 + (local_touch[1] - y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_corner = (x, y, corner)
                
                # 如果最近的控制点在检测范围内，启动形变
                if min_dist <= handle_size:
                    x, y, corner = closest_corner
                    self._drag_mode = f'scale_{corner}'
                    print(f"检测到形变操作: 角点={corner}, 距离={min_dist}, 开始形变")
                    
                    # 存储初始数据用于形变计算
                    self._initial_touch_data = {
                        'size': self.size[:],
                        'pos': self.pos[:],
                        'touch_pos': touch.pos,
                        'local_touch': local_touch,
                        'rotation': self.rotation,
                        'scale': self.scale,
                        'center': self.center[:],
                        'corner': (x, y),
                        'img_offset': (img_x, img_y),
                        'img_size': (img_width, img_height),
                        'texture_size': self.image.texture.size if self.image and self.image.texture else None,
                        'opposite_corner': (
                            img_x + img_width - x if x == img_x else img_x - x,
                            img_y + img_height - y if y == img_y else img_y - y
                        ),
                        'aspect_ratio': img_width / max(0.1, img_height)  # 防止除零错误
                    }
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
                
            elif self._drag_mode and self._drag_mode.startswith('scale_'):
                # 处理形变
                is_shift_pressed = 'shift' in Window.modifiers
                
                # 获取初始数据
                corner_name = self._drag_mode[6:]  # 'scale_TR' -> 'TR'
                initial_pos = self._initial_touch_data['pos']
                initial_size = self._initial_touch_data['size'] # 图层初始尺寸
                initial_img_offset = self._initial_touch_data.get('img_offset', (0, 0))
                initial_img_size = self._initial_touch_data.get('img_size', initial_size) # 图像内容初始尺寸
                texture_size = self._initial_touch_data.get('texture_size')
                
                img_content_initial_width = initial_img_size[0]
                img_content_initial_height = initial_img_size[1]

                # 获取图像在图层中的偏移量
                img_x, img_y = initial_img_offset
                # img_width, img_height = initial_img_size # 使用 initial_img_size 替代，避免混淆
                
                # 减少日志输出
                # print(f"形变移动: 全局位置={touch.pos}, 图层初始位置={initial_pos}, 初始尺寸={initial_size}")
                # print(f"图像偏移: ({img_x}, {img_y}), 图像尺寸: {initial_img_size}")
                
                # 根据不同的控制点计算新的图层尺寸和位置
                if corner_name == 'BL':  # 左下角，右上角固定
                    # 计算右上角的绝对坐标（固定点 - 图像内容的右上角）
                    fixed_x = initial_pos[0] + img_x + img_content_initial_width
                    fixed_y = initial_pos[1] + img_y + img_content_initial_height
                    
                    # 计算新的图像内容尺寸
                    new_content_width = fixed_x - touch.x
                    new_content_height = fixed_y - touch.y
                    # new_x = touch.x # 新的图像内容左下角 X
                    # new_y = touch.y # 新的图像内容左下角 Y
                    
                elif corner_name == 'BR':  # 右下角，左上角固定
                    # 左上角固定 (图像内容的左上角)
                    fixed_x = initial_pos[0] + img_x
                    fixed_y = initial_pos[1] + img_y + img_content_initial_height
                    
                    # 新的图像内容尺寸
                    new_content_width = touch.x - fixed_x
                    new_content_height = fixed_y - touch.y
                    # new_x = fixed_x
                    # new_y = touch.y
                    
                elif corner_name == 'TL':  # 左上角，右下角固定
                    # 右下角固定 (图像内容的右下角)
                    fixed_x = initial_pos[0] + img_x + img_content_initial_width
                    fixed_y = initial_pos[1] + img_y
                    
                    # 新的图像内容尺寸
                    new_content_width = fixed_x - touch.x
                    new_content_height = touch.y - fixed_y
                    # new_x = touch.x
                    # new_y = fixed_y
                    
                else:  # 'TR', 右上角，左下角固定
                    # 左下角固定 (图像内容的左下角)
                    fixed_x = initial_pos[0] + img_x
                    fixed_y = initial_pos[1] + img_y
                    
                    # 新的图像内容尺寸
                    new_content_width = touch.x - fixed_x
                    new_content_height = touch.y - fixed_y
                    # new_x = fixed_x
                    # new_y = fixed_y
                
                # 保持等比例缩放（如果按下了Shift键）
                if is_shift_pressed and texture_size:
                    tex_width, tex_height = texture_size
                    aspect_ratio = tex_width / max(0.1, tex_height)
                    
                    # 临时内容宽度/高度用于比较
                    temp_new_content_width = new_content_width
                    temp_new_content_height = new_content_height

                    # 根据移动距离的主导方向调整尺寸
                    # 使用初始内容尺寸进行比较，更稳定
                    width_scale_factor = temp_new_content_width / img_content_initial_width
                    height_scale_factor = temp_new_content_height / img_content_initial_height
                    
                    if abs(width_scale_factor - 1) > abs(height_scale_factor - 1): # 判断哪个方向的形变更显著
                        # 宽度变化更大，以宽度为准
                        new_content_height = temp_new_content_width / aspect_ratio
                    else:
                        # 高度变化更大，以高度为准
                        new_content_width = temp_new_content_height * aspect_ratio
                    
                    # 调整位置以保持固定点不变 (这部分是针对 new_x, new_y 的，它们是图像内容的角点)
                    # 对于BL, BR, TL, TR, 其 new_x, new_y (图像内容的活动角点) 就是 touch.x, touch.y 或 fixed_x, fixed_y
                    # 这部分不需要调整 new_x, new_y，因为它们是由 fixed_x/y 和 touch.x/y 直接决定的
                    # 而是 new_layer_x/y 的计算会基于新的 new_content_width/height 来正确定位图层
                    pass # new_x, new_y (content corner) adjustments are implicit in fixed points and touch

                # 确保内容尺寸为正
                min_dimension = 10
                if new_content_width < min_dimension:
                    if corner_name in ['BL', 'TL']: # 左侧角点，固定右边
                        # fixed_x 是右边缘, new_content_width = fixed_x - touch.x
                        # touch.x = fixed_x - new_content_width
                        pass # new_content_width 会直接被设为 min_dimension
                    new_content_width = min_dimension
                
                if new_content_height < min_dimension:
                    if corner_name in ['BL', 'BR']: # 底部角点，固定上边
                        # fixed_y 是上边缘, new_content_height = fixed_y - touch.y
                        # touch.y = fixed_y - new_content_height
                        pass # new_content_height 会直接被设为 min_dimension
                    new_content_height = min_dimension
                
                # 减少日志输出
                # print(f"形变计算: 角点={corner_name}, 固定点=({fixed_x:.1f}, {fixed_y:.1f})")
                # print(f"新内容尺寸: 宽度={new_content_width:.1f}, 高度={new_content_height:.1f}")
                
                new_layer_width = 0
                new_layer_height = 0

                if img_content_initial_width > 0 and img_content_initial_height > 0 : # 避免除零
                    scale_w = new_content_width / img_content_initial_width
                    scale_h = new_content_height / img_content_initial_height
                    
                    # 应用缩放到图层 - 使用初始图层尺寸
                    new_layer_width = initial_size[0] * scale_w
                    new_layer_height = initial_size[1] * scale_h
                else: # 如果初始内容尺寸为0，则无法计算比例，直接使用新内容尺寸作为图层尺寸
                    new_layer_width = new_content_width
                    new_layer_height = new_content_height

                # 确保图层尺寸不小于内容尺寸（通常图层会等比例放大，所以这一步可能多余，但保留以防万一）
                # new_layer_width = max(new_content_width, new_layer_width)
                # new_layer_height = max(new_content_height, new_layer_height)
                
                # 确保图层尺寸不小于最小尺寸
                new_layer_width = max(min_dimension, new_layer_width)
                new_layer_height = max(min_dimension, new_layer_height)

                # 根据控制点和新的图层尺寸计算图层的新位置 (self.pos)
                new_layer_x = 0
                new_layer_y = 0
                if corner_name == 'BL': # 左下角被拖动，图层右上角固定（基于图像内容的初始右上角）
                    # fixed_x, fixed_y 是图像内容初始右上角全局坐标
                    new_layer_x = fixed_x - new_layer_width 
                    new_layer_y = fixed_y - new_layer_height
                elif corner_name == 'BR': # 右下角被拖动，图层左上角固定
                    # fixed_x, fixed_y 是图像内容初始左上角全局坐标
                    new_layer_x = fixed_x
                    new_layer_y = fixed_y - new_layer_height 
                elif corner_name == 'TL': # 左上角被拖动，图层右下角固定
                    # fixed_x, fixed_y 是图像内容初始右下角全局坐标
                    new_layer_x = fixed_x - new_layer_width
                    new_layer_y = fixed_y
                else:  # 'TR', 右上角被拖动，图层左下角固定
                    # fixed_x, fixed_y 是图像内容初始左下角全局坐标
                    new_layer_x = fixed_x
                    new_layer_y = fixed_y
                
                # 应用新尺寸和位置到图层
                self.size = (new_layer_width, new_layer_height)
                self.pos = (new_layer_x, new_layer_y)
                
                # 更新图像尺寸以适应图层
                if self.image:
                    if texture_size:
                        # 重新计算图像尺寸，保持适当的宽高比
                        # tex_width, tex_height = texture_size
                        # tex_ratio = tex_width / max(0.1, tex_height) # 已有 aspect_ratio
                        
                        # 根据新的图层尺寸，重新计算适合的图像尺寸
                        fitted_size = self._calculate_fitted_size(texture_size, self.size)
                        self.image.size = fitted_size
                    else:
                        # 如果无法获取纹理尺寸，设置为图层尺寸
                        self.image.size = self.size
                
                # 更新控制点
                self.update_canvas()
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


class EditorCanvasWidget(RelativeLayout):
    """编辑器画布，作为所有图层的容器"""
    
    selected_layer = ObjectProperty(None, allownone=True)
    # 定义固定的画布大小
    CANVAS_WIDTH = 1920
    CANVAS_HEIGHT = 1080
    
    def __init__(self, **kwargs):
        super(EditorCanvasWidget, self).__init__(**kwargs)
        self.layers = []
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
        """更新画布显示，确保边界矩形始终可见"""
        if hasattr(self, '_bg_rect'):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size
        
        if hasattr(self, '_border_rect'):
            # 计算缩放比例，确保画布边界完全显示在窗口中并居中
            scale_x = self.width / self.CANVAS_WIDTH
            scale_y = self.height / self.CANVAS_HEIGHT
            scale = min(scale_x, scale_y)
            
            # 计算缩放后的实际画布尺寸
            display_width = self.CANVAS_WIDTH * scale
            display_height = self.CANVAS_HEIGHT * scale
            
            # 计算位置偏移，使画布居中
            offset_x = (self.width - display_width) / 2
            offset_y = (self.height - display_height) / 2
            
            # 更新画布边界显示
            self._border_rect.pos = (offset_x, offset_y)
            self._border_rect.size = (display_width, display_height)
            
            print(f"画布显示更新: 缩放比例={scale}, 位置={self._border_rect.pos}, 大小={self._border_rect.size}")
    
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

    def build(self):
        # 创建根布局
        root = FloatLayout()
        
        # 创建编辑画布 - 修复位置问题，为画布和工具栏使用绝对位置
        self.editor_canvas = EditorCanvasWidget()
        
        # 添加标题
        title = Label(
            text='Kivy Image Editor', 
            size_hint=(1, None),
            height=dp(30),
            pos_hint={'x': 0, 'top': 1},
            font_size='20sp'
        )
        
        # 计算主要区域的高度 (总高度减去标题和工具栏)
        canvas_height = Window.height - dp(30) - dp(60)  # 标题30dp, 工具栏60dp
        
        # 设置画布位置和大小
        self.editor_canvas.size_hint = (1, None)
        self.editor_canvas.height = canvas_height
        self.editor_canvas.pos_hint = {'x': 0, 'top': 1 - dp(30)/Window.height}
        
        # 创建工具栏，使用固定高度而不是相对高度
        toolbar = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=dp(60),
            pos_hint={'x': 0, 'bottom': 0},
            spacing=5,
            padding=5
        )
        
        # 添加加载图像按钮
        load_btn = Button(
            text='Load Image', 
            size_hint=(0.2, 1)
        )
        load_btn.bind(on_press=self.open_file_dialog)
        toolbar.add_widget(load_btn)
        
        # 添加图层操作按钮
        layer_controls = BoxLayout(orientation='vertical', size_hint=(0.2, 1))
        
        # 图层顺序控制
        order_controls = BoxLayout(orientation='horizontal', size_hint=(1, 0.5))
        
        up_btn = Button(text='Move Up')
        up_btn.bind(on_press=lambda x: self.editor_canvas.move_selected_layer_up())
        order_controls.add_widget(up_btn)
        
        down_btn = Button(text='Move Down')
        down_btn.bind(on_press=lambda x: self.editor_canvas.move_selected_layer_down())
        order_controls.add_widget(down_btn)
        
        layer_controls.add_widget(order_controls)
        
        # 图层显示和删除控制
        visibility_controls = BoxLayout(orientation='horizontal', size_hint=(1, 0.5))
        
        toggle_vis_btn = Button(text='Show/Hide')
        toggle_vis_btn.bind(on_press=lambda x: self.editor_canvas.toggle_selected_layer_visibility())
        visibility_controls.add_widget(toggle_vis_btn)
        
        delete_btn = Button(text='Delete')
        delete_btn.bind(on_press=lambda x: self.editor_canvas.delete_selected_layer())
        visibility_controls.add_widget(delete_btn)
        
        layer_controls.add_widget(visibility_controls)
        toolbar.add_widget(layer_controls)
        
        # --- 新增缩放控制 ---
        scaling_controls_layout = BoxLayout(orientation='vertical', size_hint_x=0.3, spacing=dp(2))

        # X Scale Slider
        x_scale_row = BoxLayout(size_hint_y=None, height=dp(25))
        x_scale_label = Label(text='Scale X:', size_hint_x=0.3, font_size='12sp')
        self.slider_x = Slider(min=0.1, max=3, value=1.0, size_hint_x=0.7, disabled=True)
        self.slider_x.bind(value=self.on_slider_x_value)
        x_scale_row.add_widget(x_scale_label)
        x_scale_row.add_widget(self.slider_x)
        scaling_controls_layout.add_widget(x_scale_row)

        # Y Scale Slider
        y_scale_row = BoxLayout(size_hint_y=None, height=dp(25))
        y_scale_label = Label(text='Scale Y:', size_hint_x=0.3, font_size='12sp')
        self.slider_y = Slider(min=0.1, max=3, value=1.0, size_hint_x=0.7, disabled=True)
        self.slider_y.bind(value=self.on_slider_y_value)
        y_scale_row.add_widget(y_scale_label)
        y_scale_row.add_widget(self.slider_y)
        scaling_controls_layout.add_widget(y_scale_row)
        
        # Lock Aspect Ratio Checkbox
        lock_row = BoxLayout(size_hint_y=None, height=dp(25)) # Even smaller height for checkbox row
        self.lock_aspect_ratio_checkbox = CheckBox(active=False, size_hint_x=0.2, disabled=True)
        # 绑定勾选框状态变化事件
        self.lock_aspect_ratio_checkbox.bind(active=self.on_lock_aspect_ratio_change)
        lock_label = Label(text='Lock Aspect Ratio', size_hint_x=0.8, font_size='12sp')
        lock_row.add_widget(self.lock_aspect_ratio_checkbox)
        lock_row.add_widget(lock_label)
        scaling_controls_layout.add_widget(lock_row)
        
        toolbar.add_widget(scaling_controls_layout)
        # --- 结束新增缩放控制 ---
        
        # 添加使用说明
        help_text = Label(
            text='Instructions:\n· Click to select layer\n· Drag to move\n· Drag center point to rotate\n· Drag corners to resize\n· Hold Shift for proportional scaling',
            size_hint=(0.4, 1),
            halign='left',
            valign='middle'
        )
        help_text.bind(size=self._update_text_size)
        toolbar.add_widget(help_text)
        
        # 按照从下到上的顺序添加元素，确保正确的堆叠顺序
        root.add_widget(toolbar)
        root.add_widget(self.editor_canvas)
        root.add_widget(title)
        
        # 绑定窗口大小变化事件，以便在窗口调整大小时更新布局
        Window.bind(on_resize=self._update_layout)
        
        # 打印调试信息，检查控件初始状态
        print(f"滑块X初始状态: disabled={self.slider_x.disabled}, value={self.slider_x.value}")
        print(f"滑块Y初始状态: disabled={self.slider_y.disabled}, value={self.slider_y.value}")
        print(f"锁定比例复选框: disabled={self.lock_aspect_ratio_checkbox.disabled}, active={self.lock_aspect_ratio_checkbox.active}")
        
        # 初始化滑块状态 - 确保在启动时滑块被禁用
        self._base_width_for_slider_scale = 0.0
        self._base_height_for_slider_scale = 0.0
        self.slider_x.disabled = True
        self.slider_y.disabled = True
        self.lock_aspect_ratio_checkbox.disabled = True
        
        # 初始化当前没有选中的图层
        self.editor_canvas.select_layer(None)
        
        print("应用初始化完成，缩放控件已设置为禁用状态")
        
        return root
    
    def _update_text_size(self, instance, size):
        """更新Label的text_size以适应其大小"""
        instance.text_size = (instance.width, None)
    
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
    
    def _update_layout(self, instance, width, height):
        """当窗口大小改变时更新布局"""
        # 更新画布高度
        canvas_height = height - dp(30) - dp(60)  # 标题30dp, 工具栏60dp
        self.editor_canvas.height = canvas_height
        self.editor_canvas.pos_hint = {'x': 0, 'top': 1 - dp(30)/height}

    def on_slider_x_value(self, instance, value):
        if self._lock_slider_event or not self.editor_canvas.selected_layer or self._base_width_for_slider_scale <= 0:
            return

        # 添加调试输出
        print(f"X轴滑块值改变: {value}, 基准宽度: {self._base_width_for_slider_scale}")

        layer = self.editor_canvas.selected_layer
        old_width, old_height = layer.size
        old_center_x = layer.x + old_width / 2
        old_center_y = layer.y + old_height / 2

        new_width = self._base_width_for_slider_scale * value
        new_height = old_height # Default if not locked

        if self.lock_aspect_ratio_checkbox and self.lock_aspect_ratio_checkbox.active and self._base_height_for_slider_scale > 0:
            # 计算宽高比以保持比例
            aspect_ratio = self._base_width_for_slider_scale / self._base_height_for_slider_scale
            if aspect_ratio > 0: # Avoid division by zero if aspect_ratio is somehow zero
                 new_height = new_width / aspect_ratio
                 self._lock_slider_event = True
                 self.slider_y.value = new_height / self._base_height_for_slider_scale if self._base_height_for_slider_scale > 0 else 1.0
                 self._lock_slider_event = False
                 print(f"锁定宽高比: 根据X轴滑块更新Y轴值为 {self.slider_y.value}")
            else: # Fallback if aspect ratio is invalid
                new_height = self._base_height_for_slider_scale * self.slider_y.value # Use current Y slider
        else: # Not locked, use current Y slider value to determine target height based on its own base
            if self._base_height_for_slider_scale > 0:
                new_height = self._base_height_for_slider_scale * self.slider_y.value
            print(f"未锁定宽高比: 保持Y值 {self.slider_y.value}, 高度 {new_height}")

        min_dimension = 10
        new_width = max(new_width, min_dimension)
        new_height = max(new_height, min_dimension)
        
        # 应用新尺寸，保持图层中心点不变
        layer.size = (new_width, new_height)
        layer.pos = (old_center_x - new_width / 2, old_center_y - new_height / 2)
        
        print(f"应用新尺寸: 宽度={new_width}, 高度={new_height}, 位置={layer.pos}")

    def on_slider_y_value(self, instance, value):
        if self._lock_slider_event or not self.editor_canvas.selected_layer or self._base_height_for_slider_scale <= 0:
            return

        # 添加调试输出
        print(f"Y轴滑块值改变: {value}, 基准高度: {self._base_height_for_slider_scale}")

        layer = self.editor_canvas.selected_layer
        old_width, old_height = layer.size
        old_center_x = layer.x + old_width / 2
        old_center_y = layer.y + old_height / 2

        new_height = self._base_height_for_slider_scale * value
        new_width = old_width # Default if not locked

        if self.lock_aspect_ratio_checkbox and self.lock_aspect_ratio_checkbox.active and self._base_width_for_slider_scale > 0:
            # 计算宽高比以保持比例
            aspect_ratio = self._base_width_for_slider_scale / self._base_height_for_slider_scale
            if aspect_ratio > 0: # Check if aspect_ratio is valid
                new_width = new_height * aspect_ratio
                self._lock_slider_event = True
                self.slider_x.value = new_width / self._base_width_for_slider_scale if self._base_width_for_slider_scale > 0 else 1.0
                self._lock_slider_event = False
                print(f"锁定宽高比: 根据Y轴滑块更新X轴值为 {self.slider_x.value}")
            else: # Fallback
                new_width = self._base_width_for_slider_scale * self.slider_x.value # Use current X slider
        else: # Not locked
            if self._base_width_for_slider_scale > 0:
                new_width = self._base_width_for_slider_scale * self.slider_x.value
            print(f"未锁定宽高比: 保持X值 {self.slider_x.value}, 宽度 {new_width}")

        min_dimension = 10
        new_width = max(new_width, min_dimension)
        new_height = max(new_height, min_dimension)

        # 应用新尺寸，保持图层中心点不变
        layer.size = (new_width, new_height)
        layer.pos = (old_center_x - new_width / 2, old_center_y - new_height / 2)
        
        print(f"应用新尺寸: 宽度={new_width}, 高度={new_height}, 位置={layer.pos}")

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


if __name__ == '__main__':
    ImageEditorApp().run() 