
好的，这是一个详细的需求案和结构设计，希望能满足您的要求。

## 图像编辑器需求案

**1. 引言**

*   **1.1 项目目标:** 开发一个基于 Kivy 和 Plyer 的图像编辑器，允许用户在 Windows 平台上通过鼠标操作，方便地载入、管理和编辑多张图像，实现图层化管理、位置调整、旋转和形变等功能。
*   **1.2 目标用户:** 需要进行基本图像合成和编辑的 Windows 用户。
*   **1.3 主要功能概述:**
    *   多图像文件载入
    *   图层化图像管理
    *   图像拖拽移动
    *   图像旋转（通过中心控制点）
    *   图像形变（通过边角控制点，支持等比例和非等比例）

**2. 功能需求 (FR)**

*   **2.1 图像载入**
    *   FR2.1.1: 用户应能通过系统文件对话框选择并载入一张或多张本地图像文件（如 PNG, JPG, BMP）。此功能可使用 `Plyer` 实现。
    *   FR2.1.2: 每张载入的图像应在编辑器中作为一个独立的图层显示。
    *   FR2.1.3: 编辑器应能处理常见的图像格式，并对无法识别或损坏的文件进行错误提示。
*   **2.2 图层管理**
    *   FR2.2.1: 图像以图层形式存在，后载入的图像默认位于最顶层。
    *   FR2.2.2: 用户应能通过点击图像选择当前活动的图层。
    *   FR2.2.3 (可选): 用户应能调整图层顺序（上移、下移、置顶、置底）。
    *   FR2.2.4 (可选): 用户应能显示/隐藏图层。
    *   FR2.2.5 (可选): 用户应能删除图层。
*   **2.3 图像显示与交互元素**
    *   FR2.3.1: 每个图层（图像）在被选中时，应显示一个紧密贴合其当前视觉边界的矩形描边（“边界框”）。
    *   FR2.3.2: 边界框的视觉中心应显示一个圆形控制点（“旋转点”）。
    *   FR2.3.3: 边界框的四个角应各显示一个控制点（“形变点”，例如小方块或小圆点）。
    *   FR2.3.4: 未选中的图层不应显示边界框及控制点，以保持界面整洁。
*   **2.4 图像移动**
    *   FR2.4.1: 用户选中一个图层后，应能通过鼠标左键按住该图像的任意可见部分（非控制点区域）并拖拽，以在画布上移动该图像。
    *   FR2.4.2: 移动操作应实时反映在画布上。
*   **2.5 图像旋转**
    *   FR2.5.1: 用户选中一个图层后，应能通过鼠标左键按住该图层的“旋转点”。
    *   FR2.5.2: 按住旋转点后，在画布上垂直或水平拖拽鼠标，图像应围绕其视觉中心点进行旋转。
    *   FR2.5.3: 旋转角度根据鼠标拖拽的方向和距离计算。
    *   FR2.5.4: 旋转操作应实时反映在画布上。
*   **2.6 图像形变**
    *   FR2.6.1: 用户选中一个图层后，应能通过鼠标左键按住该图层的任意一个“形变点”。
    *   **FR2.6.2: 非等比例形变 (单独调整X/Y轴形变):**
        *   FR2.6.2.1: 直接拖拽一个形变点时，该形变点应跟随鼠标移动。图像的形变应基于被拖拽角点的新位置，同时其对角（固定角）保持在屏幕上的位置不变。这将导致图像的宽度和高度独立变化，从而改变其宽高比和形状（保持矩形）。
    *   **FR2.6.3: 等比例形变 (Shift + 拖拽):**
        *   FR2.6.3.1: 当用户按住 `Shift` 键的同时拖拽一个形变点时，图像应以其视觉中心点（或被拖拽角点的对角）为基准，进行等比例缩放。
        *   FR2.6.3.2: 缩放比例根据鼠标拖拽的距离计算，使被拖拽的角点跟随鼠标，同时保持图像的原始宽高比。
    *   FR2.6.4: 形变操作应实时反映在画布上。
*   **2.7 画布**
    *   FR2.7.1: 提供一个主画布区域用于显示和编辑所有图像图层。
    *   FR2.7.2 (可选): 画布支持整体缩放和平移视图。
*   **2.8 用户界面**
    *   FR2.8.1: 界面简洁直观，优先考虑Windows平台下鼠标操作的便捷性。
    *   FR2.8.2: 提供清晰的视觉反馈，如当前选中图层的突出显示。
    *   FR2.8.3 (可选): 提供撤销/重做功能。

**3. 非功能需求 (NFR)**

*   **NFR3.1 平台:** 主要在 Windows 操作系统上运行。
*   **NFR3.2 技术栈:** 使用 Kivy 框架和 Plyer 库。
*   **NFR3.3 性能:** 图像操作（移动、旋转、形变）应流畅，避免明显卡顿，尤其是在处理中等数量和尺寸的图像时。
*   **NFR3.4 易用性:** 强调鼠标操作的直观性和易学性。

**4. 用例 (UC)**

*   **UC1: 载入并移动图像**
    1.  用户启动程序。
    2.  用户通过界面操作（如菜单或按钮）选择“载入图像”。
    3.  系统显示文件选择对话框。
    4.  用户选择一个或多个图像文件并确认。
    5.  图像被载入画布，每张图像为一个图层。最新载入的在最上层。
    6.  用户点击选中一张图像，该图像显示边界框和控制点。
    7.  用户按住鼠标左键拖拽该图像到新位置。
    8.  图像位置实时更新。
*   **UC2: 旋转图像**
    1.  前置条件：至少一张图像已载入并被选中。
    2.  用户将鼠标指针移动到选中图像的中心旋转点上。
    3.  用户按下鼠标左键并拖拽。
    4.  图像围绕其中心点实时旋转。
    5.  用户释放鼠标，图像保持旋转后的状态。
*   **UC3: 非等比例形变图像**
    1.  前置条件：至少一张图像已载入并被选中。
    2.  用户将鼠标指针移动到选中图像的某个边角形变点上。
    3.  用户按下鼠标左键并拖拽。
    4.  被拖拽的角点跟随鼠标移动，其对角保持屏幕位置不变，图像发生非等比例缩放/形变。
    5.  用户释放鼠标，图像保持形变后的状态。
*   **UC4: 等比例缩放图像**
    1.  前置条件：至少一张图像已载入并被选中。
    2.  用户将鼠标指针移动到选中图像的某个边角形变点上。
    3.  用户按住 `Shift` 键。
    4.  用户按下鼠标左键并拖拽该形变点。
    5.  图像以其中心（或对角）为锚点进行等比例放大或缩小。
    6.  用户释放鼠标（和 `Shift` 键），图像保持缩放后的状态。

---

## 结构设计 (伪代码 / UML 类图思路)

核心将围绕 Kivy 的 `Widget` 构建，特别是 `ScatterLayout` 或自定义类似功能的控件，因为它本身提供了移动、缩放、旋转的基础。我们需要在其上封装以实现特定的控制点和交互行为。

**主要类:**

1.  **`ImageLayerWidget(kivy.uix.scatterlayout.ScatterLayout)`**:
    *   **职责**:
        *   代表画布上的一个可交互图像图层。
        *   加载和显示图像 (内部使用 `kivy.uix.image.Image` 或直接在canvas上绘制纹理)。
        *   处理自身的移动 (主要由 `ScatterLayout` 基类提供)。
        *   自定义处理旋转和形变逻辑，通过专用的控制点。
        *   管理和绘制边界框、旋转点、形变点 (当被选中时)。
    *   **关键属性**:
        *   `source: str` (图像文件路径)
        *   `texture: kivy.graphics.texture.Texture` (图像纹理)
        *   `is_selected: bool` (标记是否被选中)
        *   `# Original Scatter properties: pos, size, scale, rotation, transform`
        *   `# Custom properties if needed for complex deformation:`
        *   `# aspect_ratio: float`
        *   `# internal_scale_x: float`, `internal_scale_y: float` (for non-uniform scaling effects)
    *   **关键方法**:
        *   `load_image(path: str)`: 加载图像纹理，设置初始尺寸。
        *   `on_touch_down(touch)`:
            *   检查点击是否在旋转点、形变点或图像本体上。
            *   根据点击位置，设置当前操作模式 (旋转、移动、特定角的形变)。
            *   `touch.grab(self)` 以捕获后续事件。
        *   `on_touch_move(touch)`:
            *   如果 `touch.grab_current` 是自身：
                *   根据当前操作模式，更新图像的 `rotation`, `pos`, 或通过修改 `transform` 矩阵（推荐用于复杂形变）来实现形变。
                *   对于旋转：计算角度变化，更新 `self.rotation`。
                *   对于形变（角点拖拽）：
                    *   **无 Shift**: 计算新的变换，使得被拖拽角点跟随鼠标，对角固定。这可能需要直接操作 `self.transform` 矩阵，以实现非均匀缩放和平移的组合，从而保持对角固定。
                    *   **有 Shift**: 计算统一缩放因子，更新 `self.scale` (Scatter的scale是统一的)。确保缩放中心正确（例如，图像中心或固定对角）。
                *   实时更新视觉元素（边界框、控制点）。
        *   `on_touch_up(touch)`:
            *   如果 `touch.grab_current` 是自身：`touch.ungrab(self)`，重置操作模式。
        *   `draw_controls()`: (在 Kivy 中通过 `canvas` 指令实现)
            *   如果 `is_selected` 为 `True`，则绘制边界框、旋转点和形变点。这些点的位置需要根据当前 `self.transform`（图像的最终变换）来计算。
        *   `update_controls_positions()`: 当图像变换（移动、旋转、缩放/形变）后，重新计算并更新控制点在屏幕上的绘制位置。
        *   `# Helper methods to convert points between local and parent coordinates.`

2.  **`EditorCanvasWidget(kivy.uix.relativelayout.RelativeLayout or kivy.uix.floatlayout.FloatLayout)`**:
    *   **职责**:
        *   作为所有 `ImageLayerWidget` 的容器和管理器。
        *   管理图层的添加、删除、选择和层级顺序。
        *   处理画布级别的事件，如点击未选中区域取消选择，或选择图层。
    *   **关键属性**:
        *   `layers: list[ImageLayerWidget]` (存储所有图层实例)
        *   `selected_layer: ImageLayerWidget | None` (当前选中的图层)
    *   **关键方法**:
        *   `add_image_as_layer(image_path: str)`: 创建一个新的 `ImageLayerWidget` 实例，加载图像，并将其添加到 `layers` 列表和画布中。
        *   `select_layer(layer_to_select: ImageLayerWidget)`:
            *   取消之前选中图层的选中状态 (`is_selected = False`)。
            *   设置新图层的选中状态 (`is_selected = True`)。
            *   更新 `self.selected_layer`。
            *   确保选中的图层在视觉上位于其他图层之上（如果需要调整Kivy子控件的绘制顺序）。
        *   `on_touch_down(touch)`:
            *   遍历所有图层，检查 `touch.pos` 是否与某个图层碰撞。
            *   如果碰撞到最上层的未选中图层，则调用 `select_layer`。
            *   如果碰撞到已选中的图层，则事件由该图层自己处理（如 `ImageLayerWidget.on_touch_down`）。
            *   如果未碰撞到任何图层，可以取消当前选中图层。

3.  **`MainApp(kivy.app.App)`**:
    *   **职责**:
        *   Kivy 应用程序的入口点。
        *   构建主界面布局，包含 `EditorCanvasWidget` 和可能的菜单栏/工具栏。
        *   处理全局操作，如通过菜单触发文件载入。
    *   **关键方法**:
        *   `build()`: 返回应用的根 Widget (例如，一个包含 `EditorCanvasWidget` 和一个文件选择按钮的布局)。
        *   `open_file_dialog()`:
            *   使用 `plyer.filechooser.open_file()` 打开文件选择对话框。
            *   获取选中的文件路径列表。
            *   对于每个文件路径，调用 `EditorCanvasWidget.add_image_as_layer()`。

**伪代码片段 (重点关注 `ImageLayerWidget` 的形变逻辑)**

```python
# Inside ImageLayerWidget:
# Pre-requisites: math, kivy.graphics.transformation.Matrix, kivy.core.window

class ImageLayerWidget(ScatterLayout):
    # ... (init, load_image, draw_controls, etc.)
    _drag_mode = None  # e.g., 'move', 'rotate', 'scale_TL', 'scale_TR', etc.
    _initial_touch_data = {} # Store data at touch_down for calculations

    def on_touch_down(self, touch):
        if not self.is_selected or not self.collide_point(*touch.pos):
            # If not selected, let EditorCanvas handle selection.
            # If selected but touch is outside, also ignore for this widget.
            return super().on_touch_down(touch) # Allow Scatter's default for movement if applicable

        # Assume controls' collision detection happens here
        # Rotation handle check:
        # if touch collides with rotation_handle_visual:
        #    self._drag_mode = 'rotate'
        #    self._initial_touch_data = {'center': self.center, 'rotation': self.rotation, 'touch_pos': touch.pos}
        #    touch.grab(self)
        #    return True

        # Scale handle check (example for Top-Right corner, index 1):
        # if touch collides with scale_handle_TR_visual:
        #    self._drag_mode = 'scale_TR' # (or scale_corner_1)
        #    # Store current transform, local coords of corners, touch pos
        #    self._initial_touch_data = {
        #        'transform': Matrix نگاهی به self.transform, # Current full transform
        #        'local_TR': (self.width, self.height), # Local coords of TR (assuming origin is BL)
        #        'local_BL': (0, 0), # Local coords of opposite corner (BL)
        #        'screen_BL': self.to_parent(0, 0), # Screen position of fixed anchor
        #        'touch_pos_screen': touch.pos
        #    }
        #    touch.grab(self)
        #    return True
        
        # If no handle is hit, it might be a drag-to-move action
        # ScatterLayout handles this by default if self.do_translation_x/y are True
        return super().on_touch_down(touch)


    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)

        if self._drag_mode == 'rotate':
            # ... (calculate angle change and update self.rotation)
            pass

        elif self._drag_mode and self._drag_mode.startswith('scale_corner_'):
            is_shift_pressed = Window.keyboard.is_pressed('shift')
            
            if is_shift_pressed:
                # Proportional scaling (e.g., around center or opposite corner)
                # Calculate scale_factor based on mouse movement from center/anchor
                # Apply uniform scale: self.scale = initial_scale * scale_factor
                # Ensure self.center or anchor point remains correct
                pass
            else:
                # Non-proportional scaling (FR2.6.2.1)
                # This is the most complex part. We need to calculate a new transform matrix.
                # Goal: dragged corner moves to touch.pos, opposite corner stays fixed on screen.
                
                # initial_fixed_corner_screen = self._initial_touch_data['screen_BL']
                # initial_dragged_corner_local = self._initial_touch_data['local_TR'] # Example
                # fixed_anchor_local = self._initial_touch_data['local_BL']       # Example

                # current_mouse_screen = touch.pos

                # 1. Calculate desired new position of dragged_corner_local in a *scaled-only* local space
                #    such that when this scaled local space is rotated and translated (to keep fixed_anchor_local
                #    at initial_fixed_corner_screen), the dragged_corner_local lands on current_mouse_screen.

                # This usually involves:
                # - Translating so the fixed_anchor_local is at the origin.
                # - Applying inverse of current rotation to align axes with screen.
                # - Calculating scale_x, scale_y based on mouse_pos relative to fixed_anchor.
                # - Building the new transform:
                #   T_fix_to_origin = Matrix().translate(-fixed_anchor_local[0], -fixed_anchor_local[1], 0)
                #   S_non_uniform = Matrix().scale(new_scale_x, new_scale_y, 1)
                #   R_current = Matrix().rotate(self.rotation, 0, 0, 1)
                #   T_place_fixed_corner = Matrix().translate(initial_fixed_corner_screen[0], initial_fixed_corner_screen[1], 0)
                #
                #   self.transform = T_place_fixed_corner * R_current * S_non_uniform * T_fix_to_origin
                #   (Order and exact calculation of new_scale_x/y need care)
                #
                #   A simpler (but potentially less robust or more complex to implement) view:
                #   - Treat the image as having a base size (e.g. texture size).
                #   - Determine the new effective width and height based on mouse drag from the fixed opposite corner,
                #     considering the current rotation.
                #   - Calculate scale_x_factor = new_eff_width / base_width, scale_y_factor = new_eff_height / base_height.
                #   - Construct a new transform matrix using these scale factors, the original rotation,
                #     and a new translation that keeps the fixed corner in place.
                #   - Set self.transform = new_matrix.
                #   This bypasses Scatter's own `scale` and `pos` properties for this operation,
                #   relying entirely on the `transform` matrix.
                print("Implement non-uniform scaling matrix manipulation here.")

        # self.update_controls_positions() # Redraw handles based on new transform
        return True
        
    # ... (on_touch_up, update_controls_positions etc.)
```

**说明:**

*   **形变实现的复杂性**: 特别是非等比例形变（拖拽角点，对角固定），在Kivy的`ScatterLayout`上直接通过`scale`, `rotation`, `pos`属性组合来实现会非常复杂且容易出错，尤其是当图像有旋转时。最稳健的方法是直接计算并设置`ScatterLayout`的`transform`属性（一个`kivy.graphics.transformation.Matrix`对象）。伪代码中对此进行了提示。
*   **控制点绘制**: 边界框和控制点需要使用Kivy的`canvas`指令（如`Line`, `Ellipse`）在`ImageLayerWidget`被选中时绘制。它们的位置必须根据`ImageLayerWidget`当前的`transform`动态计算，以确保它们总是准确地附着在形变后的图像边界和中心。

此设计为您提供了一个起点。在实际编码过程中，您会遇到更多细节问题，尤其是在变换逻辑和坐标转换方面。
