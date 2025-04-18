'''
MediaPipe 姿势估计演示程序

这是一个使用 MediaPipe 库进行人体姿势估计的 GUI 应用程序。主要功能包括：
1. 自动下载姿势检测模型（如果本地不存在）
2. 允许用户上传图像文件
3. 在图像中检测人体姿势关键点
4. 可视化显示关键点和骨架连接线
5. 提供用户友好的界面展示结果

主要组件：
- MediaPipe Pose Landmarker: 用于检测人体姿势关键点
- Tkinter: 构建图形用户界面
- OpenCV & PIL: 图像处理和显示

工作流程：
1. 程序启动时下载并初始化 MediaPipe 姿势检测模型
2. 用户通过按钮上传图像
3. 程序处理图像并检测姿势关键点
4. 在原始图像上绘制关键点和骨架线
5. 在界面上显示处理后的图像结果

技术细节：
- 支持各种常见图像格式
- 自动调整大尺寸图像以提高处理速度
- 使用半透明覆盖层显示检测结果
- 骨架显示包含主要关节连接
- 错误处理和用户反馈
'''

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import os
import numpy as np

def download_model():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("正在下载模型文件...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("模型文件下载完成")
    return model_path

def initialize_pose_detector(model_path):
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,  # 改为False以避免可能的兼容性问题
            min_pose_detection_confidence=0.5
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        print("姿势检测器初始化成功")
        return detector
    except Exception as e:
        print(f"初始化检测器时出错: {e}")
        messagebox.showerror("错误", f"初始化检测器时出错: {e}")
        return None

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        try:
            process_image(file_path)
        except Exception as e:
            print(f"处理图像时出错: {e}")
            messagebox.showerror("错误", f"处理图像时出错: {e}")

def process_image(file_path):
    print(f"正在处理图像: {file_path}")
    try:
        # 使用PIL读取图像，它能更好地处理中文路径
        pil_image = Image.open(file_path)
        # 转换为RGB模式（如果是RGBA，去掉alpha通道）
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        # 转换为OpenCV格式
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if image is None:
            print(f"无法读取图像文件: {file_path}")
            messagebox.showerror("错误", f"无法读取图像文件: {file_path}")
            return
    except Exception as e:
        print(f"读取图像时出错: {e}")
        messagebox.showerror("错误", f"读取图像时出错: {e}")
        return
        
    # 保存原始图像用于显示
    original_image = image.copy()
    
    # 确保图像不太大，以提高处理速度
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        print(f"图像已调整大小: {w}x{h} -> {int(w * scale)}x{int(h * scale)}")
    
    # 将图像转换为 RGB 格式 (MediaPipe 使用 RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建 MediaPipe Image 对象
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    try:
        # 运行姿势检测
        detection_result = detector.detect(mp_image)
        
        print(f"检测到的姿势: {len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0}")
        
        # 绘制关键点和连接线
        if detection_result.pose_landmarks:
            # 创建一个带有透明度的覆盖层
            overlay = image.copy()
            
            # 画出关键点
            for landmarks in detection_result.pose_landmarks:
                # 创建一个列表来保存所有关键点的位置
                landmark_points = []
                
                for idx, landmark in enumerate(landmarks):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmark_points.append((x, y))
                    
                    # 画出关键点
                    cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
                    
                    # 可选：显示关键点编号
                    # cv2.putText(overlay, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制骨架连接线（简化版）
                connections = [
                    # 躯干
                    (11, 12), (11, 23), (12, 24), (23, 24),
                    # 左臂
                    (11, 13), (13, 15),
                    # 右臂
                    (12, 14), (14, 16),
                    # 左腿
                    (23, 25), (25, 27),
                    # 右腿
                    (24, 26), (26, 28),
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        cv2.line(overlay, landmark_points[start_idx], landmark_points[end_idx], (255, 0, 0), 2)
            
            # 应用透明度
            alpha = 0.7
            output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        else:
            output = image
            print("未检测到姿势")
            messagebox.showinfo("信息", "未检测到姿势，请尝试不同的图像或角度")
        
        show_image(output)
        
    except Exception as e:
        print(f"姿势检测过程中出错: {e}")
        messagebox.showerror("错误", f"姿势检测失败: {e}")
        # 即使出错，也显示原始图像
        show_image(original_image)

def show_image(image):
    try:
        # 转换颜色空间
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        
        # 调整图像大小以适应窗口
        window_width = 800
        window_height = 600
        img_width, img_height = pil_image.size
        
        # 计算缩放比例
        scale = min(window_width / img_width, window_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 调整图像大小
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # 更新标签
        image_label.config(image=photo)
        # 保持引用以防止垃圾回收
        image_label.image = photo
        
        print("图像已显示")
        
    except Exception as e:
        print(f"显示图像时出错: {e}")
        messagebox.showerror("错误", f"显示图像时出错: {e}")

def main():
    global root, image_label, detector
    
    # 设置absl日志初始化
    from absl import logging
    logging.set_verbosity(logging.ERROR)
    
    # 下载并初始化模型
    model_path = download_model()
    detector = initialize_pose_detector(model_path)
    
    if detector is None:
        print("检测器初始化失败，程序无法继续")
        return
    
    # 创建GUI
    root = tk.Tk()
    root.title("MediaPipe 姿势估计演示")
    root.geometry("900x700")  # 设置初始窗口大小

    # 创建框架
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建上传按钮
    upload_button = tk.Button(frame, text="上传图片", command=upload_image, font=("Arial", 12))
    upload_button.pack(pady=10)

    # 创建图像显示标签，带有边框
    global image_label
    image_label = tk.Label(frame, borderwidth=2, relief="groove")
    image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建状态标签
    status_label = tk.Label(root, text="准备就绪。请上传图片...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # 启动主循环
    root.mainloop()

if __name__ == "__main__":
    main()
