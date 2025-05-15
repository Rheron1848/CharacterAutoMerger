
#不用mediapipe了。效果不好，所以这个脚本暂且不用了

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
import subprocess
import pkg_resources
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

def install_dependencies():
    """安装所需的Python依赖包"""
    required = {'numpy', 'opencv-python', 'pillow', 'photoshop-python-api', 'mediapipe'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print("正在安装缺失的依赖包...")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

# 在文件开头添加依赖检查
try:
    install_dependencies()
except Exception as e:
    print(f"安装依赖包时出错: {str(e)}")
    sys.exit(1)

from PIL import Image
import photoshop.api as ps

class CharacterAutoMerger:
    def __init__(self):
        # 初始化MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
    def get_pose_landmarks(self, image: np.ndarray) -> Dict:
        """使用MediaPipe获取图像的姿态关键点"""
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("未能检测到姿态关键点")
            
        # 转换关键点格式
        pose_points = np.array([[point.x * image.shape[1], point.y * image.shape[0], point.visibility] 
                               for point in results.pose_landmarks.landmark])
        
        face_points = None
        if results.face_landmarks:
            face_points = np.array([[point.x * image.shape[1], point.y * image.shape[0], point.visibility]
                                   for point in results.face_landmarks.landmark])
        
        left_hand = right_hand = None
        if results.left_hand_landmarks:
            left_hand = np.array([[point.x * image.shape[1], point.y * image.shape[0], point.visibility]
                                 for point in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            right_hand = np.array([[point.x * image.shape[1], point.y * image.shape[0], point.visibility]
                                  for point in results.right_hand_landmarks.landmark])
            
        return {
            'pose_keypoints': pose_points,
            'face_keypoints': face_points,
            'hand_keypoints': [left_hand, right_hand]
        }

    def extract_head(self, image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """提取颈部以上区域"""
        pose_points = landmarks['pose_keypoints']
        # MediaPipe姿态关键点索引
        NECK = self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value  # 使用右肩作为颈部参考点
        NOSE = self.mp_holistic.PoseLandmark.NOSE.value
        
        # 获取颈部位置
        neck_pos = pose_points[NECK]
        nose_pos = pose_points[NOSE]
        
        height, width = image.shape[:2]
        neck_y = int(neck_pos[1])
        neck_x = int(neck_pos[0])
        
        # 计算头部区域的边界框
        head_height = int((neck_pos[1] - nose_pos[1]) * 2.5)  # 留出足够空间
        head_width = int(head_height * 0.8)  # 假设头部宽高比约为0.8
        
        # 提取头部区域
        top = max(0, neck_y - head_height)
        left = max(0, neck_x - head_width // 2)
        bottom = min(height, neck_y)
        right = min(width, neck_x + head_width // 2)
        
        head_region = image[top:bottom, left:right]
        return head_region

    def extract_body_parts(self, image: np.ndarray, landmarks: Dict) -> Dict[str, np.ndarray]:
        """提取身体各部分"""
        pose_points = landmarks['pose_keypoints']
        body_parts = {}
        
        # MediaPipe姿态关键点索引
        BODY_PARTS = {
            'left_arm': [
                self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_holistic.PoseLandmark.LEFT_ELBOW.value,
                self.mp_holistic.PoseLandmark.LEFT_WRIST.value
            ],
            'right_arm': [
                self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_holistic.PoseLandmark.RIGHT_WRIST.value
            ],
            'left_leg': [
                self.mp_holistic.PoseLandmark.LEFT_HIP.value,
                self.mp_holistic.PoseLandmark.LEFT_KNEE.value,
                self.mp_holistic.PoseLandmark.LEFT_ANKLE.value
            ],
            'right_leg': [
                self.mp_holistic.PoseLandmark.RIGHT_HIP.value,
                self.mp_holistic.PoseLandmark.RIGHT_KNEE.value,
                self.mp_holistic.PoseLandmark.RIGHT_ANKLE.value
            ]
        }
        
        for part_name, indices in BODY_PARTS.items():
            body_parts[part_name] = self._extract_limb(image, pose_points, indices)
            
        return body_parts

    def _extract_limb(self, image: np.ndarray, pose_points: np.ndarray, keypoint_indices: List) -> Optional[np.ndarray]:
        """提取单个肢体部分"""
        height, width = image.shape[:2]
        points = []
        
        for idx in keypoint_indices:
            point = pose_points[idx]
            if point[2] > 0.1:  # 检查置信度
                x = int(point[0])
                y = int(point[1])
                points.append((x, y))
                
        if not points:
            return np.array([])  # 返回空数组而不是None
            
        # 创建肢体周围的边界框
        padding = 30
        min_x = max(0, min(p[0] for p in points) - padding)
        max_x = min(width, max(p[0] for p in points) + padding)
        min_y = max(0, min(p[1] for p in points) - padding)
        max_y = min(height, max(p[1] for p in points) + padding)
        
        return image[min_y:max_y, min_x:max_x]

    def merge_to_template(self, template: np.ndarray, head: np.ndarray, 
                         body_parts: Dict[str, np.ndarray], template_landmarks: Dict) -> str:
        """将所有部分合并到模板中并保存为PSD"""
        try:
            app = ps.Application()
        except Exception as e:
            print("无法初始化Photoshop. 请确保已安装Photoshop并配置了photoshop-python-api")
            raise e

        # 创建新文档
        doc = app.documents.add(template.shape[1], template.shape[0])
        
        # 添加模板图层
        template_layer = doc.artLayers.add()
        template_layer.name = "Template"
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template_pil = Image.fromarray(template_rgb)
        doc.paste(template_pil, (0, 0))  # 指定粘贴位置
        
        # 添加头部图层
        if head is not None and head.size > 0:
            head_layer = doc.artLayers.add()
            head_layer.name = "Head"
            head_rgb = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
            head_pil = Image.fromarray(head_rgb)
            # 使用模板关键点计算头部位置
            head_pos = self._calculate_head_position(template_landmarks)
            doc.paste(head_pil, head_pos)
        
        # 添加身体部分
        for part_name, part_image in body_parts.items():
            if part_image is not None and part_image.size > 0:
                layer = doc.artLayers.add()
                layer.name = part_name
                part_rgb = cv2.cvtColor(part_image, cv2.COLOR_BGR2RGB)
                part_pil = Image.fromarray(part_rgb)
                # 使用模板关键点计算部件位置
                part_pos = self._calculate_part_position(part_name, template_landmarks)
                doc.paste(part_pil, part_pos)
        
        # 保存PSD文件
        output_path = "output.psd"
        doc.save(output_path)  # 修改为save()方法
        doc.close()
        
        return output_path

    def _calculate_head_position(self, template_landmarks: Dict) -> Tuple[int, int]:
        """计算头部在模板中的位置"""
        pose_points = template_landmarks['pose_keypoints']
        neck_pos = pose_points[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        return (int(neck_pos[0]), int(neck_pos[1]))

    def _calculate_part_position(self, part_name: str, template_landmarks: Dict) -> Tuple[int, int]:
        """计算身体部件在模板中的位置"""
        pose_points = template_landmarks['pose_keypoints']
        if part_name == 'left_arm':
            point = pose_points[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        elif part_name == 'right_arm':
            point = pose_points[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        elif part_name == 'left_leg':
            point = pose_points[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        else:  # right_leg
            point = pose_points[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]
        return (int(point[0]), int(point[1]))

    def process_images(self, template_path: str, head_image_path: str, body_image_path: str) -> str:
        """处理主流程"""
        # 读取图像
        template = cv2.imread(template_path)
        head_image = cv2.imread(head_image_path)
        body_image = cv2.imread(body_image_path)
        
        # 获取姿态信息
        template_landmarks = self.get_pose_landmarks(template)
        head_landmarks = self.get_pose_landmarks(head_image)
        body_landmarks = self.get_pose_landmarks(body_image)
        
        # 提取头部
        head = self.extract_head(head_image, head_landmarks)
        
        # 提取身体部分
        body_parts = self.extract_body_parts(body_image, body_landmarks)
        
        # 合并并保存
        return self.merge_to_template(template, head, body_parts, template_landmarks)

class CharacterAutoMergerGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("角色自动合成工具")
        self.window.geometry("600x400")
        
        # 文件路径变量
        self.template_path = tk.StringVar()
        self.head_path = tk.StringVar()
        self.body_path = tk.StringVar()
        
        self._create_widgets()
        
    def _create_widgets(self):
        # 创建文件选择区域
        frame = tk.Frame(self.window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 模板图片选择
        tk.Label(frame, text="模板图片:").grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Entry(frame, textvariable=self.template_path, width=50).grid(row=0, column=1, pady=5)
        tk.Button(frame, text="选择文件", command=lambda: self._select_file(self.template_path)).grid(row=0, column=2, padx=5, pady=5)
        
        # 头部图片选择
        tk.Label(frame, text="头部图片:").grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Entry(frame, textvariable=self.head_path, width=50).grid(row=1, column=1, pady=5)
        tk.Button(frame, text="选择文件", command=lambda: self._select_file(self.head_path)).grid(row=1, column=2, padx=5, pady=5)
        
        # 身体图片选择
        tk.Label(frame, text="身体图片:").grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Entry(frame, textvariable=self.body_path, width=50).grid(row=2, column=1, pady=5)
        tk.Button(frame, text="选择文件", command=lambda: self._select_file(self.body_path)).grid(row=2, column=2, padx=5, pady=5)
        
        # 开始处理按钮
        tk.Button(frame, text="开始处理", command=self._process_images, width=20).grid(row=3, column=0, columnspan=3, pady=20)
        
        # 状态显示
        self.status_label = tk.Label(frame, text="", wraplength=500)
        self.status_label.grid(row=4, column=0, columnspan=3, pady=10)
        
    def _select_file(self, path_var):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            path_var.set(filename)
            
    def _process_images(self):
        # 检查文件是否都已选择
        if not all([self.template_path.get(), self.head_path.get(), self.body_path.get()]):
            messagebox.showerror("错误", "请选择所有需要的图片文件")
            return
            
        try:
            self.status_label.config(text="正在处理中...")
            self.window.update()
            
            merger = CharacterAutoMerger()
            output_path = merger.process_images(
                self.template_path.get(),
                self.head_path.get(),
                self.body_path.get()
            )
            
            self.status_label.config(text=f"处理完成！输出文件已保存到：{output_path}")
            messagebox.showinfo("成功", f"处理完成！\n输出文件已保存到：{output_path}")
        except Exception as e:
            self.status_label.config(text=f"处理失败：{str(e)}")
            messagebox.showerror("错误", f"处理过程中出错：\n{str(e)}")
            
    def run(self):
        self.window.mainloop()

# 修改主程序入口
if __name__ == "__main__":
    try:
        install_dependencies()
        app = CharacterAutoMergerGUI()
        app.run()
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        sys.exit(1)
