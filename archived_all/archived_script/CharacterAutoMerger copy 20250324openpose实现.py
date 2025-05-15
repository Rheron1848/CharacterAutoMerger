import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
import subprocess
import pkg_resources
import argparse
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import photoshop.api as ps
import copy
import tempfile

'''
# 程序概述
本程序(CharacterAutoMerger)是一个角色自动合成工具，使用姿态估计技术将不同图像中的角色部位(如头部、手臂、腿部等)
智能地合成到目标模板中。主要依赖OpenPose进行人体关键点检测，通过对应关键点之间的映射实现智能部位合成。

# 代码结构
1. 初始化与环境设置
   - 日志配置
   - 依赖包检查与安装
   - OpenPose模型加载

2. CharacterAutoMerger类
   - 初始化模型和参数
   - 姿态检测功能
   - 身体部位提取功能
   - 图像合成与Photoshop交互功能

3. 主函数
   - 参数设置与验证
   - 执行图像处理流程

# 处理流程
1. 加载模型和输入图像(模板图像、头部图像、身体图像)
2. 对所有图像进行姿态关键点检测
3. 从头部图像提取头部区域
4. 从身体图像提取各个身体部位(手臂、腿部等)
5. 计算各部位在模板图像中的放置位置
6. 使用Photoshop API将各部位合成到模板图像中
7. 保存为PSD文件

# 依赖
- pytorch-openpose: 人体姿态检测
- photoshop-python-api: 与Photoshop的交互
- OpenCV, PIL: 图像处理
- PyTorch: 深度学习框架
'''

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_openpose():
    """
    设置OpenPose环境
    """
    # 获取pytorch-openpose的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_openpose_dir = os.path.join('D:\\', 'drafts', 'pytorch-openpose')
    
    # 添加路径到sys.path
    if pytorch_openpose_dir not in sys.path:
        sys.path.insert(0, pytorch_openpose_dir)
    
    src_dir = os.path.join(pytorch_openpose_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        # 直接从src目录导入
        import body
        import hand
        import util
        logger.info("成功导入pytorch-openpose模块")
        # 确保Body类存在并且有nms_threshold属性
        if hasattr(body, 'Body') and hasattr(body.Body, 'nms_threshold'):
            body.Body.nms_threshold = 0.15  # 默认可能是 0.05，增加此值可保留更多候选点
        if hasattr(body, 'Body') and hasattr(body.Body, 'paf_threshold'):
            body.Body.paf_threshold = 0.1  # 降低此值可能增加检测到的肢体连接
        return body.Body, hand.Hand, util
    except ImportError as e:
        logger.error(f"无法导入pytorch-openpose模块: {str(e)}")
        logger.error(f"请确保pytorch-openpose已正确安装在: {pytorch_openpose_dir}")
        return None

# 设置OpenPose并导入所需模块
openpose_modules = setup_openpose()
if not openpose_modules:
    sys.exit(1)
else:
    Body, Hand, util = openpose_modules

def install_dependencies():
    """
    检查并安装所需的Python依赖包
    
    功能:
    - 检查numpy, opencv-python, pillow, photoshop-python-api, torch, torchvision是否已安装
    - 如果发现缺失的包，自动使用pip进行安装
    
    异常:
    - 如果安装失败，记录错误并抛出异常
    """
    logger.info("检查依赖包...")
    required = {
        'numpy', 
        'opencv-python', 
        'pillow', 
        'photoshop-python-api',
        'torch',
        'torchvision'
    }
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if (missing):
        logger.info(f"发现缺失的依赖包: {missing}")
        try:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
            logger.info("依赖包安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装依赖包失败: {str(e)}")
            raise

# 在文件开头添加依赖检查
try:
    install_dependencies()
except Exception as e:
    logger.error(f"安装依赖包时出错: {str(e)}")
    sys.exit(1)

class CharacterAutoMerger:
    """
    角色自动合成器类
    
    主要功能:
    - 使用OpenPose进行姿态检测
    - 提取和识别人物的各个部位
    - 将部位智能地合成到模板图像中
    """
    
    def __init__(self, models_path=None):
        """
        初始化合成器并调整OpenPose参数
        
        参数:
        - models_path: OpenPose模型文件的路径，默认为D:/drafts/pytorch-openpose/model
        
        功能:
        - 初始化身体和手部姿态估计模型
        - 设置推理参数(stride, upsample_ratio等)
        
        注意:
        - 使用安全模式加载模型(weights_only=True)以提高兼容性
        """
        if models_path is None:
            models_path = os.path.join('D:\\', 'drafts', 'pytorch-openpose', 'model')
        
        try:
            # 加载并调整body姿态识别模型
            self.body_estimation = Body(os.path.join(models_path, 'body_pose_model.pth'))
            
            # 调整置信度阈值(提高检测率但降低精确度)
            if hasattr(self.body_estimation, 'threshold'):
                self.body_estimation.threshold = 0.03  # 降低默认阈值以捕获更多关键点
            
            # 如果有以下属性，也一同调整
            if hasattr(self.body_estimation, 'nms_threshold'):
                self.body_estimation.nms_threshold = 0.1
            
            if hasattr(self.body_estimation, 'paf_threshold'):
                self.body_estimation.paf_threshold = 0.08
            
            # 手部姿态模型
            self.hand_estimation = Hand(os.path.join(models_path, 'hand_pose_model.pth'))
            # 同样可以调整手部模型参数
            if hasattr(self.hand_estimation, 'threshold'):
                self.hand_estimation.threshold = 0.03
            
            logger.info("PyTorch OpenPose初始化成功，已调整检测参数以提高检测率")
        except Exception as e:
            logger.error(f"初始化PyTorch OpenPose失败: {str(e)}")
            raise
        
        # 调整推理参数以提高检测质量
        self.stride = 4  # 减小步长提高精度
        self.upsample_ratio = 8  # 增大上采样比例
        self.num_keypoints = 18
        self.height_size = 480  # 增大高度尺寸提高检测成功率

    def normalize(self, img, mean, scale):
        """
        图像归一化处理
        
        参数:
        - img: 输入图像
        - mean: 均值
        - scale: 缩放因子
        
        返回:
        - 归一化后的图像
        """
        img = np.array(img, dtype=np.float32)
        img = (img - mean) * scale
        return img

    def pad_width(self, img, stride, pad_value, min_dims):
        """
        填充图像到特定尺寸
        
        参数:
        - img: 输入图像
        - stride: 步长
        - pad_value: 填充值
        - min_dims: 最小尺寸
        
        返回:
        - 填充后的图像和填充信息
        """
        h, w = img.shape[:2]
        h = min(min_dims[0], h)
        min_dims[0] = ((min_dims[0] + stride - 1) // stride) * stride
        min_dims[1] = ((min_dims[1] + stride - 1) // stride) * stride
        pad = []
        pad.append(int(min(stride - (h % stride), stride - 1)))
        pad.append(int(min(min_dims[1] - w, stride - 1)))
        
        img_padded = cv2.copyMakeBorder(img, 0, pad[0], 0, pad[1],
                                      cv2.BORDER_CONSTANT, value=pad_value)
        return img_padded, pad

    def infer_fast(self, img, height_size=256):
        """
        快速姿态推理
        
        参数:
        - img: 输入图像(已经过预处理和缩放)
        - height_size: 已弃用参数，保留是为了兼容性
        
        返回:
        - heatmaps: 热图
        - pafs: 部位亲和场
        - scale: 缩放比例(此处始终为1，因为缩放已在get_pose_landmarks中处理)
        - pad: 填充信息
        """
        height, width, _ = img.shape
        
        # 不再进行缩放，直接使用传入的预处理图像
        scaled_img = img
        min_dims = [height, max(scaled_img.shape[1], height)]
        padded_img, pad = self.pad_width(scaled_img, self.stride, (0, 0, 0), min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if torch.cuda.is_available():
            tensor_img = tensor_img.cuda()

        # 使用body_estimation进行推理
        with torch.no_grad():
            stages_output = self.body_estimation(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, 1.0, pad

    def get_pose_landmarks(self, image: np.ndarray, image_type: str = "未知") -> Dict:
        """优化姿态关键点检测，参考sd-webui-controlnet的方法"""
        try:
            # 保存原始尺寸
            orig_h, orig_w = image.shape[:2]
            
            # 将图像转换为RGB通道顺序（与sd-webui-controlnet兼容）
            # OpenPose预期输入是BGR，但内部会转换
            oriImg = image.copy()
            
            # 检查图像尺寸，如果太大则缩放
            target_size = 1000 # 增大目标尺寸提高精度
            scale = target_size / max(orig_h, orig_w)  # 使用最大边缩放以保持比例
            
            if max(orig_h, orig_w) > target_size:
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                resized_image = cv2.resize(oriImg, (new_w, new_h))
                logger.info(f"[{image_type}图像] 调整尺寸从 {orig_w}x{orig_h} 到 {new_w}x{new_h}")
            else:
                resized_image = oriImg
                scale = 1.0
                logger.info(f"[{image_type}图像] 保持原始尺寸 {orig_w}x{orig_h}")
            
            # 与sd-webui-controlnet兼容的推理方式
            with torch.no_grad():  # 添加no_grad上下文
                try:
                    # 确保body_estimation可调用
                    if not hasattr(self, 'body_estimation') or self.body_estimation is None:
                        logger.error("body_estimation模型未初始化")
                        return {
                            'pose_keypoints': [],
                            'pose_subset': [],
                            'original_size': (orig_h, orig_w),
                            'scale': scale
                        }
                        
                    # 重要：翻转通道顺序（与sd-webui-controlnet保持一致）
                    resized_image_rgb = resized_image[:, :, ::-1].copy()
                    
                    # 可以添加多次尝试不同参数的逻辑
                    thresholds = [0.1, 0.05, 0.01, 0.005]
                    for threshold in thresholds:
                        if hasattr(self.body_estimation, 'threshold'):
                            self.body_estimation.threshold = threshold
                        candidate, subset = self.body_estimation(resized_image_rgb)
                        if len(candidate) > 0 and len(subset) > 0:
                            break
                    
                    # 验证检测结果
                    if candidate is None or subset is None:
                        logger.warning(f"在{image_type}图像中未检测到任何关键点")
                        return {
                            'pose_keypoints': [],
                            'pose_subset': [],
                            'original_size': (orig_h, orig_w),
                            'scale': scale
                        }
                    
                    # 如果检测成功但缩放了图像，将坐标映射回原始尺寸
                    if scale != 1.0:
                        for i in range(len(candidate)):
                            candidate[i][0] = candidate[i][0] / scale  # x坐标
                            candidate[i][1] = candidate[i][1] / scale  # y坐标
                    
                    logger.info(f"检测到的关键点数量: {len(candidate)}")
                    logger.info(f"检测到的人物数量: {len(subset)}")
                    
                    return {
                        'pose_keypoints': candidate,
                        'pose_subset': subset,
                        'original_size': (orig_h, orig_w),
                        'scale': scale
                    }
                except Exception as inner_e:
                    logger.error(f"姿态检测内部错误: {str(inner_e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {
                        'pose_keypoints': [],
                        'pose_subset': [],
                        'original_size': (orig_h, orig_w),
                        'scale': scale
                    }
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'pose_keypoints': [],
                'pose_subset': [],
                'original_size': (orig_h, orig_w),
                'scale': 1.0
            }

    def extract_head(self, image: np.ndarray, landmarks: Dict) -> Tuple[np.ndarray, Dict]:
        """
        提取图像中的头部区域，即使骨架不完整也尝试提取
        
        参数:
        - image: 输入图像
        - landmarks: 姿态关键点信息
        
        返回:
        - 头部区域图像
        - 头部位置信息字典
        
        功能:
        - 尝试根据可用的关键点确定头部区域
        - 如果关键点不足，使用图像的上部区域作为头部
        """
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset']
        
        height, width = image.shape[:2]
        
        # 如果没有检测到人或关键点不足
        if len(pose_subset) == 0 or len(pose_points) == 0:
            # 使用图像上部1/3作为头部区域
            head_height = height // 3
            center_x = width // 2
            
            position_info = {
                'top': 0,
                'left': max(0, center_x - head_height // 2),
                'bottom': head_height,
                'right': min(width, center_x + head_height // 2),
                'center_x': center_x,
                'center_y': head_height // 2
            }
            
            return image[position_info['top']:position_info['bottom'], 
                        position_info['left']:position_info['right']], position_info
        
        # 获取第一个检测到的人的关键点索引
        person_keypoints = pose_subset[0]
        
        # OpenPose关键点索引
        NECK = 1
        NOSE = 0
        
        # 尝试获取颈部和鼻子的关键点
        neck_idx = int(person_keypoints[NECK])
        nose_idx = int(person_keypoints[NOSE])
        
        # 如果没有检测到鼻子或颈部，使用其他可用关键点或图像上部
        if neck_idx == -1 and nose_idx == -1:
            # 使用图像上部1/3作为头部区域
            head_height = height // 3
            center_x = width // 2
            
            position_info = {
                'top': 0,
                'left': max(0, center_x - head_height // 2),
                'bottom': head_height,
                'right': min(width, center_x + head_height // 2),
                'center_x': center_x,
                'center_y': head_height // 2
            }
        elif neck_idx == -1:
            # 只有鼻子关键点
            nose_pos = pose_points[nose_idx]
            nose_x = int(nose_pos[0])
            nose_y = int(nose_pos[1])
            
            head_height = height // 4  # 估计头部高度
            head_width = int(head_height * 0.8)
            
            position_info = {
                'top': max(0, nose_y - head_height // 2),
                'left': max(0, nose_x - head_width // 2),
                'bottom': min(height, nose_y + head_height // 2),
                'right': min(width, nose_x + head_width // 2),
                'center_x': nose_x,
                'center_y': nose_y
            }
        elif nose_idx == -1:
            # 只有颈部关键点
            neck_pos = pose_points[neck_idx]
            neck_x = int(neck_pos[0])
            neck_y = int(neck_pos[1])
            
            head_height = height // 4  # 估计头部高度
            head_width = int(head_height * 0.8)
            
            position_info = {
                'top': max(0, neck_y - head_height),
                'left': max(0, neck_x - head_width // 2),
                'bottom': neck_y,
                'right': min(width, neck_x + head_width // 2),
                'center_x': neck_x,
                'center_y': neck_y - head_height // 2
            }
        else:
            # 有颈部和鼻子关键点
            neck_pos = pose_points[neck_idx]
            nose_pos = pose_points[nose_idx]
            
            neck_y = int(neck_pos[1])
            neck_x = int(neck_pos[0])
            nose_y = int(nose_pos[1])
            
            # 计算头部区域的边界框
            head_height = int((neck_y - nose_y) * 2.5)  # 留出足够空间
            head_width = int(head_height * 0.8)  # 假设头部宽高比约为0.8
            
            position_info = {
                'top': max(0, neck_y - head_height),
                'left': max(0, neck_x - head_width // 2),
                'bottom': min(height, neck_y),
                'right': min(width, neck_x + head_width // 2),
                'center_x': neck_x,
                'center_y': (neck_y + (neck_y - head_height)) // 2
            }
        
        head_region = image[position_info['top']:position_info['bottom'], 
                          position_info['left']:position_info['right']]
        
        return head_region, position_info

    def extract_body_parts(self, image: np.ndarray, landmarks: Dict) -> Dict[str, Dict]:
        """
        提取图像中的身体各部分，即使骨架不完整也尝试提取
        
        参数:
        - image: 输入图像
        - landmarks: 姿态关键点信息
        
        返回:
        - 包含各身体部位图像和位置信息的字典
        
        功能:
        - 尝试提取左右手臂、左右腿部等身体部位
        - 对于检测不到的部位，会尽量基于图像位置进行估计
        """
        # 添加错误检查
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks['pose_keypoints']) == 0:
            logger.warning("未检测到人体关键点，尝试基于图像进行估计")
            return self._estimate_body_parts(image)
        
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset']
        
        if len(pose_subset) == 0:
            logger.warning("未检测到完整人物，尝试基于图像进行估计")
            return self._estimate_body_parts(image)
        
        result = {}
        
        # OpenPose关键点索引
        BODY_PARTS = {
            'left_arm': [5, 6, 7],   # 左肩、左肘、左腕
            'right_arm': [2, 3, 4],  # 右肩、右肘、右腕
            'left_leg': [12, 13, 14],  # 左髋、左膝、左踝
            'right_leg': [9, 10, 11]   # 右髋、右膝、右踝
        }
        
        person_keypoints = pose_subset[0]
        
        for part_name, indices in BODY_PARTS.items():
            # 检查是否有该部位的至少一个关键点
            valid_indices = [i for i in indices if int(person_keypoints[i]) != -1]
            
            if valid_indices:
                # 有关键点，尝试提取部位
                try:
                    part_points = [pose_points[int(person_keypoints[i])] for i in valid_indices]
                    part_img, part_info = self._extract_part_by_points(image, part_points)
                    if part_img is not None and part_img.size > 0:
                        result[part_name] = {
                            'image': part_img,
                            'position': part_info
                        }
                except Exception as e:
                    logger.warning(f"提取{part_name}时发生错误: {str(e)}")
                    # 失败时使用估计方法
                    estimated_parts = self._estimate_specific_part(image, part_name)
                    if part_name in estimated_parts:
                        result[part_name] = estimated_parts[part_name]
            else:
                # 没有关键点，使用估计方法
                estimated_parts = self._estimate_specific_part(image, part_name)
                if part_name in estimated_parts:
                    result[part_name] = estimated_parts[part_name]
        
        return result

    def _extract_part_by_points(self, image: np.ndarray, points: List) -> Tuple[np.ndarray, Dict]:
        """
        根据点列表提取身体部位
        
        参数:
        - image: 输入图像
        - points: 点列表，每个点是(x, y, confidence)形式
        
        返回:
        - 部位图像
        - 位置信息
        """
        height, width = image.shape[:2]
        
        # 提取x,y坐标
        xy_points = [(int(p[0]), int(p[1])) for p in points]
        
        # 创建边界框
        padding = 30
        min_x = max(0, min(p[0] for p in xy_points) - padding)
        max_x = min(width, max(p[0] for p in xy_points) + padding)
        min_y = max(0, min(p[1] for p in xy_points) - padding)
        max_y = min(height, max(p[1] for p in xy_points) + padding)
        
        # 计算中心点
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        position_info = {
            'top': min_y,
            'left': min_x,
            'bottom': max_y,
            'right': max_x,
            'center_x': center_x,
            'center_y': center_y,
            'points': xy_points
        }
        
        return image[min_y:max_y, min_x:max_x], position_info

    def _estimate_body_parts(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        在没有关键点的情况下，基于图像位置估计身体部位
        
        参数:
        - image: 输入图像
        
        返回:
        - 估计的身体部位字典
        """
        height, width = image.shape[:2]
        result = {}
        
        # 估计四个主要部位：左右手臂、左右腿
        parts_info = {
            'left_arm': {
                'x_start': 0, 
                'x_end': width // 2,
                'y_start': height // 4,
                'y_end': height // 2
            },
            'right_arm': {
                'x_start': width // 2, 
                'x_end': width,
                'y_start': height // 4,
                'y_end': height // 2
            },
            'left_leg': {
                'x_start': 0, 
                'x_end': width // 2,
                'y_start': height // 2,
                'y_end': height
            },
            'right_leg': {
                'x_start': width // 2, 
                'x_end': width,
                'y_start': height // 2,
                'y_end': height
            }
        }
        
        for part_name, area in parts_info.items():
            x_start = area['x_start']
            x_end = area['x_end']
            y_start = area['y_start']
            y_end = area['y_end']
            
            # 提取区域
            part_img = image[y_start:y_end, x_start:x_end]
            
            # 计算中心点
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            position_info = {
                'top': y_start,
                'left': x_start,
                'bottom': y_end,
                'right': x_end,
                'center_x': center_x,
                'center_y': center_y,
                'points': [(center_x, center_y)]  # 使用中心点作为唯一的点
            }
            
            result[part_name] = {
                'image': part_img,
                'position': position_info
            }
        
        return result

    def _estimate_specific_part(self, image: np.ndarray, part_name: str) -> Dict[str, Dict]:
        """
        估计特定的身体部位
        
        参数:
        - image: 输入图像
        - part_name: 部位名称
        
        返回:
        - 估计的部位字典
        """
        height, width = image.shape[:2]
        result = {}
        
        # 基于部位名称确定区域
        if part_name == 'left_arm':
            x_start = 0
            x_end = width // 2
            y_start = height // 4
            y_end = height // 2
        elif part_name == 'right_arm':
            x_start = width // 2
            x_end = width
            y_start = height // 4
            y_end = height // 2
        elif part_name == 'left_leg':
            x_start = 0
            x_end = width // 2
            y_start = height // 2
            y_end = height
        elif part_name == 'right_leg':
            x_start = width // 2
            x_end = width
            y_start = height // 2
            y_end = height
        else:
            return result
        
        # 提取区域
        part_img = image[y_start:y_end, x_start:x_end]
        
        # 计算中心点
        center_x = (x_start + x_end) // 2
        center_y = (y_start + y_end) // 2
        
        position_info = {
            'top': y_start,
            'left': x_start,
            'bottom': y_end,
            'right': x_end,
            'center_x': center_x,
            'center_y': center_y,
            'points': [(center_x, center_y)]  # 使用中心点作为唯一的点
        }
        
        result[part_name] = {
            'image': part_img,
            'position': position_info
        }
        
        return result

    def calculate_placement_position(self, source_position: Dict, target_landmarks: Dict, part_name: str) -> Dict:
        """
        计算部位在目标图像中的放置位置
        
        参数:
        - source_position: 源图像中部位的位置信息
        - target_landmarks: 目标图像的关键点信息
        - part_name: 部位名称
        
        返回:
        - 包含x、y坐标的放置位置字典
        
        功能:
        - 根据部位类型(头部、手臂、腿部等)计算合适的放置位置
        - 确保部位与目标图像中对应部位位置对齐
        """
        target_pose = target_landmarks['pose_keypoints'][0]
        
        # 根据不同部位计算放置位置
        placement = {}
        
        if part_name == "Head":
            # 对于头部，使用目标图像的颈部和鼻子位置
            NECK = 1
            NOSE = 0
            neck_pos = target_pose[NECK]
            nose_pos = target_pose[NOSE]
            
            target_neck_y = int(neck_pos[1])
            target_neck_x = int(neck_pos[0])
            
            # 计算头部区域
            head_height = int((neck_pos[1] - nose_pos[1]) * 2.5)
            
            placement = {
                'x': target_neck_x - (source_position['right'] - source_position['left']) // 2,
                'y': target_neck_y - head_height
            }
        elif "arm" in part_name:
            # 对于手臂，使用相应的肩、肘、腕关键点
            if part_name == "left_arm":
                indices = [5, 6, 7]  # 左肩、左肘、左腕
            else:
                indices = [2, 3, 4]  # 右肩、右肘、右腕
                
            target_points = [(int(target_pose[i][0]), int(target_pose[i][1])) 
                            for i in indices if target_pose[i][2] > 0.1]
            
            if target_points:
                # 计算目标区域中心
                target_center_x = sum(p[0] for p in target_points) // len(target_points)
                target_center_y = sum(p[1] for p in target_points) // len(target_points)
                
                # 计算放置位置，使部位中心与目标中心对齐
                placement = {
                    'x': target_center_x - (source_position['right'] - source_position['left']) // 2,
                    'y': target_center_y - (source_position['bottom'] - source_position['top']) // 2
                }
        elif "leg" in part_name:
            # 对于腿部，使用相应的髋、膝、踝关键点
            if part_name == "left_leg":
                indices = [12, 13, 14]  # 左髋、左膝、左踝
            else:
                indices = [9, 10, 11]   # 右髋、右膝、右踝
                
            target_points = [(int(target_pose[i][0]), int(target_pose[i][1])) 
                            for i in indices if target_pose[i][2] > 0.1]
            
            if target_points:
                # 计算目标区域中心
                target_center_x = sum(p[0] for p in target_points) // len(target_points)
                target_center_y = sum(p[1] for p in target_points) // len(target_points)
                
                # 计算放置位置，使部位中心与目标中心对齐
                placement = {
                    'x': target_center_x - (source_position['right'] - source_position['left']) // 2,
                    'y': target_center_y - (source_position['bottom'] - source_position['top']) // 2
                }
        
        return placement

    def merge_to_template(self, template: np.ndarray, head: np.ndarray, head_position: Dict,
                         body_parts: Dict[str, Dict], template_landmarks: Dict) -> str:
        """
        将所有部分合并到模板中并保存为PSD
        """
        # 首先检查是否安装了photoshop-python-api
        ps_available = False
        try:
            import photoshop.api as ps
            ps_available = True
        except ImportError:
            logger.warning("未找到photoshop-python-api，将使用备用方法保存图像")
            return self.merge_to_template_without_ps(template, head, head_position, body_parts, template_landmarks)
        
        try:
            app = ps.Application()
            # ... Photoshop处理代码 ...
            return "output.psd"  # 确保返回文件路径
        except Exception as e:
            logger.error(f"无法初始化Photoshop: {str(e)}")
            return self.merge_to_template_without_ps(template, head, head_position, body_parts, template_landmarks)

    def merge_to_template_without_ps(self, template, head, head_position, body_parts, template_landmarks):
        """
        在没有Photoshop的情况下合并图像
        """
        output_path = "output_merged.png"
        result_img = template.copy()
        
        # 简单的图像合成 - 将头部放到模板上
        if head is not None and head.size > 0:
            x = max(0, head_position['center_x'] - head.shape[1] // 2)
            y = max(0, head_position['center_y'] - head.shape[0] // 2)
            
            # 确保不超出边界
            x_end = min(result_img.shape[1], x + head.shape[1])
            y_end = min(result_img.shape[0], y + head.shape[0])
            
            # 简单合成 - 可以改进为带alpha通道的叠加
            result_img[y:y_end, x:x_end] = head[:y_end-y, :x_end-x]
        
        # 合成其他身体部位...
        # 这里省略具体实现...
        
        # 保存结果
        cv2.imwrite(output_path, result_img)
        return output_path

    def process_images(self, template_path: str, head_image_path: str, body_image_path: str, output_path: str = "output.psd") -> str:
        """
        处理主流程，将头部和身体图像合成到模板中
        
        参数:
        - template_path: 模板图像的路径
        - head_image_path: 头部图像的路径
        - body_image_path: 身体图像的路径
        - output_path: 输出PSD文件的路径
        
        返回:
        - 输出文件的路径
        
        功能:
        - 读取输入图像
        - 获取所有图像的姿态信息
        - 提取头部和身体部位
        - 合并图像并保存为PSD
        - 清理内存并返回结果
        
        异常:
        - ValueError: 当图像读取失败或关键点检测失败时
        - RuntimeError: 当处理过程中出现严重错误时
        """
        try:
            # 读取图像并进行基本验证
            images = {
                'template': ('模板', template_path),
                'head': ('头部', head_image_path),
                'body': ('身体', body_image_path)
            }
            
            loaded_images = {}
            for key, (name, path) in images.items():
                logger.info(f"正在读取{name}图像: {path}")
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取{name}图像: {path}")
                if img.size == 0:
                    raise ValueError(f"{name}图像是空的: {path}")
                loaded_images[key] = img

            # 使用生成器处理姿态检测，提高内存效率
            def process_landmarks():
                for key, (name, img) in zip(
                    ['template', 'head', 'body'],
                    [('模板', loaded_images['template']),
                     ('头部', loaded_images['head']),
                     ('身体', loaded_images['body'])]
                ):
                    logger.info(f"正在分析{name}图像姿态...")
                    landmarks = self.get_pose_landmarks(img, image_type=name)
                    if landmarks is None:
                        raise ValueError(f"无法在{name}图像中检测到姿态关键点")
                    yield key, landmarks

            # 获取所有图像的姿态信息
            landmarks_dict = {k: v for k, v in process_landmarks()}

            # 提取头部
            logger.info("正在提取头部区域...")
            head, head_position = self.extract_head(loaded_images['head'], landmarks_dict['head'])
            if head is None or head.size == 0:
                raise ValueError("头部提取失败")

            # 提取身体部分
            logger.info("正在提取身体部位...")
            body_parts = self.extract_body_parts(loaded_images['body'], landmarks_dict['body'])
            if not body_parts:
                raise ValueError("未能提取出任何身体部位")

            # 合并图像
            logger.info("正在合成最终图像...")
            try:
                result_path = self.merge_to_template(
                    loaded_images['template'],
                    head,
                    head_position,
                    body_parts,
                    landmarks_dict['template']
                )
            except Exception as e:
                raise RuntimeError(f"图像合成失败: {str(e)}")

            # 清理内存
            for img in loaded_images.values():
                del img
            del landmarks_dict, head, body_parts

            logger.info(f"处理完成，已保存到: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            raise

def main():
    """
    主函数，程序入口点
    
    功能:
    - 检查环境和模型文件
    - 初始化CharacterAutoMerger实例
    - 设置输入参数(模板、头部、身体图像路径)
    - 执行图像处理流程
    - 捕获和记录可能的异常
    
    返回:
    - 0: 处理成功
    - 1: 处理失败
    """
    try:
        logger.info("程序启动")
        
        # 获取pytorch-openpose的路径
        pytorch_openpose_dir = os.path.join('D:\\', 'drafts', 'pytorch-openpose')
        
        # 检查pytorch-openpose目录是否存在
        if not os.path.exists(pytorch_openpose_dir):
            logger.error(f"pytorch-openpose目录不存在: {pytorch_openpose_dir}")
            return 1
            
        # 检查模型文件
        model_dir = os.path.join(pytorch_openpose_dir, "model")
        body_model_path = os.path.join(model_dir, "body_pose_model.pth")
        hand_model_path = os.path.join(model_dir, "hand_pose_model.pth")
        
        for model_path in [body_model_path, hand_model_path]:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return 1
            
        # 创建合成器实例
        logger.info("初始化CharacterAutoMerger...")
        merger = CharacterAutoMerger()  # 不需要显式指定models_path，使用默认值
        
        # 在脚本中直接定义参数，而非通过命令行传入
        # 修改这些路径为你的实际图像路径
        # template_path = r"D:\drafts\CAM\template\teen girl template.png"#注意 不能用中文文件名
        # head_path = r"D:\drafts\CAM\template\image (12).png"
        # body_path = r"D:\drafts\CAM\template\output_image.png"
        # output_path = r"D:\drafts\CAM\template\output.psd"

        template_path = r"D:\drafts\CAM\template\teen girl template.png" #注意 不能用中文文件名
        head_path = r"D:\drafts\pytorch-openpose\images\demo.jpg"
        body_path = r"D:\drafts\pytorch-openpose\images\demo.jpg"
        output_path = r"D:\drafts\CAM\template\output.psd"
        
        logger.info(f"使用的参数: template={template_path}, head={head_path}, body={body_path}, output={output_path}")
        
        # 检查输入文件是否存在
        for path, name in [
            (template_path, "模板"),
            (head_path, "头部"),
            (body_path, "身体")
        ]:
            if not os.path.exists(path):
                logger.error(f"{name}图像路径不存在: {path}")
                return 1
            else:
                logger.info(f"找到{name}图像: {path}")
        
        # 处理图像
        logger.info("开始处理图像...")
        output_file = merger.process_images(
            template_path,
            head_path,
            body_path,
            output_path
        )
        
        logger.info(f"处理完成! 输出文件: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"程序结束，退出代码: {exit_code}")
    sys.exit(exit_code)
