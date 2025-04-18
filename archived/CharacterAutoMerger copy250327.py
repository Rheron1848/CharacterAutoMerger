import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
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
from psd_tools import PSDImage
from psd_tools.constants import ColorMode
import copy
import tempfile
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules
from mmpose.structures import merge_data_samples
from mmdet.structures import DetDataSample
from psd_tools.api.layers import Layer, Group


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentManager:
    """环境管理类，负责依赖安装和环境初始化"""
    
    BASIC_DEPENDENCIES = {
        'numpy', 
        'opencv-python', 
        'pillow', 
        'photoshop-python-api',
        'torch',
        'torchvision'
    }
    
    DWPOSE_DEPENDENCIES = {
        'mmengine>=0.7.1',
        'mmcv>=2.0.0',
        'mmdet>=3.0.0',
        'mmpose>=1.1.0'
    }
    
    MODEL_FILES = {
        'rtmdet': {
            'name': 'rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
        },
        'rtmpose': {
            'name': 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        }
    }

    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose"):
        self.dwpose_dir = dwpose_dir
        self.weights_dir = os.path.join(dwpose_dir, 'weights')
        
    def setup(self) -> Dict[str, str]:
        """完整的环境设置流程"""
        try:
            self._check_and_install_dependencies()
            self._ensure_directories()
            self._download_model_files()
            self._register_modules()
            return self._get_model_paths()
        except Exception as e:
            logger.error(f"环境设置失败: {str(e)}")
            raise

    def _check_and_install_dependencies(self) -> None:
        """检查并安装依赖"""
        logger.info("检查依赖包...")
        installed = {pkg.key for pkg in pkg_resources.working_set}
        
        # 检查基础依赖
        missing_basic = self.BASIC_DEPENDENCIES - installed
        if missing_basic:
            self._install_packages(missing_basic, "基础依赖")
            
        # 安装DWPose依赖
        self._install_packages(self.DWPOSE_DEPENDENCIES, "DWPose依赖")

    def _install_packages(self, packages: Set[str], package_type: str) -> None:
        """安装指定的包"""
        logger.info(f"开始安装{package_type}...")
        python = sys.executable
        
        for pkg in packages:
            logger.info(f"正在安装 {pkg}...")
            try:
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                result = subprocess.run(
                    [python, '-m', 'pip', 'install', pkg, '--progress-bar', 'on'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    check=True  # 这会在安装失败时抛出异常
                )
                logger.info(f"{pkg} 安装成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"{pkg} 安装失败: {e.stderr}")
                raise

    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        os.makedirs(self.weights_dir, exist_ok=True)

    def _download_model_files(self) -> None:
        """下载模型文件"""
        for model_type, info in self.MODEL_FILES.items():
            model_path = os.path.join(self.weights_dir, info['name'])
            if not os.path.exists(model_path):
                logger.info(f"正在下载{model_type}模型...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(info['url'], model_path)
                    logger.info(f"{model_type}模型下载完成")
                except Exception as e:
                    logger.error(f"下载{model_type}模型失败: {str(e)}")
                    logger.info(f"请手动下载模型文件: {info['url']}")
                    logger.info(f"并将其放置在: {model_path}")
                    raise

    def _register_modules(self) -> None:
        """注册必要的模块"""
        register_det_modules()
        register_pose_modules()

    def _get_model_paths(self) -> Dict[str, str]:
        """获取模型路径配置"""
        return {
            'det_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
            ),
            'det_checkpoint': os.path.join(
                self.weights_dir,
                self.MODEL_FILES['rtmdet']['name']
            ),
            'pose_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            ),
            'pose_checkpoint': os.path.join(
                self.weights_dir,
                self.MODEL_FILES['rtmpose']['name']
            )
        }

class Config:
    """基础配置类"""
    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose"):
        self.dwpose_dir = dwpose_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = 0.3
        self.num_keypoints = 17
        self.model_paths = self._init_model_paths()
        self._register_modules()

    def _init_model_paths(self) -> Dict[str, str]:
        """初始化模型路径"""
        weights_dir = os.path.join(self.dwpose_dir, 'weights')
        return {
            'det_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
            ),
            'det_checkpoint': os.path.join(
                weights_dir,
                'rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
            ),
            'pose_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            ),
            'pose_checkpoint': os.path.join(
                weights_dir,
                'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            )
        }

    def _register_modules(self) -> None:
        """注册必要的模块"""
        register_det_modules()
        register_pose_modules()

class BodyPartExtractor:
    """身体部位提取器，负责从图像中提取各个身体部位"""
    
    # 定义身体部位及其对应的关键点索引
    # 关键点索引：0鼻子，1颈部，2右肩，3右肘，4右腕，5左肩，6左肘，7左腕，
    # 8右髋，9右膝，10右踝，11左髋，12左膝，13左踝，14右眼，15左眼，16右耳，17左耳
    BODY_PARTS_KEYPOINTS = {
        # 颈部
        'neck': [0, 1],  # 鼻子和颈部之间
        
        # 躯干部分
        'upper_chest': [1, 2, 5],  # 颈部和双肩
        'chest_waist': [2, 5, 8, 11],  # 双肩和双髋
        'abdomen_hips': [8, 11],  # 双髋
        
        # 手臂部分
        'left_upper_arm': [5, 6],  # 左肩到左肘
        'left_lower_arm': [6, 7],  # 左肘到左腕
        'left_hand': [7],  # 左腕(加扩展)
        'right_upper_arm': [2, 3],  # 右肩到右肘
        'right_lower_arm': [3, 4],  # 右肘到右腕
        'right_hand': [4],  # 右腕(加扩展)
        
        # 腿部部分
        'left_thigh': [11, 12],  # 左髋到左膝
        'left_calf': [12, 13],  # 左膝到左踝
        'left_foot': [13],  # 左踝(加扩展)
        'right_thigh': [8, 9],  # 右髋到右膝
        'right_calf': [9, 10],  # 右膝到右踝
        'right_foot': [10]  # 右踝(加扩展)
    }
    
    # 各部位的默认位置（用于无关键点情况下）
    DEFAULT_PART_POSITIONS = None  # 将在__init__中根据图像尺寸动态设置
    
    # 特殊部位的处理方法
    SPECIAL_PARTS_HANDLERS = {
        'left_hand': '_extend_left_hand',
        'right_hand': '_extend_right_hand',
        'left_foot': '_extend_left_foot',
        'right_foot': '_extend_right_foot',
        'upper_chest': '_extend_trunk',
        'chest_waist': '_extend_trunk',
        'abdomen_hips': '_extend_trunk'
    }
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def extract_head(self, image: np.ndarray, landmarks: Dict) -> Tuple[np.ndarray, Dict]:
        """提取头部区域"""
        height, width = image.shape[:2]
        
        # 检查是否有有效的关键点
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks.get('pose_subset', [])) == 0:
            print("⚠️ 未检测到人体关键点，估计头部位置")
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
            
            head_region = image[position_info['top']:position_info['bottom'], 
                          position_info['left']:position_info['right']]
            print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}]")
            
            return head_region, position_info
        
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset'][0]  # 取第一个检测到的人
        
        # 获取关键点索引
        NECK = 1
        NOSE = 0
        
        neck_idx = int(pose_subset[NECK])
        nose_idx = int(pose_subset[NOSE])
        
        # 根据可用关键点计算头部位置
        if neck_idx == -1 and nose_idx == -1:
            # 无关键点，使用默认位置
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
            
            head_height = height // 4
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
            
            head_height = height // 4
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
            
            neck_x = int(neck_pos[0])
            neck_y = int(neck_pos[1])
            nose_y = int(nose_pos[1])
            
            # 计算头部区域
            head_height = int((neck_pos[1] - nose_pos[1]) * 2.5)
            head_width = int(head_height * 0.8)
            
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
        
        print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}]")
        
        return head_region, position_info
    
    def extract_body_parts(self, image: np.ndarray, landmarks: Dict) -> Dict[str, Dict]:
        """提取全部身体部位"""
        height, width = image.shape[:2]
        self._init_default_positions(width, height)
        
        # 检查是否有有效的关键点
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks.get('pose_subset', [])) == 0:
            print("⚠️ 未检测到人体关键点，使用默认位置估计")
            return self._estimate_all_parts(image)
        
        result = {}
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset'][0]  # 取第一个检测到的人
        
        # 逐个提取各部位
        for part_name, keypoint_indices in self.BODY_PARTS_KEYPOINTS.items():
            # 过滤有效关键点
            valid_indices = [i for i in keypoint_indices if int(pose_subset[i]) != -1]
            
            if valid_indices:
                # 有有效关键点，根据关键点提取
                try:
                    # 获取关键点坐标
                    part_points = [pose_points[int(pose_subset[i])] for i in valid_indices]
                    
                    # 处理特殊部位
                    if part_name in self.SPECIAL_PARTS_HANDLERS:
                        handler_method = getattr(self, self.SPECIAL_PARTS_HANDLERS[part_name])
                        part_points = handler_method(part_points)
                    
                    # 提取部位图像和位置信息
                    part_img, position = self._extract_by_points(image, part_points, part_name)
                    
                    if part_img is not None and part_img.size > 0:
                        result[part_name] = {
                            'image': part_img,
                            'position': position
                        }
                        print(f"✅ 成功提取{part_name}: 位置=[{position['left']},{position['top']},{position['right']},{position['bottom']}]")
                        
                except Exception as e:
                    print(f"❌ 提取{part_name}失败: {str(e)}")
                    # 使用默认位置
                    part_result = self._estimate_part(image, part_name)
                    if part_name in part_result:
                        result[part_name] = part_result[part_name]
            else:
                # 无有效关键点，使用默认位置
                print(f"⚠️ 未检测到{part_name}的关键点，使用默认位置")
                part_result = self._estimate_part(image, part_name)
                if part_name in part_result:
                    result[part_name] = part_result[part_name]
        
        return result

    def _init_default_positions(self, width, height):
        """初始化默认部位位置"""
        # 确保DEFAULT_PART_POSITIONS被正确初始化为字典而不是None
        self.DEFAULT_PART_POSITIONS = {
            # 颈部
            'neck': {
                'x_start': width // 3, 
                'x_end': 2 * width // 3,
                'y_start': height // 8,
                'y_end': height // 5
            },
            # 躯干
            'upper_chest': {
                'x_start': width // 4, 
                'x_end': 3 * width // 4,
                'y_start': height // 5,
                'y_end': height // 3
            },
            'chest_waist': {
                'x_start': width // 4, 
                'x_end': 3 * width // 4,
                'y_start': height // 3,
                'y_end': height // 2
            },
            'abdomen_hips': {
                'x_start': width // 4, 
                'x_end': 3 * width // 4,
                'y_start': height // 2,
                'y_end': 3 * height // 5
            },
            # 手臂
            'left_upper_arm': {
                'x_start': 0, 
                'x_end': width // 3,
                'y_start': height // 5,
                'y_end': 2 * height // 5
            },
            'left_lower_arm': {
                'x_start': 0, 
                'x_end': width // 4,
                'y_start': 2 * height // 5,
                'y_end': height // 2
            },
            'left_hand': {
                'x_start': 0, 
                'x_end': width // 6,
                'y_start': height // 2,
                'y_end': 3 * height // 5
            },
            'right_upper_arm': {
                'x_start': 2 * width // 3, 
                'x_end': width,
                'y_start': height // 5,
                'y_end': 2 * height // 5
            },
            'right_lower_arm': {
                'x_start': 3 * width // 4, 
                'x_end': width,
                'y_start': 2 * height // 5,
                'y_end': height // 2
            },
            'right_hand': {
                'x_start': 5 * width // 6, 
                'x_end': width,
                'y_start': height // 2,
                'y_end': 3 * height // 5
            },
            # 腿部
            'left_thigh': {
                'x_start': width // 6, 
                'x_end': width // 2,
                'y_start': 3 * height // 5,
                'y_end': 3 * height // 4
            },
            'left_calf': {
                'x_start': width // 6, 
                'x_end': width // 2,
                'y_start': 3 * height // 4,
                'y_end': 9 * height // 10
            },
            'left_foot': {
                'x_start': width // 6, 
                'x_end': width // 2,
                'y_start': 9 * height // 10,
                'y_end': height
            },
            'right_thigh': {
                'x_start': width // 2, 
                'x_end': 5 * width // 6,
                'y_start': 3 * height // 5,
                'y_end': 3 * height // 4
            },
            'right_calf': {
                'x_start': width // 2, 
                'x_end': 5 * width // 6,
                'y_start': 3 * height // 4,
                'y_end': 9 * height // 10
            },
            'right_foot': {
                'x_start': width // 2, 
                'x_end': 5 * width // 6,
                'y_start': 9 * height // 10,
                'y_end': height
            }
        }

    def _estimate_all_parts(self, image: np.ndarray) -> Dict[str, Dict]:
        """估计全部身体部位的位置"""
        result = {}
        
        for part_name in self.BODY_PARTS_KEYPOINTS.keys():
            part_result = self._estimate_part(image, part_name)
            if part_name in part_result:
                result[part_name] = part_result[part_name]
                print(f"⚠️ 使用默认位置估计{part_name}: 位置=[{result[part_name]['position']['left']},{result[part_name]['position']['top']},{result[part_name]['position']['right']},{result[part_name]['position']['bottom']}]")
        
        return result
    
    def _estimate_part(self, image: np.ndarray, part_name: str) -> Dict[str, Dict]:
        """估计特定部位的位置"""
        height, width = image.shape[:2]
        result = {}
        
        # 确保 DEFAULT_PART_POSITIONS 已初始化
        if self.DEFAULT_PART_POSITIONS is None:
            self._init_default_positions(width, height)
        
        # 使用字典的 get 方法安全地访问
        if part_name in self.DEFAULT_PART_POSITIONS:
            area = self.DEFAULT_PART_POSITIONS[part_name]
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
    
    def _extract_by_points(self, image: np.ndarray, points: List, part_name: str = None) -> Tuple[np.ndarray, Dict]:
        """根据关键点提取部位，考虑肢体方向"""
        height, width = image.shape[:2]
        
        # 提取x,y坐标
        xy_points = [(int(p[0]), int(p[1])) for p in points]
        
        # 基础填充比例
        base_padding = 0.2
        
        # 计算边界框的宽度和高度
        if len(xy_points) > 1:
            x_coords = [p[0] for p in xy_points]
            y_coords = [p[1] for p in xy_points]
            box_width = max(x_coords) - min(x_coords)
            box_height = max(y_coords) - min(y_coords)
            
            # 计算肢体方向向量（适用于手臂、腿部等）
            if part_name and ('arm' in part_name or 'thigh' in part_name or 'calf' in part_name):
                # 对于肢体部位，我们需要考虑其延伸方向
                if len(xy_points) >= 2:
                    # 简单地使用起点和终点来确定方向
                    start_point = xy_points[0]
                    end_point = xy_points[-1]
                    
                    # 计算方向向量
                    direction_x = end_point[0] - start_point[0]
                    direction_y = end_point[1] - start_point[1]
                    
                    # 向量长度
                    vector_length = (direction_x**2 + direction_y**2)**0.5
                    
                    if vector_length > 0:
                        # 归一化方向向量
                        direction_x /= vector_length
                        direction_y /= vector_length
                        
                        # 延伸系数 - 用于手和脚等需要额外空间的部位
                        extension = 0
                        if 'hand' in part_name or 'foot' in part_name:
                            extension = box_width * 0.5 if 'hand' in part_name else box_height * 0.3
                        
                        # 沿方向延伸边界框
                        if 'right' in part_name:
                            # 右侧肢体通常向右延伸
                            extra_x = max(int(extension * direction_x), 0)
                            extra_y = int(extension * direction_y)
                        else:
                            # 左侧肢体通常向左延伸
                            extra_x = min(int(extension * direction_x), 0)
                            extra_y = int(extension * direction_y)
                        
                        # 调整边界框，考虑方向
                        padding_x = max(int(box_width * base_padding), 10)
                        padding_y = max(int(box_height * base_padding), 10)
                        
                        min_x = max(0, min(x_coords) - padding_x + (extra_x if extra_x < 0 else 0))
                        max_x = min(width, max(x_coords) + padding_x + (extra_x if extra_x > 0 else 0))
                        min_y = max(0, min(y_coords) - padding_y + (extra_y if extra_y < 0 else 0))
                        max_y = min(height, max(y_coords) + padding_y + (extra_y if extra_y > 0 else 0))
                    else:
                        # 默认填充
                        padding_x = max(int(box_width * base_padding), 10)
                        padding_y = max(int(box_height * base_padding), 10)
                        
                        min_x = max(0, min(x_coords) - padding_x)
                        max_x = min(width, max(x_coords) + padding_x)
                        min_y = max(0, min(y_coords) - padding_y)
                        max_y = min(height, max(y_coords) + padding_y)
                else:
                    # 默认填充
                    padding_x = max(int(box_width * base_padding), 10)
                    padding_y = max(int(box_height * base_padding), 10)
                    
                    min_x = max(0, min(x_coords) - padding_x)
                    max_x = min(width, max(x_coords) + padding_x)
                    min_y = max(0, min(y_coords) - padding_y)
                    max_y = min(height, max(y_coords) + padding_y)
            else:
                # 对于其他部位，使用基本填充
                padding_x = max(int(box_width * base_padding), 10)
                padding_y = max(int(box_height * base_padding), 10)
                
                min_x = max(0, min(x_coords) - padding_x)
                max_x = min(width, max(x_coords) + padding_x)
                min_y = max(0, min(y_coords) - padding_y)
                max_y = min(height, max(y_coords) + padding_y)
        else:
            # 如果只有一个点，使用固定填充
            padding = 30
            min_x = max(0, xy_points[0][0] - padding)
            max_x = min(width, xy_points[0][0] + padding)
            min_y = max(0, xy_points[0][1] - padding)
            max_y = min(height, xy_points[0][1] + padding)
        
        # 确保区域有效（宽度和高度都大于0）
        if min_x >= max_x or min_y >= max_y:
            print(f"⚠️ 警告: 提取的区域无效 [{min_x}, {min_y}, {max_x}, {max_y}]，调整为最小有效区域")
            if min_x >= max_x:
                max_x = min(width, min_x + 1)
            if min_y >= max_y:
                max_y = min(height, min_y + 1)
        
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
        
    # 特殊部位处理方法
    def _extend_left_hand(self, points: List) -> List:
        """扩展左手区域"""
        if len(points) == 1:
            wrist_point = points[0]
            points.append([wrist_point[0] - 30, wrist_point[1] + 30, wrist_point[2]])
        return points
    
    def _extend_right_hand(self, points: List) -> List:
        """扩展右手区域"""
        if len(points) == 1:
            wrist_point = points[0]
            points.append([wrist_point[0] + 30, wrist_point[1] + 30, wrist_point[2]])
        return points
    
    def _extend_left_foot(self, points: List) -> List:
        """扩展左脚区域"""
        if len(points) == 1:
            ankle_point = points[0]
            points.append([ankle_point[0] - 30, ankle_point[1] + 30, ankle_point[2]])
        return points
    
    def _extend_right_foot(self, points: List) -> List:
        """扩展右脚区域"""
        if len(points) == 1:
            ankle_point = points[0]
            points.append([ankle_point[0] + 30, ankle_point[1] + 30, ankle_point[2]])
        return points
    
    def _extend_trunk(self, points: List) -> List:
        """扩展躯干区域"""
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            min_x = min(x_coords)
            max_x = max(x_coords)
            avg_y = sum(y_coords) / len(y_coords)
            
            width_extension = (max_x - min_x) * 0.2
            points.append([min_x - width_extension, avg_y, 1.0])
            points.append([max_x + width_extension, avg_y, 1.0])
        
        return points

class PositionCalculator:
    """位置计算器，负责计算身体部位在目标图像中的放置位置"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 共享同一套关键点索引定义
        self.BODY_PARTS_KEYPOINTS = BodyPartExtractor.BODY_PARTS_KEYPOINTS
        
    def calculate_placement_position(self, source_position: Dict, target_landmarks: Dict, part_name: str) -> Dict:
        """计算部位在目标图像中的放置位置"""
        # 检查目标关键点是否有效
        if not target_landmarks or 'pose_keypoints' not in target_landmarks or len(target_landmarks.get('pose_subset', [])) == 0:
            return {
                'x': source_position.get('left', 0),
                'y': source_position.get('top', 0)
            }
        
        pose_points = target_landmarks['pose_keypoints']
        pose_subset = target_landmarks['pose_subset'][0]  # 取第一个检测到的人
        
        # 处理头部特殊情况
        if part_name == "Head":
            return self._calculate_head_placement(source_position, pose_points, pose_subset)
        
        # 获取部位对应的关键点索引
        if part_name not in self.BODY_PARTS_KEYPOINTS:
            return {
                'x': source_position.get('left', 0),
                'y': source_position.get('top', 0)
            }
            
        keypoint_indices = self.BODY_PARTS_KEYPOINTS[part_name]
        
        # 过滤有效关键点
        valid_keypoints = []
        for idx in keypoint_indices:
            keypoint_idx = int(pose_subset[idx])
            if keypoint_idx != -1:
                keypoint = pose_points[keypoint_idx]
                if keypoint[2] > 0.1:  # 关键点置信度大于0.1
                    valid_keypoints.append((int(keypoint[0]), int(keypoint[1])))
        
        # 如果有有效关键点，计算放置位置
        if valid_keypoints:
            # 计算中心位置
            avg_x = sum(p[0] for p in valid_keypoints) // len(valid_keypoints)
            avg_y = sum(p[1] for p in valid_keypoints) // len(valid_keypoints)
            
            # 计算部位宽高
            part_width = source_position['right'] - source_position['left']
            part_height = source_position['bottom'] - source_position['top']
            
            # 基本放置位置
            placement = {
                'x': avg_x - part_width // 2,
                'y': avg_y - part_height // 2
            }
            
            # 根据部位类型进行特殊处理
            placement = self._adjust_placement(placement, part_name, part_width, part_height, avg_x, avg_y)
            
            return placement
        
        # 没有有效关键点，使用默认位置
        return {
            'x': source_position.get('left', 0),
            'y': source_position.get('top', 0)
        }
    
    def _calculate_head_placement(self, source_position: Dict, pose_points: List, pose_subset: List) -> Dict:
        """计算头部放置位置"""
        # 获取鼻子和颈部关键点
        NOSE = 0
        NECK = 1
        
        nose_idx = int(pose_subset[NOSE])
        neck_idx = int(pose_subset[NECK])
        
        if nose_idx != -1 and neck_idx != -1:
            nose_pos = pose_points[nose_idx]
            neck_pos = pose_points[neck_idx]
            
            target_neck_y = int(neck_pos[1])
            target_neck_x = int(neck_pos[0])
            
            # 计算头部区域
            head_height = int((neck_pos[1] - nose_pos[1]) * 2.5)
            
            return {
                'x': target_neck_x - (source_position['right'] - source_position['left']) // 2,
                'y': target_neck_y - head_height
            }
        
        # 没有有效关键点，使用默认位置
        return {
            'x': source_position.get('left', 0),
            'y': source_position.get('top', 0)
        }
    
    def _adjust_placement(self, placement: Dict, part_name: str, part_width: int, part_height: int, avg_x: int, avg_y: int) -> Dict:
        """根据部位类型调整放置位置"""
        if part_name == 'neck':
            # 颈部应该放在颈部关键点上方
            placement['y'] = avg_y - part_height * 0.7
        elif part_name == 'left_hand':
            # 左手应该放在左腕外侧
            placement['x'] = avg_x - part_width * 0.3
            placement['y'] = avg_y - part_height * 0.3
        elif part_name == 'right_hand':
            # 右手应该放在右腕外侧
            placement['x'] = avg_x - part_width * 0.7
            placement['y'] = avg_y - part_height * 0.3
        elif part_name == 'left_foot':
            # 左脚应该放在左踝下方
            placement['y'] = avg_y - part_height * 0.2
        elif part_name == 'right_foot':
            # 右脚应该放在右踝下方
            placement['y'] = avg_y - part_height * 0.2
        
        return placement

class PoseDetector:
    """姿态检测器，负责检测人体姿态关键点"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.detector = None
        self.pose_estimator = None
        self._init_models()
        
    def _init_models(self):
        """初始化检测和姿态估计模型"""
        try:
            # 1. 先初始化姿态估计模型
            register_pose_modules()  # 注册姿态模块
            import mmpose.datasets   # 导入姿态数据集
            
            self.pose_estimator = init_pose_estimator(
                self.config.model_paths['pose_config'],
                self.config.model_paths['pose_checkpoint'],
                device=self.config.device
            )
            
            # 2. 然后初始化检测器模型
            register_det_modules()   # 注册检测模块
            import mmdet.datasets    # 导入检测数据集
            
            self.detector = init_detector(
                self.config.model_paths['det_config'],
                self.config.model_paths['det_checkpoint'],
                device=self.config.device
            )
            
            self.logger.info("姿态检测模型初始化成功")
        except Exception as e:
            self.logger.error(f"姿态检测模型初始化失败: {str(e)}")
            raise
    
    def detect_pose(self, image: np.ndarray, image_type: str = "") -> Dict:
        """检测图像中的人体姿态关键点"""
        try:
            # 确保图像格式正确
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                print(f"{image_type}图像为灰度图像，已转换为BGR图像")
            
            # 确保模型已初始化
            if self.detector is None or self.pose_estimator is None:
                self.logger.error("模型未正确初始化")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 1. 使用检测器前，注册检测器相关模块
            register_det_modules()  # 在调用检测器前注册检测模块
            import mmdet.datasets    # 确保检测相关的transform被导入
            
            # 使用检测器获取人体边界框
            det_result = inference_detector(self.detector, image)
            
            # 检查结果并处理边界情况
            if not det_result or not hasattr(det_result, 'pred_instances'):
                self.logger.warning(f"{image_type}图像未检测到人体")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            if not hasattr(det_result.pred_instances, 'bboxes') or det_result.pred_instances.bboxes.shape[0] == 0:
                self.logger.warning(f"{image_type}图像未检测到人体边界框")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 获取预测实例中的边界框
            bboxes = det_result.pred_instances.bboxes.cpu().numpy()
            scores = det_result.pred_instances.scores.cpu().numpy()
            
            # 仅保留得分高于阈值的人体
            valid_indices = scores > self.config.confidence_threshold
            bboxes = bboxes[valid_indices]
            
            if len(bboxes) == 0:
                self.logger.warning(f"{image_type}图像未检测到高置信度人体")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 2. 使用姿态估计器前，注册姿态估计相关模块
            register_pose_modules()  # 在调用姿态估计器前注册姿态模块
            import mmpose.datasets   # 确保姿态相关的transform被导入
            
            # 进行姿态估计
            pose_results = inference_topdown(self.pose_estimator, image, bboxes)
            
            if not pose_results:
                self.logger.warning(f"{image_type}图像未检测到人体姿态")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 安全地获取关键点信息
            try:
                pred_instance = pose_results[0].pred_instances
                
                # 提取关键点和分数
                keypoints = []
                keypoint_scores = []
                
                if hasattr(pred_instance, 'keypoints'):
                    if torch.is_tensor(pred_instance.keypoints):
                        keypoints = pred_instance.keypoints[0].cpu().numpy()
                    else:
                        keypoints = pred_instance.keypoints[0]
                        
                    if hasattr(pred_instance, 'keypoint_scores'):
                        if torch.is_tensor(pred_instance.keypoint_scores):
                            keypoint_scores = pred_instance.keypoint_scores[0].cpu().numpy()
                        else:
                            keypoint_scores = pred_instance.keypoint_scores[0]
                    else:
                        keypoint_scores = np.ones(len(keypoints))
                    
                    # 记录日志
                    num_valid_keypoints = sum(1 for score in keypoint_scores if score > self.config.confidence_threshold)
                    self.logger.info(f"{image_type}图像检测到 {num_valid_keypoints}/{len(keypoints)} 个有效关键点")
                
                return {
                    'pose_keypoints': [
                        [float(kpt[0]), float(kpt[1]), float(score)]
                        for kpt, score in zip(keypoints, keypoint_scores)
                        if score > self.config.confidence_threshold
                    ],
                    'pose_subset': [[i for i in range(len(keypoints))]],
                    'original_size': (image.shape[0], image.shape[1]),
                    'scale': 1.0
                }
                
            except Exception as inner_e:
                self.logger.warning(f"处理关键点数据时出错: {str(inner_e)}")
                return self._empty_result((image.shape[0], image.shape[1]))
            
        except Exception as e:
            self.logger.error(f"姿态检测失败: {str(e)}")
            self.logger.exception(e)  # 添加完整的堆栈跟踪以便更好地调试
            return self._empty_result((image.shape[0], image.shape[1]))
    
    def _empty_result(self, image_size: Tuple[int, int]) -> Dict:
        """返回空结果"""
        return {
            'pose_keypoints': [],
            'pose_subset': [],
            'original_size': image_size,
            'scale': 1.0
        }

class PartExporter:
    """身体部位导出器，负责将拆分的身体部位导出到指定文件夹"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def export_parts(self, head: np.ndarray, head_position: Dict, 
                     body_parts: Dict, export_dir: str, include_position_json: bool = True) -> str:
        """
        将头部和身体部位导出到指定目录
        
        参数:
            head: 头部图像
            head_position: 头部位置信息
            body_parts: 身体部位字典 {部位名称: {'image': 图像, 'position': 位置信息}}
            export_dir: 导出目录
            include_position_json: 是否导出位置信息JSON文件
            
        返回:
            导出目录路径
        """
        try:
            # 确保导出目录存在
            os.makedirs(export_dir, exist_ok=True)
            self.logger.info(f"将部位导出到目录: {export_dir}")
            
            # 检查头部图像是否为空
            if head is None or head.size == 0:
                print(f"⚠️ 警告: 头部图像为空，跳过保存")
                self.logger.warning("头部图像为空，跳过保存")
            else:
                # 导出头部
                head_path = os.path.join(export_dir, 'head.png')
                cv2.imwrite(head_path, head)
                print(f"✅ 导出头部: {head_path}")
            
            # 导出各身体部位
            parts_info = {}
            if head is not None and head.size > 0:
                parts_info['head'] = {
                    'path': 'head.png',
                    'size': [head.shape[1], head.shape[0]],
                    'position': head_position
                }
            
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                
                # 检查部位图像是否为空
                if part_img is None or part_img.size == 0:
                    print(f"⚠️ 警告: {part_name}图像为空，跳过保存")
                    self.logger.warning(f"{part_name}图像为空，跳过保存")
                    continue
                    
                part_path = os.path.join(export_dir, f'{part_name}.png')
                cv2.imwrite(part_path, part_img)
                print(f"✅ 导出{part_name}: {part_path}")
                
                # 记录部位信息
                parts_info[part_name] = {
                    'path': f'{part_name}.png',
                    'size': [part_img.shape[1], part_img.shape[0]],
                    'position': part_info['position']
                }
            
            # 导出位置信息JSON文件
            if include_position_json and parts_info:
                # 将不可序列化的numpy数组转换为列表
                for part, info in parts_info.items():
                    if 'points' in info['position']:
                        info['position']['points'] = [
                            [int(p[0]), int(p[1])] for p in info['position']['points']
                        ]
                
                json_path = os.path.join(export_dir, 'parts_position.json')
                import json
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(parts_info, f, indent=2, ensure_ascii=False)
                print(f"✅ 导出位置信息: {json_path}")
            
            return export_dir
            
        except Exception as e:
            self.logger.error(f"导出部位失败: {str(e)}")
            raise

class PSDMerger:
    """PSD文件合成器，负责将拆分的身体部位合成为PSD文件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def merge_to_psd(self, template: np.ndarray, head: np.ndarray, head_position: Dict, 
                     body_parts: Dict, template_landmarks: Dict, output_path: str) -> str:
        """
        使用PSD-Tools合成图像
        
        参数:
            template: 模板图像
            head: 头部图像
            head_position: 头部位置信息
            body_parts: 身体部位字典 {部位名称: {'image': 图像, 'position': 位置信息}}
            template_landmarks: 模板图像的姿态关键点
            output_path: 输出PSD文件路径
            
        返回:
            输出PSD文件路径
        """
        try:
            # 检查是否可以使用PSD-Tools
            self._check_psd_tools()
            
            from psd_tools import PSDImage
            from psd_tools.constants import ColorMode
            from PIL import Image
            
            # 创建临时目录存放图像文件
            temp_dir = tempfile.gettempdir()
            
            # 创建新的PSD文档 - 使用正确的参数
            # PSDImage.new('RGB', (宽, 高))
            psd = PSDImage.new('RGB', (template.shape[1], template.shape[0]))
            
            # 添加模板图层
            template_path = os.path.join(temp_dir, 'template.png')
            # 转换BGR到RGB并保存临时文件
            cv2.imwrite(template_path, cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            template_img = Image.open(template_path)
            
            # 创建模板图层组和子图层
            template_layer = psd.create_group("Template")
            template_sub = psd.create_artboard("TemplateLayer", (0, 0, template.shape[1], template.shape[0]))
            template_sub.paste(template_img, (0, 0))
            template_layer.append(template_sub)
            
            # 添加头部图层
            head_path = os.path.join(temp_dir, 'head.png')
            cv2.imwrite(head_path, cv2.cvtColor(head, cv2.COLOR_BGR2RGB))
            head_img = Image.open(head_path)
            
            # 创建头部位置计算器
            position_calculator = PositionCalculator(None)  # 传入None，因为我们只需要计算功能
            
            # 计算头部放置位置
            head_placement = position_calculator.calculate_placement_position(
                head_position, template_landmarks, "Head"
            )
            
            # 添加头部图层
            head_layer = psd.create_artboard("Head", 
                (int(head_placement.get('x', 0)), 
                 int(head_placement.get('y', 0)),
                 int(head_placement.get('x', 0)) + head.shape[1],
                 int(head_placement.get('y', 0)) + head.shape[0]))
            head_layer.paste(head_img, (0, 0))
            
            # 添加身体部位图层
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                part_position = part_info['position']
                
                # 保存临时图像文件
                part_path = os.path.join(temp_dir, f'{part_name}.png')
                cv2.imwrite(part_path, cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB))
                part_pil_img = Image.open(part_path)
                
                # 计算部位放置位置
                placement = position_calculator.calculate_placement_position(
                    part_position, template_landmarks, part_name
                )
                
                # 添加图层
                part_layer = psd.create_artboard(part_name,
                    (int(placement.get('x', 0)),
                     int(placement.get('y', 0)),
                     int(placement.get('x', 0)) + part_img.shape[1],
                     int(placement.get('y', 0)) + part_img.shape[0]))
                part_layer.paste(part_pil_img, (0, 0))
            
            # 保存PSD文件
            psd.save(output_path)
            
            # 清理临时文件
            for temp_file in [template_path, head_path] + [os.path.join(temp_dir, f'{part}.png') for part in body_parts]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"PSD合成失败: {str(e)}")
            raise
    
    def _check_psd_tools(self) -> bool:
        """检查是否可以使用PSD-Tools"""
        try:
            import psd_tools
            return True
        except ImportError:
            error_msg = "未检测到PSD-Tools，无法进行PSD合成"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

class CharacterAutoMerger:
    """角色自动合成器"""
    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose", export_dir: str = r"D:\drafts\CAM\export"):
        self.config = Config(dwpose_dir)
        self.pose_detector = PoseDetector(self.config)
        self.body_part_extractor = BodyPartExtractor(self.config)
        self.position_calculator = PositionCalculator(self.config)
        self.part_exporter = PartExporter()  # 实例化部位导出器
        self.psd_merger = PSDMerger()  # 实例化PSD合成器
        self.export_dir = export_dir  # 设置默认导出目录
        logger.info("CharacterAutoMerger 初始化成功")
    
    def set_export_dir(self, export_dir: str) -> None:
        """设置导出目录"""
        self.export_dir = export_dir
        
    def process_images(self, template_path: str, head_path: str, body_path: str, output_path: str = "output.psd") -> str:
        """处理图像并合成PSD文件"""
        try:
            print("\n========== 开始处理图像 ==========")
            print(f"模板图像: {template_path}")
            print(f"头部图像: {head_path}")
            print(f"身体图像: {body_path}")
            print(f"输出路径: {output_path}")
            
            # 加载图像
            template = cv2.imread(template_path)
            head_image = cv2.imread(head_path)
            body_image = cv2.imread(body_path)
            
            print(f"\n模板图像尺寸: {template.shape[1]}x{template.shape[0]}")
            print(f"头部图像尺寸: {head_image.shape[1]}x{head_image.shape[0]}")
            print(f"身体图像尺寸: {body_image.shape[1]}x{body_image.shape[0]}")
            
            # 获取姿态信息
            print("\n检测各图像姿态信息...")
            template_landmarks = self.pose_detector.detect_pose(template, "模板")
            head_landmarks = self.pose_detector.detect_pose(head_image, "头部")
            body_landmarks = self.pose_detector.detect_pose(body_image, "身体")
            
            # 提取部位 - 使用body_part_extractor
            print("\n开始提取头部和身体部位...")
            head, head_position = self.body_part_extractor.extract_head(head_image, head_landmarks)
            body_parts = self.body_part_extractor.extract_body_parts(body_image, body_landmarks)
            
            print(f"\n共提取了 {1 + len(body_parts)} 个部位:")
            print(f"- 头部: {head.shape[1]}x{head.shape[0]}")
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                print(f"- {part_name}: {part_img.shape[1]}x{part_img.shape[0]}")
            
            # 合成图像
            print("\n开始进行图像合成...")
            result = self.psd_merger.merge_to_psd(
                template, head, head_position, body_parts, template_landmarks, output_path
            )
            print(f"\n✅ 图像合成完成: {result}")
            print("========== 处理完成 ==========\n")
            return result
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            logging.error(f"处理失败: {str(e)}")
            raise
        
    def extract_and_export(self, template_path: str, head_path: str, body_path: str, export_dir: str = None) -> str:
        """
        提取身体部位并导出到指定目录
        
        参数:
            template_path: 模板图像路径
            head_path: 头部图像路径
            body_path: 身体图像路径
            export_dir: 导出目录，如果为None则使用默认导出目录
            
        返回:
            导出目录路径
        """
        try:
            print("\n========== 开始提取和导出身体部位 ==========")
            print(f"模板图像: {template_path}")
            print(f"头部图像: {head_path}")
            print(f"身体图像: {body_path}")
            
            # 使用指定导出目录或默认导出目录
            actual_export_dir = export_dir if export_dir is not None else self.export_dir
            
            # 如果没有指定导出目录，则在当前目录下创建export文件夹
            if not actual_export_dir:
                actual_export_dir = os.path.join(os.getcwd(), "export")
            
            print(f"导出目录: {actual_export_dir}")
            
            # 加载图像
            template = cv2.imread(template_path)
            head_image = cv2.imread(head_path)
            body_image = cv2.imread(body_path)
            
            print(f"\n模板图像尺寸: {template.shape[1]}x{template.shape[0]}")
            print(f"头部图像尺寸: {head_image.shape[1]}x{head_image.shape[0]}")
            print(f"身体图像尺寸: {body_image.shape[1]}x{body_image.shape[0]}")
            
            # 获取姿态信息
            print("\n检测各图像姿态信息...")
            template_landmarks = self.pose_detector.detect_pose(template, "模板")
            head_landmarks = self.pose_detector.detect_pose(head_image, "头部")
            body_landmarks = self.pose_detector.detect_pose(body_image, "身体")
            
            # 提取部位 - 使用body_part_extractor
            print("\n开始提取头部和身体部位...")
            head, head_position = self.body_part_extractor.extract_head(head_image, head_landmarks)
            body_parts = self.body_part_extractor.extract_body_parts(body_image, body_landmarks)
            
            # 检查提取的部位是否有效
            valid_parts_count = 0
            if head is not None and head.size > 0:
                valid_parts_count += 1
                print(f"- 头部: {head.shape[1]}x{head.shape[0]}")
            else:
                print("⚠️ 警告: 提取的头部无效")
            
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                if part_img is not None and part_img.size > 0:
                    valid_parts_count += 1
                    print(f"- {part_name}: {part_img.shape[1]}x{part_img.shape[0]}")
                else:
                    print(f"⚠️ 警告: 提取的{part_name}无效")
            
            print(f"\n共提取了 {valid_parts_count} 个有效部位")
            
            # 导出部位
            print("\n开始导出身体部位...")
            result = self.part_exporter.export_parts(head, head_position, body_parts, actual_export_dir)
            
            print(f"\n✅ 身体部位导出完成: {result}")
            print("========== 处理完成 ==========\n")
            return result
            
        except Exception as e:
            print(f"❌ 提取和导出失败: {str(e)}")
            logging.error(f"提取和导出失败: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 配置参数
        SETUP_ENV = False  # 是否需要设置环境
        DWPOSE_DIR = r"D:\drafts\DWPose"
        
        # 配置输入和输出路径
        TEMPLATE_PATH = r"D:\drafts\CAM\template\teen girl template.png"
        HEAD_PATH = r"D:\drafts\CAM\template\head_raw.png"
        BODY_PATH = r"D:\drafts\CAM\template\body_raw.png"
        OUTPUT_PATH = r"D:\drafts\CAM\output\merged_character.psd"  # 指定输出PSD文件路径
        EXPORT_DIR = r"D:\drafts\CAM\export"  # 指定导出目录
        
        # 处理模式: "merge" 合并到PSD, "export" 导出各部分
        PROCESS_MODE = "export"  # 改为 "export" 即可切换到导出模式
        
        # 确保输出目录存在
        if PROCESS_MODE == "merge":
            output_dir = os.path.dirname(OUTPUT_PATH)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"创建输出目录: {output_dir}")
        
        # 如果需要设置环境
        if SETUP_ENV:
            env_manager = EnvironmentManager(DWPOSE_DIR)
            try:
                env_manager.setup()
                logger.info("环境设置完成")
            except Exception as e:
                logger.error(f"环境设置失败: {str(e)}")
                return 1

        # 初始化合成器
        merger = CharacterAutoMerger(DWPOSE_DIR, EXPORT_DIR)
        
        # 根据处理模式选择不同的处理方法
        if PROCESS_MODE == "merge":
            # 合并到PSD
            result = merger.process_images(
                template_path=TEMPLATE_PATH,
                head_path=HEAD_PATH,
                body_path=BODY_PATH,
                output_path=OUTPUT_PATH
            )
        else:  # "export" 模式
            # 导出各部分
            result = merger.extract_and_export(
                template_path=TEMPLATE_PATH,
                head_path=HEAD_PATH,
                body_path=BODY_PATH
            )
            
        print(f"处理完成: {result}")
        return 0
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
