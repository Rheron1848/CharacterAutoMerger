# python version = 3.8.10
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
import traceback

# 添加U-Net分割所需的导入
try:
    from rembg import remove, new_session
except ImportError:
    remove = None
    new_session = None


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
        self.confidence_threshold = 0.3        # 人体检测置信度阈值
        self.head_keypoint_threshold = 0.4     # 头部图像关键点置信度阈值
        self.body_keypoint_threshold = 0.2     # 身体图像关键点置信度阈值
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
    
    """
    DWPose关键点与早期版本的区别说明：
    
    早期版本使用的关键点顺序(18个关键点，0-17)：
    0鼻子，1颈部，2右肩，3右肘，4右腕，5左肩，6左肘，7左腕，
    8右髋，9右膝，10右踝，11左髋，12左膝，13左踝，14右眼，15左眼，16右耳，17左耳
    
    当前DWPose使用的关键点顺序(17个关键点，0-16)：
    0鼻子，1左眼，2右眼，3左耳，4右耳，5左肩，6右肩，
    7左肘，8右肘，9左手腕，10右手腕，11左髋，12右髋，
    13左膝，14右膝，15左踝，16右踝
    
    主要区别：
    1. DWPose没有颈部关键点
    2. 眼睛和耳朵的顺序不同
    3. 左右肢体的关键点索引完全不同
    4. DWPose总共17个关键点，而不是18个
    """
    
    # 定义身体部位及其对应的关键点索引
    # DWPose关键点索引：
    # 0鼻子，1左眼，2右眼，3左耳，4右耳，5左肩，6右肩，
    # 7左肘，8右肘，9左手腕，10右手腕，11左髋，12右髋，
    # 13左膝，14右膝，15左踝，16右踝
    # 总共17个关键点(0-16)
    BODY_PARTS_KEYPOINTS = {
        # 头部
        'head': [0, 1, 2, 3, 4],  # 鼻子、双眼、双耳
        
        # 躯干上部（胸部）
        'torso': [5, 6, 11, 12],  # 双肩和双髋
        
        # 躯干下部（腰部和髋部）
        'waist_hip': [5, 6, 11, 12],  # 双肩和双髋
        
        # 左腿（包含大腿、小腿和脚）
        'left_leg': [11, 13, 15],  # 左髋到左踝
        
        # 右腿（包含大腿、小腿和脚）
        'right_leg': [12, 14, 16],  # 右髋到右踝
        
        # 左臂（包含上臂、前臂和手）
        'left_arm': [5, 7, 9],  # 左肩到左腕
        
        # 右臂（包含上臂、前臂和手）
        'right_arm': [6, 8, 10]  # 右肩到右腕
    }
    
    # 各部位的默认位置（用于无关键点情况下）
    DEFAULT_PART_POSITIONS = None  # 将在__init__中根据图像尺寸动态设置
    
    # 特殊部位的处理方法
    SPECIAL_PARTS_HANDLERS = {
        'left_arm': '_extend_left_arm',
        'right_arm': '_extend_right_arm',
        'left_leg': '_extend_left_leg',
        'right_leg': '_extend_right_leg',
        'torso': '_extend_trunk',
        'waist_hip': '_extend_trunk'
    }
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 立即初始化默认位置字典，避免None类型错误
        self.DEFAULT_PART_POSITIONS = {}  # 初始化为空字典而不是None
        
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
            
            head_region_bgr = image[position_info['top']:position_info['bottom'], 
                          position_info['left']:position_info['right']]
            
            # 确保 head_region_bgr 不为空
            if head_region_bgr.size == 0:
                self.logger.warning("提取的头部区域为空，返回一个默认的小透明图像。")
                # 返回一个10x10的透明图像作为占位符
                return np.zeros((10, 10, 4), dtype=np.uint8), position_info

            head_region_bgra = cv2.cvtColor(head_region_bgr, cv2.COLOR_BGR2BGRA)

            # 尝试使用 rembg 去除背景
            try:
                if remove is not None and new_session is not None:
                    # self.logger.info("尝试使用rembg移除头部背景...")
                    session = new_session("u2net") # 或者其他适合的模型如 u2net_human_seg
                    # 对 head_region_bgr 进行操作，因为它没有alpha通道
                    alpha_output = remove(
                        head_region_bgr, # 输入应该是没有alpha通道的图像
                        session=session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=5 # 可以调整腐蚀大小
                    )
                    if alpha_output is not None and alpha_output.shape[2] == 4:
                        # self.logger.info("rembg成功移除头部背景。")
                        head_region_bgra = alpha_output # alpha_output 已经是 BGRA
                    else:
                        # self.logger.warning("rembg未能成功处理头部图像或返回格式不正确，将使用原始提取区域（不透明）。")
                        # 如果 rembg 失败，确保 alpha 通道是不透明的
                        head_region_bgra[:, :, 3] = 255
                else:
                    # self.logger.info("rembg不可用，头部区域将不透明。")
                    head_region_bgra[:, :, 3] = 255 # rembg 不可用，设为不透明
            except Exception as e:
                self.logger.error(f"使用rembg处理头部时发生错误: {str(e)}。头部区域将不透明。")
                head_region_bgra[:, :, 3] = 255 # 出错则设为不透明

            print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}], 尺寸: {head_region_bgra.shape[1]}x{head_region_bgra.shape[0]}")
            
            return head_region_bgra, position_info
        
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset'][0]  # 取第一个检测到的人
        
        # DWPose关键点索引
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
        
        # 获取头部关键点索引
        nose_idx = int(pose_subset[NOSE])
        left_eye_idx = int(pose_subset[LEFT_EYE])
        right_eye_idx = int(pose_subset[RIGHT_EYE])
        left_ear_idx = int(pose_subset[LEFT_EAR])
        right_ear_idx = int(pose_subset[RIGHT_EAR])
        
        # 收集有效的头部关键点
        valid_points = []
        if nose_idx != -1:
            valid_points.append(pose_points[nose_idx])
        if left_eye_idx != -1:
            valid_points.append(pose_points[left_eye_idx])
        if right_eye_idx != -1:
            valid_points.append(pose_points[right_eye_idx])
        if left_ear_idx != -1:
            valid_points.append(pose_points[left_ear_idx])
        if right_ear_idx != -1:
            valid_points.append(pose_points[right_ear_idx])
        
        # 根据可用关键点计算头部位置
        if not valid_points:
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
        else:
            # 计算头部中心点
            x_coords = [int(p[0]) for p in valid_points]
            y_coords = [int(p[1]) for p in valid_points]
            
            center_x = sum(x_coords) // len(x_coords)
            center_y = sum(y_coords) // len(y_coords)
            
            # 计算头部大小
            # 如果有鼻子和至少一只眼睛，可以更准确地估计头部尺寸
            if nose_idx != -1 and (left_eye_idx != -1 or right_eye_idx != -1):
                nose_y = int(pose_points[nose_idx][1])
                eye_y = int(pose_points[left_eye_idx][1]) if left_eye_idx != -1 else int(pose_points[right_eye_idx][1])
                
                # 以鼻子到眼睛距离的5倍作为头部高度估计
                eye_nose_distance = abs(nose_y - eye_y)
                head_height = eye_nose_distance * 5
                head_width = int(head_height * 0.8)  # 头部宽高比约为0.8
            else:
                # 默认头部尺寸
                head_height = height // 4
                head_width = int(head_height * 0.8)
            
            # 确保头部区域在图像范围内
            position_info = {
                'top': max(0, center_y - head_height // 2),
                'left': max(0, center_x - head_width // 2),
                'bottom': min(height, center_y + head_height // 2),
                'right': min(width, center_x + head_width // 2),
                'center_x': center_x,
                'center_y': center_y
            }
        
        head_region_bgr = image[position_info['top']:position_info['bottom'], 
                                position_info['left']:position_info['right']]
        
        # 确保 head_region_bgr 不为空
        if head_region_bgr.size == 0:
            self.logger.warning("提取的头部区域为空，返回一个默认的小透明图像。")
            # 返回一个10x10的透明图像作为占位符
            return np.zeros((10, 10, 4), dtype=np.uint8), position_info

        head_region_bgra = cv2.cvtColor(head_region_bgr, cv2.COLOR_BGR2BGRA)

        # 尝试使用 rembg 去除背景
        try:
            if remove is not None and new_session is not None:
                # self.logger.info("尝试使用rembg移除头部背景...")
                session = new_session("u2net") # 或者其他适合的模型如 u2net_human_seg
                # 对 head_region_bgr 进行操作，因为它没有alpha通道
                alpha_output = remove(
                    head_region_bgr, # 输入应该是没有alpha通道的图像
                    session=session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=5 # 可以调整腐蚀大小
                )
                if alpha_output is not None and alpha_output.shape[2] == 4:
                    # self.logger.info("rembg成功移除头部背景。")
                    head_region_bgra = alpha_output # alpha_output 已经是 BGRA
                else:
                    # self.logger.warning("rembg未能成功处理头部图像或返回格式不正确，将使用原始提取区域（不透明）。")
                    # 如果 rembg 失败，确保 alpha 通道是不透明的
                    head_region_bgra[:, :, 3] = 255
            else:
                # self.logger.info("rembg不可用，头部区域将不透明。")
                head_region_bgra[:, :, 3] = 255 # rembg 不可用，设为不透明
        except Exception as e:
            self.logger.error(f"使用rembg处理头部时发生错误: {str(e)}。头部区域将不透明。")
            head_region_bgra[:, :, 3] = 255 # 出错则设为不透明

        print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}], 尺寸: {head_region_bgra.shape[1]}x{head_region_bgra.shape[0]}")
        
        return head_region_bgra, position_info
    
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
        
        # 获取pose_subset的实际长度
        pose_subset_len = len(pose_subset)
        
        # 逐个提取各部位
        for part_name, keypoint_indices in self.BODY_PARTS_KEYPOINTS.items():
            # 过滤有效关键点，添加边界检查
            valid_indices = [i for i in keypoint_indices if i < pose_subset_len and int(pose_subset[i]) != -1]
            
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
            # 头部
            'head': {
                'x_start': width // 3, 
                'x_end': 2 * width // 3,
                'y_start': 0,
                'y_end': height // 4
            },
            
            # 躯干上部（胸部）
            'torso': {
                'x_start': width // 4, 
                'x_end': 3 * width // 4,
                'y_start': height // 5,
                'y_end': 2 * height // 5
            },
            
            # 躯干下部（腰部和髋部）
            'waist_hip': {
                'x_start': width // 4, 
                'x_end': 3 * width // 4,
                'y_start': 2 * height // 5,
                'y_end': 3 * height // 5
            },
            
            # 左腿
            'left_leg': {
                'x_start': width // 6, 
                'x_end': width // 2,
                'y_start': 3 * height // 5,
                'y_end': height
            },
            
            # 右腿
            'right_leg': {
                'x_start': width // 2, 
                'x_end': 5 * width // 6,
                'y_start': 3 * height // 5,
                'y_end': height
            },
            
            # 左臂
            'left_arm': {
                'x_start': 0, 
                'x_end': width // 4,
                'y_start': height // 5,
                'y_end': 3 * height // 5
            },
            
            # 右臂
            'right_arm': {
                'x_start': 3 * width // 4, 
                'x_end': width,
                'y_start': height // 5,
                'y_end': 3 * height // 5
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
        if self.DEFAULT_PART_POSITIONS is None or not isinstance(self.DEFAULT_PART_POSITIONS, dict):
            self._init_default_positions(width, height)
        
        # 使用字典的 get 方法安全地访问
        area = self.DEFAULT_PART_POSITIONS.get(part_name)
        if area:
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
    
    def _extract_by_points(self, image: np.ndarray, points: List, part_name: str = "") -> Tuple[np.ndarray, Dict]:
        """根据关键点提取部位，结合语义分割和骨架分析"""
        height, width = image.shape[:2]
        
        # 提取x,y坐标
        xy_points = [(int(p[0]), int(p[1])) for p in points]
        
        # 如果没有有效点，返回空结果
        if not xy_points:
            return np.zeros((1, 1, 3), dtype=np.uint8), {
                'top': 0, 'left': 0, 'bottom': 1, 'right': 1,
                'center_x': 0, 'center_y': 0, 'points': []
            }
        
        # 修正颜色值，解决cv2函数参数错误
        LINE_COLOR = (255, 255, 255)  # 使用正确的颜色格式
        
        # 1. 创建部位区域蒙版
        # 基于骨架创建初始蒙版
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 对于多个点，绘制连接线形成骨架
        if len(xy_points) > 1:
            for i in range(len(xy_points) - 1):
                pt1 = xy_points[i]
                pt2 = xy_points[i + 1]
                cv2.line(mask, pt1, pt2, LINE_COLOR, thickness=max(int(width * 0.03), 5))
        else:
            # 单点情况下绘制一个圆
            pt = xy_points[0]
            radius = max(int(width * 0.02), 10)
            cv2.circle(mask, pt, radius, LINE_COLOR, -1)
        
        # 2. 为不同部位应用特定的处理逻辑
        if part_name == 'head':
            # 头部需要较大区域
            kernel_size = max(int(width * 0.03), 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=4)
            
            # 计算头部中心
            center_x = sum(p[0] for p in xy_points) // len(xy_points)
            center_y = sum(p[1] for p in xy_points) // len(xy_points)
            
            # 绘制椭圆形头部
            head_width = int(width * 0.15)
            head_height = int(head_width * 1.3)
            cv2.ellipse(mask, (center_x, center_y), (head_width, head_height), 
                         0, 0, 360, (255, 255, 255), -1)
            
        elif part_name == 'left_arm' or part_name == 'right_arm':
            # 手臂需要细长的区域
            kernel_size = max(int(width * 0.02), 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            # 添加手部区域
            wrist_point = xy_points[-1]  # 假设最后一个点是手腕
            
            # 根据左右手确定延伸方向
            extension_x = 30
            if 'left' in part_name:
                extension_x = -extension_x
                
            # 在手腕处添加一个圆形区域代表手掌
            hand_center = (wrist_point[0] + extension_x, wrist_point[1] + 20)
            cv2.circle(mask, hand_center, max(int(width * 0.04), 20), (255, 255, 255), -1)
            cv2.line(mask, wrist_point, hand_center, (255, 255, 255), thickness=max(int(width * 0.025), 4))
            
        elif part_name == 'left_leg' or part_name == 'right_leg':
            # 腿部需要较宽区域
            kernel_size = max(int(width * 0.03), 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=4)
            
            # 添加脚部区域
            ankle_point = xy_points[-1]  # 假设最后一个点是踝关节
            
            # 根据左右脚确定延伸方向
            extension_x = 30
            if 'left' in part_name:
                extension_x = -extension_x
                
            # 添加脚部区域
            foot_center = (ankle_point[0] + extension_x, ankle_point[1] + 30)
            cv2.circle(mask, foot_center, max(int(width * 0.035), 15), (255, 255, 255), -1)
            cv2.line(mask, ankle_point, foot_center, (255, 255, 255), thickness=max(int(width * 0.03), 8))
            
        elif part_name == 'torso' or part_name == 'waist_hip':
            # 躯干需要较大区域
            kernel_size = max(int(width * 0.04), 7)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=5)
            
            # 计算区域中心点
            center_x = sum(p[0] for p in xy_points) // len(xy_points)
            center_y = sum(p[1] for p in xy_points) // len(xy_points)
            
            # 添加横向扩展
            width_extension = int(width * 0.15)
            left_point = (center_x - width_extension, center_y)
            right_point = (center_x + width_extension, center_y)
            cv2.line(mask, left_point, right_point, (255, 255, 255), thickness=max(int(width * 0.05), 10))
        else:
            # 默认扩展
            kernel_size = max(int(width * 0.025), 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
        
        # 3. 应用边缘检测来改进蒙版
        try:
            # 在目标区域上应用边缘检测
            x_coords = [p[0] for p in xy_points]
            y_coords = [p[1] for p in xy_points]
            
            # 计算关键点周围的区域，扩大一些以确保包含完整身体部位
            padding_factor = 1.5
            min_x = max(0, int(min(x_coords) - width * 0.1 * padding_factor))
            max_x = min(width, int(max(x_coords) + width * 0.1 * padding_factor))
            min_y = max(0, int(min(y_coords) - height * 0.1 * padding_factor))
            max_y = min(height, int(max(y_coords) + height * 0.1 * padding_factor))
            
            # 确保区域有效
            if min_x >= max_x:
                max_x = min(width, min_x + 10)
            if min_y >= max_y:
                max_y = min(height, min_y + 10)
            
            # 提取感兴趣区域
            roi = image[min_y:max_y, min_x:max_x]
            
            # 应用Canny边缘检测
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            edges = cv2.Canny(blurred_roi, 30, 100)
            
            # 在边缘上应用膨胀操作以连接断开的边缘
            edge_kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, edge_kernel, iterations=2)
            
            # 使用洪水填充算法填充封闭区域
            filled_edges = dilated_edges.copy()
            fill_mask = np.zeros((filled_edges.shape[0] + 2, filled_edges.shape[1] + 2), np.uint8)
            
            # 定义身体部位的中心点在ROI坐标中的位置
            center_x_roi = int((sum(x_coords) / len(x_coords)) - min_x)
            center_y_roi = int((sum(y_coords) / len(y_coords)) - min_y)
            
            # 确保中心点在ROI范围内
            center_x_roi = max(0, min(center_x_roi, filled_edges.shape[1]-1))
            center_y_roi = max(0, min(center_y_roi, filled_edges.shape[0]-1))
            
            # 从中心点开始填充，255是填充值，边缘为屏障(255)
            cv2.floodFill(filled_edges, fill_mask, (center_x_roi, center_y_roi), 255, 
                           loDiff=1, upDiff=1, flags=4 | (255 << 8))
            
            # 将填充后的边缘整合到原始蒙版中
            edge_mask = np.zeros((height, width), dtype=np.uint8)
            edge_mask[min_y:max_y, min_x:max_x] = filled_edges
            
            # 合并原始蒙版和边缘增强的蒙版
            combined_mask = cv2.bitwise_or(mask, edge_mask)
            mask = combined_mask
            
        except Exception as e:
            print(f"边缘检测处理失败: {str(e)}")
            # 如果边缘检测失败，继续使用原始蒙版
        
        # 4. 尝试使用U-Net前景分割改进蒙版
        try:
            if remove is not None and new_session is not None:
                # 获取关键点范围
                padding_factor = 2.0  # 扩大范围以确保包含完整身体部位
                min_x = max(0, int(min(x_coords) - width * 0.15 * padding_factor))
                max_x = min(width, int(max(x_coords) + width * 0.15 * padding_factor))
                min_y = max(0, int(min(y_coords) - height * 0.15 * padding_factor))
                max_y = min(height, int(max(y_coords) + height * 0.15 * padding_factor))
                
                # 确保区域有效
                if min_x >= max_x:
                    max_x = min(width, min_x + 20)
                if min_y >= max_y:
                    max_y = min(height, min_y + 20)
                
                # 提取ROI
                roi = image[min_y:max_y, min_x:max_x].copy()
                
                # 使用U-Net去除背景
                session = new_session("u2net")  # 使用u2net模型
                result = remove(roi, session=session, alpha_matting=True,
                               alpha_matting_foreground_threshold=240,
                               alpha_matting_background_threshold=10,
                               alpha_matting_erode_size=10)
                
                # 从结果中提取alpha通道作为蒙版
                if result is not None and result.shape[2] == 4:  # 确保是RGBA图像
                    alpha_channel = result[:, :, 3]
                    
                    # 创建U-Net分割蒙版
                    unet_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # 放置alpha通道蒙版到对应位置
                    roi_height, roi_width = alpha_channel.shape[:2]
                    # 确保不会超出边界
                    roi_height = min(roi_height, max_y - min_y)
                    roi_width = min(roi_width, max_x - min_x)
                    unet_mask[min_y:min_y+roi_height, min_x:min_x+roi_width] = alpha_channel[:roi_height, :roi_width]
                    
                    # 将U-Net蒙版与当前蒙版结合
                    # 为U-Net结果赋予更高权重
                    _, combined_mask = cv2.threshold(unet_mask, 128, 255, cv2.THRESH_BINARY)
                    kernel = np.ones((5, 5), np.uint8)
                    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
                    
                    # 与原始骨架蒙版结合，确保关键点位置被包含
                    final_mask = cv2.bitwise_or(mask, combined_mask)
                    mask = final_mask
        except Exception as e:
            print(f"U-Net分割处理失败: {str(e)}")
            # 如果U-Net分割失败，继续使用当前蒙版
        
        # 5. 找到蒙版的边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_x_bbox, min_y_bbox, max_x_bbox, max_y_bbox = width, height, 0, 0

        if not contours:
            # 如果没有轮廓，回退到基础方法，但这种情况应该尽量避免，确保mask有效
            self.logger.warning(f"部位 {part_name} 未找到有效轮廓，将使用基于关键点的粗略边界。")
            x_coords_pts = [p[0] for p in xy_points]
            y_coords_pts = [p[1] for p in xy_points]
            min_x_bbox = max(0, min(x_coords_pts) - 20)
            max_x_bbox = min(width, max(x_coords_pts) + 20)
            min_y_bbox = max(0, min(y_coords_pts) - 20)
            max_y_bbox = min(height, max(y_coords_pts) + 20)
        else:
            # 合并所有轮廓的边界框
            # 排除太小的轮廓(可能是噪声)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # 阈值可以根据图像内容和大小调整
                # 对于非常小的部位，这个阈值可能需要降低
                min_area_threshold = max(50, int(width * height * 0.0001)) 
                if area > min_area_threshold:  
                    valid_contours.append(contour)
            
            if not valid_contours:
                self.logger.warning(f"部位 {part_name} 未找到足够大的轮廓，将使用所有检测到的轮廓。")
                valid_contours = contours # 如果没有大的，就用所有小的
            
            if not valid_contours: # 如果还是没有（例如mask完全是黑的）
                self.logger.error(f"部位 {part_name} 的mask为空或无效，无法计算边界框。使用基于关键点的粗略边界。")
                x_coords_pts = [p[0] for p in xy_points]
                y_coords_pts = [p[1] for p in xy_points]
                min_x_bbox = max(0, min(x_coords_pts) - 20)
                max_x_bbox = min(width, max(x_coords_pts) + 20)
                min_y_bbox = max(0, min(y_coords_pts) - 20)
                max_y_bbox = min(height, max(y_coords_pts) + 20)
            else:
                for contour in valid_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    min_x_bbox = min(min_x_bbox, x)
                    min_y_bbox = min(min_y_bbox, y)
                    max_x_bbox = max(max_x_bbox, x + w)
                    max_y_bbox = max(max_y_bbox, y + h)
        
        # 确保边界合法
        min_x_bbox = max(0, min_x_bbox)
        min_y_bbox = max(0, min_y_bbox)
        max_x_bbox = min(width, max(min_x_bbox + 1, max_x_bbox)) # 确保至少1像素宽
        max_y_bbox = min(height, max(min_y_bbox + 1, max_y_bbox)) # 确保至少1像素高

        # 检查计算出的边界框是否有效
        if min_x_bbox >= max_x_bbox or min_y_bbox >= max_y_bbox:
            self.logger.error(f"部位 {part_name} 的计算边界框无效 ({min_x_bbox},{min_y_bbox})-({max_x_bbox},{max_y_bbox})。将使用基于关键点的粗略边界。")
            # Fallback to a small region around the first keypoint if exists
            if xy_points:
                fallback_center_x, fallback_center_y = xy_points[0]
                min_x_bbox = max(0, fallback_center_x - 10)
                min_y_bbox = max(0, fallback_center_y - 10)
                max_x_bbox = min(width, fallback_center_x + 10)
                max_y_bbox = min(height, fallback_center_y + 10)
            else: # 如果连关键点都没有，只能返回一个极小的空区域
                return np.zeros((1, 1, 4), dtype=np.uint8), {
                    'top': 0, 'left': 0, 'bottom': 1, 'right': 1,
                    'center_x': 0, 'center_y': 0, 'points': []
                }

        # 6. 应用额外填充 (可选，但对于确保不丢失边缘像素可能重要)
        #    这一步需要小心，因为我们想要精确形状，过多的填充可能引入不必要的背景
        #    对于精确抠图，可能不需要或只需要非常小的填充来确保轮廓完整
        padding_pixels = 5 # 例如，固定的小填充
        
        eff_min_x = max(0, min_x_bbox - padding_pixels)
        eff_min_y = max(0, min_y_bbox - padding_pixels)
        eff_max_x = min(width, max_x_bbox + padding_pixels)
        eff_max_y = min(height, max_y_bbox + padding_pixels)

        # 如果填充后区域无效，则不使用填充
        if eff_min_x >= eff_max_x or eff_min_y >= eff_max_y:
            eff_min_x, eff_min_y, eff_max_x, eff_max_y = min_x_bbox, min_y_bbox, max_x_bbox, max_y_bbox

        # 计算中心点 (基于有效边界框)
        center_x = (eff_min_x + eff_max_x) // 2
        center_y = (eff_min_y + eff_max_y) // 2
        
        # 7. 创建输出：提取图像区域并应用mask
        position_info = {
            'top': eff_min_y,
            'left': eff_min_x,
            'bottom': eff_max_y,
            'right': eff_max_x,
            'center_x': center_x,
            'center_y': center_y,
            'points': xy_points,
            'part_name': part_name
        }
        
        # 从原始图像中提取包含部位的矩形区域 (BGR)
        part_image_bgr = image[eff_min_y:eff_max_y, eff_min_x:eff_max_x].copy()
        
        # 从全局二值mask中提取对应的区域 (单通道)
        mask_roi = mask[eff_min_y:eff_max_y, eff_min_x:eff_max_x].copy()

        if part_image_bgr.size == 0 or mask_roi.size == 0 or part_image_bgr.shape[:2] != mask_roi.shape[:2]:
             self.logger.error(f"部位 {part_name} 的图像或掩码ROI为空或尺寸不匹配。BGR: {part_image_bgr.shape}, Mask ROI: {mask_roi.shape}")
             # 返回一个小的透明图像作为错误指示
             return np.zeros((10, 10, 4), dtype=np.uint8), position_info


        # 创建BGRA图像
        part_image_bgra = cv2.cvtColor(part_image_bgr, cv2.COLOR_BGR2BGRA)
        
        # 将mask_roi (0或255) 应用为Alpha通道
        part_image_bgra[:, :, 3] = mask_roi
        
        # 原先的 rembg 调用被移除，因为我们现在使用内部生成的精确 mask
        # # 8. 生成透明背景版本 (可选)
        # try:
        #     if remove is not None and new_session is not None:
        #         # 使用U-Net去除背景，生成带透明度的部位图像
        #         session = new_session("u2net")
        #         alpha_part_image = remove(
        #             part_image, 
        #             session=session,
        #             alpha_matting=True,
        #             alpha_matting_foreground_threshold=240,
        #             alpha_matting_background_threshold=10,
        #             alpha_matting_erode_size=10
        #         )
        #         if alpha_part_image is not None:
        #             part_image = alpha_part_image
        # except Exception as e:
        #     print(f"生成透明背景失败: {str(e)}")
        
        return part_image_bgra, position_info

    # 特殊部位处理方法
    def _extend_left_arm(self, points: List) -> List:
        """扩展左臂区域，包括添加手部"""
        if len(points) >= 2:
            # 获取手腕位置(假设是最后一个点)
            wrist_point = points[-1]
            # 添加手掌位置
            points.append([wrist_point[0] - 30, wrist_point[1] + 30, wrist_point[2]])
        return points
    
    def _extend_right_arm(self, points: List) -> List:
        """扩展右臂区域，包括添加手部"""
        if len(points) >= 2:
            # 获取手腕位置(假设是最后一个点)
            wrist_point = points[-1]
            # 添加手掌位置
            points.append([wrist_point[0] + 30, wrist_point[1] + 30, wrist_point[2]])
        return points
    
    def _extend_left_leg(self, points: List) -> List:
        """扩展左腿区域，包括添加脚部"""
        if len(points) >= 2:
            # 获取踝关节位置(假设是最后一个点)
            ankle_point = points[-1]
            # 添加脚部位置
            points.append([ankle_point[0] - 30, ankle_point[1] + 40, ankle_point[2]])
        return points
    
    def _extend_right_leg(self, points: List) -> List:
        """扩展右腿区域，包括添加脚部"""
        if len(points) >= 2:
            # 获取踝关节位置(假设是最后一个点)
            ankle_point = points[-1]
            # 添加脚部位置
            points.append([ankle_point[0] + 30, ankle_point[1] + 40, ankle_point[2]])
        return points
    
    def _extend_trunk(self, points: List) -> List:
        """扩展躯干区域"""
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            min_x = min(x_coords)
            max_x = max(x_coords)
            avg_y = sum(y_coords) / len(y_coords)
            
            width_extension = (max_x - min_x) * 0.3
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
        if part_name == "head" or part_name == "Head":
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
        # DWPose关键点索引：0鼻子，1左眼，2右眼
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        
        nose_idx = int(pose_subset[NOSE])
        left_eye_idx = int(pose_subset[LEFT_EYE])
        right_eye_idx = int(pose_subset[RIGHT_EYE])
        
        if nose_idx != -1:
            # 优先使用鼻子位置
            nose_pos = pose_points[nose_idx]
            nose_x = int(nose_pos[0])
            nose_y = int(nose_pos[1])
            
            # 估计头部位置
            head_width = source_position['right'] - source_position['left']
            head_height = source_position['bottom'] - source_position['top']
            
            # 鼻子位置大约在头部高度的60%处
            head_top_offset = int(head_height * 0.6)
            
            return {
                'x': nose_x - head_width // 2,
                'y': nose_y - head_top_offset
            }
        elif left_eye_idx != -1 and right_eye_idx != -1:
            # 使用双眼确定位置
            left_eye_pos = pose_points[left_eye_idx]
            right_eye_pos = pose_points[right_eye_idx]
            
            # 计算双眼中心点
            center_x = int((left_eye_pos[0] + right_eye_pos[0]) / 2)
            center_y = int((left_eye_pos[1] + right_eye_pos[1]) / 2)
            
            # 估计头部位置
            head_width = source_position['right'] - source_position['left']
            head_height = source_position['bottom'] - source_position['top']
            
            # 双眼位置大约在头部高度的50%处
            head_top_offset = int(head_height * 0.5)
            
            return {
                'x': center_x - head_width // 2,
                'y': center_y - head_top_offset
            }
            
        # 没有有效关键点，使用默认位置
        return {
            'x': source_position.get('left', 0),
            'y': source_position.get('top', 0)
        }
    
    def _adjust_placement(self, placement: Dict, part_name: str, part_width: int, part_height: int, avg_x: int, avg_y: int) -> Dict:
        """根据部位类型调整放置位置"""
        # 更精确的偏移计算
        if part_name == 'torso':
            # 胸部应该放在颈部和躯干中间
            placement['x'] = avg_x - part_width // 2
            placement['y'] = avg_y - part_height // 2
        elif part_name == 'waist_hip':
            # 腰部和髋部应该居中放置
            placement['x'] = avg_x - part_width // 2
            placement['y'] = avg_y - part_height // 2
        elif part_name == 'left_arm':
            # 左臂应该对准左肩和左腕
            placement['x'] = avg_x - part_width * 0.5
            placement['y'] = avg_y - part_height * 0.5
        elif part_name == 'right_arm':
            # 右臂应该对准右肩和右腕
            placement['x'] = avg_x - part_width * 0.5
            placement['y'] = avg_y - part_height * 0.5
        elif part_name == 'left_leg':
            # 左腿应该对准左髋和左踝
            placement['x'] = avg_x - part_width * 0.5
            placement['y'] = avg_y - part_height * 0.5
        elif part_name == 'right_leg':
            # 右腿应该对准右髋和右踝
            placement['x'] = avg_x - part_width * 0.5
            placement['y'] = avg_y - part_height * 0.5
        
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
            # 根据图像类型设置不同的置信度阈值
            if image_type.lower() == "头部":
                # 对头部图像使用较高的置信度阈值
                keypoint_confidence_threshold = self.config.head_keypoint_threshold
                print(f"{image_type}图像使用较高的关键点置信度阈值: {keypoint_confidence_threshold*100}%")
            else:
                # 对其他图像使用较低的置信度阈值
                keypoint_confidence_threshold = self.config.body_keypoint_threshold
            
            # 更新配置的置信度阈值
            self.config.confidence_threshold = keypoint_confidence_threshold
            
            # 增加图像预处理以提高检测质量
            if image.shape[0] < 300 or image.shape[1] < 300:
                # 对小图像进行放大以提高检测质量
                scale_factor = max(1, 300 / min(image.shape[0], image.shape[1]))
                image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
                print(f"{image_type}图像已放大 {scale_factor} 倍以提高检测质量")
            
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
                
                # 确保所有检测到的关键点都被包含，即使置信度低
                return {
                    'pose_keypoints': [
                        [float(kpt[0]), float(kpt[1]), float(score)]
                        for kpt, score in zip(keypoints, keypoint_scores)
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
                if head.shape[2] != 4:
                    self.logger.warning(f"头部图像 'head.png' 不是BGRA格式 (shape: {head.shape})，可能不会有透明度。尝试转换...")
                    try:
                        head = cv2.cvtColor(head, cv2.COLOR_BGR2BGRA)
                        # 假设如果转换前是3通道，转换后alpha应为不透明
                        if head.shape[2] == 4 and np.all(head[:,:,3]==0):
                             head[:,:,3] = 255 # 如果alpha全0，设为不透明
                    except Exception as e:
                        self.logger.error(f"转换头部图像到BGRA失败: {e}，按原样保存。")
                # 导出头部
                head_path = os.path.join(export_dir, 'head.png')
                cv2.imwrite(head_path, head) # imwrite可以直接处理BGRA的PNG
                print(f"✅ 导出头部: {head_path}")
            
            # 导出各身体部位
            parts_info = {}
            if head is not None and head.size > 0: # 确保使用可能已转换的head
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
                
                if part_img.shape[2] != 4:
                    self.logger.warning(f"部位图像 '{part_name}.png' 不是BGRA格式 (shape: {part_img.shape})，可能不会有透明度。尝试转换...")
                    try:
                        part_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2BGRA)
                        if part_img.shape[2] == 4 and np.all(part_img[:,:,3]==0):
                             part_img[:,:,3] = 255 # 如果alpha全0，设为不透明
                    except Exception as e:
                        self.logger.error(f"转换部位 {part_name} 图像到BGRA失败: {e}，按原样保存。")

                part_path = os.path.join(export_dir, f'{part_name}.png')
                cv2.imwrite(part_path, part_img) # imwrite可以直接处理BGRA的PNG
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
            
            # 创建新的PSD文档 - RGBA模式以支持透明度
            psd = PSDImage.new('RGBA', (template.shape[1], template.shape[0]))
            
            # 1. 添加模板图层 (通常是不透明的背景)
            template_rgb_path = os.path.join(temp_dir, 'template_rgb.png')
            # 模板图像是BGR，转换为RGB给PIL
            cv2.imwrite(template_rgb_path, cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            template_pil_img = Image.open(template_rgb_path)
            
            template_layer = psd.add_layer(name="TemplateBackground")
            template_layer.left, template_layer.top = 0, 0
            template_layer.size = (template.shape[1], template.shape[0])
            template_layer.image_data = template_pil_img
            
            # 清理模板的临时文件
            if os.path.exists(template_rgb_path):
                os.remove(template_rgb_path)

            # 初始化PositionCalculator，尝试使用self.config
            # (PSDMerger实例本身可能没有config，但CharacterAutoMerger会传递它)
            # PositionCalculator 的 __init__ 应该能处理 config 为 None 的情况
            position_calculator_config = getattr(self, 'config', None)
            position_calculator = PositionCalculator(position_calculator_config)
            
            # 2. 添加头部图层
            if head is not None and head.size > 0:
                head_for_psd = head # 预期是BGRA
                if head.shape[2] != 4:
                    self.logger.warning("头部图像在PSD合并前不是BGRA，尝试转换。")
                    try:
                        head_for_psd = cv2.cvtColor(head, cv2.COLOR_BGR2BGRA)
                        # 如果转换后alpha全0，则设为不透明，避免图层不可见
                        if np.all(head_for_psd[:,:,3] == 0): 
                            head_for_psd[:,:,3] = 255
                    except Exception as e_conv:
                        self.logger.error(f"转换头部图像到BGRA失败: {e_conv}，跳过此图层。")
                        head_for_psd = None # 标记为None以跳过
                
                if head_for_psd is not None:
                    head_rgba_pil = cv2.cvtColor(head_for_psd, cv2.COLOR_BGRA2RGBA)
                    head_pil_image = Image.fromarray(head_rgba_pil, 'RGBA')
                    
                    head_placement = position_calculator.calculate_placement_position(
                        head_position, template_landmarks, "Head"
                    )
                    
                    head_layer_psd = psd.add_layer(name="Head")
                    head_layer_psd.left = int(head_placement.get('x', 0))
                    head_layer_psd.top = int(head_placement.get('y', 0))
                    head_layer_psd.size = (head_for_psd.shape[1], head_for_psd.shape[0])
                    head_layer_psd.image_data = head_pil_image
            else:
                self.logger.warning("头部图像为空，跳过在PSD中添加头部图层。")

            # 3. 添加身体部位图层
            for part_name, part_data in body_parts.items():
                part_img_bgra = part_data.get('image')
                part_pos_info = part_data.get('position')

                if part_img_bgra is None or part_img_bgra.size == 0 or part_pos_info is None:
                    self.logger.warning(f"身体部位 '{part_name}' 数据不完整，跳过在PSD中添加此图层。")
                    continue

                part_for_psd = part_img_bgra # 预期是BGRA
                if part_img_bgra.shape[2] != 4:
                    self.logger.warning(f"身体部位 '{part_name}' 在PSD合并前不是BGRA，尝试转换。")
                    try:
                        part_for_psd = cv2.cvtColor(part_img_bgra, cv2.COLOR_BGR2BGRA)
                        if np.all(part_for_psd[:,:,3] == 0): 
                            part_for_psd[:,:,3] = 255
                    except Exception as e_conv_part:
                        self.logger.error(f"转换身体部位 '{part_name}' 到BGRA失败: {e_conv_part}，跳过此图层。")
                        part_for_psd = None
                
                if part_for_psd is not None:
                    part_rgba_pil = cv2.cvtColor(part_for_psd, cv2.COLOR_BGRA2RGBA)
                    part_pil_image = Image.fromarray(part_rgba_pil, 'RGBA')

                    placement = position_calculator.calculate_placement_position(
                        part_pos_info, template_landmarks, part_name
                    )
                    
                    part_layer_psd = psd.add_layer(name=part_name)
                    part_layer_psd.left = int(placement.get('x', 0))
                    part_layer_psd.top = int(placement.get('y', 0))
                    part_layer_psd.size = (part_for_psd.shape[1], part_for_psd.shape[0])
                    part_layer_psd.image_data = part_pil_image
            
            # 保存PSD文件
            psd.save(output_path)
            self.logger.info(f"PSD文件已合成并保存到: {output_path}")
            return output_path
            
        except ImportError: # ImportError from psd_tools or PIL
            error_msg = "未检测到 PSD-Tools 或 Pillow，无法进行PSD合成。请确保已安装这些库 (pip install psd-tools Pillow)。"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) # Re-raise as RuntimeError for consistent error handling
        except Exception as e:
            self.logger.error(f"PSD合成失败: {str(e)}")
            self.logger.exception("PSD合成时发生异常") # Log full traceback
            raise # Re-raise the original exception

    def _check_psd_tools(self) -> bool:
        """检查是否可以使用PSD-Tools和Pillow"""
        try:
            import psd_tools
            from PIL import Image # Also check Pillow
            return True
        except ImportError:
            return False # The main method will catch this and log a more detailed message

    def visualize_keypoints(self, image: np.ndarray, landmarks: Dict, output_path: str) -> None:
        """可视化关键点，帮助调试"""
        debug_img = image.copy()
        
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks.get('pose_subset', [])) == 0:
            print("⚠️ 无法可视化关键点：未检测到关键点")
            return
        
        pose_points = landmarks['pose_keypoints']
        
        # DWPose关键点名称列表，用于显示
        dwpose_keypoint_names = [
            "nose", "left eye", "right eye", "left ear", "right ear",
            "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
            "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
        ]
        
        # 直接使用点列表而不是subset
        for i, point in enumerate(pose_points):
            # 检查点是否有效（通常第三个值是置信度）
            if len(point) < 3 or point[2] < 0.2:  # 使用低置信度阈值以显示更多点
                continue
                
            x, y = int(point[0]), int(point[1])
            
            # 绘制关键点
            cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
            
            # 添加关键点编号和名称标签
            # 格式: "#编号:名称 (置信度)"
            if i < len(dwpose_keypoint_names):
                confidence = int(point[2] * 100)
                label = f"#{i}:{dwpose_keypoint_names[i]} ({confidence}%)"
            else:
                label = f"#{i}"
                
            # 显示关键点编号在点上方
            cv2.putText(debug_img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存调试图像
        cv2.imwrite(output_path, debug_img)
        print(f"✅ 关键点可视化已保存到: {output_path}")
        print(f"✓ 识别到 {len(pose_points)} 个关键点")

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
        
    def set_keypoint_thresholds(self, head_threshold: float = None, body_threshold: float = None) -> None:
        """设置关键点置信度阈值
        
        参数:
            head_threshold: 头部图像关键点置信度阈值(0.0-1.0)
            body_threshold: 身体图像关键点置信度阈值(0.0-1.0)
        """
        if head_threshold is not None:
            if 0.0 <= head_threshold <= 1.0:
                self.config.head_keypoint_threshold = head_threshold
                print(f"头部关键点置信度阈值已设置为: {head_threshold*100}%")
            else:
                print(f"⚠️ 无效的头部置信度阈值 {head_threshold}，必须在0.0-1.0之间")
        
        if body_threshold is not None:
            if 0.0 <= body_threshold <= 1.0:
                self.config.body_keypoint_threshold = body_threshold
                print(f"身体关键点置信度阈值已设置为: {body_threshold*100}%")
            else:
                print(f"⚠️ 无效的身体置信度阈值 {body_threshold}，必须在0.0-1.0之间")
    
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
            import traceback
            error_msg = f"❌ 处理失败: {str(e)}\n"
            error_msg += traceback.format_exc()
            print(error_msg)
            logging.error(error_msg)
            raise
        
    def extract_and_export(self, template_path: str, head_path: str, body_path: str, export_dir: Optional[str] = None) -> str:
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
            # 提取head_image中的头部
            head_from_head_img, head_from_head_position = self.body_part_extractor.extract_head(head_image, head_landmarks)
            # 提取body_image中的头部和身体部位
            head_from_body_img, head_from_body_position = self.body_part_extractor.extract_head(body_image, body_landmarks)
            body_parts = self.body_part_extractor.extract_body_parts(body_image, body_landmarks)
            
            # 检查提取的部位是否有效
            valid_parts_count = 0
            
            # 检查从head_image提取的头部
            if head_from_head_img is not None and head_from_head_img.size > 0:
                valid_parts_count += 1
                print(f"- 从头部图像提取的头部: {head_from_head_img.shape[1]}x{head_from_head_img.shape[0]}")
            else:
                print("⚠️ 警告: 从头部图像提取的头部无效")
            
            # 检查从body_image提取的头部
            if head_from_body_img is not None and head_from_body_img.size > 0:
                valid_parts_count += 1
                print(f"- 从身体图像提取的头部: {head_from_body_img.shape[1]}x{head_from_body_img.shape[0]}")
            else:
                print("⚠️ 警告: 从身体图像提取的头部无效")
            
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
            # 确保导出目录存在
            os.makedirs(actual_export_dir, exist_ok=True)
            
            # 导出从head_image提取的头部
            if head_from_head_img is not None and head_from_head_img.size > 0:
                head_from_head_path = os.path.join(actual_export_dir, 'head_from_head.png')
                cv2.imwrite(head_from_head_path, head_from_head_img)
                print(f"✅ 导出从头部图像提取的头部: {head_from_head_path}")
            
            # 使用part_exporter导出从body_image提取的头部和身体部位
            result = self.part_exporter.export_parts(head_from_body_img, head_from_body_position, body_parts, actual_export_dir)
            
            print(f"\n✅ 身体部位导出完成: {result}")
            print("========== 处理完成 ==========\n")
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 提取和导出失败: {str(e)}\n"
            error_msg += traceback.format_exc()
            print(error_msg)
            logging.error(error_msg)
            raise

    def visualize_keypoints(self, image: np.ndarray, landmarks: Dict, output_path: str) -> None:
        """可视化关键点，帮助调试"""
        debug_img = image.copy()
        
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks.get('pose_subset', [])) == 0:
            print("⚠️ 无法可视化关键点：未检测到关键点")
            return
        
        pose_points = landmarks['pose_keypoints']
        
        # DWPose关键点名称列表，用于显示
        dwpose_keypoint_names = [
            "nose", "left eye", "right eye", "left ear", "right ear",
            "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
            "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
        ]
        
        # 直接使用点列表而不是subset
        for i, point in enumerate(pose_points):
            # 检查点是否有效（通常第三个值是置信度）
            if len(point) < 3 or point[2] < 0.2:  # 使用低置信度阈值以显示更多点
                continue
                
            x, y = int(point[0]), int(point[1])
            
            # 绘制关键点
            cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
            
            # 添加关键点编号和名称标签
            # 格式: "#编号:名称 (置信度)"
            if i < len(dwpose_keypoint_names):
                confidence = int(point[2] * 100)
                label = f"#{i}:{dwpose_keypoint_names[i]} ({confidence}%)"
            else:
                label = f"#{i}"
                
            # 显示关键点编号在点上方
            cv2.putText(debug_img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存调试图像
        cv2.imwrite(output_path, debug_img)
        print(f"✅ 关键点可视化已保存到: {output_path}")
        print(f"✓ 识别到 {len(pose_points)} 个关键点")

def debug_run(merger, template_path, head_path, body_path):
    """用于调试的运行函数，会捕获并打印完整错误信息"""
    try:
        # 加载图像
        template = cv2.imread(template_path)
        head_image = cv2.imread(head_path)
        body_image = cv2.imread(body_path)
        
        if template is None:
            print(f"无法加载模板图像: {template_path}")
            return
        if head_image is None:
            print(f"无法加载头部图像: {head_path}")
            return
        if body_image is None:
            print(f"无法加载身体图像: {body_path}")
            return
            
        print(f"模板图像尺寸: {template.shape}")
        print(f"头部图像尺寸: {head_image.shape}")
        print(f"身体图像尺寸: {body_image.shape}")
        
        # 检测姿态
        print("\n正在检测模板图像姿态...")
        template_landmarks = merger.pose_detector.detect_pose(template, "模板")
        print("\n正在检测头部图像姿态...")
        head_landmarks = merger.pose_detector.detect_pose(head_image, "头部")
        print("\n正在检测身体图像姿态...")
        body_landmarks = merger.pose_detector.detect_pose(body_image, "身体")
        
        # 提取部位
        print("\n正在提取头部...")
        head, head_position = merger.body_part_extractor.extract_head(head_image, head_landmarks)
        print("\n正在提取身体部位...")
        body_parts = merger.body_part_extractor.extract_body_parts(body_image, body_landmarks)
        
        print("\n提取完成")
        return "调试运行完成"
        
    except Exception as e:
        import traceback
        error_msg = f"调试失败: {str(e)}\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return None

def setup_detailed_logging():
    """设置详细的日志记录"""
    import logging
    log_file = 'debug_detailed.log'
    
    # 创建一个详细的文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 获取根日志记录器并添加处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    print(f"详细日志将被记录到: {log_file}")
    return log_file

def main():
    """主函数"""
    try:
        log_file = setup_detailed_logging()
        # 配置参数
        SETUP_ENV = False  # 是否需要设置环境
        DWPOSE_DIR = r"D:\drafts\DWPose"
        
        # 配置输入和输出路径
        TEMPLATE_PATH = r"D:\drafts\CAM\template\teen girl template.png"
        HEAD_PATH = r"D:\drafts\CAM\template\head_raw.png"
        BODY_PATH = r"D:\drafts\CAM\template\body_raw.png"
        OUTPUT_PATH = r"D:\drafts\CAM\output\merged_character.psd"  # 指定输出PSD文件路径
        EXPORT_DIR = r"D:\drafts\CAM\export"  # 指定导出目录
        
        # 处理模式: "merge" 合并到PSD, "export" 导出各部分 debug 检测报错
        PROCESS_MODE = "debug"  # 改为 "export" 即可切换到导出模式
        
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
        
        # 设置关键点置信度阈值，可以根据需要调整
        # 头部图像使用0.4的阈值来过滤低置信度的点，身体图像使用0.2的阈值
        merger.set_keypoint_thresholds(head_threshold=0.4, body_threshold=0.2)
        
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
            
        # 添加调试输出
        if PROCESS_MODE == "debug":
            # 加载图像
            template = cv2.imread(TEMPLATE_PATH)
            head_image = cv2.imread(HEAD_PATH)
            body_image = cv2.imread(BODY_PATH)
            
            # 创建调试目录
            debug_dir = os.path.join(EXPORT_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 检测姿态
            pose_detector = PoseDetector(Config(DWPOSE_DIR))
            template_landmarks = pose_detector.detect_pose(template, "模板")
            head_landmarks = pose_detector.detect_pose(head_image, "头部")
            body_landmarks = pose_detector.detect_pose(body_image, "身体")
            
            # 可视化关键点
            visualize_keypoints = getattr(merger, "visualize_keypoints", None)
            if visualize_keypoints:
                visualize_keypoints(template, template_landmarks, os.path.join(debug_dir, "template_keypoints.png"))
                visualize_keypoints(head_image, head_landmarks, os.path.join(debug_dir, "head_keypoints.png"))
                visualize_keypoints(body_image, body_landmarks, os.path.join(debug_dir, "body_keypoints.png"))
            
            print(f"调试图像已保存到: {debug_dir}")
        
        print(f"处理完成: {result}")
        return 0
    except Exception as e:
        import traceback
        error_msg = f"处理失败: {str(e)}\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        logging.error(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())
