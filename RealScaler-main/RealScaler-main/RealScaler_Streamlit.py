import os
import sys
import json
import time
import shutil
import subprocess
from functools import cache
from threading import Thread
from multiprocessing import Process, Queue
from os import (
    environ as os_environ,
    makedirs as os_makedirs,
    devnull as os_devnull,
    sep as os_separator,
    listdir as os_listdir,
    remove as os_remove
)

from os.path import (
    basename as os_path_basename,
    dirname as os_path_dirname,
    abspath as os_path_abspath,
    join as os_path_join,
    exists as os_path_exists,
    splitext as os_path_splitext,
    expanduser as os_path_expanduser
)

# Third-party library imports
import streamlit as st
from natsort import natsorted
from moviepy.video.io import ImageSequenceClip 
from onnxruntime import InferenceSession
from PIL import Image
from PIL.Image import (
    open as pillow_image_open,
    fromarray as pillow_image_fromarray
)

import cv2
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_GRAY2RGB,
    COLOR_BGR2RGBA,
    COLOR_RGB2GRAY,
    IMREAD_UNCHANGED,
    INTER_AREA,
    VideoCapture as opencv_VideoCapture,
    cvtColor as opencv_cvtColor,
    imdecode as opencv_imdecode,
    imencode as opencv_imencode,
    addWeighted as opencv_addWeighted,
    resize as opencv_resize,
)

import numpy as np
from numpy import (
    ndarray as numpy_ndarray,
    frombuffer as numpy_frombuffer,
    concatenate as numpy_concatenate, 
    transpose as numpy_transpose,
    full as numpy_full, 
    zeros as numpy_zeros, 
    expand_dims as numpy_expand_dims,
    squeeze as numpy_squeeze,
    clip as numpy_clip,
    mean as numpy_mean,
    repeat as numpy_repeat,
    max as numpy_max, 
    float32,
    float16,
    uint8
)

from multiprocessing.pool import ThreadPool
from itertools import repeat
from json import load as json_load
from webbrowser import open as open_browser
from time import time as timer
from time import sleep
from subprocess import run as subprocess_run
from typing import Callable

# Constants and configuration
app_name = "Video Enhancer"



very_low_VRAM = 4
low_VRAM = 3
medium_VRAM = 2.2
very_high_VRAM = 0.6

AI_LIST_SEPARATOR = ["----"]
SRVGGNetCompact_models_list = ['RealESR_Gx4', 'RealSRx4_Anime']
RealESRGAN_models_list = ['RealESRGANx4', 'RealESRNetx4']

AI_models_list = (SRVGGNetCompact_models_list + AI_LIST_SEPARATOR + RealESRGAN_models_list)
AI_multithreading_list = ["1 threads", "2 threads", "3 threads", "4 threads", "5 threads", "6 threads"]
interpolation_list = ["Disabled", "Low", "Medium", "High"]
gpus_list = ["Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4"]
keep_frames_list = ["Disabled", "Enabled"]
image_extension_list = [".png", ".jpg", ".bmp", ".tiff"]
video_extension_list = [".mp4 (x264)", ".mp4 (x265)", ".avi"]

OUTPUT_PATH_CODED = "Same path as input files"
DOCUMENT_PATH = os_path_join(os_path_expanduser('~'), 'Documents')

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)

USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_UserPreference.json")
FFMPEG_EXE_PATH = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH = find_by_relative_path(f"Assets{os_separator}exiftool.exe")

ECTRACTION_FRAMES_FOR_CPU = 25
MULTIPLE_FRAMES_TO_SAVE = 8
MULTIPLE_FRAMES_TO_SAVE_MULTITHREAD = MULTIPLE_FRAMES_TO_SAVE/2

COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"

# Load user preferences
if os_path_exists(FFMPEG_EXE_PATH): 
    print(f"[{app_name}] External ffmpeg.exe file found")
    os_environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

if os_path_exists(USER_PREFERENCE_PATH):
    print(f"[{app_name}] Preference file exist")
    with open(USER_PREFERENCE_PATH, "r") as json_file:
        json_data = json_load(json_file)
        default_AI_model = json_data.get("default_AI_model", AI_models_list[0])
        default_AI_multithreading = json_data.get("default_AI_multithreading", AI_multithreading_list[0])
        default_gpu = json_data.get("default_gpu", gpus_list[0])
        default_keep_frames = json_data.get("default_keep_frames", keep_frames_list[0])
        default_image_extension = json_data.get("default_image_extension", image_extension_list[0])
        default_video_extension = json_data.get("default_video_extension", video_extension_list[0])
        default_interpolation = json_data.get("default_interpolation", interpolation_list[1])
        default_output_path = json_data.get("default_output_path", OUTPUT_PATH_CODED)
        default_resize_factor = json_data.get("default_resize_factor", str(50))
        default_VRAM_limiter = json_data.get("default_VRAM_limiter", str(4))
        default_cpu_number = json_data.get("default_cpu_number", str(4))
else:
    print(f"[{app_name}] Preference file does not exist, using default coded value")
    default_AI_model = AI_models_list[0]
    default_AI_multithreading = AI_multithreading_list[0]
    default_gpu = gpus_list[0]
    default_keep_frames = keep_frames_list[0]
    default_image_extension = image_extension_list[0]
    default_video_extension = video_extension_list[0]
    default_interpolation = interpolation_list[1]
    default_output_path = OUTPUT_PATH_CODED
    default_resize_factor = str(50)
    default_VRAM_limiter = str(4)
    default_cpu_number = str(4)

supported_file_extensions = [
    '.heic', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png',
    '.PNG', '.webp', '.WEBP', '.bmp', '.BMP', '.tif',
    '.tiff', '.TIF', '.TIFF', '.mp4', '.MP4', '.webm',
    '.WEBM', '.mkv', '.MKV', '.flv', '.FLV', '.gif',
    '.GIF', '.m4v', ',M4V', '.avi', '.AVI', '.mov',
    '.MOV', '.qt', '.3gp', '.mpg', '.mpeg', ".vob"
]

supported_video_extensions = [
    '.mp4', '.MP4', '.webm', '.WEBM', '.mkv', '.MKV',
    '.flv', '.FLV', '.gif', '.GIF', '.m4v', ',M4V',
    '.avi', '.AVI', '.mov', '.MOV', '.qt', '.3gp',
    '.mpg', '.mpeg', ".vob"
]

# AI Class
class AI:
    # CLASS INIT FUNCTIONS
    def __init__(
            self, 
            AI_model_name: str, 
            directml_gpu: str, 
            resize_factor: int,
            max_resolution: int
            ):
        
        # Passed variables
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.resize_factor = resize_factor
        self.max_resolution = max_resolution

        # Calculated variables
        self.AI_model_path = find_by_relative_path(f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        self.upscale_factor = self._get_upscale_factor()
        self.inferenceSession = self._load_inferenceSession()

    def _get_upscale_factor(self) -> int:
        if "x1" in self.AI_model_name: return 1
        elif "x2" in self.AI_model_name: return 2
        elif "x4" in self.AI_model_name: return 4

    def _load_inferenceSession(self) -> InferenceSession:
        providers = ['DmlExecutionProvider']

        match self.directml_gpu:
            case 'Auto': provider_options = [{"performance_preference": "high_performance"}]
            case 'GPU 1': provider_options = [{"device_id": "0"}]
            case 'GPU 2': provider_options = [{"device_id": "1"}]
            case 'GPU 3': provider_options = [{"device_id": "2"}]
            case 'GPU 4': provider_options = [{"device_id": "3"}]

        inference_session = InferenceSession(
            path_or_bytes = self.AI_model_path, 
            providers = providers,
            provider_options = provider_options
            )

        return inference_session

    # INTERNAL CLASS FUNCTIONS
    def get_image_mode(self, image: numpy_ndarray) -> str:
        match image.shape:
            case (rows, cols):
                return "Grayscale"
            case (rows, cols, channels) if channels == 3:
                return "RGB"
            case (rows, cols, channels) if channels == 4:
                return "RGBA"

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width = image.shape[1]
        return height, width 

    def calculate_target_resolution(self, image: numpy_ndarray) -> tuple:
        height, width = self.get_image_resolution(image)
        target_height = height * self.upscale_factor
        target_width = width * self.upscale_factor
        return target_height, target_width

    def resize_image_with_resize_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        old_height, old_width = self.get_image_resolution(image)
        new_width = int(old_width * self.resize_factor)
        new_height = int(old_height * self.resize_factor)

        match self.resize_factor:
            case factor if factor > 1:
                return opencv_resize(image, (new_width, new_height))
            case factor if factor < 1:
                return opencv_resize(image, (new_width, new_height), interpolation=INTER_AREA)
            case _:
                return image

    def resize_image_with_target_resolution(
            self,
            image: numpy_ndarray, 
            t_height: int,
            t_width: int
            ) -> numpy_ndarray:
        
        old_height, old_width = self.get_image_resolution(image)
        old_resolution = old_height + old_width
        new_resolution = t_height + t_width

        if new_resolution > old_resolution:
            return opencv_resize(image, (t_width, t_height))
        else:
            return opencv_resize(image, (t_width, t_height), interpolation=INTER_AREA) 

    # VIDEO CLASS FUNCTIONS
    def calculate_multiframes_supported_by_gpu(self, video_frame_path: str) -> int:
        resized_video_frame = self.resize_image_with_resize_factor(image_read(video_frame_path))
        height, width = self.get_image_resolution(resized_video_frame)
        image_pixels = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        frames_simultaneously = max_supported_pixels // image_pixels 
        print(f" Frames supported simultaneously by GPU: {frames_simultaneously}")
        return frames_simultaneously

    # TILLING FUNCTIONS
    def video_need_tilling(self, video_frame_path: str) -> bool:       
        resized_video_frame = self.resize_image_with_resize_factor(image_read(video_frame_path))
        height, width = self.get_image_resolution(resized_video_frame)
        image_pixels = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        if image_pixels > max_supported_pixels:
            return True
        else:
            return False

    def image_need_tilling(self, image: numpy_ndarray) -> bool:
        height, width = self.get_image_resolution(image)
        image_pixels = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        if image_pixels > max_supported_pixels:
            return True
        else:
            return False

    def add_alpha_channel(self, image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full((image.shape[0], image.shape[1], 1), 255, dtype=uint8)
            image = numpy_concatenate((image, alpha), axis=2)
        return image

    def calculate_tiles_number(
            self, 
            image: numpy_ndarray, 
            ) -> tuple:
        
        height, width = self.get_image_resolution(image)
        tiles_x = (width + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution
        return tiles_x, tiles_y
    
    def split_image_into_tiles(
            self,
            image: numpy_ndarray, 
            tiles_x: int, 
            tiles_y: int
            ) -> list[numpy_ndarray]:

        img_height, img_width = self.get_image_resolution(image)
        tile_width = img_width // tiles_x
        tile_height = img_height // tiles_y
        tiles = []

        for y in range(tiles_y):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height

            for x in range(tiles_x):
                x_start = x * tile_width
                x_end = (x + 1) * tile_width
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append(tile)

        return tiles

    def combine_tiles_into_image(
            self,
            image: numpy_ndarray,
            tiles: list[numpy_ndarray], 
            t_height: int, 
            t_width: int,
            num_tiles_x: int, 
            ) -> numpy_ndarray:

        match self.get_image_mode(image):
            case "Grayscale": tiled_image = numpy_zeros((t_height, t_width, 3), dtype=uint8)
            case "RGB": tiled_image = numpy_zeros((t_height, t_width, 3), dtype=uint8)
            case "RGBA": tiled_image = numpy_zeros((t_height, t_width, 4), dtype=uint8)

        for tile_index in range(len(tiles)):
            actual_tile = tiles[tile_index]
            tile_height, tile_width = self.get_image_resolution(actual_tile)

            row = tile_index // num_tiles_x
            col = tile_index % num_tiles_x
            y_start = row * tile_height
            y_end = y_start + tile_height
            x_start = col * tile_width
            x_end = x_start + tile_width

            match self.get_image_mode(image):
                case "Grayscale": tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGB": tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGBA": tiled_image[y_start:y_end, x_start:x_end] = self.add_alpha_channel(actual_tile)

        return tiled_image

    # AI CLASS FUNCTIONS
    def normalize_image(self, image: numpy_ndarray) -> tuple:
        range = 255
        if numpy_max(image) > 256: range = 65535
        normalized_image = image / range
        return normalized_image, range
    
    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_transpose(image, (2, 0, 1))
        image = numpy_expand_dims(image, axis=0)
        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:
        onnx_input = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]
        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))
        return onnx_output.astype(float32)

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:    
        match max_range:
            case 255: return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)

    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        image = image.astype(float32)
        image_mode = self.get_image_mode(image)
        image, range = self.normalize_image(image)

        match image_mode:
            case "RGB":
                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                output_image = self.de_normalize_image(onnx_output, range)
                return output_image
            
            case "RGBA":
                alpha = image[:, :, 3]
                image = image[:, :, :3]
                image = opencv_cvtColor(image, COLOR_BGR2RGB)

                image = image.astype(float32)
                alpha = alpha.astype(float32)

                # Image
                image = self.preprocess_image(image)
                onnx_output_image = self.onnxruntime_inference(image)
                onnx_output_image = self.postprocess_output(onnx_output_image)
                onnx_output_image = opencv_cvtColor(onnx_output_image, COLOR_BGR2RGBA)

                # Alpha
                alpha = numpy_expand_dims(alpha, axis=-1)
                alpha = numpy_repeat(alpha, 3, axis=-1)
                alpha = self.preprocess_image(alpha)
                onnx_output_alpha = self.onnxruntime_inference(alpha)
                onnx_output_alpha = self.postprocess_output(onnx_output_alpha)
                onnx_output_alpha = opencv_cvtColor(onnx_output_alpha, COLOR_RGB2GRAY)

                # Fusion Image + Alpha
                onnx_output_image[:, :, 3] = onnx_output_alpha
                output_image = self.de_normalize_image(onnx_output_image, range)
                return output_image
            
            case "Grayscale":
                image = opencv_cvtColor(image, COLOR_GRAY2RGB)
                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                output_image = opencv_cvtColor(onnx_output, COLOR_RGB2GRAY)
                output_image = self.de_normalize_image(onnx_output, range)
                return output_image

    def AI_upscale_with_tilling(self, image: numpy_ndarray) -> numpy_ndarray:
        t_height, t_width = self.calculate_target_resolution(image)
        tiles_x, tiles_y = self.calculate_tiles_number(image)
        tiles_list = self.split_image_into_tiles(image, tiles_x, tiles_y)
        tiles_list = [self.AI_upscale(tile) for tile in tiles_list]
        return self.combine_tiles_into_image(image, tiles_list, t_height, t_width, tiles_x)

    # EXTERNAL FUNCTION
    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:
        resized_image = self.resize_image_with_resize_factor(image)
        
        if self.image_need_tilling(resized_image):
            return self.AI_upscale_with_tilling(resized_image)
        else:
            return self.AI_upscale(resized_image)

# File Utils functions
def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir): 
        shutil.rmtree(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def image_read(file_path: str) -> numpy_ndarray: 
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_frombuffer(file.read(), uint8), IMREAD_UNCHANGED)

def image_write(file_path: str, file_data: numpy_ndarray, file_extension: str = ".jpg") -> None: 
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def copy_file_metadata(
        original_file_path: str, 
        upscaled_file_path: str
        ) -> None:
    
    exiftool_cmd = [
        EXIFTOOL_EXE_PATH, 
        '-fast', 
        '-TagsFromFile', 
        original_file_path, 
        '-overwrite_original', 
        '-all:all',
        '-unsafe',
        '-largetags', 
        upscaled_file_path
    ]
    
    try: 
        subprocess_run(exiftool_cmd, check=True, shell="False")
    except:
        pass

def prepare_output_image_filename(
        image_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_image_extension: str,
        selected_interpolation_factor: float
        ) -> str:
        
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(image_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(image_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected image extension
    to_append += f"{selected_image_extension}"
        
    output_path += to_append
    return output_path

def prepare_output_video_directory_name(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
    
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    output_path += to_append
    return output_path

def prepare_output_video_frame_filename(
        frame_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
    
    file_path_no_extension, _ = os_path_splitext(frame_path)
    output_path = file_path_no_extension
    
    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected image extension
    to_append += ".jpg"
        
    output_path += to_append
    return output_path

def prepare_output_video_filename(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_video_extension: str,
        selected_interpolation_factor: float
        ) -> str:
    
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"
    
    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected video extension
    match selected_video_extension:
        case ".mp4 (x264)": selected_video_extension = ".mp4"
        case ".mp4 (x265)": selected_video_extension = ".mp4"
        case ".avi":        selected_video_extension = ".avi"
            
    to_append += f"{selected_video_extension}"
        
    output_path += to_append
    return output_path

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate

def get_image_resolution(image: numpy_ndarray) -> tuple:
    height = image.shape[0]
    width = image.shape[1]
    return height, width 

def check_if_file_is_video(file: str) -> bool:
    return any(video_extension in file for video_extension in supported_video_extensions)

def interpolate_images_and_save(
        target_path: str,
        starting_image: numpy_ndarray,
        upscaled_image: numpy_ndarray,
        starting_image_importance: float,
        file_extension: str = ".jpg"
        ) -> None:
    
    def add_alpha_channel(image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full((image.shape[0], image.shape[1], 1), 255, dtype=uint8)
            image = numpy_concatenate((image, alpha), axis=2)
        return image
    
    def get_image_mode(image: numpy_ndarray) -> str:
        match image.shape:
            case (rows, cols):
                return "Grayscale"
            case (rows, cols, channels) if channels == 3:
                return "RGB"
            case (rows, cols, channels) if channels == 4:
                return "RGBA"

    ZERO = 0
    upscaled_image_importance = 1 - starting_image_importance
    starting_height, starting_width = get_image_resolution(starting_image)
    target_height, target_width = get_image_resolution(upscaled_image)

    starting_resolution = starting_height + starting_width
    target_resolution = target_height + target_width

    if starting_resolution > target_resolution:
        starting_image = opencv_resize(starting_image, (target_width, target_height), INTER_AREA)
    else:
        starting_image = opencv_resize(starting_image, (target_width, target_height))

    try: 
        if get_image_mode(starting_image) == "RGBA":
            starting_image = add_alpha_channel(starting_image)
            upscaled_image = add_alpha_channel(upscaled_image)

        interpolated_image = opencv_addWeighted(starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, ZERO)
        image_write(target_path, interpolated_image, file_extension)
    except:
        image_write(target_path, upscaled_image, file_extension)

# Streamlit UI
def streamlit_ui():
    st.set_page_config(
        page_title=f"{app_name}",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    st.sidebar.subheader("AI Model Settings")
    
    # AI Model Selection
    
    selected_AI_model = st.sidebar.selectbox(
        "AI Model", 
        AI_models_list,
        index=AI_models_list.index(default_AI_model) if default_AI_model in AI_models_list else 0,
        help="Select the AI model for upscaling"
    )
    
    selected_gpu = st.sidebar.selectbox(
        "GPU", 
        gpus_list,
        index=gpus_list.index(default_gpu) if default_gpu in gpus_list else 0,
        help="Select which GPU to use for processing"
    )
    
    selected_AI_multithreading = st.sidebar.selectbox(
        "AI Multithreading", 
        AI_multithreading_list,
        index=AI_multithreading_list.index(default_AI_multithreading) if default_AI_multithreading in AI_multithreading_list else 0,
        help="Select how many threads to use for video processing"
    )
    
    # Image/Video Settings
    st.sidebar.subheader("Processing Settings")
    resize_factor = st.sidebar.slider(
        "Input Resolution %", 
        min_value=1, 
        max_value=100, 
        value=int(default_resize_factor),
        help="Adjust the input resolution percentage"
    )
    
    selected_interpolation = st.sidebar.selectbox(
        "AI Interpolation", 
        interpolation_list,
        index=interpolation_list.index(default_interpolation) if default_interpolation in interpolation_list else 0,
        help="Select interpolation level between original and upscaled image"
    )
    
    selected_VRAM_limiter = st.sidebar.number_input(
        "GPU VRAM (GB)", 
        min_value=1, 
        max_value=24, 
        value=int(default_VRAM_limiter),
        help="Set the VRAM limit for processing"
    )
    
    selected_cpu_number = st.sidebar.number_input(
        "CPU Threads", 
        min_value=1, 
        max_value=32, 
        value=int(default_cpu_number),
        help="Set the number of CPU threads to use"
    )
    
    # Output Settings
    st.sidebar.subheader("Output Settings")
    selected_image_extension = st.sidebar.selectbox(
        "Image Output Format", 
        image_extension_list,
        index=image_extension_list.index(default_image_extension) if default_image_extension in image_extension_list else 0,
        help="Select the output format for images"
    )
    
    selected_video_extension = st.sidebar.selectbox(
        "Video Output Format", 
        video_extension_list,
        index=video_extension_list.index(default_video_extension) if default_video_extension in video_extension_list else 0,
        help="Select the output format for videos"
    )
    
    selected_keep_frames = st.sidebar.selectbox(
        "Keep Video Frames", 
        keep_frames_list,
        index=keep_frames_list.index(default_keep_frames) if default_keep_frames in keep_frames_list else 0,
        help="Choose whether to keep extracted video frames after processing"
    ) == "Enabled"
    
    # Output path selection
    output_path_options = [OUTPUT_PATH_CODED, "Custom output path"]
    selected_output_path_option = st.sidebar.selectbox(
        "Output Path", 
        output_path_options,
        index=0 if default_output_path == OUTPUT_PATH_CODED else 1,
        help="Select where to save the upscaled files"
    )
    
    if selected_output_path_option == "Custom output path":
        selected_output_path = st.sidebar.text_input(
            "Custom Output Path", 
            value=default_output_path if default_output_path != OUTPUT_PATH_CODED else DOCUMENT_PATH,
            help="Enter a custom path to save the upscaled files"
        )
    else:
        selected_output_path = OUTPUT_PATH_CODED
    
    # Main content area
    st.title(f"{app_name}")
    st.markdown("AI Image/Video Upscaler")
    
    # File upload section
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload images or videos to upscale", 
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv"]
    )
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("Files to Process")
        file_info = []
        
        for file in uploaded_files:
            # Save uploaded file temporarily
            temp_file_path = os.path.join("temp", file.name)
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Display file info
            if check_if_file_is_video(temp_file_path):
                cap = opencv_VideoCapture(temp_file_path)
                width = round(cap.get(CAP_PROP_FRAME_WIDTH))
                height = round(cap.get(CAP_PROP_FRAME_HEIGHT))
                num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
                frame_rate = cap.get(CAP_PROP_FPS)
                duration = num_frames/frame_rate
                minutes = int(duration/60)
                seconds = duration % 60
                cap.release()
                
                # Calculate upscaled dimensions
                resize_factor_decimal = resize_factor / 100
                upscale_factor = 4 if "x4" in selected_AI_model else 2 if "x2" in selected_AI_model else 1
                
                resized_height = int(height * resize_factor_decimal)
                resized_width = int(width * resize_factor_decimal)
                upscaled_height = int(resized_height * upscale_factor)
                upscaled_width = int(resized_width * upscale_factor)
                
                st.write(f"**{file.name}** - Video")
                st.write(f"Resolution: {width}x{height} • Duration: {minutes}m:{round(seconds)}s • {num_frames} frames")
                st.write(f"AI input {resize_factor}% → {resized_width}x{resized_height}")
                st.write(f"AI output x{upscale_factor} → {upscaled_width}x{upscaled_height}")
            else:
                # For images
                img = image_read(temp_file_path)
                height, width = get_image_resolution(img)
                
                # Calculate upscaled dimensions
                resize_factor_decimal = resize_factor / 100
                upscale_factor = 4 if "x4" in selected_AI_model else 2 if "x2" in selected_AI_model else 1
                
                resized_height = int(height * resize_factor_decimal)
                resized_width = int(width * resize_factor_decimal)
                upscaled_height = int(resized_height * upscale_factor)
                upscaled_width = int(resized_width * upscale_factor)
                
                st.write(f"**{file.name}** - Image")
                st.write(f"Resolution: {width}x{height}")
                st.write(f"AI input {resize_factor}% → {resized_width}x{resized_height}")
                st.write(f"AI output x{upscale_factor} → {upscaled_width}x{upscaled_height}")
            
            # Add to file list
            file_info.append({
                "path": temp_file_path,
                "name": file.name,
                "is_video": check_if_file_is_video(temp_file_path)
            })
        
        # Process button
        if st.button("Upscale Files", type="primary"):
            # Calculate VRAM-based tile resolution
            match float(selected_VRAM_limiter):
                case vram if vram <= 2: tiles_resolution = int(512 * very_high_VRAM)
                case vram if vram <= 4: tiles_resolution = int(512 * medium_VRAM)
                case vram if vram <= 6: tiles_resolution = int(512 * low_VRAM)
                case vram if vram <= 8: tiles_resolution = int(512 * very_low_VRAM)
                case _: tiles_resolution = 512
            
            # Get interpolation factor
            match selected_interpolation:
                case "Low": selected_interpolation_factor = 0.3
                case "Medium": selected_interpolation_factor = 0.5
                case "High": selected_interpolation_factor = 0.7
                case _: selected_interpolation_factor = 0
            
            # Get AI multithreading value
            selected_AI_multithreading_value = int(selected_AI_multithreading.split()[0])
            
            # Convert resize factor to decimal
            resize_factor_decimal = resize_factor / 100
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for i, file in enumerate(file_info):
                file_path = file["path"]
                file_number = i + 1
                status_text.write(f"Processing file {file_number}/{len(file_info)}: {file['name']}")
                
                # Initialize AI model
                if i == 0:
                    status_text.write(f"Loading AI model: {selected_AI_model}")
                    AI_instance = AI(selected_AI_model, selected_gpu, resize_factor_decimal, tiles_resolution)
                
                # Process file based on type
                if file["is_video"]:
                    # Create directory for video frames
                    target_directory = prepare_output_video_directory_name(
                        file_path, 
                        selected_output_path,
                        selected_AI_model, 
                        resize_factor_decimal, 
                        selected_interpolation_factor
                    )
                    
                    # Extract frames
                    status_text.write(f"Extracting frames from {file['name']}")
                    os.makedirs(target_directory, exist_ok=True)
                    
                    # Extract frames (simplified for demo)
                    video_capture = opencv_VideoCapture(file_path)
                    frame_count = int(video_capture.get(CAP_PROP_FRAME_COUNT))
                    extracted_frames_paths = []
                    
                    for frame_number in range(frame_count):
                        success, frame = video_capture.read()
                        if success:
                            frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}.jpg"
                            image_write(frame_path, frame)
                            extracted_frames_paths.append(frame_path)
                            
                            # Update progress
                            extraction_progress = (frame_number + 1) / frame_count
                            progress_bar.progress(extraction_progress * 0.3)  # Extraction is 30% of process
                            status_text.write(f"Extracting frames: {int(extraction_progress * 100)}%")
                    
                    video_capture.release()
                    
                    # Upscale frames
                    upscaled_frame_paths = []
                    for j, frame_path in enumerate(extracted_frames_paths):
                        # Prepare output path
                        upscaled_frame_path = prepare_output_video_frame_filename(
                            frame_path, 
                            selected_AI_model, 
                            resize_factor_decimal, 
                            selected_interpolation_factor
                        )
                        upscaled_frame_paths.append(upscaled_frame_path)
                        
                        # Upscale frame
                        starting_frame = image_read(frame_path)
                        upscaled_frame = AI_instance.AI_orchestration(starting_frame)
                        
                        # Apply interpolation if needed
                        if selected_interpolation_factor > 0:
                            interpolate_images_and_save(
                                upscaled_frame_path, 
                                starting_frame, 
                                upscaled_frame, 
                                selected_interpolation_factor
                            )
                        else:
                            image_write(upscaled_frame_path, upscaled_frame)
                        
                        # Update progress
                        upscale_progress = (j + 1) / len(extracted_frames_paths)
                        progress_bar.progress(0.3 + (upscale_progress * 0.6))  # Upscaling is 60% of process
                        status_text.write(f"Upscaling frame {j+1}/{len(extracted_frames_paths)}")
                    
                    # Encode video
                    status_text.write(f"Encoding video: {file['name']}")
                    video_output_path = prepare_output_video_filename(
                        file_path, 
                        selected_output_path,
                        selected_AI_model, 
                        resize_factor_decimal, 
                        selected_video_extension,
                        selected_interpolation_factor
                    )
                    
                    # Encode video (simplified)
                    video_fps = get_video_fps(file_path)
                    video_clip = ImageSequenceClip.ImageSequenceClip(
                        sequence=upscaled_frame_paths, 
                        fps=video_fps
                    )
                    
                    # Determine codec
                    match selected_video_extension:
                        case ".mp4 (x264)": codec = "libx264"
                        case ".mp4 (x265)": codec = "libx265"
                        case ".avi": codec = "png"
                    
                    video_clip.write_videofile(
                        filename=video_output_path,
                        fps=video_fps,
                        codec=codec,
                        threads=selected_cpu_number,
                        logger=None,
                        audio=None,
                        bitrate="12M",
                        preset="ultrafast"
                    )
                    
                    # Copy metadata
                    copy_file_metadata(file_path, video_output_path)
                    
                    # Clean up frames if not keeping them
                    if not selected_keep_frames and os.path.exists(target_directory):
                        shutil.rmtree(target_directory)
                    
                else:
                    # Process image
                    status_text.write(f"Upscaling image: {file['name']}")
                    
                    # Prepare output path
                    upscaled_image_path = prepare_output_image_filename(
                        file_path, 
                        selected_output_path,
                        selected_AI_model, 
                        resize_factor_decimal, 
                        selected_image_extension,
                        selected_interpolation_factor
                    )
                    
                    # Upscale image
                    starting_image = image_read(file_path)
                    upscaled_image = AI_instance.AI_orchestration(starting_image)
                    
                    # Apply interpolation if needed
                    if selected_interpolation_factor > 0:
                        interpolate_images_and_save(
                            upscaled_image_path, 
                            starting_image, 
                            upscaled_image, 
                            selected_interpolation_factor, 
                            selected_image_extension
                        )
                    else:
                        image_write(upscaled_image_path, upscaled_image, selected_image_extension)
                    
                    # Copy metadata
                    copy_file_metadata(file_path, upscaled_image_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(file_info))
                
                # Update overall progress
                progress_bar.progress((i + 1) / len(file_info))
            
            # Processing complete
            progress_bar.progress(1.0)
            status_text.write("All files processed successfully!")
            st.success(f"Upscaling complete! Files saved to: {selected_output_path if selected_output_path != OUTPUT_PATH_CODED else 'Same location as input files'}")
    
    # About section
    

# Run the app
if __name__ == "__main__":
    streamlit_ui()