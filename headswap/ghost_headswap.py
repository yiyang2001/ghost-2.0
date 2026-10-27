"""
Inference-only GHOST 2.0 head swap for images and videos.

Loads only the inference parts of the official checkpoints (aligner embedder + generator,
blender generator) and replaces training-only dependencies (insightface, pytorch3d/DECA/EMOCA,
imgaug, wandb, diffusers) so it runs on Windows without a compiler. Face keypoints come from
FaceFusion's yoloface_8n model.

Usage:
  python ghost_headswap.py --source me.jpg [more.jpg ...] --target frame.jpg --output out.png
  python ghost_headswap.py --source me.jpg --target clip.mp4 --output out.mp4
"""
import os
import sys
import gc
import time
import types
import shutil
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALLER_CWD = os.getcwd()
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('TORCH_HOME', os.path.join(ROOT, '.torch'))
os.environ.setdefault('LAMA_MODEL', os.path.join(ROOT, 'weights', 'big-lama.pt'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import onnxruntime as ort
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale

YOLOFACE_PATH = os.environ.get('YOLOFACE_PATH', r'D:\nsfw\facefusion\facefusion\.assets\models\yoloface_8n.onnx')
MASK2FORMER_CONFIG = os.path.join(ROOT, 'weights', 'mask2former-swin-tiny-coco-instance')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')
ort.set_default_logger_severity(3)


def install_stubs():
	# imgaug is only used for training augmentations in the embedder
	class Noop:
		def __init__(self, *args, **kwargs):
			pass

		def __call__(self, *args, **kwargs):
			return args[0] if args else None

	imgaug = types.ModuleType('imgaug')
	augmenters = types.ModuleType('imgaug.augmenters')
	augmenters.Sometimes = augmenters.Affine = augmenters.Sequential = Noop
	imgaug.augmenters = augmenters
	sys.modules['imgaug'] = imgaug
	sys.modules['imgaug.augmenters'] = augmenters

	# EMOCA helper imported by src.utils.crops but only used by emoca_crop (training)
	def bbox2point(left, right, top, bottom, type = 'bbox'):
		if type == 'kpt68':
			old_size = (right - left + bottom - top) / 2 * 1.1
			center = np.array([ right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ]).T
		else:
			old_size = (right - left + bottom - top) / 2
			center = np.array([ right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12 ]).T
		return old_size, center

	names = [ 'repos.emoca', 'repos.emoca.gdl', 'repos.emoca.gdl.datasets', 'repos.emoca.gdl.datasets.ImageDatasetHelpers' ]
	for name in names:
		sys.modules[name] = types.ModuleType(name)
	sys.modules[names[-1]].bbox2point = bbox2point

	# ImageNet backbone weights are overwritten by the GHOST checkpoints, skip downloading them
	for name in [ 'resnext50_32x4d', 'mobilenet_v2', 'vgg19' ]:
		constructor = getattr(torchvision.models, name)
		setattr(torchvision.models, name, (lambda fn: lambda *args, weights = None, **kwargs: fn(*args, weights = None, **kwargs))(constructor))

	# StyleMatte only needs the Mask2Former architecture, its checkpoint holds all weights
	from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

	def from_local_config(cls, *args, **kwargs):
		return cls(Mask2FormerConfig.from_pretrained(MASK2FORMER_CONFIG))

	Mask2FormerForUniversalSegmentation.from_pretrained = classmethod(from_local_config)


install_stubs()

from src.aligner import Embedder, Generator
from src.blender.generator import BlenderGenerator
import src.blender.generator


class FlatMorphology:
	""" flat-kernel dilation/erosion/closing matching src.utils.kornia_morphology (geodesic border),
	done with OpenCV on the CPU: the unfold version needs ~4 GB for the 61x61 glasses kernel """

	@staticmethod
	def apply(operation, tensor, kernel):
		structuring = kernel.detach().cpu().numpy().astype(np.uint8)
		array = tensor.detach().float().cpu().numpy()
		result = np.empty_like(array)
		for index in np.ndindex(array.shape[:-2]):
			result[index] = operation(array[index], structuring)
		return torch.from_numpy(result).to(device = tensor.device, dtype = tensor.dtype)

	@staticmethod
	def dilation(tensor, kernel, **kwargs):
		return FlatMorphology.apply(cv2.dilate, tensor, kernel)

	@staticmethod
	def erosion(tensor, kernel, **kwargs):
		return FlatMorphology.apply(cv2.erode, tensor, kernel)

	@staticmethod
	def closing(tensor, kernel, **kwargs):
		return FlatMorphology.erosion(FlatMorphology.dilation(tensor, kernel), kernel)


# upstream bug: the glasses code path in the blender uses kornia_morphology and np without importing them
src.blender.generator.kornia_morphology = FlatMorphology
src.blender.generator.np = np
from src.utils.crops import norm_crop, wide_src_map
from src.utils.inference import normalize_and_torch
from src.utils.inpainter import LamaInpainter
from src.utils.preblending import get_mask, dilate_torch
from skimage import transform as skimage_transform
from repos.stylematte.stylematte.models import StyleMatte


class Aligner(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.embedder = Embedder(**cfg.model.embed)
		self.gen = Generator(d_por = cfg.model.embed.d_por, d_id = cfg.model.embed.d_id, d_pose = cfg.model.embed.d_pose, d_exp = cfg.model.embed.d_exp, **cfg.model.gen)

	def forward(self, X_dict):
		return self.gen(self.embedder(X_dict))


class Blender(nn.Module):
	def __init__(self):
		super().__init__()
		self.gen = BlenderGenerator()

	def forward(self, batch, inpainter, copy_source_attrb = False):
		oup, gen_h, gen_i, M_Ah, I_tb, M_Ai, I_ag = self.gen(
			batch['face_source'], batch['gray_source'], batch['face_target'],
			batch['mask_source'], batch['mask_target'],
			gt = batch['face_target'],
			M_a_noise = None, M_t_noise = None,
			cycle = False, train = False,
			return_inputs = True,
			copy_source_attrb = copy_source_attrb,
			inpainter = inpainter
		)
		return oup, M_Ah, M_Ai


def load_submodules(module, checkpoint_path, prefixes):
	checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
	state_dict = checkpoint.get('state_dict', checkpoint)
	state_dict = { key: value for key, value in state_dict.items() if key.split('.')[0] in prefixes }
	missing, _ = module.load_state_dict(state_dict, strict = False)
	if missing:
		raise RuntimeError(f'{checkpoint_path}: {len(missing)} missing keys, e.g. {missing[:5]}')
	del checkpoint, state_dict
	gc.collect()


def calc_pseudo_target_bg(target, target_parsing):
	# same result as src.utils.preblending.calc_pseudo_target_bg, without the full pairwise distance matrix
	gaussian_blur = transforms.GaussianBlur(3)
	por_mask = dilate_torch(get_mask(target_parsing), dilate_kernel = 15)
	inp_mask = dilate_torch(por_mask, dilate_kernel = 3) - por_mask
	inp_mask_idxes = inp_mask[0, 0].nonzero()
	por_mask_idxes = por_mask[0, 0].nonzero()
	pseudo_target = target.clone()
	if len(inp_mask_idxes) == 0 or len(por_mask_idxes) == 0:
		return pseudo_target
	inp_mask_colors = gaussian_blur(target)[0, :, inp_mask_idxes[:, 0], inp_mask_idxes[:, 1]].T
	inp_points = inp_mask_idxes.float()
	min_dist_idx = torch.cat([ torch.cdist(chunk.float(), inp_points).argmin(dim = 1) for chunk in por_mask_idxes.split(8192) ])
	pseudo_target[0, :, por_mask_idxes[:, 0], por_mask_idxes[:, 1]] = inp_mask_colors[min_dist_idx].T
	return pseudo_target


def read_image(path):
	image = cv2.imdecode(np.fromfile(path, dtype = np.uint8), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f'cannot read image: {path}')
	return image


def write_image(path, image):
	cv2.imencode(os.path.splitext(path)[1] or '.png', image)[1].tofile(path)


def ort_providers():
	return [ ('CUDAExecutionProvider', { 'device_id': 0, 'arena_extend_strategy': 'kSameAsRequested', 'cudnn_conv_algo_search': 'DEFAULT' }), 'CPUExecutionProvider' ]


class FaceDetector:
	def __init__(self, model_path, size = 640, score = 0.5):
		self.session = ort.InferenceSession(model_path, providers = ort_providers())
		self.input_name = self.session.get_inputs()[0].name
		self.size = size
		self.score = score

	def __call__(self, frame, previous_kps = None):
		height, width = frame.shape[:2]
		scale = min(1.0, self.size / height, self.size / width)
		temp_frame = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
		detect_frame = np.zeros((self.size, self.size, 3), dtype = np.float32)
		detect_frame[:temp_frame.shape[0], :temp_frame.shape[1]] = temp_frame
		detect_frame = detect_frame.transpose(2, 0, 1)[None] / 255.0
		detection = np.squeeze(self.session.run(None, { self.input_name: detect_frame.astype(np.float32) })[0]).T
		keep = detection[:, 4] > self.score
		if not np.any(keep):
			return None
		detection = detection[keep]
		boxes = detection[:, :4] / scale
		kps = detection[:, 5:].reshape(-1, 5, 3)[:, :, :2] / scale
		if previous_kps is not None:
			# follow the same person across frames
			distances = np.linalg.norm(kps.mean(axis = 1) - previous_kps.mean(axis = 0), axis = 1)
			index = int(np.argmin(distances - detection[:, 4]))
		else:
			index = int(np.argmax(boxes[:, 2] * boxes[:, 3] * detection[:, 4]))
		return kps[index].astype(np.float32)


def estimate_wide_matrix(kps, previous_index = None, hysteresis = 1.15):
	templates = wide_src_map[512]
	tform = skimage_transform.SimilarityTransform()
	kps_homogeneous = np.insert(kps, 2, values = np.ones(5), axis = 1)
	matrices, errors = [], []
	for template in templates:
		tform.estimate(kps, template)
		matrix = tform.params[0:2, :]
		projected = np.dot(matrix, kps_homogeneous.T).T
		matrices.append(matrix)
		errors.append(np.sum(np.sqrt(np.sum((projected - template) ** 2, axis = 1))))
	index = int(np.argmin(errors))
	# stick with the previous pose template unless another one is clearly better, avoids crop jumps
	if previous_index is not None and errors[previous_index] <= errors[index] * hysteresis:
		index = previous_index
	return matrices[index], index


class HeadSwapper:
	def __init__(self, aligner_ckpt, blender_ckpt):
		started = time.time()
		cfg_a = OmegaConf.load('./configs/aligner.yaml')
		self.aligner = Aligner(cfg_a)
		load_submodules(self.aligner, aligner_ckpt, { 'embedder', 'gen' })
		self.aligner.eval().cuda()

		self.blender = Blender()
		load_submodules(self.blender, blender_ckpt, { 'gen' })
		self.blender.eval().cuda()

		self.inpainter = LamaInpainter()

		self.segment_model = StyleMatte()
		self.segment_model.load_state_dict(torch.load('./repos/stylematte/stylematte/checkpoints/stylematte_synth.pth', map_location = 'cpu'))
		self.segment_model.eval().cuda()

		# onnxruntime's CUDA convs fall back to a very slow path with torch's cuDNN, so run SegFormer in torch
		self.parsing_model = None
		self.parsing_session = None
		try:
			import onnx
			from onnx2torch import convert
			# pass the loaded proto, converting from a path uses a temp file Windows cannot reopen
			self.parsing_model = convert(onnx.load('./weights/segformer_B5_ce.onnx')).eval().cuda()
		except Exception as exception:
			print(f'[ghost] SegFormer torch conversion failed ({exception}), using onnxruntime')
			self.parsing_session = ort.InferenceSession('./weights/segformer_B5_ce.onnx', providers = ort_providers())
			self.parsing_input = self.parsing_session.get_inputs()[0].name
			self.parsing_outputs = [ output.name for output in self.parsing_session.get_outputs() ]
		self.parsing_mean = torch.tensor([ 0.51315393, 0.48064056, 0.46301059 ], device = 'cuda')[None, :, None, None]
		self.parsing_std = torch.tensor([ 0.21438347, 0.20799829, 0.20304542 ], device = 'cuda')[None, :, None, None]

		self.detector = FaceDetector(YOLOFACE_PATH)
		self.imagenet_normalize = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
		self.source = None
		self.source_embeds = None
		self.pose_embed = None
		self.pose_smooth = 0.0
		self.keep_source_colors = False
		print(f'[ghost] models loaded in {time.time() - started:.1f}s, VRAM allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB')

	def infer_parsing(self, img):
		img = ((img[:, [2, 1, 0], ...] / 2 + 0.5 - self.parsing_mean) / self.parsing_std).float()
		if self.parsing_model is not None:
			output = self.parsing_model(img)
			return (output[0] if isinstance(output, (list, tuple)) else output).float()
		return torch.tensor(self.parsing_session.run(self.parsing_outputs, { self.parsing_input: img.cpu().numpy() })[0], device = 'cuda', dtype = torch.float32)

	def calc_mask(self, img):
		if isinstance(img, np.ndarray):
			img = torch.from_numpy(img).permute(2, 0, 1).cuda()
		if img.max() > 1.:
			img = img / 255.0
		input_t = self.imagenet_normalize(img).unsqueeze(0).float()
		return self.segment_model(input_t)[0][0]

	@torch.no_grad()
	def set_source(self, paths):
		wides, arcs = [], []
		for path in paths:
			image = read_image(path)
			kps = self.detector(image)
			if kps is None:
				raise RuntimeError(f'no face found in source image: {path}')
			M, _ = estimate_wide_matrix(kps)
			wide = cv2.warpAffine(image, M, (512, 512), borderValue = 0.0)
			mask = self.calc_mask(wide)
			wides.append((normalize_and_torch(wide) * mask).unsqueeze(1))
			arcs.append(normalize_and_torch(norm_crop(image, kps)).unsqueeze(1))
		self.source = {
			'face_arc': torch.cat(arcs, dim = 1),
			'face_wide': torch.cat(wides, dim = 1),
			'face_wide_mask': mask
		}
		self.source_embeds = None
		self.reset_temporal()

	def reset_temporal(self):
		self.pose_embed = None

	def embed(self, X_dict):
		""" the source embeddings never change, so compute them once; the pose embedding of the target
		drives every AdaIN parameter of the generator (identity look included), so smooth it over time """
		embedder = self.aligner.embedder
		if self.source_embeds is None:
			embeds = embedder(X_dict)
			self.source_embeds = (embeds['por_embed'], embeds['id_embed'])
			pose_embed = embeds['pose_embed']
		else:
			pose_embed = embedder.pose_encoder(X_dict['target']['face_wide'])
		if self.pose_embed is not None and self.pose_smooth > 0:
			pose_embed = self.pose_smooth * self.pose_embed + (1 - self.pose_smooth) * pose_embed
		self.pose_embed = pose_embed
		return { 'por_embed': self.source_embeds[0], 'id_embed': self.source_embeds[1], 'pose_embed': pose_embed }

	@torch.no_grad()
	def swap_crop(self, frame, kps, M):
		""" returns the blended 512x512 BGR crop and the mask of the changed head region """
		wide = cv2.warpAffine(frame, M, (512, 512), borderValue = 0.0)
		mask_target = self.calc_mask(wide)
		wide_target = normalize_and_torch(wide)
		X_dict = {
			'source': self.source,
			'target': {
				'face_arc': normalize_and_torch(norm_crop(frame, kps)),
				'face_wide': wide_target * mask_target,
				'face_wide_mask': mask_target
			}
		}
		output = self.aligner.gen(self.embed(X_dict))

		target_parsing = self.infer_parsing(wide_target)
		pseudo_norm_target = calc_pseudo_target_bg(wide_target, target_parsing)
		soft_mask = self.calc_mask(((output['fake_rgbs'] * output['fake_segm'])[0, [2, 1, 0], :, :] + 1) / 2)[None]
		new_source = output['fake_rgbs'] * soft_mask[:, None, ...] + pseudo_norm_target * (1 - soft_mask[:, None, ...])

		blender_input = {
			'face_source': new_source,
			'gray_source': rgb_to_grayscale(new_source[0][[2, 1, 0], ...]).unsqueeze(0),
			'face_target': wide_target,
			'mask_source': self.infer_parsing(output['fake_rgbs'] * output['fake_segm']),
			'mask_target': target_parsing
		}
		oup, M_Ah, M_Ai = self.blender(blender_input, inpainter = self.inpainter, copy_source_attrb = self.keep_source_colors)

		crop = oup[0].float().cpu().numpy().transpose(1, 2, 0)
		crop = np.clip((crop / 2 + 0.5) * 255, 0, 255).astype(np.uint8)
		changed = (M_Ah + M_Ai).clamp(0, 1)[0, 0].float().cpu().numpy()
		return crop, changed


def paste_back(frame, crop, changed_mask, M, mode = 'head', dilate = 15, blur = 11, border = 8):
	if mode == 'head':
		mask = cv2.dilate(changed_mask, np.ones((dilate, dilate), np.uint8))
	else:
		mask = np.ones((512, 512), np.float32)
	# keep away from the crop border where the warp has black padding
	mask[:border, :] = mask[-border:, :] = 0
	mask[:, :border] = mask[:, -border:] = 0
	mask = cv2.GaussianBlur(mask, (0, 0), blur)
	M_inv = cv2.invertAffineTransform(M)
	size = (frame.shape[1], frame.shape[0])
	crop_back = cv2.warpAffine(crop, M_inv, size, flags = cv2.INTER_CUBIC, borderValue = 0.0)
	mask_back = cv2.warpAffine(mask, M_inv, size, flags = cv2.INTER_LINEAR, borderValue = 0.0)[..., None]
	return (frame * (1 - mask_back) + crop_back * mask_back).clip(0, 255).astype(np.uint8)


def run_image(swapper, args):
	frame = read_image(args.target)
	kps = swapper.detector(frame)
	if kps is None:
		raise RuntimeError(f'no face found in target image: {args.target}')
	M, _ = estimate_wide_matrix(kps)
	started = time.time()
	crop, changed = swapper.swap_crop(frame, kps, M)
	result = paste_back(frame, crop, changed, M, mode = args.paste)
	write_image(args.output, result)
	if args.save_crop:
		write_image(os.path.splitext(args.output)[0] + '_crop.png', crop)
	print(f'[ghost] image done in {time.time() - started:.2f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB -> {args.output}')


def run_video(swapper, args):
	ffmpeg = shutil.which('ffmpeg')
	if ffmpeg is None:
		raise RuntimeError('ffmpeg not found on PATH')
	capture = cv2.VideoCapture(args.target)
	if not capture.isOpened():
		raise RuntimeError(f'cannot open video: {args.target}')
	fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	start_frame = int(args.start * fps)
	end_frame = min(total, int(args.end * fps)) if args.end else total
	capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	command = [ ffmpeg, '-y', '-loglevel', 'error',
		'-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps}', '-i', '-',
		'-ss', f'{start_frame / fps}', '-i', args.target,
		'-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-crf', str(args.crf), '-preset', 'medium', '-pix_fmt', 'yuv420p',
		'-c:a', 'aac', '-b:a', '192k', '-shortest', args.output ]
	encoder = subprocess.Popen(command, stdin = subprocess.PIPE)

	swapper.pose_smooth = getattr(args, 'pose_smooth', 0.6)
	swapper.reset_temporal()
	smoothed_kps = None
	template_index = None
	smoothed_mask = None
	previous_crop = None
	missed = 0
	started = time.time()
	frame_count = end_frame - start_frame
	progress = getattr(args, 'progress', None)
	try:
		for index in range(frame_count):
			if progress:
				progress(index, frame_count)
			ok, frame = capture.read()
			if not ok:
				break
			kps = swapper.detector(frame, smoothed_kps)
			if kps is None:
				# no face: pass the frame through and restart smoothing
				missed += 1
				smoothed_kps = template_index = smoothed_mask = previous_crop = None
				swapper.reset_temporal()
				encoder.stdin.write(frame.tobytes())
				continue
			if smoothed_kps is None:
				smoothed_kps = kps
			else:
				# adaptive EMA: strong smoothing for jitter, follows quickly on real motion
				face_size = np.linalg.norm(kps[0] - kps[4]) + 1e-6
				motion = np.linalg.norm(kps - smoothed_kps, axis = 1).mean() / face_size
				alpha = float(np.clip(motion / args.motion_threshold, args.smooth, 1.0))
				smoothed_kps = alpha * kps + (1 - alpha) * smoothed_kps
			M, template_index = estimate_wide_matrix(smoothed_kps, template_index)
			try:
				crop, changed = swapper.swap_crop(frame, smoothed_kps, M)
			except torch.cuda.OutOfMemoryError:
				torch.cuda.empty_cache()
				print(f'[ghost] out of VRAM on frame {start_frame + index}, frame passed through')
				encoder.stdin.write(frame.tobytes())
				continue
			if smoothed_mask is not None:
				changed = np.maximum(changed, args.mask_smooth * smoothed_mask)
			smoothed_mask = changed
			if args.temporal_blend > 0 and previous_crop is not None:
				crop = cv2.addWeighted(crop, 1 - args.temporal_blend, previous_crop, args.temporal_blend, 0)
			previous_crop = crop
			encoder.stdin.write(paste_back(frame, crop, changed, M, mode = args.paste).tobytes())
			if (index + 1) % 10 == 0 or index + 1 == frame_count:
				elapsed = time.time() - started
				print(f'[ghost] frame {index + 1}/{frame_count}  {(index + 1) / elapsed:.2f} fps  peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB', flush = True)
	except BaseException:
		# do not leave a half written file and a hanging encoder behind (the web UI keeps the process alive)
		capture.release()
		# ffmpeg may be a shim that spawns the real encoder, closing stdin is what makes the child exit
		try:
			encoder.stdin.close()
		except OSError:
			pass
		encoder.kill()
		encoder.wait()
		raise
	capture.release()
	encoder.stdin.close()
	encoder.wait()
	print(f'[ghost] video done in {time.time() - started:.1f}s, frames without a face: {missed} -> {args.output}')


def main():
	parser = argparse.ArgumentParser(description = 'GHOST 2.0 full head swap (inference only)')
	parser.add_argument('--source', nargs = '+', required = True, help = 'one or more photos of the head to insert')
	parser.add_argument('--target', required = True, help = 'target image or video')
	parser.add_argument('--output', required = True)
	parser.add_argument('--ckpt-a', default = './aligner_checkpoints/aligner_1020_gaze_final.ckpt')
	parser.add_argument('--ckpt-b', default = './blender_checkpoints/blender_lama.ckpt')
	parser.add_argument('--paste', choices = [ 'head', 'rect' ], default = 'head', help = 'paste only the changed head region or the whole crop')
	parser.add_argument('--start', type = float, default = 0.0, help = 'video start in seconds')
	parser.add_argument('--end', type = float, default = 0.0, help = 'video end in seconds (0 = until the end)')
	parser.add_argument('--smooth', type = float, default = 0.35, help = 'minimum landmark EMA factor, lower = steadier')
	parser.add_argument('--motion-threshold', type = float, default = 0.08, help = 'relative motion above which smoothing is disabled')
	parser.add_argument('--pose-smooth', type = float, default = 0.6, help = 'carry-over of the previous pose embedding (0-0.9), higher = steadier head look but laggier expressions')
	parser.add_argument('--mask-smooth', type = float, default = 0.5, help = 'carry-over of the previous paste mask')
	parser.add_argument('--temporal-blend', type = float, default = 0.0, help = 'blend with previous crop (0-0.5), reduces flicker but can ghost')
	parser.add_argument('--keep-source-colors', action = argparse.BooleanOptionalAction, default = True, help = 'keep hair/brows/lips colours of the source instead of recolouring them from the target (upstream default: off)')
	parser.add_argument('--crf', type = int, default = 16)
	parser.add_argument('--save-crop', action = 'store_true')
	args = parser.parse_args()

	args.source = [ os.path.join(CALLER_CWD, path) for path in args.source ]
	args.target = os.path.join(CALLER_CWD, args.target)
	args.output = os.path.join(CALLER_CWD, args.output)
	swapper = HeadSwapper(args.ckpt_a, args.ckpt_b)
	swapper.keep_source_colors = args.keep_source_colors
	swapper.set_source(args.source)
	if args.target.lower().endswith(VIDEO_EXTENSIONS):
		run_video(swapper, args)
	else:
		run_image(swapper, args)


if __name__ == '__main__':
	main()
