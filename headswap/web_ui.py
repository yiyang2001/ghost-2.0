"""
Local web UI for GHOST 2.0 head swap. Standard library only, so it runs in the existing ghost2 env.

The models are loaded once and stay on the GPU; open http://<this PC's LAN address>:7860 from a phone on the same Wi-Fi.

Usage:
  python web_ui.py [--host 0.0.0.0] [--port 7860]
"""
import os
import re
import sys
import json
import time
import uuid
import queue
import socket
import argparse
import mimetypes
import threading
import traceback
from types import SimpleNamespace
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPLOAD_DIR = os.path.join(ROOT, 'web_data', 'uploads')
OUTPUT_DIR = os.path.join(ROOT, 'output', 'web')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')
UPLOAD_MAX_AGE = 24 * 3600
SAFE_NAME = re.compile(r'^[\w\-]+\.[A-Za-z0-9]{2,5}$')

JOBS = {}
JOBS_LOCK = threading.Lock()
QUEUE = queue.Queue()
ENGINE = { 'state': 'loading', 'message': 'importing libraries', 'module': None, 'swapper': None }


class JobCancelled(Exception):
	pass


def load_page():
	with open(os.path.join(HERE, 'web_ui.html'), encoding = 'utf-8') as page:
		return page.read()


def safe_extension(name):
	extension = os.path.splitext(name)[1].lower()
	return extension if re.fullmatch(r'\.[a-z0-9]{2,5}', extension) else ''


def public_job(job):
	return { key: value for key, value in job.items() if key not in ('args', 'cancel') }


def clean_uploads():
	now = time.time()
	for name in os.listdir(UPLOAD_DIR):
		path = os.path.join(UPLOAD_DIR, name)
		if os.path.isfile(path) and now - os.path.getmtime(path) > UPLOAD_MAX_AGE:
			os.remove(path)


def run_job(job):
	module = ENGINE['module']
	swapper = ENGINE['swapper']
	options = job['options']
	is_video = job['kind'] == 'video'

	def progress(done, total):
		if job['cancel'].is_set():
			raise JobCancelled()
		elapsed = time.time() - job['started']
		job['done'], job['total'] = done, total
		job['fps'] = round(done / elapsed, 2) if elapsed > 0 and done else 0
		job['eta'] = round((total - done) / job['fps']) if job['fps'] else None

	args = SimpleNamespace(
		source = job['args']['sources'],
		target = job['args']['target'],
		output = job['args']['output'],
		paste = options['paste'],
		start = options['start'],
		end = options['end'],
		smooth = 0.35,
		motion_threshold = 0.08,
		mask_smooth = 0.5,
		pose_smooth = 0.6,
		temporal_blend = options['temporal_blend'],
		keep_source_colors = options['keep_source_colors'],
		crf = options['crf'],
		save_crop = False,
		progress = progress
	)
	swapper.keep_source_colors = args.keep_source_colors
	swapper.set_source(args.source)
	job['started'] = time.time()
	job['status'] = 'running'
	if is_video:
		module.run_video(swapper, args)
	else:
		module.run_image(swapper, args)
	job['result'] = os.path.basename(args.output)


def worker():
	try:
		sys.path.insert(0, HERE)
		import ghost_headswap
		ENGINE['module'] = ghost_headswap
		ENGINE['message'] = 'loading models'
		ENGINE['swapper'] = ghost_headswap.HeadSwapper('./aligner_checkpoints/aligner_1020_gaze_final.ckpt', './blender_checkpoints/blender_lama.ckpt')
		ENGINE['state'] = 'ready'
		ENGINE['message'] = ''
	except BaseException as exception:
		traceback.print_exc()
		ENGINE['state'] = 'error'
		ENGINE['message'] = f'{type(exception).__name__}: {exception}'
		return
	while True:
		job = QUEUE.get()
		if job['cancel'].is_set():
			job['status'] = 'cancelled'
			continue
		try:
			run_job(job)
			job['status'] = 'done'
		except JobCancelled:
			print('[web] job cancelled', flush = True)
			job['status'] = 'cancelled'
			remove_quietly(job['args']['output'])
		except BaseException as exception:
			traceback.print_exc()
			job['status'] = 'error'
			job['error'] = f'{type(exception).__name__}: {exception}'
			remove_quietly(job['args']['output'])
		finally:
			job['finished'] = time.time()
			try:
				import torch
				torch.cuda.empty_cache()
			except Exception:
				pass


def remove_quietly(path):
	# Windows keeps the file locked for a moment after the encoder is killed
	for _ in range(20):
		try:
			os.remove(path)
			return
		except FileNotFoundError:
			return
		except OSError as exception:
			print(f'[web] cannot remove {path}: {exception}', flush = True)
			time.sleep(0.25)


def create_job(payload):
	sources = [ resolve_upload(identifier) for identifier in payload.get('sources', []) ]
	target = resolve_upload(payload.get('target', ''))
	if not sources or not target:
		raise ValueError('need at least one source photo and a target')
	kind = 'video' if target.lower().endswith(VIDEO_EXTENSIONS) else 'image'
	raw = payload.get('options', {})
	options = {
		'paste': 'rect' if raw.get('paste') == 'rect' else 'head',
		'keep_source_colors': bool(raw.get('keep_source_colors', True)),
		'start': max(0.0, float(raw.get('start') or 0)),
		'end': max(0.0, float(raw.get('end') or 0)),
		'temporal_blend': min(0.5, max(0.0, float(raw.get('temporal_blend') or 0))),
		'crf': min(30, max(10, int(raw.get('crf') or 16)))
	}
	identifier = uuid.uuid4().hex[:12]
	output = os.path.join(OUTPUT_DIR, f'{time.strftime("%Y%m%d-%H%M%S")}-{identifier[:6]}.{"mp4" if kind == "video" else "png"}')
	job = {
		'id': identifier, 'status': 'queued', 'kind': kind, 'options': options, 'created': time.time(),
		'started': time.time(), 'finished': None, 'done': 0, 'total': 0, 'fps': 0, 'eta': None,
		'result': None, 'error': None, 'cancel': threading.Event(),
		'args': { 'sources': sources, 'target': target, 'output': output }
	}
	with JOBS_LOCK:
		JOBS[identifier] = job
	QUEUE.put(job)
	return job


def resolve_upload(identifier):
	if not isinstance(identifier, str) or not SAFE_NAME.match(identifier):
		return None
	path = os.path.join(UPLOAD_DIR, identifier)
	return path if os.path.isfile(path) else None


def list_history():
	items = []
	for name in os.listdir(OUTPUT_DIR):
		path = os.path.join(OUTPUT_DIR, name)
		if os.path.isfile(path) and SAFE_NAME.match(name):
			items.append({ 'name': name, 'time': os.path.getmtime(path), 'size': os.path.getsize(path), 'video': name.lower().endswith(VIDEO_EXTENSIONS) })
	running = { job['args']['output'] for job in JOBS.values() if job['status'] in ('queued', 'running') }
	items = [ item for item in items if os.path.join(OUTPUT_DIR, item['name']) not in running ]
	return sorted(items, key = lambda item: item['time'], reverse = True)


class Handler(BaseHTTPRequestHandler):
	protocol_version = 'HTTP/1.1'

	def log_message(self, format, *args):
		pass

	def send_json(self, data, status = 200):
		body = json.dumps(data).encode('utf-8')
		self.send_response(status)
		self.send_header('Content-Type', 'application/json; charset=utf-8')
		self.send_header('Content-Length', str(len(body)))
		self.send_header('Cache-Control', 'no-store')
		self.end_headers()
		self.wfile.write(body)

	def read_json(self):
		length = int(self.headers.get('Content-Length') or 0)
		return json.loads(self.rfile.read(length) or b'{}')

	def do_GET(self):
		url = urlparse(self.path)
		try:
			if url.path == '/':
				body = load_page().encode('utf-8')
				self.send_response(200)
				self.send_header('Content-Type', 'text/html; charset=utf-8')
				self.send_header('Content-Length', str(len(body)))
				self.send_header('Cache-Control', 'no-store')
				self.end_headers()
				self.wfile.write(body)
			elif url.path == '/api/state':
				with JOBS_LOCK:
					jobs = sorted(JOBS.values(), key = lambda job: job['created'], reverse = True)[:20]
					self.send_json({ 'engine': { 'state': ENGINE['state'], 'message': ENGINE['message'] }, 'jobs': [ public_job(job) for job in jobs ], 'history': list_history() })
			elif url.path.startswith('/files/'):
				self.serve_file(os.path.join(OUTPUT_DIR, url.path[len('/files/'):]), download = parse_qs(url.query).get('download') == [ '1' ])
			else:
				self.send_json({ 'error': 'not found' }, 404)
		except (BrokenPipeError, ConnectionResetError):
			pass

	def serve_file(self, path, download):
		name = os.path.basename(path)
		if not SAFE_NAME.match(name) or os.path.dirname(path) != OUTPUT_DIR or not os.path.isfile(path):
			return self.send_json({ 'error': 'not found' }, 404)
		size = os.path.getsize(path)
		start, end = 0, size - 1
		match = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range') or '')
		status = 200
		if match and (match.group(1) or match.group(2)):
			# phones need range requests to play and seek videos
			if match.group(1):
				start = int(match.group(1))
				end = int(match.group(2)) if match.group(2) else end
			else:
				start = max(0, size - int(match.group(2)))
			end = min(end, size - 1)
			if start > end:
				self.send_response(416)
				self.send_header('Content-Range', f'bytes */{size}')
				self.send_header('Content-Length', '0')
				self.end_headers()
				return
			status = 206
		self.send_response(status)
		self.send_header('Content-Type', mimetypes.guess_type(name)[0] or 'application/octet-stream')
		self.send_header('Accept-Ranges', 'bytes')
		self.send_header('Content-Length', str(end - start + 1))
		if status == 206:
			self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
		if download:
			self.send_header('Content-Disposition', f'attachment; filename="{name}"')
		self.end_headers()
		with open(path, 'rb') as file:
			file.seek(start)
			remaining = end - start + 1
			while remaining > 0:
				chunk = file.read(min(1 << 20, remaining))
				if not chunk:
					break
				self.wfile.write(chunk)
				remaining -= len(chunk)

	def do_POST(self):
		url = urlparse(self.path)
		try:
			if url.path == '/api/upload':
				self.handle_upload(url)
			elif url.path == '/api/jobs':
				try:
					job = create_job(self.read_json())
				except (ValueError, TypeError, json.JSONDecodeError) as exception:
					return self.send_json({ 'error': str(exception) }, 400)
				self.send_json(public_job(job))
			elif url.path == '/api/cancel':
				job = JOBS.get(self.read_json().get('id'))
				if job:
					job['cancel'].set()
				self.send_json({ 'ok': bool(job) })
			elif url.path == '/api/delete':
				name = self.read_json().get('name', '')
				if SAFE_NAME.match(name):
					remove_quietly(os.path.join(OUTPUT_DIR, name))
				self.send_json({ 'ok': True })
			else:
				self.send_json({ 'error': 'not found' }, 404)
		except (BrokenPipeError, ConnectionResetError):
			pass

	def handle_upload(self, url):
		extension = safe_extension(parse_qs(url.query).get('name', [ '' ])[0])
		length = int(self.headers.get('Content-Length') or 0)
		if extension not in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS or length <= 0:
			return self.send_json({ 'error': 'unsupported or empty file' }, 400)
		identifier = uuid.uuid4().hex[:16] + extension
		path = os.path.join(UPLOAD_DIR, identifier)
		remaining = length
		with open(path, 'wb') as file:
			while remaining > 0:
				chunk = self.rfile.read(min(1 << 20, remaining))
				if not chunk:
					break
				file.write(chunk)
				remaining -= len(chunk)
		if remaining:
			remove_quietly(path)
			return
		self.send_json({ 'id': identifier })


def lan_addresses():
	addresses = set()
	try:
		for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
			addresses.add(info[4][0])
	except OSError:
		pass
	try:
		probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		probe.connect(('10.255.255.255', 1))
		addresses.add(probe.getsockname()[0])
		probe.close()
	except OSError:
		pass
	return sorted(address for address in addresses if not address.startswith('127.'))


def main():
	parser = argparse.ArgumentParser(description = 'GHOST 2.0 head swap web UI')
	parser.add_argument('--host', default = '0.0.0.0', help = '0.0.0.0 = reachable from the LAN, 127.0.0.1 = this PC only')
	parser.add_argument('--port', type = int, default = 7860)
	args = parser.parse_args()
	os.makedirs(UPLOAD_DIR, exist_ok = True)
	os.makedirs(OUTPUT_DIR, exist_ok = True)
	clean_uploads()
	threading.Thread(target = worker, daemon = True).start()
	server = ThreadingHTTPServer((args.host, args.port), Handler)
	server.daemon_threads = True
	print(f'[web] this PC : http://127.0.0.1:{args.port}')
	if args.host == '0.0.0.0':
		for address in lan_addresses():
			print(f'[web] phone   : http://{address}:{args.port}')
	print('[web] models are loading in the background, the page shows when they are ready. Ctrl+C to stop.', flush = True)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
	main()
