# This extension works with [https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
# version: v1.5.1

LOG_PREFIX = '[Prompt-Travel]'

import os
from pathlib import Path
from PIL.Image import Image as PILImage
from enum import Enum
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Callable, Any, Union, Optional, Generic, TypeVar
from traceback import print_exc, format_exc
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms

import gradio as gr
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.editor import concatenate_videoclips, ImageClip
except ImportError:
    print(f'{LOG_PREFIX} package moviepy not installed, will not be able to generate video')

import modules.scripts as scripts
from modules.script_callbacks import on_before_image_saved, ImageSaveParams, on_cfg_denoiser, CFGDenoiserParams, remove_callbacks_for_function
from modules.ui import gr_show
from modules.shared import state, opts, sd_upscalers
from modules.processing import process_images, get_fixed_seed
from modules.processing import Processed, StableDiffusionProcessing as Processing, StableDiffusionProcessingTxt2Img as ProcessingTxt2Img, StableDiffusionProcessingImg2Img as ProcessingImg2Img
from modules.images import resize_image
from modules.sd_samplers_common import single_sample_to_image

try:
    from modules.prompt_parser import DictWithShape
except ImportError:
    '''
    DictWithShape {
    'crossattn': Tensor,
    'vector': Tensor,
    }
    '''
    class DictWithShape(dict):
        def __init__(self, x, shape):
            super().__init__()
            self.update(x)

        @property
        def shape(self):
            return self["crossattn"].shape

Cond = Union[Tensor, DictWithShape]

class Mode(Enum):
    LINEAR  = 'linear'
    REPLACE = 'replace'

class LerpMethod(Enum):
    LERP  = 'lerp'
    SLERP = 'slerp'

class ModeReplaceDim(Enum):
    TOKEN   = 'token'
    CHANNEL = 'channel'
    RANDOM  = 'random'

class ModeReplaceOrder(Enum):
    SIMILAR   = 'similar'
    DIFFERENT = 'different'
    RANDOM    = 'random'

class Gensis(Enum):
    FIXED      = 'fixed'
    SUCCESSIVE = 'successive'
    EMBRYO     = 'embryo'

class VideoFormat(Enum):
    MP4  = 'mp4'
    GIF  = 'gif'
    WEBM = 'webm'

if 'typing':
    T = TypeVar('T')
    @dataclass
    class Ref(Generic[T]): value: T = None

    CondRef = Ref[Tensor]
    StrRef = Ref[str]
    PILImages = List[PILImage]
    RunResults = Tuple[PILImages, str]

if 'consts':
    __ = lambda key, value=None: opts.data.get(f'customscript/prompt_travel.py/txt2img/{key}/value', value)

    LABEL_MODE              = 'Travel mode'
    LABEL_STEPS             = 'Travel steps between stages'
    LABEL_GENESIS           = 'Frame genesis'
    LABEL_DENOISE_W         = 'Denoise strength'
    LABEL_EMBRYO_STEP       = 'Denoise steps for embryo'
    LABEL_LERP_METH         = 'Linear interp method'
    LABEL_REPLACE_DIM       = 'Replace dimension'
    LABEL_REPLACE_ORDER     = 'Replace order'
    LABEL_VIDEO             = 'Ext. export video'
    LABEL_VIDEO_FPS         = 'Video FPS'
    LABEL_VIDEO_FMT         = 'Video file format'
    LABEL_VIDEO_PAD         = 'Pad begin/end frames'
    LABEL_VIDEO_PICK        = 'Pick frame by slice'
    LABEL_UPSCALE           = 'Ext. upscale'
    LABEL_UPSCALE_METH      = 'Upscaler'
    LABEL_UPSCALE_RATIO     = 'Upscale ratio'
    LABEL_UPSCALE_WIDTH     = 'Upscale width'
    LABEL_UPSCALE_HEIGHT    = 'Upscale height'
    LABEL_DEPTH             = 'Ext. depth-image-io (for depth2img models)'
    LABEL_DEPTH_IMG         = 'Depth image file'

    DEFAULT_MODE            = __(LABEL_MODE, Mode.LINEAR.value)
    DEFAULT_STEPS           = __(LABEL_STEPS, 30)
    DEFAULT_GENESIS         = __(LABEL_GENESIS, Gensis.FIXED.value)
    DEFAULT_DENOISE_W       = __(LABEL_DENOISE_W, 1.0)
    DEFAULT_EMBRYO_STEP     = __(LABEL_EMBRYO_STEP, 8)
    DEFAULT_LERP_METH       = __(LABEL_LERP_METH, LerpMethod.LERP.value)
    DEFAULT_REPLACE_DIM     = __(LABEL_REPLACE_DIM, ModeReplaceDim.TOKEN.value)
    DEFAULT_REPLACE_ORDER   = __(LABEL_REPLACE_ORDER, ModeReplaceOrder.RANDOM.value)
    DEFAULT_UPSCALE         = __(LABEL_UPSCALE, False)
    DEFAULT_UPSCALE_METH    = __(LABEL_UPSCALE_METH, 'Lanczos')
    DEFAULT_UPSCALE_RATIO   = __(LABEL_UPSCALE_RATIO, 2.0)
    DEFAULT_UPSCALE_WIDTH   = __(LABEL_UPSCALE_WIDTH, 0)
    DEFAULT_UPSCALE_HEIGHT  = __(LABEL_UPSCALE_HEIGHT, 0)
    DEFAULT_VIDEO           = __(LABEL_VIDEO, True)
    DEFAULT_VIDEO_FPS       = __(LABEL_VIDEO_FPS, 10)
    DEFAULT_VIDEO_FMT       = __(LABEL_VIDEO_FMT, VideoFormat.MP4.value)
    DEFAULT_VIDEO_PAD       = __(LABEL_VIDEO_PAD, 0)
    DEFAULT_VIDEO_PICK      = __(LABEL_VIDEO_PICK, '')
    DEFAULT_DEPTH           = __(LABEL_DEPTH, False)

    CHOICES_MODE            = [x.value for x in Mode]
    CHOICES_LERP_METH       = [x.value for x in LerpMethod]
    CHOICES_GENESIS         = [x.value for x in Gensis]
    CHOICES_REPLACE_DIM     = [x.value for x in ModeReplaceDim]
    CHOICES_REPLACE_ORDER   = [x.value for x in ModeReplaceOrder]
    CHOICES_UPSCALER        = [x.name for x in sd_upscalers]
    CHOICES_VIDEO_FMT       = [x.value for x in VideoFormat]

    EPS = 1e-6


def wrap_cond_align(fn:Callable[..., Cond]):
    def cond_align(condA:Cond, condB:Cond) -> Tuple[Cond, Cond]:
        def align_tensor(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
            d = x.shape[0] - y.shape[0]
            if   d < 0: x = F.pad(x, (0, 0, 0, -d))
            elif d > 0: y = F.pad(y, (0, 0, 0,  d))
            return x, y

        if isinstance(condA, dict):     # SDXL
            for key in condA:
                condA[key], condB[key] = align_tensor(condA[key], condB[key])
        else:
            condA, condB = align_tensor(condA, condB)
        return condA, condB

    def wrapper(condA:Cond, condB:Cond, *args, **kwargs) -> Cond:
        condA, condB = cond_align(condA, condB)
        if isinstance(condA, dict):     # SDXL
            stacked = { key: fn(condA[key], condB[key], *args, **kwargs) for key in condA }
            return DictWithShape(stacked, stacked['crossattn'].shape)
        else:
            return fn(condA, condB, *args, **kwargs)
    return wrapper

@wrap_cond_align
def weighted_sum(A:Tensor, B:Tensor, alpha:float) -> Tensor:
    ''' linear interpolate on latent space of condition '''

    return (1 - alpha) * A + (alpha) * B

@wrap_cond_align
def geometric_slerp(A:Tensor, B:Tensor, alpha:float) -> Tensor:
    ''' spherical linear interpolation on latent space of condition, ref: https://en.wikipedia.org/wiki/Slerp '''

    A_n = A / torch.norm(A, dim=-1, keepdim=True)   # [T=77, D=768]
    B_n = B / torch.norm(B, dim=-1, keepdim=True)

    dot = (A_n * B_n).sum(dim=-1, keepdim=True)     # [T=77, D=1]
    omega = torch.acos(dot)                         # [T=77, D=1]
    so = torch.sin(omega)                           # [T=77, D=1]

    slerp = (torch.sin((1 - alpha) * omega) / so) * A + (torch.sin(alpha * omega) / so) * B

    mask: Tensor = dot > 0.9995                     # [T=77, D=1]
    if not mask.any():
        return slerp
    else:
        lerp = (1 - alpha) * A + (alpha) * B
        return torch.where(mask, lerp, slerp)       # use simple lerp when angle very close to avoid NaN

@wrap_cond_align
def replace_until_match(A:Tensor, B:Tensor, count:int, dist:Tensor, order:str=ModeReplaceOrder.RANDOM) -> Tensor:
    ''' value substite on condition tensor; will inplace modify `dist` '''

    def index_tensor_to_tuple(index:Tensor) -> Tuple[Tensor, ...]:
        return tuple([index[..., i] for i in range(index.shape[-1])])       # tuple([nDiff], ...)

    # mask: [T=77, D=768], [T=77] or [D=768]
    mask = dist > EPS
    # idx_diff: [nDiff, nDim=2] or [nDiff, nDim=1]
    idx_diff = torch.nonzero(mask)
    n_diff = len(idx_diff)

    if order == ModeReplaceOrder.RANDOM:
        sel = np.random.choice(range(n_diff), size=count, replace=False) if n_diff > count else slice(None)
    else:
        val_diff = dist[index_tensor_to_tuple(idx_diff)]    # [nDiff]

        if order == ModeReplaceOrder.SIMILAR:
            sorted_index = val_diff.argsort()
        elif order == ModeReplaceOrder.DIFFERENT:
            sorted_index = val_diff.argsort(descending=True)
        else: raise ValueError(f'unknown replace_order: {order}')

        sel = sorted_index[:count]

    idx_diff_sel = idx_diff[sel, ...]       # [cnt] => [cnt, nDim]
    idx_diff_sel_tp = index_tensor_to_tuple(idx_diff_sel)
    dist[idx_diff_sel_tp] = 0.0
    mask[idx_diff_sel_tp] = False

    if mask.shape != A.shape:   # cond.shape = [T=77, D=768]
        mask_len = mask.shape[0]
        if   mask_len == A.shape[0]: mask = mask.unsqueeze(1)
        elif mask_len == A.shape[1]: mask = mask.unsqueeze(0)
        else: raise ValueError(f'unknown mask.shape: {mask.shape}')
        mask = mask.expand_as(A)

    return mask * A + ~mask * B


def get_next_sequence_number(path:str) -> int:
    """ Determines and returns the next sequence number to use when saving an image in the specified directory. The sequence starts at 0. """
    result = -1
    dir = Path(path)
    for file in dir.iterdir():
        if not file.is_dir(): continue
        try:
            num = int(file.name)
            if num > result: result = num
        except ValueError:
            pass
    return result + 1

def update_img2img_p(p:Processing, imgs:PILImages, denoising_strength:float=0.75) -> ProcessingImg2Img:
    if isinstance(p, ProcessingImg2Img):
        p.init_images = imgs
        p.denoising_strength = denoising_strength
        return p

    if isinstance(p, ProcessingTxt2Img):
        KNOWN_KEYS = [      # see `StableDiffusionProcessing.__init__()`
            'sd_model',
            'outpath_samples',
            'outpath_grids',
            'prompt',
            'styles',
            'seed',
            'subseed',
            'subseed_strength',
            'seed_resize_from_h',
            'seed_resize_from_w',
            'seed_enable_extras',
            'sampler_name',
            'batch_size',
            'n_iter',
            'steps',
            'cfg_scale',
            'width',
            'height',
            'restore_faces',
            'tiling',
            'do_not_save_samples',
            'do_not_save_grid',
            'extra_generation_params',
            'overlay_images',
            'negative_prompt',
            'eta',
            'do_not_reload_embeddings',
            #'denoising_strength',
            'ddim_discretize',
            's_min_uncond',
            's_churn',
            's_tmax',
            's_tmin',
            's_noise',
            'override_settings',
            'override_settings_restore_afterwards',
            'sampler_index',
            'script_args',
        ]
        kwargs = { k: getattr(p, k) for k in dir(p) if k in KNOWN_KEYS }    # inherit params
        return ProcessingImg2Img(
            init_images=imgs,
            denoising_strength=denoising_strength,
            **kwargs,
        )

def parse_slice(picker:str) -> Optional[slice]:
    if not picker.strip(): return None
    
    to_int = lambda s: None if not s else int(s)
    segs = [to_int(x.strip()) for x in picker.strip().split(':')]
    
    start, stop, step = None, None, None
    if   len(segs) == 1:        stop,      = segs
    elif len(segs) == 2: start, stop       = segs
    elif len(segs) == 3: start, stop, step = segs
    else: raise ValueError
    
    return slice(start, stop, step)

def parse_resolution(width:int, height:int, upscale_ratio:float, upscale_width:int, upscale_height:int) -> Tuple[bool, Tuple[int, int]]:
    if upscale_width == upscale_height == 0:
        if upscale_ratio == 1.0:
            return False, (width, height)
        else:
            return True, (round(width * upscale_ratio), round(height * upscale_ratio))
    else:
        if upscale_width  == 0: upscale_width  = round(width  * upscale_height / height)
        if upscale_height == 0: upscale_height = round(height * upscale_width  / width)
        return (width != upscale_width and height != upscale_height), (upscale_width, upscale_height)


def upscale_image(img:PILImage, width:int, height:int, upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int) -> PILImage:
    if upscale_meth == 'None': return img
    need_upscale, (tgt_w, tgt_h) = parse_resolution(width, height, upscale_ratio, upscale_width, upscale_height)
    if need_upscale:
        if 'show_debug': print(f'>> upscale: ({width}, {height}) => ({tgt_w}, {tgt_h})')

        if max(tgt_w / width, tgt_h / height) > 4:      # must split into two rounds for NN model capatibility
            hf_w, hf_h = round(width * 4), round(height * 4)
            img = resize_image(0, img, hf_w, hf_h, upscaler_name=upscale_meth)
        img = resize_image(0, img, tgt_w, tgt_h, upscaler_name=upscale_meth)
    return img

def save_video(imgs:PILImages, video_slice:slice, video_pad:int, video_fps:float, video_fmt:VideoFormat, fbase:str):
    if len(imgs) <= 1 or 'ImageSequenceClip' not in globals(): return

    try:
        # arrange frames
        if video_slice:   imgs = imgs[video_slice]
        if video_pad > 0: imgs = [imgs[0]] * video_pad + imgs + [imgs[-1]] * video_pad

        # export video
        seq: List[np.ndarray] = [np.asarray(img) for img in imgs]
        try:
            clip = ImageSequenceClip(seq, fps=video_fps)
        except:     # images may have different size (small probability due to upscaler)
            clip = concatenate_videoclips([ImageClip(img, duration=1/video_fps) for img in seq], method='compose')
            clip.fps = video_fps
        if   video_fmt == VideoFormat.MP4:  clip.write_videofile(fbase + '.mp4',  verbose=False, audio=False)
        elif video_fmt == VideoFormat.WEBM: clip.write_videofile(fbase + '.webm', verbose=False, audio=False)
        elif video_fmt == VideoFormat.GIF:  clip.write_gif      (fbase + '.gif',  loop=True)
    except: print_exc()


class on_cfg_denoiser_wrapper:
    def __init__(self, callback_fn:Callable):
        self.callback_fn = callback_fn
    def __enter__(self):
        on_cfg_denoiser(self.callback_fn)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        remove_callbacks_for_function(self.callback_fn)

class p_steps_overrider:
    def __init__(self, p:Processing, steps:int=1):
        self.p = p
        self.steps = steps
        self.steps_saved = self.p.steps
    def __enter__(self):
        self.p.steps = self.steps
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.steps = self.steps_saved

class p_save_samples_overrider:
    def __init__(self, p:Processing, save:bool=True):
        self.p = p
        self.save = save
        self.do_not_save_samples_saved = self.p.do_not_save_samples
    def __enter__(self):
        self.p.do_not_save_samples = not self.save
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.do_not_save_samples = self.do_not_save_samples_saved

def get_cond_callback(refs:List[CondRef], params:CFGDenoiserParams):
    if params.sampling_step > 0: return
    values: List[Cond] = [
        params.text_cond,        # [B=1, L= 77, D=768/2048]
        params.text_uncond,      # [B=1, L=231, D=768/2048]
    ]
    for i, ref in enumerate(refs):
        ref.value = values[i]

def set_cond_callback(refs:List[CondRef], params:CFGDenoiserParams):
    values: List[Cond] = [
        params.text_cond,        # [B=1, L= 77, D=768/2048]
        params.text_uncond,      # [B=1, L=231, D=768/2048]
    ]
    for i, ref in enumerate(refs):
        refv = ref.value
        if isinstance(refv, dict):   # SDXL
            for key in refv:
                values[i][key].data = refv[key]
        else:
            values[i].data = refv

def get_latent_callback(ref:CondRef, embryo_step:int, params:CFGDenoiserParams):
    if params.sampling_step != embryo_step: return
    ref.value = params.x

def set_latent_callback(ref:CondRef, embryo_step:int, params:CFGDenoiserParams):
    if params.sampling_step != embryo_step: return
    params.x.data = ref.value


def switch_to_stage_binding_(self:'Script', i:int):
    if 'show_debug':
        print(f'[stage {i+1}/{self.n_stages}]')
        print(f'  pos prompt: {self.pos_prompts[i]}')
        if hasattr(self, 'neg_prompts'):
            print(f'  neg prompt: {self.neg_prompts[i]}')
    self.p.prompt = self.pos_prompts[i]
    if hasattr(self, 'neg_prompts'):
        self.p.negative_prompt = self.neg_prompts[i]
    self.p.subseed = self.subseed

def process_p_binding_(self:'Script', append:bool=True, save:bool=True) -> PILImages:
    assert hasattr(self, 'images') and hasattr(self, 'info'), 'unknown logic, "images" and "info" not initialized'
    with p_save_samples_overrider(self.p, save):
        proc = process_images(self.p)
    if save:
        if not self.info.value: self.info.value = proc.info
        if append: self.images.extend(proc.images)
        if self.genesis == Gensis.SUCCESSIVE:
            self.p = update_img2img_p(self.p, self.images[-1:], self.denoise_w)
        return proc.images


class Script(scripts.Script):

    def title(self):
        return 'Prompt Travel'

    def describe(self):
        return 'Travel from one prompt to another in the text encoder latent space.'

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Row(variant='compact') as tab_mode:
            mode          = gr.Radio   (label=LABEL_MODE,          value=lambda: DEFAULT_MODE,          choices=CHOICES_MODE)
            lerp_meth     = gr.Dropdown(label=LABEL_LERP_METH,     value=lambda: DEFAULT_LERP_METH,     choices=CHOICES_LERP_METH)
            replace_dim   = gr.Dropdown(label=LABEL_REPLACE_DIM,   value=lambda: DEFAULT_REPLACE_DIM,   choices=CHOICES_REPLACE_DIM,   visible=False)
            replace_order = gr.Dropdown(label=LABEL_REPLACE_ORDER, value=lambda: DEFAULT_REPLACE_ORDER, choices=CHOICES_REPLACE_ORDER, visible=False)

            def switch_mode(mode:str):
                show_meth = Mode(mode) == Mode.LINEAR
                show_repl = Mode(mode) == Mode.REPLACE
                return [gr_show(x) for x in [show_meth, show_repl, show_repl]]
            mode.change(switch_mode, inputs=[mode], outputs=[lerp_meth, replace_dim, replace_order], show_progress=False)

        with gr.Row(variant='compact') as tab_param:
            steps       = gr.Text    (label=LABEL_STEPS,       value=lambda: DEFAULT_STEPS, max_lines=1)
            genesis     = gr.Dropdown(label=LABEL_GENESIS,     value=lambda: DEFAULT_GENESIS, choices=CHOICES_GENESIS)
            denoise_w   = gr.Slider  (label=LABEL_DENOISE_W,   value=lambda: DEFAULT_DENOISE_W, minimum=0.0, maximum=1.0, visible=False)
            embryo_step = gr.Text    (label=LABEL_EMBRYO_STEP, value=lambda: DEFAULT_EMBRYO_STEP, max_lines=1, visible=False)

            def switch_genesis(genesis:str):
                show_dw = Gensis(genesis) == Gensis.SUCCESSIVE    # show 'denoise_w' for 'successive'
                show_es = Gensis(genesis) == Gensis.EMBRYO        # show 'embryo_step' for 'embryo'
                return [gr_show(x) for x in [show_dw, show_es]]
            genesis.change(switch_genesis, inputs=[genesis], outputs=[denoise_w, embryo_step], show_progress=False)

        with gr.Row(variant='compact', visible=DEFAULT_DEPTH) as tab_ext_depth:
            depth_img = gr.Image(label=LABEL_DEPTH_IMG, source='upload', type='pil', image_mode=None)

        with gr.Row(variant='compact', visible=DEFAULT_UPSCALE) as tab_ext_upscale:
            upscale_meth   = gr.Dropdown(label=LABEL_UPSCALE_METH,   value=lambda: DEFAULT_UPSCALE_METH,   choices=CHOICES_UPSCALER)
            upscale_ratio  = gr.Slider  (label=LABEL_UPSCALE_RATIO,  value=lambda: DEFAULT_UPSCALE_RATIO,  minimum=1.0, maximum=16.0, step=0.1)
            upscale_width  = gr.Slider  (label=LABEL_UPSCALE_WIDTH,  value=lambda: DEFAULT_UPSCALE_WIDTH,  minimum=0, maximum=2048, step=8)
            upscale_height = gr.Slider  (label=LABEL_UPSCALE_HEIGHT, value=lambda: DEFAULT_UPSCALE_HEIGHT, minimum=0, maximum=2048, step=8)

        with gr.Row(variant='compact', visible=DEFAULT_VIDEO) as tab_ext_video:
            video_fmt  = gr.Dropdown(label=LABEL_VIDEO_FMT,  value=lambda: DEFAULT_VIDEO_FMT, choices=CHOICES_VIDEO_FMT)
            video_fps  = gr.Number  (label=LABEL_VIDEO_FPS,  value=lambda: DEFAULT_VIDEO_FPS)
            video_pad  = gr.Number  (label=LABEL_VIDEO_PAD,  value=lambda: DEFAULT_VIDEO_PAD,  precision=0)
            video_pick = gr.Text    (label=LABEL_VIDEO_PICK, value=lambda: DEFAULT_VIDEO_PICK, max_lines=1)

        with gr.Row(variant='compact') as tab_ext:
            ext_video   = gr.Checkbox(label=LABEL_VIDEO,   value=lambda: DEFAULT_VIDEO) 
            ext_upscale = gr.Checkbox(label=LABEL_UPSCALE, value=lambda: DEFAULT_UPSCALE) 
            ext_depth   = gr.Checkbox(label=LABEL_DEPTH,   value=lambda: DEFAULT_DEPTH)
        
            ext_video  .change(gr_show, inputs=ext_video,   outputs=tab_ext_video,   show_progress=False)
            ext_upscale.change(gr_show, inputs=ext_upscale, outputs=tab_ext_upscale, show_progress=False)
            ext_depth  .change(gr_show, inputs=ext_depth,   outputs=tab_ext_depth,   show_progress=False)

        with gr.Accordion(label="Structual Similarity Index Metric", open=False):
            gr.Markdown(
                "If this is set to something other than 0, the script will first"
                " generate the steps you've specified above,                 but then"
                " take a second pass and fill in the gaps between images that differ"
                " too much according to Structual Similarity Index Metric \n           "
                "         *Only implemented for linear travel and only for fixed or"
                " successive frame genesis*"
            )
            ssim_diff = gr.Slider(
                label="SSIM threshold", value=0.0, minimum=0.0, maximum=1.0, step=0.01
            )
            ssim_ccrop = gr.Slider(
                label="SSIM CenterCrop%", value=0, minimum=0, maximum=100, step=1
            )
            substep_min = gr.Number(label="SSIM minimum step", value=0.0001)
            ssim_diff_min = gr.Slider(
                label="SSIM min threshold", value=75, minimum=0, maximum=100, step=1
            )

        return [
            mode, lerp_meth, replace_dim, replace_order,
            steps, genesis, denoise_w, embryo_step,
            depth_img,
            upscale_meth, upscale_ratio, upscale_width, upscale_height,
            video_fmt, video_fps, video_pad, video_pick,
            ext_video, ext_upscale, ext_depth, ssim_diff, ssim_ccrop,
            substep_min, ssim_diff_min
        ]

    def run(self, p:Processing, 
            mode:str, lerp_meth:str, replace_dim:str, replace_order:str,
            steps:str, genesis:str, denoise_w:float, embryo_step:str,
            depth_img:PILImage,
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int,
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            ext_video:bool, ext_upscale:bool, ext_depth:bool,
            ssim_diff: float, ssim_ccrop:int,
            substep_min:float, ssim_diff_min:int
        ):
        
        # enum lookup
        mode: Mode                      = Mode(mode)
        lerp_meth: LerpMethod           = LerpMethod(lerp_meth)
        replace_dim: ModeReplaceDim     = ModeReplaceDim(replace_dim)
        replace_order: ModeReplaceOrder = ModeReplaceOrder(replace_order)
        genesis: Gensis                 = Gensis(genesis)
        video_fmt: VideoFormat          = VideoFormat(video_fmt)

        # Param check & type convert
        if ext_video:
            if video_pad <  0: return Processed(p, [], p.seed, f'video_pad must >= 0, but got {video_pad}')
            if video_fps <= 0: return Processed(p, [], p.seed, f'video_fps must > 0, but got {video_fps}')
            try: video_slice = parse_slice(video_pick)
            except: return Processed(p, [], p.seed, 'syntax error in video_slice')
        if genesis == Gensis.EMBRYO:
            try: x = float(embryo_step)
            except: return Processed(p, [], p.seed, f'embryo_step is not a number: {embryo_step}')
            if x <= 0: return Processed(p, [], p.seed, f'embryo_step must > 0, but got {embryo_step}')
            embryo_step: int = round(x * p.steps if x < 1.0 else x) ; del x

        # Prepare prompts & steps
        prompt_pos = p.prompt.strip()
        if not prompt_pos: return Processed(p, [], p.seed, 'positive prompt should not be empty :(')
        pos_prompts = [p.strip() for p in prompt_pos.split('\n') if p.strip()]
        if len(pos_prompts) == 1: return Processed(p, [], p.seed, 'should specify at least two lines of prompt to travel between :(')
        if genesis == Gensis.EMBRYO and len(pos_prompts) > 2: return Processed(p, [], p.seed, 'processing with "embryo" genesis takes exactly two lines of prompt :(')
        prompt_neg = p.negative_prompt.strip()
        neg_prompts = [p.strip() for p in prompt_neg.split('\n') if p.strip()]
        if len(neg_prompts) == 0: neg_prompts = ['']
        n_stages = max(len(pos_prompts), len(neg_prompts))
        while len(pos_prompts) < n_stages: pos_prompts.append(pos_prompts[-1])
        while len(neg_prompts) < n_stages: neg_prompts.append(neg_prompts[-1])

        try: steps: List[int] = [int(s.strip()) for s in steps.strip().split(',')]
        except: return Processed(p, [], p.seed, f'cannot parse steps option: {steps}')
        if len(steps) == 1:
            steps = [steps[0]] * (n_stages - 1)
        elif len(steps) != n_stages - 1:
            return Processed(p, [], p.seed, f'stage count mismatch: you have {n_stages} prompt stages, but specified {len(steps)} steps; should assure len(steps) == len(stages) - 1')
        n_frames = sum(steps) + n_stages
        if 'show_debug':
            print('n_stages:', n_stages)
            print('n_frames:', n_frames)
            print('steps:', steps)
        steps.insert(0, -1)     # fixup the first stage

        # Custom saving path
        travel_path = os.path.join(p.outpath_samples, 'prompt_travel')
        os.makedirs(travel_path, exist_ok=True)
        travel_number = get_next_sequence_number(travel_path)
        self.log_dp = os.path.join(travel_path, f'{travel_number:05}')
        p.outpath_samples = self.log_dp
        os.makedirs(self.log_dp, exist_ok=True)
        #self.log_fp = os.path.join(self.log_dp, 'log.txt')

        # Force batch count and batch size to 1
        p.n_iter     = 1
        p.batch_size = 1

        # Random unified const seed
        p.seed = get_fixed_seed(p.seed)     # fix it to assure all processes using the same major seed
        self.subseed = p.subseed            # stash it to allow using random subseed for each process (when -1)
        if 'show_debug':
            print('seed:',             p.seed)
            print('subseed:',          p.subseed)
            print('subseed_strength:', p.subseed_strength)

        # Start job
        state.job_count = n_frames

        # Pack parameters
        self.pos_prompts    = pos_prompts
        self.neg_prompts    = neg_prompts
        self.steps          = steps
        self.genesis        = genesis
        self.denoise_w      = denoise_w
        self.embryo_step    = embryo_step
        self.lerp_meth      = lerp_meth
        self.replace_dim    = replace_dim
        self.replace_order  = replace_order
        self.n_stages       = n_stages
        self.n_frames       = n_frames
        self.ssim_diff      = ssim_diff
        self.ssim_ccrop     = ssim_ccrop
        self.substep_min    = substep_min
        self.ssim_diff_min  = ssim_diff_min

        def upscale_image_callback(params:ImageSaveParams):
            params.image = upscale_image(params.image, p.width, p.height, upscale_meth, upscale_ratio, upscale_width, upscale_height)

        # Dispatch
        self.p: Processing = p
        self.images: PILImages = []
        self.info: StrRef = Ref()
        try:
            if ext_upscale: on_before_image_saved(upscale_image_callback)
            if ext_depth: self.ext_depth_preprocess(p, depth_img)

            runner = getattr(self, f'run_{mode.value}')
            if not runner: return Processed(p, [], p.seed, f'no runner found for mode: {mode.value}')
            runner()
        except:
            e = format_exc()
            print(e)
            self.info.value = e
        finally:
            if ext_depth: self.ext_depth_postprocess(p, depth_img)
            if ext_upscale: remove_callbacks_for_function(upscale_image_callback)

        # Save video
        if ext_video: save_video(self.images, video_slice, video_pad, video_fps, video_fmt, os.path.join(self.log_dp, f'travel-{travel_number:05}'))

        return Processed(p, self.images, p.seed, self.info.value)

    def run_linear(self):
        # dispatch for special case
        if self.genesis == Gensis.EMBRYO: return self.run_linear_embryo()

        lerp_fn = weighted_sum if self.lerp_meth == LerpMethod.LERP else geometric_slerp

        if 'auxiliary':
            switch_to_stage = partial(switch_to_stage_binding_, self)
            process_p = partial(process_p_binding_, self)

            from_pos_hidden:  CondRef = Ref()
            from_neg_hidden:  CondRef = Ref()
            to_pos_hidden:    CondRef = Ref()
            to_neg_hidden:    CondRef = Ref()
            inter_pos_hidden: CondRef = Ref()
            inter_neg_hidden: CondRef = Ref()

        # Step 1: draw the init image
        switch_to_stage(0)
        with on_cfg_denoiser_wrapper(partial(get_cond_callback, [from_pos_hidden, from_neg_hidden])):
            process_p()

        # travel through stages
        for i in range(1, self.n_stages):
            if state.interrupted: break

            state.job = f'{i}/{self.n_frames}'
            state.job_no = i + 1

            # only change target prompts
            switch_to_stage(i)
            with on_cfg_denoiser_wrapper(partial(get_cond_callback, [to_pos_hidden, to_neg_hidden])):
                if self.genesis == Gensis.FIXED:
                    imgs = process_p(append=False)      # stash it to make order right
                elif self.genesis == Gensis.SUCCESSIVE:
                    with p_steps_overrider(self.p, steps=1):    # ignore final image, only need cond
                        process_p(save=False, append=False)
                else: raise ValueError(f'invalid genesis: {self.genesis.value}')

            # Step 2: draw the interpolated images
            is_break_iter = False
            n_inter = self.steps[i]
            for t in range(1, n_inter + (1 if self.genesis == Gensis.SUCCESSIVE else 0)):
                if state.interrupted: is_break_iter = True ; break

                alpha = t / n_inter     # [1/T, 2/T, .. T-1/T] (+ [T/T])?
                self.interpolate(
                    lerp_fn=lerp_fn,
                    from_pos_hidden=from_pos_hidden,
                    from_neg_hidden=from_neg_hidden,
                    to_pos_hidden=to_pos_hidden,
                    to_neg_hidden=to_neg_hidden,
                    inter_pos_hidden=inter_pos_hidden,
                    inter_neg_hidden=inter_neg_hidden,
                    alpha=alpha,
                )
                with on_cfg_denoiser_wrapper(partial(set_cond_callback, [inter_pos_hidden, inter_neg_hidden])):
                    process_p()

            if is_break_iter: break

            # Step 3: append the final stage
            if self.genesis != Gensis.SUCCESSIVE: self.images.extend(imgs)

            if self.ssim_diff > 0:
                # SSIM
                (
                    skip_count,
                    not_better,
                    skip_ssim_min,
                    min_step,
                    interpolated_images,
                ) = self.ssim_loop(
                    p=self.p,
                    ssim_diff=self.ssim_diff,
                    ssim_ccrop=self.ssim_ccrop,
                    ssim_diff_min=self.ssim_diff_min,
                    substep_min=self.substep_min,
                    prompt_images=self.images[-(n_inter + 1) :],
                    lerp_fn=lerp_fn,
                    process_p=process_p,
                    from_pos_hidden=from_pos_hidden,
                    from_neg_hidden=from_neg_hidden,
                    to_pos_hidden=to_pos_hidden,
                    to_neg_hidden=to_neg_hidden,
                    inter_pos_hidden=inter_pos_hidden,
                    inter_neg_hidden=inter_neg_hidden,
                )
                self.images = self.images[: -(n_inter + 1)] + interpolated_images

            
            # move to next stage
            from_pos_hidden.value, from_neg_hidden.value = to_pos_hidden.value, to_neg_hidden.value
            inter_pos_hidden.value, inter_neg_hidden.value = None, None

    def interpolate(
        self,
        lerp_fn,
        from_pos_hidden,
        from_neg_hidden,
        to_pos_hidden,
        to_neg_hidden,
        inter_pos_hidden,
        inter_neg_hidden,
        alpha,
    ):
        inter_pos_hidden.value = lerp_fn(
            from_pos_hidden.value, to_pos_hidden.value, alpha
        )
        inter_neg_hidden.value = lerp_fn(
            from_neg_hidden.value, to_neg_hidden.value, alpha
        )

    def run_linear_embryo(self):
        ''' NOTE: this procedure has special logic, we separate it from run_linear() so far '''

        lerp_fn  = weighted_sum if self.lerp_meth == LerpMethod.LERP else geometric_slerp
        n_frames = self.steps[1] + 2

        if 'auxiliary':
            switch_to_stage = partial(switch_to_stage_binding_, self)
            process_p = partial(process_p_binding_, self)

            from_pos_hidden:  CondRef = Ref()
            to_pos_hidden:    CondRef = Ref()
            inter_pos_hidden: CondRef = Ref()
            embryo:           CondRef = Ref()     # latent image, the common half-denoised prototype of all frames

        # Step 1: get starting & ending condition
        switch_to_stage(0)
        with on_cfg_denoiser_wrapper(partial(get_cond_callback, [from_pos_hidden])):
            with p_steps_overrider(self.p, steps=1):
                process_p(save=False)
        switch_to_stage(1)
        with on_cfg_denoiser_wrapper(partial(get_cond_callback, [to_pos_hidden])):
            with p_steps_overrider(self.p, steps=1):
                process_p(save=False)

        # Step 2: get the condition middle-point as embryo then hatch it halfway
        inter_pos_hidden.value = lerp_fn(from_pos_hidden.value, to_pos_hidden.value, 0.5)
        with on_cfg_denoiser_wrapper(partial(set_cond_callback, [inter_pos_hidden])):
            with on_cfg_denoiser_wrapper(partial(get_latent_callback, embryo, self.embryo_step)):
                process_p(save=False)
        try:
            img: PILImage = single_sample_to_image(embryo.value[0], approximation=-1)     # the data is duplicated, just get first item
            img.save(os.path.join(self.log_dp, 'embryo.png'))
        except: pass

        # Step 3: derive the embryo towards each interpolated condition
        for t in range(0, n_frames+1):
            if state.interrupted: break

            alpha = t / n_frames     # [0, 1/T, 2/T, .. T-1/T, 1]
            inter_pos_hidden.value = lerp_fn(from_pos_hidden.value, to_pos_hidden.value, alpha)
            with on_cfg_denoiser_wrapper(partial(set_cond_callback, [inter_pos_hidden])):
                with on_cfg_denoiser_wrapper(partial(set_latent_callback, embryo, self.embryo_step)):
                    process_p()

    def run_replace(self):
        ''' yet another replace method, but do replacing on the condition tensor by token dim or channel dim '''

        if self.genesis == Gensis.EMBRYO: raise NotImplementedError(f'genesis {self.genesis.value!r} is only supported in linear mode currently :(')

        if 'auxiliary':
            switch_to_stage = partial(switch_to_stage_binding_, self)
            process_p = partial(process_p_binding_, self)

            from_pos_hidden:  CondRef = Ref()
            to_pos_hidden:    CondRef = Ref()
            inter_pos_hidden: CondRef = Ref()

        # Step 1: draw the init image
        switch_to_stage(0)
        with on_cfg_denoiser_wrapper(partial(get_cond_callback, [from_pos_hidden])):
            process_p()
        
        # travel through stages
        for i in range(1, self.n_stages):
            if state.interrupted: break

            state.job = f'{i}/{self.n_frames}'
            state.job_no = i + 1

            # only change target prompts
            switch_to_stage(i)
            with on_cfg_denoiser_wrapper(partial(get_cond_callback, [to_pos_hidden])):
                if self.genesis == Gensis.FIXED:
                    imgs = process_p(append=False)      # stash it to make order right
                elif self.genesis == Gensis.SUCCESSIVE:
                    with p_steps_overrider(self.p, steps=1):    # ignore final image, only need cond
                        process_p(save=False, append=False)
                else: raise ValueError(f'invalid genesis: {self.genesis.value}')

            # ========== ↓↓↓ major differences from run_linear() ↓↓↓ ==========
            
            # decide change portion in each iter
            L1 = torch.abs(from_pos_hidden.value - to_pos_hidden.value)
            if   self.replace_dim == ModeReplaceDim.RANDOM:
                dist = L1                  # [T=77, D=768]
            elif self.replace_dim == ModeReplaceDim.TOKEN:
                dist = L1.mean(axis=1)     # [T=77]
            elif self.replace_dim == ModeReplaceDim.CHANNEL:
                dist = L1.mean(axis=0)     # [D=768]
            else: raise ValueError(f'unknown replace_dim: {self.replace_dim}')
            mask = dist > EPS
            dist = torch.where(mask, dist, 0.0)
            n_diff = mask.sum().item()            # when value differs we have mask==True
            n_inter = self.steps[i] + 1
            replace_count = int(n_diff / n_inter) + 1    # => accumulative modifies [1/T, 2/T, .. T-1/T] of total cond

            # Step 2: draw the replaced images
            inter_pos_hidden.value = from_pos_hidden.value
            is_break_iter = False
            for _ in range(1, n_inter):
                if state.interrupted: is_break_iter = True ; break

                inter_pos_hidden.value = replace_until_match(inter_pos_hidden.value, to_pos_hidden.value, replace_count, dist=dist, order=self.replace_order)
                with on_cfg_denoiser_wrapper(partial(set_cond_callback, [inter_pos_hidden])):
                    process_p()
            
            # ========== ↑↑↑ major differences from run_linear() ↑↑↑ ==========

            if is_break_iter: break

            # Step 3: append the final stage
            if self.genesis != Gensis.SUCCESSIVE: self.images.extend(imgs)
            # move to next stage
            from_pos_hidden.value = to_pos_hidden.value
            inter_pos_hidden.value = None

    def ssim_loop(
        self,
        p,
        ssim_diff,
        ssim_ccrop,
        ssim_diff_min,
        substep_min,
        prompt_images,
        lerp_fn,
        process_p,
        from_pos_hidden,
        from_neg_hidden,
        to_pos_hidden,
        to_neg_hidden,
        inter_pos_hidden,
        inter_neg_hidden,
    ):
        """Copied from shift-attentions plugin: https://github.com/yownas/shift-attention/blob/0129f6b99109f6f7c9e4e2bee0d1dc5f96e62506/scripts/shift_attention.py#L268"""

        dist_per_image = 1 / (len(prompt_images) - 1)
        dists = [dist_per_image * (i) for i, _ in enumerate(prompt_images)]

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        if ssim_ccrop == 0:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [
                    transforms.CenterCrop(
                        (
                            p.height * (ssim_ccrop / 100),
                            p.width * (ssim_ccrop / 100),
                        )
                    ),
                    transforms.ToTensor(),
                ]
            )

        check = True
        skip_count = 0
        not_better = 0
        skip_ssim_min = 1.0
        min_step = 1.0

        done = 0
        while check:
            if state.interrupted:
                break
            check = False
            for i in range(done, len(prompt_images) - 1):
                # Check distance between i and i+1
                a = transform(prompt_images[i]).unsqueeze(0)
                b = transform(prompt_images[i + 1]).unsqueeze(0)
                d = ssim(a, b)

                if d < ssim_diff and (dists[i + 1] - dists[i]) > substep_min:
                    print(
                        f"SSIM: {dists[i]} <-> {dists[i+1]} ="
                        f" ({dists[i+1] - dists[i]}) {d}"
                    )

                    # Add image and run check again
                    check = True

                    new_dist = (dists[i] + dists[i + 1]) / 2.0

                    self.interpolate(
                        lerp_fn=lerp_fn,
                        from_pos_hidden=from_pos_hidden,
                        from_neg_hidden=from_neg_hidden,
                        to_pos_hidden=to_pos_hidden,
                        to_neg_hidden=to_neg_hidden,
                        inter_pos_hidden=inter_pos_hidden,
                        inter_neg_hidden=inter_neg_hidden,
                        alpha=new_dist,
                    )
                    with on_cfg_denoiser_wrapper(
                        partial(set_cond_callback, [inter_pos_hidden, inter_neg_hidden])
                    ):
                        # SSIM stats for the new image

                        print(f"Process: {new_dist}")
                        image = process_p(append=False)[0]

                    # Check if this was an improvment
                    c = transform(image).unsqueeze(0)
                    d2 = ssim(a, c)

                    if d2 > d or d2 < ssim_diff * ssim_diff_min / 100.0:
                        # Keep image if it is improvment or hasn't reached desired min ssim_diff
                        prompt_images.insert(i + 1, image)
                        dists.insert(i + 1, new_dist)

                    else:
                        print(
                            f"Did not find improvment: {d2} < {d} ({d-d2}) Taking"
                            " shortcut."
                        )
                        not_better += 1
                        done = i + 1

                    break
                else:
                    # DEBUG
                    if d > ssim_diff:
                        if i > done:
                            print(
                                f"Done: {dists[i+1]*100}% ({d}) {len(dists)} frames.   "
                            )
                    else:
                        print(
                            f"Reached minimum step limit @{dists[i]} (Skipping) SSIM ="
                            f" {d}   "
                        )
                        if skip_ssim_min > d:
                            skip_ssim_min = d
                        skip_count += 1
                    done = i
            # DEBUG
        print("SSIM done!")

        if skip_count > 0:
            print(
                f"Minimum step limits reached: {skip_count} Worst: {skip_ssim_min} No"
                f" improvment: {not_better}"
            )

        return skip_count, not_better, skip_ssim_min, min_step, prompt_images

    ''' ↓↓↓ extension support ↓↓↓ '''

    def ext_depth_preprocess(self, p:Processing, depth_img:PILImage):  # copy from repo `AnonymousCervine/depth-image-io-for-SDWebui`
        from types import MethodType
        from einops import repeat, rearrange
        import modules.shared as shared
        import modules.devices as devices

        def sanitize_pil_image_mode(img):
            if img.mode in {'P', 'CMYK', 'HSV'}:
                img = img.convert(mode='RGB')
            return img

        def alt_depth_image_conditioning(self, source_image):
            with devices.autocast():
                conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(source_image))
            depth_data = np.array(sanitize_pil_image_mode(depth_img))

            if len(np.shape(depth_data)) == 2:
                depth_data = rearrange(depth_data, "h w -> 1 1 h w")
            else:
                depth_data = rearrange(depth_data, "h w c -> c 1 1 h w")[0]
            depth_data = torch.from_numpy(depth_data).to(device=shared.device).to(dtype=torch.float32)
            depth_data = repeat(depth_data, "1 ... -> n ...", n=self.batch_size)
            
            conditioning = torch.nn.functional.interpolate(
                depth_data,
                size=conditioning_image.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            (depth_min, depth_max) = torch.aminmax(conditioning)
            conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
            return conditioning
        
        p.depth2img_image_conditioning = MethodType(alt_depth_image_conditioning, p)
        
        def alt_txt2img_image_conditioning(self, x, width=None, height=None):
            fake_img = torch.zeros(1, 3, height or self.height, width or self.width).to(shared.device).type(self.sd_model.dtype)
            return self.depth2img_image_conditioning(fake_img)

        p.txt2img_image_conditioning = MethodType(alt_txt2img_image_conditioning, p)

    def ext_depth_postprocess(self, p:Processing, depth_img:PILImage):
        depth_img.close()
