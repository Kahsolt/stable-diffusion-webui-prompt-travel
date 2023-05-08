from scripts.prompt_travel import *

import gc
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding

if 'global consts':
    LABEL_REF_DIR   = 'Reference image folder (one ref image per stage :)'

    DEFAULT_STEPS   = 2
    DEFAULT_REF_DIR = ''

if 'global vars':
    interp_alpha: float = 0
    interp_ip: int = 0
    from_unet_latent: List[Tensor] = []
    to_unet_latent: List[Tensor] = []


# ↓↓↓ the following is modified from 'stable-diffusion\ldm\modules\diffusionmodules\openaimodel.py' ↓↓↓

from modules.sd_hijack_unet import th

def UNetModel_forward(self:UNetModel, x:Tensor, timesteps:Tensor=None, context:Tensor=None, y:Tensor=None, **kwargs):
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """
    assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    global interp_alpha, interp_ip, from_unet_latent, to_unet_latent
    device = devices.device

    if self.num_classes is not None:
        assert y.shape == (x.shape[0],)
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)

    if interp_alpha == 0.0:     # collect tensors on key frames
        to_unet_latent.append(h.cpu().clone())
    else:                       # interp with cached tensors
        h = weighted_sum(from_unet_latent[interp_ip].to(device), to_unet_latent[interp_ip].to(device), interp_alpha)
        interp_ip += 1

    h = self.middle_block(h, emb, context)

    if interp_alpha == 0.0:     # collect tensors on key frames
        to_unet_latent.append(h.cpu().clone())
    else:                       # interp with cached tensors
        h = weighted_sum(from_unet_latent[interp_ip].to(device), to_unet_latent[interp_ip].to(device), interp_alpha)
        interp_ip += 1

    for module in self.output_blocks:
        h = th.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)

    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)

# ↑↑↑ the above is modified from 'stable-diffusion\ldm\modules\diffusionmodules\openaimodel.py' ↑↑↑


class Script(scripts.Script):

    def title(self):
        return 'UNet Travel'

    def describe(self):
        return 'Blend frames in UNet lantent.'

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row(variant='compact'):
            steps = gr.Text(label=LABEL_STEPS, value=lambda: DEFAULT_STEPS, max_lines=1)

        with gr.Row(variant='compact'):
            ref_dir = gr.Text(label=LABEL_REF_DIR, value=lambda: DEFAULT_REF_DIR, max_lines=1)

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
        
            ext_video  .change(fn=lambda x: gr_show(x), inputs=ext_video,   outputs=tab_ext_video,   show_progress=False)
            ext_upscale.change(fn=lambda x: gr_show(x), inputs=ext_upscale, outputs=tab_ext_upscale, show_progress=False)

        return [steps, ref_dir,
                upscale_meth, upscale_ratio, upscale_width, upscale_height,
                video_fmt, video_fps, video_pad, video_pick,
                ext_video, ext_upscale]

    def run(self, p:StableDiffusionProcessingImg2Img, 
            steps:str, ref_dir:str, 
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int,
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            ext_video:bool, ext_upscale:bool):

        # enum lookup
        video_fmt: VideoFormat = VideoFormat(video_fmt)

        # Param check & type convert
        if ext_video:
            if video_pad <  0: return Processed(p, [], p.seed, f'video_pad must >= 0, but got {video_pad}')
            if video_fps <= 0: return Processed(p, [], p.seed, f'video_fps must > 0, but got {video_fps}')
            try: video_slice = parse_slice(video_pick)
            except: return Processed(p, [], p.seed, 'syntax error in video_slice')

        # Prepare ref-images
        if not ref_dir: return Processed(p, [], p.seed, f'invalid image folder path: {ref_dir}')
        ref_dir: Path  = Path(ref_dir)
        if not ref_dir.is_dir(): return Processed(p, [], p.seed, f'invalid image folder path: {ref_dir}(')
        self.ref_fps = [fp for fp in list(ref_dir.iterdir()) if fp.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        n_stages = len(self.ref_fps)
        if n_stages == 0: return Processed(p, [], p.seed, f'no images file (*.jpg/*.png/*.bmp/*.webp) found in folder path: {ref_dir}')
        if n_stages == 1: return Processed(p, [], p.seed, 'requires at least two images to travel between, but found only 1 :(')

        # Prepare steps (n_interp)
        try: steps: List[int] = [int(s.strip()) for s in steps.strip().split(',')]
        except: return Processed(p, [], p.seed, f'cannot parse steps options: {steps}')
        if   len(steps) == 1: steps = [steps[0]] * (n_stages - 1)
        elif len(steps) != n_stages - 1: return Processed(p, [], p.seed, f'stage count mismatch: len_steps({len(steps)}) != n_stages({n_stages} - 1))')
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

        # Force Batch Count and Batch Size to 1
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

        # Pack params
        self.n_stages = n_stages
        self.steps    = steps

        def save_image_hijack(params:ImageSaveParams):
            if not ext_upscale: return
            params.image = upscale_image(params.image, p.width, p.height, upscale_meth, upscale_ratio, upscale_width, upscale_height)

        images = []
        info = ''
        try:
            self.unet_hijack(p)
            on_before_image_saved(save_image_hijack)

            images, info = self.run_linear(p)
        except StopIteration: pass
        finally:
            remove_callbacks_for_function(save_image_hijack)
            self.unet_unhijack(p)

            devices.torch_gc()
            gc.collect()

        # Save video
        if ext_video: save_video(images, video_slice, video_pad, video_fps, video_fmt, os.path.join(self.log_dp, f'travel-{travel_number:05}'))

        return Processed(p, images, p.seed, info)

    def run_linear(self, p:StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        global from_unet_latent, to_unet_latent, interp_alpha, interp_ip

        images = []
        info = None
        def gen_image(append=True):
            nonlocal p, images, info
            proc = process_images(p)
            if not info: info = proc.info
            if append: images.extend(proc.images)
            else: return proc.images

        # Step 1: draw the init image
        p.init_images = [Image.open(self.ref_fps[0])]
        interp_alpha = 0.0
        gen_image()

        # travel through stages
        for i in range(1, self.n_stages):
            if state.interrupted: raise StopIteration

            # Setp 3: move to next stage
            from_unet_latent = [t for t in to_unet_latent] ; to_unet_latent.clear()
            p.init_images = [Image.open(self.ref_fps[i])]
            interp_alpha = 0.0
            cached_images = gen_image(append=False)

            # Step 2: draw the interpolated images
            n_inter = self.steps[i] + 1
            for t in range(1, n_inter):
                if state.interrupted: raise StopIteration

                interp_alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]
                interp_ip = 0
                gen_image()

            # adjust order
            images.extend(cached_images)

        return images, info

    def unet_hijack(self, p:StableDiffusionProcessingImg2Img):
        unet: UNetModel = p.sd_model.model.diffusion_model
        self.saved_unet_forward = unet.forward
        unet.forward = lambda *args, **kwargs: UNetModel_forward(unet, *args, **kwargs)
        self.hooked = True

    def unet_unhijack(self, p:StableDiffusionProcessingImg2Img):
        if not self.hooked: return

        unet: UNetModel = p.sd_model.model.diffusion_model
        unet.forward = self.saved_unet_forward
        self.hooked = False
