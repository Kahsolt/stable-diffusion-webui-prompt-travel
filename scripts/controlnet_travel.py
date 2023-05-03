from scripts.prompt_travel import *
from manager import run_cmd

from modules.scripts import ScriptRunner

class InterpMethod(Enum):
    LINEAR = 'linear (weight sum)'
    RIFE   = 'rife (optical flow)'

if 'global consts':
    __ = lambda key, value=None: opts.data.get(f'customscript/controlnet_travel.py/txt2img/{key}/value', value)

    WEBUI_PATH = Path.cwd()
    PTRAVEL_PATH = WEBUI_PATH / 'extensions' / 'stable-diffusion-webui-prompt-travel'

    LABEL_CTRLNET_REF_DIR   = 'Reference image folder (one ref image per stage :)'
    LABEL_INTERP_METH       = 'Interpolate method'
    LABEL_SKIP_FUSE         = 'Ext. skip latent fusion'
    LABEL_DEBUG_RIFE        = 'Save RIFE intermediates'

    DEFAULT_STEPS           = 10
    DEFAULT_CTRLNET_REF_DIR = str(PTRAVEL_PATH / 'img' / 'ref_ctrlnet')
    DEFAULT_INTERP_METH     = __(LABEL_INTERP_METH, InterpMethod.LINEAR.value)
    DEFAULT_SKIP_FUSE       = __(LABEL_SKIP_FUSE, False)
    DEFAULT_DEBUG_RIFE      = __(LABEL_DEBUG_RIFE, False)

    CHOICES_INTERP_METH     = [x.value for x in InterpMethod]

if 'global vars':
    skip_fuse_plan: List[bool] = []                 # n_blocks (13)

    interp_alpha: float = 0
    interp_ip: int = 0                              # 0 ~ n_sampling_step-1
    from_hint_cond: List[Tensor] = []               # n_contrlnet_set
    to_hint_cond: List[Tensor] = []
    mid_hint_cond: List[Tensor] = []
    from_control_tensors: List[List[Tensor]] = []   # n_sampling_step x n_blocks
    to_control_tensors: List[List[Tensor]] = []


# ↓↓↓ the following is modified from 'sd-webui-controlnet/scripts/hook.py' ↓↓↓

try:
    from scripts.hook import th, cond_cast_unet, timestep_embedding
    from scripts.hook import UNetModel, UnetHook, ControlParams
    controlnet_found = True
except:
    controlnet_found = False

def cfg_based_adder(self, base:Tensor, x:Tensor, require_autocast:bool):
    self: UnetHook
    
    if isinstance(x, float):
        return base + x
    
    if require_autocast:
        zeros = torch.zeros_like(base)
        zeros[:, :x.shape[1], ...] = x
        x = zeros

    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]
    if base_h != xh or base_w != xw:
        x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
    
    return base + x

def forward(outer, x:Tensor, timesteps:Tensor=None, context:Tensor=None, **kwargs):
    ''' NOTE: This function is called `sampling_steps*2` times (once for cond & uncond respectively) '''
    outer: UnetHook

    total_control = [0.0] * 13
    total_adapter = [0.0] * 4
    total_extra_cond = None
    only_mid_control = outer.only_mid_control
    require_inpaint_hijack = False

    # NOTE: declare globals
    global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, mid_hint_cond, interp_alpha, interp_ip
    self: UNetModel = shared.sd_model.model.diffusion_model
    x: Tensor           # [1, 4, 64, 64]
    context: Tensor     # [1, 77, 768]

    # High-res fix
    is_in_high_res_fix = False
    for param in outer.control_params:
        # select which hint_cond to use
        param.used_hint_cond = param.hint_cond
        # has high-res fix
        if param.hr_hint_cond is not None and x.ndim == 4 and param.hint_cond.ndim == 3 and param.hr_hint_cond.ndim == 3:
            _, h_lr, w_lr = param.hint_cond.shape
            _, h_hr, w_hr = param.hr_hint_cond.shape
            _, _, h, w = x.shape
            h, w = h * 8, w * 8
            if abs(h - h_lr) < abs(h - h_hr):
                # we are in low-res path
                param.used_hint_cond = param.hint_cond
            else:
                # we are in high-res path
                param.used_hint_cond = param.hr_hint_cond
                is_in_high_res_fix = True
                if shared.opts.data.get("control_net_high_res_only_mid", False):
                    only_mid_control = True

    # handle external cond
    for param in outer.control_params:
        if param.guidance_stopped or not param.is_extra_cond:
            continue
        param.control_model.to(devices.get_device_for("controlnet"))
        query_size = int(x.shape[0])
        control = param.control_model(x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context)
        uc_mask = param.generate_uc_mask(query_size, dtype=x.dtype, device=x.device)[:, None, None]
        control = torch.concatenate([control.clone() for _ in range(query_size)], dim=0)
        control *= param.weight
        control *= uc_mask
        if total_extra_cond is None:
            total_extra_cond = control.clone()
        else:
            total_extra_cond = torch.cat([total_extra_cond, control.clone()], dim=1)
        
    if total_extra_cond is not None:
        context = torch.cat([context, total_extra_cond], dim=1)

    # handle unet injection stuff
    for i, param in enumerate(outer.control_params):
        if param.guidance_stopped or param.is_extra_cond:
            continue

        param.control_model.to(devices.get_device_for("controlnet"))
        # inpaint model workaround
        x_in = x
        control_model = param.control_model.control_model
        if not param.is_adapter and x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9: 
            # inpaint_model: 4 data + 4 downscaled image + 1 mask
            x_in = x[:, :4, ...]
            require_inpaint_hijack = True

        # NOTE: perform hint shallow fusion here
        if interp_alpha == 0.0:     # collect hind_cond on key frames
            if len(to_hint_cond) < len(outer.control_params):
                to_hint_cond.append(param.used_hint_cond.cpu().clone())
        else:                       # interp with cached hind_cond
            param.used_hint_cond = mid_hint_cond[i].to(x_in.device)

        assert param.used_hint_cond is not None, f"Controlnet is enabled but no input image is given"
        control = param.control_model(x=x_in, hint=param.used_hint_cond, timesteps=timesteps, context=context)
        control_scales = ([param.weight] * 13)
        
        if outer.lowvram:
            param.control_model.to("cpu")

        if param.cfg_injection or param.global_average_pooling:
            query_size = int(x.shape[0])
            if param.is_adapter:
                control = [torch.concatenate([c.clone() for _ in range(query_size)], dim=0) for c in control]
            uc_mask = param.generate_uc_mask(query_size, dtype=x.dtype, device=x.device)[:, None, None, None]
            control = [c * uc_mask for c in control]

        if param.soft_injection or is_in_high_res_fix:
            # important! use the soft weights with high-res fix can significantly reduce artifacts.
            if param.is_adapter:
                control_scales = [param.weight * x for x in (0.25, 0.62, 0.825, 1.0)]
            else:    
                control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]

        if param.advanced_weighting is not None:
            control_scales = param.advanced_weighting
            
        control = [c * scale for c, scale in zip(control, control_scales)]
        if param.global_average_pooling:
            control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            
        for idx, item in enumerate(control):
            target = total_adapter if param.is_adapter else total_control
            target[idx] = item + target[idx]

    # NOTE: perform latent fusion here
    if interp_alpha == 0.0:     # collect control tensors on key frames
        tensors: List[Tensor] = []
        for i, t in enumerate(total_control):
            if len(skip_fuse_plan) and skip_fuse_plan[i]:
                tensors.append(None)
            else:
                tensors.append(t.cpu().clone())
        to_control_tensors.append(tensors)
    else:                       # interp with cached control tensors
        device = total_control[0].device
        for i, (ctrlA, ctrlB) in enumerate(zip(from_control_tensors[interp_ip], to_control_tensors[interp_ip])):
            if ctrlA is not None and ctrlB is not None:
                ctrlC = weighted_sum(ctrlA.to(device), ctrlB.to(device), interp_alpha)
                #print('  ctrl diff:', (ctrlC - total_control[i]).abs().mean().item())
                total_control[i].data = ctrlC
        interp_ip += 1

    control = total_control
    assert timesteps is not None, ValueError(f"insufficient timestep: {timesteps}")
    hs = []
    with th.no_grad():
        t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            
            # t2i-adatper, same as openaimodel.py:744
            if ((i+1) % 3 == 0) and len(total_adapter):
                h = cfg_based_adder(outer, h, total_adapter.pop(0), require_inpaint_hijack)
                
            hs.append(h)
        h = self.middle_block(h, emb, context)

    control_in = control.pop()
    h = cfg_based_adder(outer, h, control_in, require_inpaint_hijack)

    for i, module in enumerate(self.output_blocks):
        if only_mid_control:
            hs_input = hs.pop()
            h = th.cat([h, hs_input], dim=1)
        else:
            hs_input, control_input = hs.pop(), control.pop()
            h = th.cat([h, cfg_based_adder(outer, hs_input, control_input, require_inpaint_hijack)], dim=1)
        h = module(h, emb, context)

    h = h.type(x.dtype)
    return self.out(h)

def forward2(self, *args, **kwargs):
    # webui will handle other compoments 
    self: UnetHook
    try:
        if shared.cmd_opts.lowvram:
            lowvram.send_everything_to_cpu()
                                    
        return forward(self, *args, **kwargs)
    finally:
        if self.lowvram:
            [param.control_model.to("cpu") for param in self.control_params]

# ↑↑↑ the above is modified from 'sd-webui-controlnet/scripts/hook.py' ↑↑↑

def reset_cuda():
    devices.torch_gc()
    import gc; gc.collect()

    try:
        import os
        import psutil
        mem = psutil.Process(os.getpid()).memory_info()
        print(f'[Mem] rss: {mem.rss/2**30:.3f} GB, vms: {mem.vms/2**30:.3f} GB')
        from modules.shared import mem_mon as vram_mon
        free, total = vram_mon.cuda_mem_get_info()
        print(f'[VRAM] free: {free/2**30:.3f} GB, total: {total/2**30:.3f} GB')
    except:
        pass


class Script(scripts.Script):

    def title(self):
        return 'ControlNet Travel'

    def describe(self):
        return 'Travel from one controlnet hint to another in the latent space.'

    def show(self, is_img2img):
        if not controlnet_found: print('extension Mikubill/sd-webui-controlnet not found, ControlNet Travel ignored')
        return controlnet_found

    def ui(self, is_img2img):
        with gr.Row(variant='compact'):
            interp_meth = gr.Dropdown(label=LABEL_INTERP_METH, value=lambda: DEFAULT_INTERP_METH, choices=CHOICES_INTERP_METH)
            steps       = gr.Text    (label=LABEL_STEPS,       value=lambda: DEFAULT_STEPS, max_lines=1)
            
            reset = gr.Button(value='Reset Cuda', variant='tool')
            reset.click(fn=reset_cuda, show_progress=False)

        with gr.Row(variant='compact'):
            ctrlnet_ref_dir = gr.Text(label=LABEL_CTRLNET_REF_DIR, value=lambda: DEFAULT_CTRLNET_REF_DIR, max_lines=1)

        with gr.Group(visible=DEFAULT_SKIP_FUSE) as tab_ext_skip_fuse:
            with gr.Row(variant='compact'):
                skip_in_0  = gr.Checkbox(label='in_0')
                skip_in_3  = gr.Checkbox(label='in_3')
                skip_out_0 = gr.Checkbox(label='out_0')
                skip_out_3 = gr.Checkbox(label='out_3')
            with gr.Row(variant='compact'):
                skip_in_1  = gr.Checkbox(label='in_1')
                skip_in_4  = gr.Checkbox(label='in_4')
                skip_out_1 = gr.Checkbox(label='out_1')
                skip_out_4 = gr.Checkbox(label='out_4')
            with gr.Row(variant='compact'):
                skip_in_2  = gr.Checkbox(label='in_2')
                skip_in_5  = gr.Checkbox(label='in_5')
                skip_out_2 = gr.Checkbox(label='out_2')
                skip_out_5 = gr.Checkbox(label='out_5')
            with gr.Row(variant='compact'):
                skip_mid   = gr.Checkbox(label='mid')

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
            ext_video     = gr.Checkbox(label=LABEL_VIDEO,      value=lambda: DEFAULT_VIDEO)
            ext_upscale   = gr.Checkbox(label=LABEL_UPSCALE,    value=lambda: DEFAULT_UPSCALE)
            ext_skip_fuse = gr.Checkbox(label=LABEL_SKIP_FUSE,  value=lambda: DEFAULT_SKIP_FUSE)
            dbg_rife      = gr.Checkbox(label=LABEL_DEBUG_RIFE, value=lambda: DEFAULT_DEBUG_RIFE)
        
            ext_video    .change(fn=lambda x: gr_show(x), inputs=ext_video,     outputs=tab_ext_video,     show_progress=False)
            ext_upscale  .change(fn=lambda x: gr_show(x), inputs=ext_upscale,   outputs=tab_ext_upscale,   show_progress=False)
            ext_skip_fuse.change(fn=lambda x: gr_show(x), inputs=ext_skip_fuse, outputs=tab_ext_skip_fuse, show_progress=False)

        skip_fuses = [
            skip_in_0,
            skip_in_1,
            skip_in_2,
            skip_in_3,
            skip_in_4,
            skip_in_5,
            skip_mid,
            skip_out_0,
            skip_out_1,
            skip_out_2,
            skip_out_3,
            skip_out_4,
            skip_out_5,
        ]
        return [interp_meth, steps, ctrlnet_ref_dir,
                upscale_meth, upscale_ratio, upscale_width, upscale_height,
                video_fmt, video_fps, video_pad, video_pick,
                ext_video, ext_upscale, ext_skip_fuse, dbg_rife,
                *skip_fuses]

    def run(self, p:StableDiffusionProcessing, 
            interp_meth:str, steps:str, ctrlnet_ref_dir:str, 
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int,
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            ext_video:bool, ext_upscale:bool, ext_skip_fuse:bool, dbg_rife:bool,
            *skip_fuses:bool):

        # Prepare ControlNet
        self.controlnet_script = None
        self.hooked = None
        try:
            #from scripts.controlnet import Script as ControlNetScript
            from scripts.external_code import ControlNetUnit
            for script in p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    script_args: Tuple[ControlNetUnit] = p.script_args[script.args_from:script.args_to]
                    if not any([u.enabled for u in script_args]): return Processed(p, [], p.seed, 'sd-webui-controlnet not enabled')
                    #self.controlnet_script: ControlNetScript = script
                    self.controlnet_script = script
                    break
        except ImportError:
            return Processed(p, [], p.seed, 'sd-webui-controlnet not installed')
        if not self.controlnet_script: return Processed(p, [], p.seed, 'sd-webui-controlnet not loaded')

        # enum lookup
        interp_meth: InterpMethod = InterpMethod(interp_meth)
        video_fmt:   VideoFormat  = VideoFormat (video_fmt)

        # Param check & type convert
        if ext_video:
            if video_pad <  0: return Processed(p, [], p.seed, f'video_pad must >= 0, but got {video_pad}')
            if video_fps <= 0: return Processed(p, [], p.seed, f'video_fps must > 0, but got {video_fps}')
            try: video_slice = parse_slice(video_pick)
            except: return Processed(p, [], p.seed, 'syntax error in video_slice')
        if ext_skip_fuse:
            global skip_fuse_plan
            skip_fuse_plan = skip_fuses

        # Prepare ref-images
        if not ctrlnet_ref_dir: return Processed(p, [], p.seed, f'invalid image folder path: {ctrlnet_ref_dir}')
        ctrlnet_ref_dir: Path  = Path(ctrlnet_ref_dir)
        if not ctrlnet_ref_dir.is_dir(): return Processed(p, [], p.seed, f'invalid image folder path: {ctrlnet_ref_dir}(')
        self.ctrlnet_ref_fps = [fp for fp in list(ctrlnet_ref_dir.iterdir()) if fp.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        n_stages = len(self.ctrlnet_ref_fps)
        if n_stages == 0: return Processed(p, [], p.seed, f'no images file (*.jpg/*.png/*.bmp/*.webp) found in folder path: {ctrlnet_ref_dir}')
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
        self.tmp_dp = Path(self.log_dp) / 'ctrl_cond'       # cache for rife
        self.tmp_fp = self.tmp_dp / 'tmp.png'               # cache for rife

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
        self.n_stages    = n_stages
        self.steps       = steps
        self.interp_meth = interp_meth
        self.dbg_rife    = dbg_rife

        def save_image_hijack(params:ImageSaveParams):
            if not ext_upscale: return
            params.image = upscale_image(params.image, p.width, p.height, upscale_meth, upscale_ratio, upscale_width, upscale_height)

        images = []
        info = ''
        script_runner: ScriptRunner = p.scripts
        caches: List[List[Tensor]] = [from_hint_cond, to_hint_cond, mid_hint_cond, from_control_tensors, to_control_tensors]
        try:
            if self not in script_runner.alwayson_scripts: script_runner.alwayson_scripts.append(self)
            on_before_image_saved(save_image_hijack)

            [c.clear() for c in caches]
            images, info = self.run_linear(p)
        finally:
            if self.tmp_fp.exists(): os.unlink(self.tmp_fp)
            [c.clear() for c in caches]

            remove_callbacks_for_function(save_image_hijack)
            if self in script_runner.alwayson_scripts: script_runner.alwayson_scripts.remove(self)

            self.controlnet_script.input_image = None
            if self.controlnet_script.latest_network:
                self.controlnet_script.latest_network.restore(p.sd_model.model.diffusion_model)
                self.controlnet_script.latest_network = None
            self.postprocess_batch(p)    # assure unhook

            reset_cuda()

        # Save video
        if ext_video: save_video(images, video_slice, video_pad, video_fps, video_fmt, os.path.join(self.log_dp, f'travel-{travel_number:05}'))

        return Processed(p, images, p.seed, info)

    def run_linear(self, p:StableDiffusionProcessing) -> Tuple[List[PILImage], str]:
        global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, interp_alpha, interp_ip

        images = []
        info = None
        def gen_image(append=True):
            nonlocal p, images, info
            proc = process_images(p)
            if not info: info = proc.info
            if append: images.extend(proc.images)
            else: return proc.images

        ''' ↓↓↓ rife interp utils ↓↓↓ '''
        def save_ctrl_cond(idx:int):
            self.tmp_dp.mkdir(exist_ok=True)
            for i, x in enumerate(to_hint_cond):
                if len(x.shape) == 3:
                    if   x.shape[0] == 1: x = x.squeeze_(0)         # [C=1, H, W] => [H, W]
                    elif x.shape[0] == 3: x = x.permute([1, 2, 0])  # [C=3, H, W] => [H, W, C]
                    else: raise ValueError(f'unknown cond shape: {x.shape}')
                else:
                    raise ValueError(f'unknown cond shape: {x.shape}')
                im = (x.detach().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(im).save(self.tmp_dp / f'{idx}-{i}.png')
        def rife_interp(i:int, j:int, k:int, alpha:float) -> Tensor:
            ''' interp between i-th and j-th cond of the k-th ctrlnet set '''
            fp0 = self.tmp_dp / f'{i}-{k}.png'
            fp1 = self.tmp_dp / f'{j}-{k}.png'
            fpo = self.tmp_dp / f'{i}-{j}-{alpha:.3f}.png' if self.dbg_rife else self.tmp_fp
            assert run_cmd(f'rife-ncnn-vulkan -m rife-v4 -s {alpha:.3f} -0 "{fp0}" -1 "{fp1}" -o "{fpo}"')
            x = torch.from_numpy(np.asarray(Image.open(fpo)) / 255.0)
            if   len(x.shape) == 2: x = x.unsqueeze_(0)             # [H, W] => [C=1, H, W]
            elif len(x.shape) == 3: x = x.permute([2, 0, 1])        # [H, W, C] => [C, H, W]
            else: raise ValueError(f'unknown cond shape: {x.shape}')
            return x
        ''' ↑↑↑ rife interp utils ↑↑↑ '''

        ''' ↓↓↓ filename reorder utils ↓↓↓ '''
        iframe = 0
        def rename_image_filename(idx:int, param: ImageSaveParams):
            fn = param.filename
            stem, suffix = os.path.splitext(os.path.basename(fn))
            param.filename = os.path.join(os.path.dirname(fn), f'{idx:05d}' + suffix)
        class save_image_hijack:
            def __init__(self, callback_fn, idx):
                self.callback_fn = lambda *args, **kwargs: callback_fn(idx, *args, **kwargs)
            def __enter__(self):
                on_before_image_saved(self.callback_fn)
            def __exit__(self, exc_type, exc_value, exc_traceback):
                remove_callbacks_for_function(self.callback_fn)

        ''' ↑↑↑ filename reorder utils ↑↑↑ '''

        # Step 1: draw the init image
        setattr(p, 'init_images', [Image.open(self.ctrlnet_ref_fps[0])])
        interp_alpha = 0.0
        with save_image_hijack(rename_image_filename, 0):
            gen_image()
            iframe += 1
        save_ctrl_cond(0)

        # travel through stages
        for i in range(1, self.n_stages):
            if state.interrupted: break

            # Setp 3: move to next stage
            from_hint_cond       = [t for t in to_hint_cond]       ; to_hint_cond      .clear()
            from_control_tensors = [t for t in to_control_tensors] ; to_control_tensors.clear()
            setattr(p, 'init_images', [Image.open(self.ctrlnet_ref_fps[i])])
            interp_alpha = 0.0

            with save_image_hijack(rename_image_filename, iframe + self.steps[i]):
                cached_images = gen_image(append=False)
            save_ctrl_cond(i)

            # Step 2: draw the interpolated images
            is_interrupted = False
            n_inter = self.steps[i] + 1
            for t in range(1, n_inter):
                if state.interrupted: is_interrupted = True ; break

                interp_alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]

                mid_hint_cond.clear()
                device = devices.get_device_for("controlnet")
                if self.interp_meth == InterpMethod.LINEAR:
                    for hintA, hintB in zip(from_hint_cond, to_hint_cond):
                        hintC = weighted_sum(hintA.to(device), hintB.to(device), interp_alpha)
                        mid_hint_cond.append(hintC)
                elif self.interp_meth == InterpMethod.RIFE:
                    dtype = to_hint_cond[0].dtype
                    for k in range(len(to_hint_cond)):
                        hintC = rife_interp(i-1, i, k, interp_alpha).to(device, dtype)
                        mid_hint_cond.append(hintC)
                else: raise ValueError(f'unknown interp_meth: {self.interp_meth}')

                interp_ip = 0
                with save_image_hijack(rename_image_filename, iframe):
                    gen_image()
                    iframe += 1

            # adjust order
            images.extend(cached_images)
            iframe += 1

            if is_interrupted: break

        return images, info

    def process_batch(self, p:StableDiffusionProcessing, *args, **kwargs):
        ''' hijack over ControlNet's hijack, controlnet hooks `process()` so we have to hijack after it :) '''

        unethook: UnetHook = self.controlnet_script.latest_network
        assert unethook, 'Error: UnetHook is None! You silly forgot to enable or set up ControlNet?'
        unet: UNetModel = p.sd_model.model.diffusion_model
        assert unet._original_forward,  'Error: Unet did not hook on? You silly forgot to enable or set up ControlNet?'
        control_params: List[ControlParams] = unethook.control_params
        assert control_params, 'Error: ControlParams is None! You silly forgot to enable or set up ControlNet?'
        
        self.saved_unet_forward = unet._original_forward
        unet.forward = lambda *args, **kwargs: forward2(unethook, *args, **kwargs)
        self.hooked = True
        setattr(p, 'detected_map', None)

    def postprocess_batch(self, p:StableDiffusionProcessing, *args, **kwargs):
        if not self.hooked: return

        unet: UNetModel = p.sd_model.model.diffusion_model
        unet.forward = self.saved_unet_forward
        self.hooked = False
        setattr(p, 'detected_map', None)
