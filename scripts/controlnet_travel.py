import gc

from scripts.prompt_travel import *

class InterpMethod(Enum):
    LINEAR = 'linear (weight sum)'
    #RIFE   = 'rife (optical flow)'

if 'global consts':
    __ = lambda key, value=None: opts.data.get(f'customscript/controlnet_travel.py/txt2img/{key}/value', value)

    LABEL_CTRLNET_REF_DIR   = 'Reference image folder (one ref image per stage :)'
    LABEL_INTERP_METH       = 'Interpolate method'

    DEFAULT_CTRLNET_REF_DIR = __(LABEL_CTRLNET_REF_DIR, '')
    DEFAULT_INTERP_METH     = __(LABEL_INTERP_METH, InterpMethod.LINEAR.value)

    CHOICES_INTERP_METH     = [x.value for x in InterpMethod]


import matplotlib.pyplot as plt

def _dbg_hint_cond(hint_cond:Tensor):
    plt.imshow(hint_cond.permute([2, 1, 0]).detach().cpu().numpy())
    plt.show()


# ↓↓↓ the following is modified from 'sd-webui-controlnet/scripts/hook.py' ↓↓↓

try:
    from scripts.hook import th, cond_cast_unet, timestep_embedding
    from scripts.hook import UNetModel, UnetHook, ControlParams
    controlnet_found = True
except:
    controlnet_found = False

def cfg_based_adder(base, x, require_autocast, is_adapter=False):
    if isinstance(x, float):
        return base + x
    
    if require_autocast:
        zeros = torch.zeros_like(base)
        zeros[:, :x.shape[1], ...] = x
        x = zeros
        
    # assume the input format is [cond, uncond] and they have same shape
    # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/0cc0ee1bcb4c24a8c9715f66cede06601bfc00c8/modules/sd_samplers_kdiffusion.py#L114
    if base.shape[0] % 2 == 0 and (self.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False)):
        if self.is_vanilla_samplers:  
            uncond, cond = base.chunk(2)
            if x.shape[0] % 2 == 0:
                _, x_cond = x.chunk(2)
                return torch.cat([uncond, cond + x_cond], dim=0)
            if is_adapter:
                return torch.cat([uncond, cond + x], dim=0)
        else:
            cond, uncond = base.chunk(2)
            if x.shape[0] % 2 == 0:
                x_cond, _ = x.chunk(2)
                return torch.cat([cond + x_cond, uncond], dim=0)
            if is_adapter:
                return torch.cat([cond + x, uncond], dim=0)
    
    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]
    if base_h != xh or base_w != xw:
        x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
    
    return base + x

def forward(outer: UnetHook, x:Tensor, timesteps:Tensor=None, context:Tensor=None, **kwargs):
    ''' NOTE: This function is called `sampling_steps*2` times (once for cond & uncond respectively) '''

    total_control = [0.0] * 13
    total_adapter = [0.0] * 4
    total_extra_cond = torch.zeros([0, context.shape[-1]]).to(devices.get_device_for("controlnet"))
    only_mid_control = outer.only_mid_control
    require_inpaint_hijack = False

    # NOTE: declare globals
    global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, mid_hint_cond, interp_alpha, interp_ip, interp_fn
    self: UNetModel = shared.sd_model.model.diffusion_model
    x: Tensor           # [1, 4, 64, 64]
    context: Tensor     # [1, 77, 768]

    # handle external cond first
    for param in outer.control_params:                              # do nothing due to no extra_cond
        if param.guidance_stopped or not param.is_extra_cond:
            continue
        if outer.lowvram:
            param.control_model.to(devices.get_device_for("controlnet"))

        control = param.control_model(x=x, hint=param.hint_cond, timesteps=timesteps, context=context)
        total_extra_cond = torch.cat([total_extra_cond, control.clone().squeeze(0) * param.weight])

    # check if it's non-batch-cond mode (lowvram, edit model etc)
    if context.shape[0] % 2 != 0 and outer.batch_cond_available:    # True
        outer.batch_cond_available = False
        if len(total_extra_cond) > 0 or outer.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False):
            print("Warning: StyleAdapter and cfg/guess mode may not works due to non-batch-cond inference")

    # concat styleadapter to cond, pad uncond to same length
    if len(total_extra_cond) > 0 and outer.batch_cond_available:    # False
        total_extra_cond = torch.repeat_interleave(total_extra_cond.unsqueeze(0), context.shape[0] // 2, dim=0)
        if outer.is_vanilla_samplers:  
            uncond, cond = context.chunk(2)
            cond = torch.cat([cond, total_extra_cond], dim=1)
            uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
            context = torch.cat([uncond, cond], dim=0)
        else:
            cond, uncond = context.chunk(2)
            cond = torch.cat([cond, total_extra_cond], dim=1)
            uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
            context = torch.cat([cond, uncond], dim=0)

    # handle unet injection stuff
    for i, param in enumerate(outer.control_params):
        if param.guidance_stopped or param.is_extra_cond:
            continue
        if outer.lowvram:
            param.control_model.to(devices.get_device_for("controlnet"))

        # hires stuffs
        # note that this method may not works if hr_scale < 1.1
        if abs(x.shape[-1] - param.hint_cond.shape[-1] // 8) > 8:
            only_mid_control = shared.opts.data.get("control_net_only_midctrl_hires", True)
            # If you want to completely disable control net, uncomment this.
            # return self._original_forward(x, timesteps=timesteps, context=context, **kwargs)
         
        # inpaint model workaround
        x_in = x
        control_model = param.control_model.control_model
        if not param.is_adapter and x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9: 
            # inpaint_model: 4 data + 4 downscaled image + 1 mask
            x_in = x[:, :4, ...]
            require_inpaint_hijack = True

        # NOTE: perform hint shallow fusion here
        if interp_alpha == 0.0:     # collect hind_cond on key frames
            if len(to_hint_cond) < i + 1:
                to_hint_cond.append(param.hint_cond.cpu().clone())
        else:                       # interp with cached hind_cond
            param.hint_cond = mid_hint_cond[i]

        assert param.hint_cond is not None, f"Controlnet is enabled but no input image is given"
        control = param.control_model(x=x_in, hint=param.hint_cond, timesteps=timesteps, context=context)
        control_scales = ([param.weight] * 13)

        if outer.lowvram:
            param.control_model.to("cpu")
        if param.guess_mode:
            if param.is_adapter:
                # see https://github.com/Mikubill/sd-webui-controlnet/issues/269
                control_scales = param.weight * [0.25, 0.62, 0.825, 1.0]
            else:    
                control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]
        if param.advanced_weighting is not None:
            control_scales = param.advanced_weighting
            
        control = [c * scale for c, scale in zip(control, control_scales)]
        for idx, item in enumerate(control):
            target = total_adapter if param.is_adapter else total_control
            target[idx] += item

    # NOTE: perform latent fusion here
    if interp_alpha == 0.0:     # collect control tensors on key frames
        to_control_tensors.append([t.cpu().clone() for t in total_control])
    else:                       # interp with cached control tensors
        device = total_control[0].device
        for i, (ctrlA, ctrlB) in enumerate(zip(from_control_tensors[interp_ip], to_control_tensors[interp_ip])):
            ctrlC = interp_fn(ctrlA.to(device), ctrlB.to(device), interp_alpha)
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
            if ((i+1)%3 == 0) and len(total_adapter):
                h = cfg_based_adder(h, total_adapter.pop(0), require_inpaint_hijack, is_adapter=True)
                
            hs.append(h)
        h = self.middle_block(h, emb, context)

    control_in = control.pop()
    h = cfg_based_adder(h, control_in, require_inpaint_hijack)

    for i, module in enumerate(self.output_blocks):
        if only_mid_control:
            hs_input = hs.pop()
            h = th.cat([h, hs_input], dim=1)
        else:
            hs_input, control_input = hs.pop(), control.pop()
            h = th.cat([h, cfg_based_adder(hs_input, control_input, require_inpaint_hijack)], dim=1)
        h = module(h, emb, context)

    h = h.type(x.dtype)
    return self.out(h)

def forward2(self: UnetHook, *args, **kwargs):
    # webui will handle other compoments 
    try:
        if shared.cmd_opts.lowvram:
            lowvram.send_everything_to_cpu()
                                    
        return forward(self, *args, **kwargs)
    finally:
        if self.lowvram:
            [param.control_model.to("cpu") for param in self.control_params]

# ↑↑↑ the above is modified from 'sd-webui-controlnet/scripts/hook.py' ↑↑↑


# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

from modules.processing import *

def process_images(p: StableDiffusionProcessing) -> Processed:
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        for k, v in p.override_settings.items():
            setattr(opts, k, v)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        res = process_images_inner(p)

    finally:
        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)
                if k == 'sd_model_checkpoint':
                    sd_models.reload_model_weights()

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res

def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)
    
    # NOTE: hijack over ControlNet's hijack
    global controlnet_script
    assert controlnet_script, 'Error: controlnet script not found'
    unethook: UnetHook = controlnet_script.latest_network
    assert unethook, 'Error: UnetHook is None! You silly forgot to enable or set up ControlNet?'
    unet: UNetModel = p.sd_model.model.diffusion_model
    assert unet._original_forward,  'Error: Unet did not hook on? You silly forgot to enable or set up ControlNet?'
    control_params: List[ControlParams] = unethook.control_params
    assert control_params, 'Error: ControlParams is None! You silly forgot to enable or set up ControlNet?'
    saved_unet_forward = unet.forward
    unet.forward = lambda *args, **kwargs: forward2(unethook, *args, **kwargs)
    setattr(p, 'detected_map', None)

    infotexts = []
    output_images = []

    cached_uc = [None, None]
    cached_c = [None, None]

    def get_conds_with_caching(function, required_prompts, steps, cache):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.
        """

        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)

        cache[0] = (required_prompts, steps)
        return cache[1]

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None
        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            if len(prompts) == 0:
                break

            prompts, extra_network_data = extra_networks.parse_prompts(prompts)

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [], p.seed, "")
                    file.write(processed.infotext(p, 0))

            uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, p.steps, cached_uc)
            c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, cached_c)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            for x in x_samples_ddim:
                devices.test_for_nans(x, "vae")

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay:
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), p.mask_for_overlay.convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)
                    
                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, extra_network_data)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext(), comments="".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    # NOTE:hijack ControlNet's hijack
    unet.forward = saved_unet_forward

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

# ↑↑↑ the above is modified from 'modules/processing.py' ↑↑↑


controlnet_script = None
interp_fn: Callable = None
interp_alpha: float = 0
interp_ip: int = 0
from_hint_cond: List[Tensor] = []
to_hint_cond: List[Tensor] = []
mid_hint_cond: List[Tensor] = []
from_control_tensors: List[List[Tensor]] = []
to_control_tensors: List[List[Tensor]] = []

class Script(scripts.Script):

    def title(self):
        return 'ControlNet Travel'

    def describe(self):
        return 'Travel from one controlnet hint to another in the latent space.'

    def show(self, is_img2img):
        global controlnet_found
        if not controlnet_found: print('extension Mikubill/sd-webui-controlnet not found, ControlNet Travel ignored')
        return controlnet_found

    def ui(self, is_img2img):
        with gr.Row(variant='compact'):
            interp_meth = gr.Dropdown(label=LABEL_INTERP_METH, value=lambda: DEFAULT_INTERP_METH, choices=CHOICES_INTERP_METH)
            steps       = gr.Text    (label=LABEL_STEPS,       value=lambda: DEFAULT_STEPS, max_lines=1)

        with gr.Row(variant='compact'):
            ctrlnet_ref_dir = gr.Text(label=LABEL_CTRLNET_REF_DIR, value=lambda: DEFAULT_CTRLNET_REF_DIR, max_lines=1)

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

        return [interp_meth, steps, ctrlnet_ref_dir,
                upscale_meth, upscale_ratio, upscale_width, upscale_height,
                video_fmt, video_fps, video_pad, video_pick,
                ext_video, ext_upscale]

    def run(self, p:StableDiffusionProcessing, 
            interp_meth:str, steps:str, ctrlnet_ref_dir:str, 
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int,
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            ext_video:bool, ext_upscale:bool):

        # enum lookup
        interp_meth: InterpMethod = InterpMethod(interp_meth)
        video_fmt:   VideoFormat  = VideoFormat (video_fmt)

        # Param check & type convert
        if ext_video:
            if video_pad <  0: return Processed(p, [], p.seed, f'video_pad must >= 0, but got {video_pad}')
            if video_fps <= 0: return Processed(p, [], p.seed, f'video_fps must > 0, but got {video_fps}')
            try: video_slice = parse_slice(video_pick)
            except: return Processed(p, [], p.seed, 'syntax error in video_slice')

        # Prepare ControlNet
        global controlnet_script
        try:
            from scripts.cldm import ControlNet
            for script in p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    controlnet_script = script
                    break
        except ImportError:
            pass
        if not controlnet_script: return Processed(p, [], p.seed, 'extension Mikubill/sd-webui-controlnet not running, why?')

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

        # Pack parameters
        self.lerp_meth  = interp_meth
        self.steps      = steps
        self.n_stages   = n_stages
        self.n_frames   = n_frames

        global interp_fn
        if interp_meth == InterpMethod.LINEAR:
            interp_fn = weighted_sum
        elif interp_meth == InterpMethod.RIFE:
            pass

        def save_image_hijack(params:ImageSaveParams):
            if not ext_upscale: return
            params.image = upscale_image(params.image, p.width, p.height, upscale_meth, upscale_ratio, upscale_width, upscale_height)

        images = []
        info = None
        def gen_image(append=True):
            nonlocal p, images, info
            proc = process_images(p)
            if not info: info = proc.info
            if append: images.extend(proc.images)
            else: return proc.images

        try:
            on_before_image_saved(save_image_hijack)

            global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, interp_alpha, interp_ip
            from_hint_cond      .clear()
            to_hint_cond        .clear()
            from_control_tensors.clear()
            to_control_tensors  .clear()

            # Step 1: draw the init image
            setattr(p, 'init_images', [Image.open(self.ctrlnet_ref_fps[0])])
            interp_alpha = 0.0
            gen_image()

            # travel through stages
            for i in range(1, n_stages):
                if state.interrupted: raise StopIteration

                # Setp 3: move to next stage
                from_control_tensors = [t for t in to_control_tensors] ; to_control_tensors.clear()
                from_hint_cond       = [t for t in to_hint_cond]       ; to_hint_cond      .clear()
                setattr(p, 'init_images', [Image.open(self.ctrlnet_ref_fps[i])])
                interp_alpha = 0.0
                cached_images = gen_image(append=False)

                # Step 2: draw the interpolated images
                n_inter = steps[i] + 1
                for t in range(1, n_inter):
                    if state.interrupted: raise StopIteration

                    mid_hint_cond.clear()
                    device = devices.get_device_for("controlnet")
                    for hintA, hintB in zip(from_hint_cond, to_hint_cond):
                        hintC = interp_fn(hintA.to(device), hintB.to(device), interp_alpha)
                        #print('  hint diff:', (hintC - param.hint_cond).abs().mean().item())
                        mid_hint_cond.append(hintC)

                    interp_alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]
                    interp_ip = 0
                    gen_image()

                # adjust order
                images.extend(cached_images)

        except StopIteration: pass
        finally:
            remove_callbacks_for_function(save_image_hijack)

            from_hint_cond      .clear()
            to_hint_cond        .clear()
            from_control_tensors.clear()
            to_control_tensors  .clear()
            devices.torch_gc()
            gc.collect()

        # Save video
        if ext_video: save_video(images, video_slice, video_pad, video_fps, video_fmt, os.path.join(self.log_dp, f'travel-{travel_number:05}'))

        return Processed(p, images, p.seed, info)
