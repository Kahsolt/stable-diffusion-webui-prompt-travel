import os
import random
from pathlib import Path
from copy import deepcopy
from PIL import Image
from typing import List, Tuple, Union
from traceback import print_exc

import gradio as gr
import torch.nn.functional as F
import numpy as np
try: from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError: print('package moviepy not installed, will not be able to generate video')

import modules.scripts as scripts
from modules.processing import Processed, StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning
from modules.shared import state, opts

__ = lambda key, value=None: opts.data.get(f'customscript/prompt_travel.py/txt2img/{key}/value', value)

DEFAULT_MODE           = __('Travel mode', 'linear')
DEFAULT_GENESIS        = __('Frame genesis', 'fixed')
DEFAULT_STEPS          = __('Travel steps between stages', 30)
DEFAULT_DENOISE_W      = __('Denoise strength', 1.0)
DEFAULT_REPLACE_ORDER  = __('Replace order', 'grad_min')
DEFAULT_GRAD_ALPHA     = __('Step size', 0.01)
DEFAULT_GRAD_ITER      = __('Step count', 1)
DEFAULT_GRAD_METH      = __('Step method', 'clip')
DEFAULT_GRAD_W_LATENT  = __('Weight for latent match', 1)
DEFAULT_GRAD_W_COND    = __('Weight for cond match', 1)
DEFAULT_VIDEO_FPS      = __('Video FPS', 10)
DEFAULT_VIDEO_FMT      = __('Video file format', 'mp4')
DEFAULT_VIDEO_PAD      = __('Pad begin/end frames', 0)
DEFAULT_VIDEO_PICK     = __('Pick frame by slice', '')
DEFAULT_VIDEO_RTRIM    = __('Video drop last frame', False)
DEFAULT_DEBUG          = __('Show console debug', True)

CHOICES_GENESIS        = ['fixed', 'successive']
CHOICES_MODE           = ['linear', 'replace', 'grad']
CHOICES_REPLACE_ORDER  = ['random', 'most_similar', 'most_different', 'grad_min', 'grad_max']
CHOICES_GRAD_METH      = ['clip', 'sign', 'tanh']
CHOICES_VIDEO_FMT      = ['mp4', 'gif']

T_tokens  = List[List[float]]
T_weights = List[List[int]]

# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

from modules.processing import create_infotext, decode_first_stage, get_fixed_seed

import torch
import numpy as np
from PIL import Image

import modules.sd_hijack
from modules import devices, lowvram
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae


def process_images_before(p: StableDiffusionProcessing):
    try:
        for k, v in p.override_settings.items():
            setattr(opts, k, v)
            if k == 'sd_hypernetwork': shared.reload_hypernetworks()  # make onchange call for changing hypernet
            if k == 'sd_model_checkpoint': sd_models.reload_model_weights()  # make onchange call for changing SD model
            if k == 'sd_vae': sd_vae.reload_vae_weights()  # make onchange call for changing VAE
    except:
        pass

def process_images_after(p: StableDiffusionProcessing):
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    for k, v in stored_opts.items():
        setattr(opts, k, v)
        if k == 'sd_hypernetwork': shared.reload_hypernetworks()
        if k == 'sd_model_checkpoint': sd_models.reload_model_weights()
        if k == 'sd_vae': sd_vae.reload_vae_weights()


def process_images_prompt_to_cond(p: StableDiffusionProcessing, ret_token_and_weight=False) -> tuple:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed    = p.seed
    subseed = p.subseed

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

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

    with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if hasattr(p, 'scripts') and p.scripts is not None:
        p.scripts.process(p)

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        if state.skipped:
            state.skipped = False
        
        if state.interrupted:
            return

        n = 0
        prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
        negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
        seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
        subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

        if len(prompts) == 0:
            return

        if hasattr(p, 'scripts') and p.scripts is not None:
            p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

        with devices.autocast():
            # 'prompt string' => tensor([T, D])
            uc, uc_tokens, uc_weights = get_learned_conditioning(shared.sd_model, negative_prompts, p.steps)
            c, c_tokens, c_weights = get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)

        devices.torch_gc()

        if ret_token_and_weight:
            return c, uc, prompts, seeds, subseeds, (c_tokens, c_weights, uc_tokens, uc_weights)
        else:
            return c, uc, prompts, seeds, subseeds

def process_images_cond_to_image(p: StableDiffusionProcessing, c, uc, prompts, seeds, subseeds) -> Processed:
    comments = {}
    infotexts = []
    output_images = []

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

        # [B=1, C=4, H=64,  W=64] => [B=1, C=3, H=512, W=512]
        x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
        x_samples_ddim = torch.stack(x_samples_ddim).float()
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        del samples_ddim

        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        devices.torch_gc()

        n = 0       # batch count for legacy compatible
        for i, x_sample in enumerate(x_samples_ddim):
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)

            if p.restore_faces:
                if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                    images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, 
                                      info=infotext(n, i), p=p, suffix="-before-face-restoration")
                devices.torch_gc()
                x_sample = modules.face_restoration.restore_faces(x_sample)
                devices.torch_gc()

            image = Image.fromarray(x_sample)

            if opts.samples_save and not p.do_not_save_samples:
                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

            text = infotext(n, i)
            infotexts.append(text)
            if opts.enable_pnginfo: image.info["parameters"] = text
            output_images.append(image)

        del x_samples_ddim 

        devices.torch_gc()
        state.nextjob()

    devices.torch_gc()

    if len(model_hijack.comments) > 0:
        for comment in model_hijack.comments:
            comments[comment] = 1

    res = Processed(p, output_images, p.all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]), 
                    subseed=p.all_subseeds[0], all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds, 
                    index_of_first_image=0, infotexts=infotexts)

    if hasattr(p, 'scripts') and p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

# ↑↑↑ the above is modified from 'modules/processing.py' ↑↑↑


# ↓↓↓ the following is modified from 'modules/prompt_parser.py' ↓↓↓

from modules.prompt_parser import get_learned_conditioning_prompt_schedules, get_multicond_prompt_list
from modules.prompt_parser import ComposableScheduledPromptConditioning

def get_learned_conditioning(model, prompts, steps) -> Tuple[ScheduledPromptConditioning, T_tokens, T_weights]:
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):      # forced to be lengthed 1
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds, tokens, weights = LatentDiffusion_get_learned_conditioning(model, texts)

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res, tokens, weights

def get_multicond_learned_conditioning(model, prompts, steps) -> Tuple[MulticondLearnedConditioning, T_tokens, T_weights]:
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning, tokens, weights = get_learned_conditioning(model, prompt_flat_list, steps)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res), tokens, weights

# ↑↑↑ the above is modified from 'modules/prompt_parser.py' ↑↑↑


# ↓↓↓ the following is modified from 'ldm.models.diffusion/ddpm.py' ↓↓↓

#from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase
#from ldm.models.diffusion import LatentDiffusion
#from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

def get_latent_loss(sd_model, latent:torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
    # forward(self:LatentDiffusion, x, c, *args, **kwargs)
    self, x, c = sd_model, latent, cond

    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()  # [B=1]
    if self.model.conditioning_key is not None:     # 'crossattn'
        assert c is not None
        if self.cond_stage_trainable:               # False
            c = self.get_learned_conditioning(c)
        if self.shorten_cond_schedule:              # False
            tc = self.cond_ids[t].to(self.device)
            c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    
    # p_losses(self:LatentDiffusion, x_start, cond, t, noise=None)
    x_start, cond, t = x, c, t

    noise = torch.randn_like(x_start)                               # [B=1, C=4, H=64, W=64]
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      # [B=1, C=4, H=64, W=64], diffsusion step
    model_output = self.apply_model(x_noisy, t, cond)               # [B=1, C=4, H=64, W=64], inverse diffusion step

    if    self.parameterization == "x0":  target = x_start
    elif  self.parameterization == "eps": target = noise            # it goes this way
    else: raise NotImplementedError()

    loss_simple = self.get_loss(model_output, target, mean=False)   # self.loss_type == 'l2'; the model shoud predict the noise that is added
    logvar_t = self.logvar[t].to(self.device)                       # self.logvar == torch.zeros([1000])
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    loss = self.l_simple_weight * loss                              # self.l_simple_weight == 1.0

    loss_vlb = self.get_loss(model_output, target, mean=False)
    loss_vlb = (self.lvlb_weights[t] * loss_vlb)                    # self.lvlb_weights is non-zeros
    loss += (self.original_elbo_weight * loss_vlb)                  # but self.original_elbo_weight == 0.0, I don't know why :(

    return loss                                                     # [B=1, C=4, H=64, W=64]

def text_to_token(self, text:List[str]) -> tuple:
    # FrozenCLIPEmbedderWithCustomWords.FrozenCLIPEmbedder.CLIPTokenizer

    with devices.autocast('cuda'):
        if opts.use_old_emphasis_implementation:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text_old(text)
        else:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text(text)
    
    return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

def token_to_cond(self, batch_multipliers:T_weights, remade_batch_tokens:T_tokens, used_custom_terms, hijack_comments, hijack_fixes) -> torch.Tensor:
    self.hijack.comments += hijack_comments

    if len(used_custom_terms) > 0:
        self.hijack.comments.append("Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

    if opts.use_old_emphasis_implementation:
        self.hijack.fixes = hijack_fixes
        with torch.no_grad(), devices.autocast():
            return self.process_tokens(remade_batch_tokens, batch_multipliers)

    # allow length > 75
    z = None
    i = 0
    while max(map(len, remade_batch_tokens)) != 0:
        rem_tokens = [x[75:] for x in remade_batch_tokens]
        rem_multipliers = [x[75:] for x in batch_multipliers]

        self.hijack.fixes = []
        for unfiltered in hijack_fixes:
            fixes = []
            for fix in unfiltered:
                if fix[0] == i:
                    fixes.append(fix[1])
            self.hijack.fixes.append(fixes)

        tokens = []
        multipliers = []
        for j in range(len(remade_batch_tokens)):
            if len(remade_batch_tokens[j]) > 0:
                tokens.append(remade_batch_tokens[j][:75])
                multipliers.append(batch_multipliers[j][:75])
            else:
                tokens.append([self.wrapped.tokenizer.eos_token_id] * 75)
                multipliers.append([1.0] * 75)

        with torch.no_grad(), devices.autocast():
            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

        remade_batch_tokens = rem_tokens
        batch_multipliers = rem_multipliers
        i += 1

    return z

def token_to_text(self, tokens:T_tokens) -> str:
    id_2_word = { v: k for k, v in self.wrapped.tokenizer.get_vocab().items() }

    words = []
    for tk in tokens[0]:    # force B=1
        w = id_2_word.get(tk, '<unk>')
        if w in ['<|startoftext|>', '<|endoftext|>']: continue
        if w.endswith('</w>'): w = w[:-4]
        words.append(w)

    return ' '.join(words)

def FrozenCLIPEmbedderWithCustomWords_forward(self, text:List[str]) -> Tuple[torch.Tensor, T_tokens, T_weights]:
    batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = text_to_token(self, text)
    cond = token_to_cond(self, batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes)

    return cond, remade_batch_tokens, batch_multipliers

def LatentDiffusion_get_learned_conditioning(sd_model, cond:List[str]) -> Tuple[torch.Tensor, T_tokens, T_weights]:
    self, c = sd_model, cond

    if self.cond_stage_forward is None:
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if 'DiagonalGaussianDistribution' in str(type(c)):
                c = c.mode()
        else:
            #c = self.cond_stage_model(c)    # => goes this way, [B=1, T=77*n, D=768], [[]], [[]]
            c, tokens, weights = FrozenCLIPEmbedderWithCustomWords_forward(self.cond_stage_model, c)
    else:
        assert hasattr(self.cond_stage_model, self.cond_stage_forward)
        c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)

    return c, tokens, weights
    
# ↑↑↑ the above is modified from 'ldm.models.diffusion/ddpm.py' ↑↑↑


def image_to_latent(model, img: Image) -> torch.Tensor:
    #from ldm.models.diffusion import LatentDiffusion
    # type(model) == LatentDiffusion

    im = np.array(img).astype(np.uint8)
    im = (im / 127.5 - 1.0).astype(np.float32)      # value range [-1.0, 1.0]
    x = torch.from_numpy(im)
    x = torch.moveaxis(x, 2, 0)
    x = x.unsqueeze(dim=0)          # [B=1, C=3, H=512, W=512]
    x = x.to(model.device)
    
    latent = model.get_first_stage_encoding(model.encode_first_stage(x))    # [B=1, C=4, H=64, W=64]
    return latent

def spc_get_cond(c:List[List[ScheduledPromptConditioning]]) -> torch.Tensor:
    return c[0][0].cond

def spc_replace_cond(c:List[List[ScheduledPromptConditioning]], cond: torch.Tensor) -> ScheduledPromptConditioning:
    r = deepcopy(c)
    spc = r[0][0]
    r[0][0] = ScheduledPromptConditioning(spc.end_at_step, cond=cond)
    return r

def mlc_get_cond(c:MulticondLearnedConditioning) -> torch.Tensor:
    return c.batch[0][0].schedules[0].cond      # [B=1, T=77, D=768]

def mlc_replace_cond(c:MulticondLearnedConditioning, cond: torch.Tensor) -> MulticondLearnedConditioning:
    r = deepcopy(c)
    spc = r.batch[0][0].schedules[0]
    r.batch[0][0].schedules[0] = ScheduledPromptConditioning(spc.end_at_step, cond=cond)
    return r


def update_img2img_p(p: StableDiffusionProcessing, img:Image, denoising_strength:float=0.75) -> StableDiffusionProcessingImg2Img:
    if isinstance(p, StableDiffusionProcessingImg2Img):
        p.init_images = [img]
        p.denoising_strength = denoising_strength
        return p

    if isinstance(p, StableDiffusionProcessingTxt2Img):
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
            's_churn',
            's_tmax',
            's_tmin',
            's_noise',
            'override_settings',
            'sampler_index',
        ]
        kwargs = { k: getattr(p, k) for k in dir(p) if k in KNOWN_KEYS }    # inherit params
        return StableDiffusionProcessingImg2Img(
            init_images=[img],
            denoising_strength=denoising_strength,
            **kwargs,
        )

def parse_slice(picker:str) -> Union[slice, None]:
    if not picker.strip(): return None
    
    to_int = lambda s: None if not s else int(s)
    segs = [to_int(x.strip()) for x in picker.strip().split(':')]
    
    start, stop, step = None, None, None
    if   len(segs) == 1:        stop,      = segs
    elif len(segs) == 2: start, stop       = segs
    elif len(segs) == 3: start, stop, step = segs
    else: raise ValueError
    
    return slice(start, stop, step)


class Script(scripts.Script):

    def title(self):
        return 'Prompt Travel'

    def describe(self):
        return 'Travel from one prompt to another in the latent space.'

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Row():
                mode    = gr.Radio(label='Travel mode',   value=lambda: DEFAULT_MODE,    choices=CHOICES_MODE)
                genesis = gr.Radio(label='Frame genesis', value=lambda: DEFAULT_GENESIS, choices=CHOICES_GENESIS)
            with gr.Row():
                steps = gr.Text(label='Travel steps between stages', value=lambda: DEFAULT_STEPS, max_lines=1)
                denoise_strength = gr.Slider(label='Denoise strength', value=lambda: DEFAULT_DENOISE_W, minimum=0.0, maximum=1.0, visible=True)
    
        with gr.Group(visible=False) as tab_replace:
            replace_order = gr.Dropdown(label='Replace order', value=lambda: DEFAULT_REPLACE_ORDER, choices=CHOICES_REPLACE_ORDER)

        with gr.Group(visible=False) as tab_grad:
            with gr.Row():
                grad_alpha = gr.Number(label='Step size',  value=lambda: DEFAULT_GRAD_ALPHA)
                grad_iter  = gr.Number(label='Step count', value=lambda: DEFAULT_GRAD_ITER, precision=0)
            with gr.Row():
                grad_meth     = gr.Dropdown(label='Step method', value=lambda: DEFAULT_GRAD_METH, choices=CHOICES_GRAD_METH)
                grad_w_latent = gr.Number  (label='Weight for latent match', value=lambda: DEFAULT_GRAD_W_LATENT)
                grad_w_cond   = gr.Number  (label='Weight for cond match',   value=lambda: DEFAULT_GRAD_W_COND)

        with gr.Group():
            with gr.Row():
                video_fmt  = gr.Dropdown(label='Video file format',     value=lambda: DEFAULT_VIDEO_FMT, choices=CHOICES_VIDEO_FMT)
                video_fps  = gr.Number  (label='Video FPS',             value=lambda: DEFAULT_VIDEO_FPS)
                video_pad  = gr.Number  (label='Pad begin/end frames',  value=lambda: DEFAULT_VIDEO_PAD,  precision=0)
                video_pick = gr.Text    (label='Pick frame by slice',   value=lambda: DEFAULT_VIDEO_PICK, max_lines=1)

        with gr.Group():
            show_debug = gr.Checkbox(label='Show console debug', value=lambda: DEFAULT_DEBUG)

        def switch_mode(mode, replace_order):
            requires_grad = mode == 'grad' or replace_order.startswith('grad')
            show_replace  = mode == 'replace'
            if mode == 'linear': requires_grad = show_replace = False
            return [
                { 'visible': show_replace,  '__type__': 'update' },
                { 'visible': requires_grad, '__type__': 'update' },
            ]

        mode         .change(fn=switch_mode, inputs=[mode, replace_order], outputs=[tab_replace, tab_grad])
        replace_order.change(fn=switch_mode, inputs=[mode, replace_order], outputs=[tab_replace, tab_grad])

        # This not work, do not know why...
        #def switch_genesis(genesis):
        #    show = genesis == 'successive'
        #    return [
        #        { 'visible': show, '__type__': 'update' }
        #    ]
        #
        #genesis.change(fn=switch_genesis, inputs=[genesis], outputs=[denoise_strength])

        return [genesis, denoise_strength, mode, steps, 
            replace_order,
            grad_alpha, grad_iter, grad_meth, grad_w_latent, grad_w_cond,
            video_fmt, video_fps, video_pad, video_pick,
            show_debug]
    
    def get_next_sequence_number(path):
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

    def run(self, p:StableDiffusionProcessing, 
            genesis:str, denoise_strength:float, mode:str, steps:str, 
            replace_order: str,
            grad_alpha:float, grad_iter:int, grad_meth:str, grad_w_latent:float, grad_w_cond:float,
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            show_debug:bool):
        
        # Param check
        if grad_iter <= 0: return Processed(p, [], p.seed, 'grad_iter must > 0')
        if video_pad  < 0: return Processed(p, [], p.seed, 'video_pad must >= 0')
        if video_fps  < 0: return Processed(p, [], p.seed, 'video_fps must >= 0')
        try: video_slice = parse_slice(video_pick)
        except: return Processed(p, [], p.seed, 'syntax error in video_slice')
        
        # Prepare prompts
        prompt_pos = p.prompt.strip()
        if not prompt_pos: return Processed(p, [], p.seed, 'positive prompt should not be empty :(')
        pos_prompts = [p.strip() for p in prompt_pos.split('\n') if p.strip()]
        if len(pos_prompts) == 1:
            # NOTE: if only single stage specified, we double it to allow wandering around :)
            if mode == 'grad' or p.subseed == -1: pos_prompts = pos_prompts * 2
            else: return Processed(p, [], p.seed, 'should specify at least two lines of prompt to travel between :)')
        prompt_neg = p.negative_prompt.strip()
        neg_prompts = [p.strip() for p in prompt_neg.split('\n') if p.strip()]
        if len(neg_prompts) == 0: neg_prompts = ['']
        n_stages = max(len(pos_prompts), len(neg_prompts))
        while len(pos_prompts) < n_stages: pos_prompts.append(pos_prompts[-1])
        while len(neg_prompts) < n_stages: neg_prompts.append(neg_prompts[-1])

        try: steps = [int(s.strip()) for s in steps.strip().split(',')]
        except: return Processed(p, [], p.seed, f'cannot parse steps options: {steps}')
    
        if len(steps) == 1:
            steps = [steps[0]] * (n_stages - 1)
        elif len(steps) != n_stages - 1:
            info = (f'stage count mismatch: you have {n_stages} prompt stages, but specified {len(steps)} steps; should assure len(steps) == len(stages) - 1')
            return Processed(p, [], p.seed, info)
        count = sum(steps) + n_stages
        if show_debug: print(f'n_stages={n_stages}, steps={steps}')
        steps.insert(0, -1)     # fixup the first stage

        # Custom saving path
        travel_path = os.path.join(p.outpath_samples, 'prompt_travel')
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        self.log_dp = os.path.join(travel_path, f'{travel_number:05}')
        p.outpath_samples = self.log_dp
        os.makedirs(self.log_dp, exist_ok=True)
        self.log_fp = os.path.join(self.log_dp, 'log.txt')

        # Force Batch Count and Batch Size to 1
        p.n_iter     = 1
        p.batch_size = 1

        # Random unified const seed
        p.seed = get_fixed_seed(p.seed)     # fix it to assure all processes using the same major seed
        self.subseed = p.subseed            # stash it to allow using random subseed for each process (when -1)
        if show_debug:
            print('seed:',             p.seed)
            print('subseed:',          p.subseed)
            print('subseed_strength:', p.subseed_strength)

        # Start job
        state.job_count = count
        print(f'Generating {count} images.')

        # Pack parameters
        self.p                = p
        self.pos_prompts      = pos_prompts
        self.neg_prompts      = neg_prompts
        self.steps            = steps
        self.genesis          = genesis
        self.denoise_strength = denoise_strength
        self.replace_order    = replace_order
        self.grad_alpha       = grad_alpha
        self.grad_iter        = grad_iter
        self.grad_meth        = grad_meth
        self.grad_w_latent    = grad_w_latent
        self.grad_w_cond      = grad_w_cond
        self.show_debug       = show_debug

        # Dispatch
        process_images_before(p)
        if   mode == 'linear' : images, info = self.run_linear()
        elif mode == 'replace': images, info = self.run_replace()
        elif mode == 'grad'   : images, info = self.run_grad()
        process_images_after(p)

        # Save video
        if video_fps > 0 and len(images) > 1:
            try:
                seq = [np.asarray(t) for t in images]
                if video_slice:   seq = seq[video_slice]
                if video_pad > 0: seq = [seq[0]] * video_pad + seq + [seq[-1]] * video_pad
                clip = ImageSequenceClip(seq, fps=video_fps)
                fbase = os.path.join(self.log_dp, f'travel-{travel_number:05}')
                if video_fmt == 'mp4':
                    clip.write_videofile(fbase + '.mp4', verbose=False, audio=False)
                elif video_fmt == 'gif':
                    clip.write_gif(fbase + '.gif', loop=True)
            except NameError: pass
            except: print_exc()

        return Processed(p, images, p.seed, info)

    def run_linear(self):
        p:StableDiffusionProcessing = self.p
        genesis:str                 = self.genesis
        denoise_strength:float      = self.denoise_strength
        pos_prompts:List[str]       = self.pos_prompts
        neg_prompts:List[str]       = self.neg_prompts
        steps:List[int]             = self.steps
        show_debug:bool             = self.show_debug

        n_stages = len(steps)
        n_frames = sum(steps) + n_stages - 1
        initial_info = None
        images = []

        def weighted_sum(A, B, alpha:float, kind:str) -> Union[ScheduledPromptConditioning, MulticondLearnedConditioning]:
            ''' linear interpolate on latent space of condition '''
            if kind == 'pos':
                condA = mlc_get_cond(A)
                condB = mlc_get_cond(B)
                condC = (1 - alpha) * condA + (alpha) * condB
                C = mlc_replace_cond(A, condC)
            if kind == 'neg':
                condA = spc_get_cond(A)
                condB = spc_get_cond(B)
                condC = (1 - alpha) * condA + (alpha) * condB
                C = spc_replace_cond(A, condC)
            return C

        def gen_image(pos_hidden, neg_hidden, prompts, seeds, subseeds):
            nonlocal images, initial_info, p
            proc = process_images_cond_to_image(p, pos_hidden, neg_hidden, prompts, seeds, subseeds)
            if initial_info is None: initial_info = proc.info
            img = proc.images[0]
            if genesis == 'successive': p = update_img2img_p(p, img, denoise_strength)
            images += [img]

        # Step 1: draw the init image
        if show_debug:
            print(f'[stage 1/{n_stages}]')
            print(f'  pos prompts: {pos_prompts[0]}')
            print(f'  neg prompts: {neg_prompts[0]}')
        p.prompt          = pos_prompts[0]
        p.negative_prompt = neg_prompts[0]
        p.subseed         = self.subseed
        from_pos_hidden, from_neg_hidden, prompts, seeds, subseeds = process_images_prompt_to_cond(p)
        gen_image(from_pos_hidden, from_neg_hidden, prompts, seeds, subseeds)
        
        # travel through stages
        i_frames = 1
        for i in range(1, n_stages):
            if state.interrupted: break
            devices.torch_gc()

            state.job = f'{i_frames}/{n_frames}'
            state.job_no = i_frames + 1
            i_frames += 1

            # only change target prompts
            if show_debug:
                print(f'[stage {i+1}/{n_stages}]')
                print(f'  pos prompts: {pos_prompts[i]}')
                print(f'  neg prompts: {neg_prompts[i]}')
            p.prompt           = pos_prompts[i]
            p.negative_prompt  = neg_prompts[i]
            p.subseed          = self.subseed
            to_pos_hidden, to_neg_hidden, prompts, seeds, subseeds = process_images_prompt_to_cond(p)

            # Step 2: draw the interpolated images
            is_break_iter = False
            n_inter = steps[i] + 1
            for t in range(1, n_inter):
                if state.interrupted: is_break_iter = True ; break
                devices.torch_gc()

                alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]
                inter_pos_hidden = weighted_sum(from_pos_hidden, to_pos_hidden, alpha, kind='pos')
                inter_neg_hidden = weighted_sum(from_neg_hidden, to_neg_hidden, alpha, kind='neg')
                gen_image(inter_pos_hidden, inter_neg_hidden, prompts, seeds, subseeds)

            if is_break_iter: break

            # Step 3: draw the fianl stage
            gen_image(to_pos_hidden, to_neg_hidden, prompts, seeds, subseeds)
            
            # move to next stage
            from_pos_hidden, from_neg_hidden = to_pos_hidden, to_neg_hidden

        return images, initial_info

    def run_replace(self):
        p:StableDiffusionProcessing = self.p
        genesis:str                 = self.genesis
        denoise_strength:float      = self.denoise_strength
        pos_prompts:List[str]       = self.pos_prompts
        steps:List[int]             = self.steps
        replace_order:str           = self.replace_order
        grad_w_latent:float         = self.grad_w_latent
        grad_w_cond:float           = self.grad_w_cond
        show_debug:bool             = self.show_debug

        clip_model = p.sd_model.cond_stage_model
        n_stages = len(steps)
        n_frames = sum(steps) + n_stages - 1
        initial_info = None
        images = []

        def gen_image(pos_hidden, neg_hidden, prompts, seeds, subseeds, ret_img=False):
            nonlocal images, initial_info, p
            proc = process_images_cond_to_image(p, pos_hidden, neg_hidden, prompts, seeds, subseeds)
            if initial_info is None: initial_info = proc.info
            img = proc.images[0]
            if genesis == 'successive': p = update_img2img_p(p, img, denoise_strength)
            if ret_img: return img
            else: images += [img]
        
        # Step 1: draw init image
        if show_debug: print(f'[stage 1/{n_stages}] prompts: {pos_prompts[0]}')
        p.prompt = pos_prompts[0]
        p.subseed = self.subseed
        c, uc, prompts, seeds, subseeds, (c_tokens, c_weights, uc_tokens, uc_weights) = process_images_prompt_to_cond(p, ret_token_and_weight=True)
        gen_image(c, uc, prompts, seeds, subseeds)

        # make log
        log_fh = open(self.log_fp, 'w', encoding='utf-8')
        log_fh.write(f'replace_order = {replace_order}\n')
        log_fh.write('\n')

        text_rev = token_to_text(clip_model, c_tokens)
        log_fh.write(f'tokens: {text_rev}\n')
        log_fh.write('\n')
        
        # travel between stages
        i_frames = 1
        for i in range(1, n_stages):
            if state.interrupted: break
            devices.torch_gc()

            # Step 2: draw the stage target
            if show_debug: print(f'[stage {i+1}/{n_stages}] prompts: {pos_prompts[i]}')
            p.prompt = pos_prompts[i]
            p.subseed = self.subseed
            *params, (tgt_c_tokens, tgt_c_weights, tgt_uc_tokens, tgt_uc_weights) = process_images_prompt_to_cond(p, ret_token_and_weight=True)
            _, _, used_custom_terms, hijack_comments, hijack_fixes, _ = text_to_token(clip_model, [p.prompt])
            target_image = gen_image(*params, ret_img=True)     # cache it here to make video sequence order right
            
            with torch.no_grad(), devices.autocast():
                if replace_order.startswith('grad'):
                    target_latent = image_to_latent(p.sd_model, target_image)   # [B=1, C=4, H=64, W=64]
                    target_cond   = mlc_get_cond(params[0]).unsqueeze(0)        # [B=1, T=77, D=768]
                else:
                    embed_layer  = clip_model.wrapped.transformer.get_input_embeddings()     # transformers.models.clip.CLIPTextModel
                    source_embed = embed_layer(torch.LongTensor(c_tokens)    .to(p.sd_model.device))        # [B=1, T=75, D=768]
                    target_embed = embed_layer(torch.LongTensor(tgt_c_tokens).to(p.sd_model.device))
                    L1_dist      = F.l1_loss(source_embed, target_embed, reduction='none').squeeze(dim=0).mean(dim=-1).cpu().numpy()    # [T=75]

            # Step 3: draw the inter-mediums
            is_break_step = False
            for _ in range(steps[i]):
                if state.interrupted: is_break_step = True ; break
                devices.torch_gc()

                state.job = f'{i_frames}/{n_frames}'
                state.job_no = i_frames + 1
                i_frames += 1

                mask = np.asarray(c_tokens[0]) != np.asarray(tgt_c_tokens[0])   # [T=75]
                n_replaces = sum(mask)
                if n_replaces == 0: break
                cnt = max(n_replaces // steps[i], 1)
                if show_debug: print(f'need to replace {n_replaces} tokens, {cnt} tokens per travel step')

                # token inverse
                text_rev = token_to_text(clip_model, c_tokens)
                log_fh.write(f'tokens: {text_rev}\n')
                tokens = text_rev.split(' ')

                def _replace_tokens(sorted_indexes, c_tokens, tgt_c_tokens):
                    nonlocal mask, cnt
                    k, done = 0, 0
                    while done < cnt and sum(mask) > 0 and k < len(sorted_indexes):
                        idx = sorted_indexes[k]
                        if mask[idx]:
                            c_tokens[0][idx] = tgt_c_tokens[0][idx]
                            done += 1
                        k += 1
                
                if replace_order.startswith('grad'):
                    with devices.autocast():
                        current_cond = mlc_get_cond(c).unsqueeze(0).clone()     # [B=1, T=77, D=768]

                        current_cond .requires_grad = True
                        target_cond  .requires_grad = True
                        target_latent.requires_grad = True

                        loss_latent = get_latent_loss(p.sd_model, target_latent, current_cond)                      # [B=1, C=4, H=64, W=64]
                        grad_latent = torch.autograd.grad(loss_latent, current_cond, grad_outputs=loss_latent)[0]   # [B=1, T=77, D=768]
                        loss_cond   = F.l1_loss(current_cond, target_cond, reduction='none')                        # [B=1, T=77, D=768]
                        grad_cond   = torch.autograd.grad(loss_cond, current_cond, grad_outputs=loss_cond)[0]       # [B=1, T=77, D=768]
                        grad = grad_latent * grad_w_latent + grad_cond * grad_w_cond                                # [B=1, T=77, D=768]
                        
                        grad_trim = grad.squeeze(dim=0)[1:-1, :]    # [T=75, D=768]  
                        grad_token = grad_trim.mean(dim=1)          # [T=75]

                    sorted_indexes_grad_ascending = grad_token.argsort().tolist()
                    if replace_order == 'grad_min':
                        sorted_indexes = sorted_indexes_grad_ascending
                    elif replace_order == 'grad_max':
                        sorted_indexes = sorted_indexes_grad_ascending[::-1]
                    _replace_tokens(sorted_indexes, c_tokens, tgt_c_tokens)
                
                else:
                    sorted_indexes_L1_ascending = L1_dist.argsort()
                    if replace_order == 'random':
                        neq_indexes = [i for i, m in enumerate(mask) if m]
                        random.shuffle(neq_indexes)
                        _replace_tokens(neq_indexes, c_tokens, tgt_c_tokens)
                    else:
                        if replace_order == 'most_similar':
                            sorted_indexes = sorted_indexes_L1_ascending
                        elif replace_order == 'most_different':
                            sorted_indexes = sorted_indexes_L1_ascending[::-1]
                        _replace_tokens(sorted_indexes, c_tokens, tgt_c_tokens)

                # log token importance (?
                if replace_order.startswith('grad'):
                    tokens_grad_asc = [tokens[idx] for idx in sorted_indexes_grad_ascending if idx < len(tokens)]
                    log_fh.write(f'  >> grad ascend: {" ".join(tokens_grad_asc)}\n')
                else:
                    tokens_l1_asc = [tokens[idx] for idx in sorted_indexes_L1_ascending if idx < len(tokens)]
                    log_fh.write(f'  >> embed L1-distance ascend: {" ".join(tokens_l1_asc)}\n')
                log_fh.write(f'\n')
                log_fh.flush()

                # move to new 'c' (one travel step!)
                # FIXME: we do not walk on 'uc' so far
                cond = token_to_cond(clip_model, c_weights, c_tokens, used_custom_terms, hijack_comments, hijack_fixes)
                c = mlc_replace_cond(c, cond.detach().squeeze(0))
                gen_image(c, uc, prompts, seeds, subseeds)

            if is_break_step: break

            # append the finishing image for current stage 
            images += [target_image]

            # shift: last stage's final info becomes new stage's init info
            c, uc, prompts, seeds, subseeds = params
            c_tokens, c_weights, uc_tokens, uc_weights = tgt_c_tokens, tgt_c_weights, tgt_uc_tokens, tgt_uc_weights

        # save log
        log_fh.close()

        return images, initial_info

    def run_grad(self):
        p:StableDiffusionProcessing = self.p
        genesis:str                 = self.genesis
        denoise_strength:float      = self.denoise_strength
        pos_prompts:List[str]       = self.pos_prompts
        steps:List[int]             = self.steps
        grad_alpha:float            = self.grad_alpha
        grad_iter:int               = self.grad_iter
        grad_meth:str               = self.grad_meth
        grad_w_latent:float         = self.grad_w_latent
        grad_w_cond:float           = self.grad_w_cond
        show_debug:bool             = self.show_debug

        n_stages = len(steps)
        n_frames = sum(steps) + n_stages - 1
        initial_info = None
        images = []

        def gen_image(pos_hidden, neg_hidden, prompts, seeds, subseeds, ret_img=False):
            nonlocal images, initial_info, p
            proc = process_images_cond_to_image(p, pos_hidden, neg_hidden, prompts, seeds, subseeds)
            if initial_info is None: initial_info = proc.info
            img = proc.images[0]
            if genesis == 'successive': p = update_img2img_p(p, img, denoise_strength)
            if ret_img: return img
            else: images += [img]
        
        # Step 1: draw init image
        if show_debug: print(f'[stage 1/{n_stages}] prompts: {pos_prompts[0]}')
        p.prompt = pos_prompts[0]
        p.subseed = self.subseed
        c, uc, prompts, seeds, subseeds = process_images_prompt_to_cond(p)
        gen_image(c, uc, prompts, seeds, subseeds)

        # make log
        log_fh = open(self.log_fp, 'w', encoding='utf-8')
        log_fh.write(f'grad_alpha   = {grad_alpha}\n')
        log_fh.write(f'grad_iter    = {grad_iter}\n')
        log_fh.write(f'grad_meth    = {grad_meth}\n')
        log_fh.write(f'grad_w_grad  = {grad_w_latent}\n')
        log_fh.write(f'grad_w_match = {grad_w_cond}\n')
        log_fh.write('\n')

        # travel between stages
        i_frames = 1
        for i in range(1, n_stages):
            if state.interrupted: break
            devices.torch_gc()

            # Step 2: draw the stage target
            if show_debug: print(f'[stage {i+1}/{n_stages}] prompts: {pos_prompts[i]}')
            p.prompt = pos_prompts[i]
            p.subseed = self.subseed
            params = process_images_prompt_to_cond(p)
            target_image = gen_image(*params, ret_img=True)     # cache it here to make video sequence order right
            
            with torch.no_grad(), devices.autocast():
                target_latent = image_to_latent(p.sd_model, target_image)   # [B=1, C=4, H=64, W=64]
                target_cond   = mlc_get_cond(params[0]).unsqueeze(0)        # [B=1, T=77, D=768]
                source_cond   = mlc_get_cond(c).unsqueeze(0)
                L1_dist       = F.l1_loss(source_cond, target_cond).item()
            
            # Step 3: draw the inter-mediums
            is_break_step = False
            for _ in range(steps[i]):
                if state.interrupted: is_break_step = True ; break
                devices.torch_gc()

                state.job = f'{i_frames}/{n_frames}'
                state.job_no = i_frames + 1
                i_frames += 1

                with devices.autocast():
                    current_cond = mlc_get_cond(c).unsqueeze(0).clone()  # [B=1, T=77, D=768]

                    # simple PGD attack on cond (src prompt) to match latent (dst image)
                    is_break_iter = False
                    for _ in range(grad_iter):
                        if state.interrupted: is_break_iter = True ; break
                        devices.torch_gc()

                        current_cond .requires_grad = True
                        target_cond  .requires_grad = True
                        target_latent.requires_grad = True

                        loss_latent = get_latent_loss(p.sd_model, target_latent, current_cond)                      # [B=1, C=4, H=64, W=64]
                        grad_latent = torch.autograd.grad(loss_latent, current_cond, grad_outputs=loss_latent)[0]   # [B=1, T=77, D=768]
                        loss_cond   = F.l1_loss(current_cond, target_cond, reduction='none')                        # [B=1, T=77, D=768]
                        grad_cond   = torch.autograd.grad(loss_cond, current_cond, grad_outputs=loss_cond)[0]       # [B=1, T=77, D=768]
                        grad = grad_latent * grad_w_latent + grad_cond * grad_w_cond                                # [B=1, T=77, D=768]
                        
                        EPS = 1e-5
                        GAMMA = 0.2         # NOTE: if still gets oscilattion, try decreasing this
                        methods = {
                            'clip': lambda grad: (grad > EPS) * grad.clamp(GAMMA, 1.0) + (grad < -EPS) * grad.clamp(-1.0, -GAMMA),
                            'sign': lambda grad: grad.sign(),
                            'tanh': lambda grad: grad.tanh(),
                        }
                        current_cond = current_cond.detach() - methods[grad_meth](grad) * grad_alpha

                        with torch.no_grad():
                            l_latent = loss_latent.mean().item()
                            l_match  = loss_cond  .mean().item()
                            l_total  = l_latent + l_match
                            L1_from  = F.l1_loss(current_cond, source_cond).item()
                            L1_to    = F.l1_loss(current_cond, target_cond).item()    # FIXME: stop early when L1_to < grad_alpha / 2
                            grad_abs = grad.abs()
                            info = [
                                f'loss: {l_total} (l_grad: {l_latent}, l_match: {l_match})',
                                f'  |grad|.avg: {grad_abs.mean()}, |grad|.max: {grad_abs.max()}',
                                f'  L1 from src: {L1_from}, L1 to dst: {L1_to}, L1 total: {L1_dist}',
                            ]
                            log_fh.write('\n'.join(info))
                            log_fh.write('\n')
                            log_fh.flush()
                    
                    if is_break_iter: break

                # move to new 'c' (one travel step!)
                # FIXME: we do not walk on 'uc' so far
                c = mlc_replace_cond(c, current_cond.detach().squeeze(0))
                gen_image(c, uc, prompts, seeds, subseeds)

            if is_break_step: break

            # append the finishing image for current stage 
            images += [target_image]

            # shift: last stage's final info becomes new stage's init info
            c, uc, prompts, seeds, subseeds = params

        # save log
        log_fh.close()

        return images, initial_info
