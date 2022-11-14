import os
from copy import deepcopy

import gradio as gr
import torch.nn.functional as F
import numpy as np
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    print(f"moviepy python module not installed. Will not be able to generate video.")

import modules.scripts as scripts
from modules.processing import Processed, StableDiffusionProcessing
from modules.processing import *
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning
from modules.shared import state

DEFAULT_MODE           = 'linear'
DEFAULT_STEPS          = 30
DEFAULT_REPLACE_ORDER  = 'random'
DEFAULT_GRAD_ALPHA     = 0.01
DEFAULT_GRAD_ITERS     = 1
DEFAULT_GRAD_METH      = 'clip'
DEFAULT_GRAD_W_LATENT    = 1
DEFAULT_GRAD_W_COND   = 1
DEFAULT_FPS            = 10
DEFAULT_DEBUG          = True

CHOICES_MODE          = ['linear', 'replace', 'grad']
CHOICES_REPLACE_ORDER = ['random', 'similar', 'different']
CHOICES_GRAD_METH     = ['clip', 'sign', 'tanh']


# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

import torch
import numpy as np
from PIL import Image

import modules.sd_hijack
from modules import devices, prompt_parser, lowvram
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.face_restoration
import modules.images as images
import modules.styles

def process_images_inner_half_A(p: StableDiffusionProcessing) -> tuple:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    assert p.prompt is not None

    with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    devices.torch_gc()

    seed    = p.seed
    subseed = p.subseed

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    shared.prompt_styles.apply_styles(p)

    p.all_prompts  = p.batch_size * 1 * [p.prompt]
    p.all_seeds    = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
    p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if hasattr(p, 'scripts') and p.scripts is not None:
        p.scripts.process(p)

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = 1

        n = 0       # batch count for legacy compatible
        prompts  = p.all_prompts [n * p.batch_size : (n + 1) * p.batch_size]
        seeds    = p.all_seeds   [n * p.batch_size : (n + 1) * p.batch_size]
        subseeds = p.all_subseeds[n * p.batch_size : (n + 1) * p.batch_size]

        if hasattr(p, 'scripts') and p.scripts is not None:
            p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(shared.sd_model, len(prompts) * [p.negative_prompt], p.steps)
            c  = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)

            return c, uc, prompts, seeds, subseeds

def process_images_inner_half_B(p: StableDiffusionProcessing, c, uc, prompts, seeds, subseeds):
    comments = {}
    infotexts = []
    output_images = []

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

        samples_ddim = samples_ddim.to(devices.dtype_vae)
        x_samples_ddim = decode_first_stage(p.sd_model, samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        del samples_ddim

        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        devices.torch_gc()

        if opts.filter_nsfw:
            import modules.safety as safety
            x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)
        
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

            if p.color_corrections is not None and i < len(p.color_corrections):
                if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                    image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                    images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, 
                                      info=infotext(n, i), p=p, suffix="-before-color-correction")
                image = apply_color_correction(p.color_corrections[i], image)

            image = apply_overlay(image, p.paste_to, i, p.overlay_images)

            if opts.samples_save and not p.do_not_save_samples:
                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

            text = infotext(n, i)
            infotexts.append(text)
            if opts.enable_pnginfo: image.info["parameters"] = text
            output_images.append(image)

        del x_samples_ddim 

        devices.torch_gc()
        state.nextjob()

        p.color_corrections = None

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


# ↓↓↓ the following is modified from 'ldm.models.diffusion/ddpm.py' ↓↓↓

def get_latent_loss(sd_model, latent:torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
    #from ldm.models.diffusion import LatentDiffusion
    # type(sd_model) == LatentDiffusion

    # forward(self, x, c, *args, **kwargs)
    self, x, c = sd_model, latent, cond

    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()  # [B=1]
    if self.model.conditioning_key is not None:     # 'crossattn'
        assert c is not None
        if self.cond_stage_trainable:               # False
            c = self.get_learned_conditioning(c)
        if self.shorten_cond_schedule:              # False
            tc = self.cond_ids[t].to(self.device)
            c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    
    # p_losses(self, x_start, cond, t, noise=None)
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

# ↑↑↑ the above is modified from 'ldm.models.diffusion/ddpm.py' ↑↑↑


class Script(scripts.Script):

    def title(self):
        return 'Prompt Travel'

    def describe(self):
        return "Gradually travels from one prompt to another in the semantical latent space."

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        mode          = gr.Radio  (label='Travel mode',                         value=lambda: DEFAULT_MODE, choices=CHOICES_MODE)
        steps         = gr.Textbox(label='Travel steps between stages',         value=lambda: DEFAULT_STEPS)
        
        replace_order = gr.Dropdown(label='Replace order (replace mode)',       value=lambda: DEFAULT_REPLACE_ORDER, choices=CHOICES_REPLACE_ORDER)
        grad_alpha    = gr.Number  (label='Step size (grad mode)',              value=lambda: DEFAULT_GRAD_ALPHA)
        grad_iter     = gr.Number  (label='Step count (grad mode)',             value=lambda: DEFAULT_GRAD_ITERS, precision=0)
        grad_meth     = gr.Dropdown(label='Step method (grad mode)',            value=lambda: DEFAULT_GRAD_METH, choices=CHOICES_GRAD_METH)
        grad_w_latent = gr.Number  (label='Loss for latent match (grad mode)',  value=lambda: DEFAULT_GRAD_W_LATENT)
        grad_w_cond   = gr.Number  (label='Loss for cond match (grad mode)',    value=lambda: DEFAULT_GRAD_W_COND)
        
        video_fps     = gr.Number  (label='Video FPS',                          value=lambda: DEFAULT_FPS)
        show_debug    = gr.Checkbox(label='Show verbose debug info at console', value=lambda: DEFAULT_DEBUG)

        return [mode, steps, 
            replace_order,
            grad_alpha, grad_iter, grad_meth, grad_w_latent, grad_w_cond,
            video_fps, show_debug]
    
    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
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

    def run(self, p:StableDiffusionProcessing, mode:str, steps:int, 
            replace_order: str,
            grad_alpha:float, grad_iter:int, grad_meth:str, grad_w_latent:float, grad_w_cond:float,
            video_fps:int, show_debug:bool):
        
        # Prepare prompts
        prompt_pos = p.prompt.strip()
        if not prompt_pos: return Processed(p, [], p.seed, 'positive prompt should not be empty')
        pos_prompts = prompt_pos.split('\n')
        if len(pos_prompts) == 1:
            # NOTE: if only single stage specified, we double it to allow wandering around :)
            if mode == 'grad': pos_prompts = pos_prompts * 2
            else: return Processed(p, [], p.seed, 'should specify at least two lines of prompt to travel between')
        prompt_neg = p.negative_prompt.strip()
        neg_prompts = prompt_neg.split('\n')
        n_stages = max(len(pos_prompts), len(neg_prompts))
        while len(pos_prompts) < n_stages: pos_prompts.append(pos_prompts[-1])
        while len(neg_prompts) < n_stages: neg_prompts.append(neg_prompts[-1])

        try: steps = [int(s.strip()) for s in steps.strip().split(',')]
        except: return Processed(p, [], p.seed, f'cannot parse steps options: {steps}')
    
        if len(steps) == 1:
            steps = [steps[0]] * (n_stages - 1)
        elif len(steps) != n_stages - 1:
            info = (f'stage count mismatch: you have {n_stages} prompt stages, but specified {len(steps)} steps; '
                    'should be len(steps) == len(stages) - 1')
            return Processed(p, [], p.seed, info)
        count = sum(steps) + n_stages
        if show_debug: print(f'n_stages={n_stages}, steps={steps}')
        steps.insert(0, -1)     # fixup the first stage

        # Custom saving path
        travel_path = os.path.join(p.outpath_samples, 'prompt_travel')
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        travel_path = os.path.join(travel_path, f"{travel_number:05}")
        p.outpath_samples = travel_path
        os.makedirs(travel_path, exist_ok=True)
        self.log_fp = os.path.join(travel_path, 'log.txt')

        # Force Batch Count and Batch Size to 1.
        p.n_iter     = 1
        p.batch_size = 1

        # Random unified const seed
        p.seed             = get_fixed_seed(p.seed)
        p.subseed          = p.seed
        p.subseed_strength = 0.0
        if show_debug: print('seed:', p.seed)

        # Start job
        state.job_count = count
        print(f"Generating {count} images.")

        # Implementation dispatcher
        if   mode == 'linear' : images, info = self.run_linear (p, pos_prompts, neg_prompts, steps, show_debug)
        elif mode == 'replace': images, info = self.run_replace(p, pos_prompts, neg_prompts, steps, replace_order, show_debug)
        elif mode == 'grad'   : images, info = self.run_grad   (p, pos_prompts, neg_prompts, steps, grad_alpha, grad_iter, grad_meth, grad_w_latent, grad_w_cond, show_debug)

        # Save video
        if video_fps > 0 and len(images) > 1:
            try:
                clip = ImageSequenceClip([np.asarray(t) for t in images], fps=video_fps)
                clip.write_videofile(os.path.join(travel_path, f"travel-{travel_number:05}.mp4"), verbose=False, audio=False)
            except: pass

        return Processed(p, images, p.seed, info)

    def run_linear(self, p:StableDiffusionProcessing, pos_prompts:List[str], neg_prompts:List[str], steps:List[int], show_debug:bool):
        n_stages = len(steps)
        initial_info = None
        images = []

        def weighted_sum(A, B, alpha, kind):
            ''' linear interpolate on latent space '''
            C = deepcopy(A)
            if kind == 'pos':
                condA = A.batch[0][0].schedules[0].cond
                condB = B.batch[0][0].schedules[0].cond
                condC = (1 - alpha) * condA + alpha * condB
                end_at_step = C.batch[0][0].schedules[0].end_at_step
                C.batch[0][0].schedules[0] = ScheduledPromptConditioning(end_at_step, condC)
            if kind == 'neg':
                condA = A[0][0].cond
                condB = B[0][0].cond
                condC = (1 - alpha) * condA + alpha * condB
                end_at_step = C[0][0].end_at_step
                C[0][0] = ScheduledPromptConditioning(end_at_step, condC)
            return C

        def draw_by_cond(pos_hidden, neg_hidden, prompts, seeds, subseeds):
            nonlocal images, initial_info, p
            proc = process_images_inner_half_B(p, pos_hidden, neg_hidden, prompts, seeds, subseeds)
            if initial_info is None: initial_info = proc.info
            images += proc.images

        # Step 1: draw the init image
        if show_debug:
            print(f'[stage 1/{n_stages}]')
            print(f'  pos prompts: {pos_prompts[0]}')
            print(f'  neg prompts: {neg_prompts[0]}')
        p.prompt           = pos_prompts[0]
        p.negative_prompt  = neg_prompts[0]
        from_pos_hidden, from_neg_hidden, prompts, seeds, subseeds = process_images_inner_half_A(p)
        draw_by_cond(from_pos_hidden, from_neg_hidden, prompts, seeds, subseeds)
        
        # travel through stages
        for i in range(1, n_stages):
            if state.interrupted: break

            # only change target prompts
            if show_debug:
                print(f'[stage {i+1}/{n_stages}]')
                print(f'  pos prompts: {pos_prompts[i]}')
                print(f'  neg prompts: {neg_prompts[i]}')
            p.prompt           = pos_prompts[i]
            p.negative_prompt  = neg_prompts[i]
            to_pos_hidden, to_neg_hidden, prompts, seeds, subseeds = process_images_inner_half_A(p)

            # Step 2: draw the interpolated images
            n_inter = steps[i] + 1
            for t in range(1, n_inter):
                if state.interrupted: break

                alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]
                inter_pos_hidden = weighted_sum(from_pos_hidden, to_pos_hidden, alpha, kind='pos')
                inter_neg_hidden = weighted_sum(from_neg_hidden, to_neg_hidden, alpha, kind='neg')
                draw_by_cond(inter_pos_hidden, inter_neg_hidden, prompts, seeds, subseeds)

            # Step 3: draw the fianl stage
            draw_by_cond(to_pos_hidden, to_neg_hidden, prompts, seeds, subseeds)
            
            # move to next stage
            from_pos_hidden, from_neg_hidden = to_pos_hidden, to_neg_hidden

        return images, initial_info

    def run_replace(self, p:StableDiffusionProcessing, pos_prompts:List[str], neg_prompts:List[str], steps:List[int], replace_order:str, show_debug:bool):
        n_stages = len(steps)
        initial_info = None
        images = []

        initial_info = '你先别急，这个还没有实现……'

        return images, initial_info

    def run_grad(self, p:StableDiffusionProcessing, pos_prompts:List[str], neg_prompts:List[str], steps:List[int], grad_alpha:float, grad_iter:int, grad_meth:str, grad_w_latent:float, grad_w_cond:float, show_debug:bool):
        n_stages = len(steps)
        initial_info = None
        images = []

        def image_to_latent(img: Image) -> torch.Tensor:
            nonlocal p
            model = p.sd_model

            im = np.array(img).astype(np.uint8)
            im = (im / 127.5 - 1.0).astype(np.float32)
            x = torch.from_numpy(im)
            x = torch.moveaxis(x, 2, 0)
            x = x.unsqueeze(dim=0)          # [B=1, C=3, H=512, W=512]
            x = x.to(model.device)
            
            latent = model.get_first_stage_encoding(model.encode_first_stage(x))    # [B=1, C=4, H=64, W=64]
            return latent

        def mlc_get_cond(c:MulticondLearnedConditioning) -> torch.Tensor:
            return c.batch[0][0].schedules[0].cond      # [B=1, T=77, D=768]

        def mlc_replace_cond(c:MulticondLearnedConditioning, cond: torch.Tensor) -> MulticondLearnedConditioning:
            r = deepcopy(c)
            spc = r.batch[0][0].schedules[0]
            r.batch[0][0].schedules[0] = ScheduledPromptConditioning(spc.end_at_step, cond)
            return r

        # Step 1: draw init image
        if show_debug: print(f'[stage 1/{n_stages}] prompts: {pos_prompts[0]}')
        p.prompt = pos_prompts[0]
        c, uc, prompts, seeds, subseeds = process_images_inner_half_A(p)
        proc = process_images_inner_half_B(p, c, uc, prompts, seeds, subseeds)
        if initial_info is None: initial_info = proc.info
        images += proc.images

        # make log
        log_fh = open(self.log_fp, 'w', encoding='utf-8')
        log_fh.write(f'grad_alpha   = {grad_alpha}\n')
        log_fh.write(f'grad_iter    = {grad_iter}\n')
        log_fh.write(f'grad_meth    = {grad_meth}\n')
        log_fh.write(f'grad_w_grad  = {grad_w_latent}\n')
        log_fh.write(f'grad_w_match = {grad_w_cond}\n')
        log_fh.write('\n')

        # travel between stages
        for i in range(1, n_stages):
            if state.interrupted: break

            # Step 2: draw the stage target
            if show_debug: print(f'[stage {i+1}/{n_stages}] prompts: {pos_prompts[i]}')
            p.prompt = pos_prompts[i]
            params = process_images_inner_half_A(p)
            proc = process_images_inner_half_B(p, *params)
            if initial_info is None: initial_info = proc.info
            target_image = proc.images[0]     # cache it here to make video sequence order right
            
            with torch.no_grad(), devices.autocast():
                target_latent = image_to_latent(target_image)           # [B=1, C=4, H=64, W=64]
                target_cond   = mlc_get_cond(params[0]).unsqueeze(0)    # [B=1, T=77, D=768]
                source_cond   = mlc_get_cond(c).unsqueeze(0)
                L1_dist       = F.l1_loss(source_cond, target_cond).item()
            
            # Step 3: draw the inter-mediums
            for _ in range(steps[i]):
                if state.interrupted: break

                with devices.autocast():
                    current_cond = mlc_get_cond(c).unsqueeze(0).clone()  # [B=1, T=77, D=768]

                    # simple PGD attack on cond (src prompt) to match latent (dst image)
                    for _ in range(grad_iter):
                        if state.interrupted: break

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

                        if show_debug:
                            with torch.no_grad():
                                l_latent = loss_latent.mean().item()
                                l_match  = loss_cond .mean().item()
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

                # move to new 'c' (one travel step!)
                # FIXME: we do not walk on 'uc' so far
                c = mlc_replace_cond(c, current_cond.detach().squeeze(0))

                proc = process_images_inner_half_B(p, c, uc, prompts, seeds, subseeds)
                if initial_info is None: initial_info = proc.info
                images += proc.images

            # append the finishing image for current stage 
            images += [target_image]

            # shift: last stage's final info becomes new stage's init info
            c, uc, prompts, seeds, subseeds = params

        # save log
        log_fh.close()

        return images, initial_info
