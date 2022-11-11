import os
import random
from copy import deepcopy

import gradio as gr
import numpy as np
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    print(f"moviepy python module not installed. Will not be able to generate video.")

import modules.scripts as scripts
from modules.processing import Processed, StableDiffusionProcessing
from modules.processing import *
from modules.prompt_parser import ScheduledPromptConditioning
from modules.shared import state

DEFAULT_STEPS          = 10
DEFAULT_SAVE           = True
DEFAULT_FPS            = 10
DEFAULT_DEBUG          = True


# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓
import torch
import numpy as np
from PIL import Image
import random

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


class Script(scripts.Script):

    def title(self):
        return 'Prompt Travel'

    def describe(self):
        return "Gradually travels from one prompt to another in the semantical latent space."

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        steps       = gr.Textbox(label='Steps between prompts', value=lambda: DEFAULT_STEPS, precision=0)
        
        video_save  = gr.Checkbox(label='Save results as video', value=lambda: DEFAULT_SAVE)
        video_fps   = gr.Number(label='Frames per second', value=lambda: DEFAULT_FPS)

        show_debug  = gr.Checkbox(label='Show verbose debug info at console', value=lambda: DEFAULT_DEBUG)

        return [steps, video_save, video_fps, show_debug]
    
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

    def run(self, p:StableDiffusionProcessing, steps:str, video_save:bool, video_fps:int, show_debug:bool):
        initial_info = None
        images = []
        
        prompt_pos = p.prompt         .strip()
        prompt_neg = p.negative_prompt.strip()
        if not prompt_pos:
            print('positive prompt should not be empty')
            return Processed(p, images, p.seed)

        # prepare prompts
        pos_prompts = prompt_pos.split('\n')
        neg_prompts = prompt_neg.split('\n')
        n_stages = max(len(pos_prompts), len(neg_prompts))
        while len(pos_prompts) < n_stages: pos_prompts.append(pos_prompts[-1])
        while len(neg_prompts) < n_stages: neg_prompts.append(neg_prompts[-1])

        steps = steps.strip()
        try:
            steps = [int(s.strip()) for s in steps.split(',')]
        except:
            print(f'cannot parse steps options: {steps}')
            return Processed(p, images, p.seed)
    
        if len(steps) == 1:
            steps = [steps[0]] * (n_stages - 1)
        elif len(steps) != n_stages - 1:
            print(f'stage count mismatch: you have {n_stages} prompt stages, but specified {len(steps)} steps; should be len(steps) = len(stages) - 1')
            return Processed(p, images, p.seed)
        count = sum(steps) + n_stages
        print(f'n_stages={n_stages}, steps={steps}')
        steps.insert(0, -1)     # fixup the first stage

        # Custom seed travel saving
        travel_path = os.path.join(p.outpath_samples, 'prompt_travel')
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        travel_path = os.path.join(travel_path, f"{travel_number:05}")
        p.outpath_samples = travel_path
        os.makedirs(travel_path, exist_ok=True)

        # Force Batch Count and Batch Size to 1.
        p.n_iter     = 1
        p.batch_size = 1

        # Random unified const seed
        p.seed             = get_fixed_seed(p.seed)
        p.subseed          = p.seed
        p.subseed_strength = 0.0
        print('seed:', p.seed)

        # Start job
        state.job_count = count
        print(f"Generating {count} images.")

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

            if show_debug:
                print(f'[stage {i+1}/{n_stages}]')
                print(f'  pos prompts: {pos_prompts[i]}')
                print(f'  neg prompts: {neg_prompts[i]}')

            # only change target prompts
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

        if video_save and len(images) > 1:
            try:
                clip = ImageSequenceClip([np.asarray(t) for t in images], fps=video_fps)
                clip.write_videofile(os.path.join(travel_path, f"travel-{travel_number:05}.mp4"), verbose=False, audio=False, logger=None)
            except: pass

        return Processed(p, images, p.seed, initial_info)
