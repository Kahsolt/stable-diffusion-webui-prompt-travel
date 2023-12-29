# This extension works with [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
# version: v1.1.424

LOG_PREFIX = '[ControlNet-Travel]'

# ↓↓↓ EXIT EARLY IF EXTERNAL REPOSITORY NOT FOUND ↓↓↓

CTRLNET_REPO_NAME = 'Mikubill/sd-webui-controlnet'
if 'externel repo sanity check':
    from pathlib import Path
    from modules.scripts import basedir
    from traceback import print_exc

    ME_PATH = Path(basedir())
    CTRLNET_PATH = ME_PATH.parent / 'sd-webui-controlnet'

    controlnet_found = False
    try:
        import sys ; sys.path.append(str(CTRLNET_PATH))
        #from scripts.controlnet import Script as ControlNetScript  # NOTE: this will mess up the import order
        from scripts.external_code import ControlNetUnit
        from scripts.hook import UNetModel, UnetHook, ControlParams
        from scripts.hook import *

        controlnet_found = True
        print(f'{LOG_PREFIX} extension {CTRLNET_REPO_NAME} found, ControlNet-Travel loaded :)')
    except ImportError:
        print(f'{LOG_PREFIX} extension {CTRLNET_REPO_NAME} not found, ControlNet-Travel ignored :(')
    except:
        print_exc()

# ↑↑↑ EXIT EARLY IF EXTERNAL REPOSITORY NOT FOUND ↑↑↑

TOOL_PATH = ME_PATH / 'tools'
paths_ext = []
paths_ext.append(str(TOOL_PATH))
paths_ext.append(str(TOOL_PATH / 'rife-ncnn-vulkan'))
import os
os.environ['PATH'] += os.path.pathsep + os.path.pathsep.join(paths_ext)

import sys
from subprocess import Popen
from PIL import Image

from ldm.models.diffusion.ddpm import LatentDiffusion
from modules import shared, devices, lowvram
from modules.processing import StableDiffusionProcessing as Processing
from modules.script_callbacks import ImageSaveParams, on_before_image_saved

from scripts.prompt_travel import *


class InterpMethod(Enum):
    LINEAR = 'linear (weight sum)'
    RIFE   = 'rife (optical flow)'

if 'consts':
    __ = lambda key, value=None: opts.data.get(f'customscript/controlnet_travel.py/txt2img/{key}/value', value)


    LABEL_CTRLNET_REF_DIR   = 'Reference image folder (one ref image per stage :)'
    LABEL_INTERP_METH       = 'Interpolate method'
    LABEL_SKIP_FUSE         = 'Ext. skip latent fusion'
    LABEL_DEBUG_RIFE        = 'Save RIFE intermediates'

    DEFAULT_STEPS           = 10
    DEFAULT_CTRLNET_REF_DIR = str(ME_PATH / 'img' / 'ref_ctrlnet')
    DEFAULT_INTERP_METH     = __(LABEL_INTERP_METH, InterpMethod.LINEAR.value)
    DEFAULT_SKIP_FUSE       = __(LABEL_SKIP_FUSE, False)
    DEFAULT_DEBUG_RIFE      = __(LABEL_DEBUG_RIFE, False)

    CHOICES_INTERP_METH     = [x.value for x in InterpMethod]

if 'vars':
    skip_fuse_plan:       List[bool]         = []   # n_blocks (13)

    interp_alpha:         float              = 0.0
    interp_ip:            int                = 0    # 0 ~ n_sampling_step-1
    from_hint_cond:       List[Tensor]       = []   # n_contrlnet_set
    to_hint_cond:         List[Tensor]       = []
    mid_hint_cond:        List[Tensor]       = []
    from_control_tensors: List[List[Tensor]] = []   # n_sampling_step x n_blocks
    to_control_tensors:   List[List[Tensor]] = []

    caches: List[list] = [from_hint_cond, to_hint_cond, mid_hint_cond, from_control_tensors, to_control_tensors]


def run_cmd(cmd:str) -> bool:
  try:
    print(f'[exec] {cmd}')
    Popen(cmd, shell=True, encoding='utf-8').wait()
    return True
  except:
    return False


# ↓↓↓ the following is modified from 'sd-webui-controlnet/scripts/hook.py' ↓↓↓

def hook_hijack(self:UnetHook, model:UNetModel, sd_ldm:LatentDiffusion, control_params:List[ControlParams], process:Processing, batch_option_uint_separate=False, batch_option_style_align=False):
    self.model = model
    self.sd_ldm = sd_ldm
    self.control_params = control_params

    model_is_sdxl = getattr(self.sd_ldm, 'is_sdxl', False)

    outer = self

    def process_sample(*args, **kwargs):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        mark_prompt_context(kwargs.get('conditioning', []), positive=True)
        mark_prompt_context(kwargs.get('unconditional_conditioning', []), positive=False)
        mark_prompt_context(getattr(process, 'hr_c', []), positive=True)
        mark_prompt_context(getattr(process, 'hr_uc', []), positive=False)
        return process.sample_before_CN_hack(*args, **kwargs)

    # NOTE: ↓↓↓ only hack this method ↓↓↓
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        is_sdxl = y is not None and model_is_sdxl
        total_t2i_adapter_embedding = [0.0] * 4
        if is_sdxl:
            total_controlnet_embedding = [0.0] * 10
        else:
            total_controlnet_embedding = [0.0] * 13
        require_inpaint_hijack = False
        is_in_high_res_fix = False
        batch_size = int(x.shape[0])

        # NOTE: declare globals
        global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, mid_hint_cond, interp_alpha, interp_ip
        x: Tensor           # [1, 4, 64, 64]
        timesteps: Tensor   # [1]
        context: Tensor     # [1, 78, 768]
        kwargs: dict        # {}

        # Handle cond-uncond marker
        cond_mark, outer.current_uc_indices, outer.current_c_indices, context = unmark_prompt_context(context)
        outer.model.cond_mark = cond_mark
        # logger.info(str(cond_mark[:, 0, 0, 0].detach().cpu().numpy().tolist()) + ' - ' + str(outer.current_uc_indices))

        # Revision
        if is_sdxl:
            revision_y1280 = 0

            for param in outer.control_params:
                if param.guidance_stopped:
                    continue
                if param.control_model_type == ControlModelType.ReVision:
                    if param.vision_hint_count is None:
                        k = torch.Tensor([int(param.preprocessor['threshold_a'] * 1000)]).to(param.hint_cond).long().clip(0, 999)
                        param.vision_hint_count = outer.revision_q_sampler.q_sample(param.hint_cond, k)
                    revision_emb = param.vision_hint_count
                    if isinstance(revision_emb, torch.Tensor):
                        revision_y1280 += revision_emb * param.weight

            if isinstance(revision_y1280, torch.Tensor):
                y[:, :1280] = revision_y1280 * cond_mark[:, :, 0, 0]
                if any('ignore_prompt' in param.preprocessor['name'] for param in outer.control_params) \
                        or (getattr(process, 'prompt', '') == '' and getattr(process, 'negative_prompt', '') == ''):
                    context = torch.zeros_like(context)

        # High-res fix
        for param in outer.control_params:
            # select which hint_cond to use
            if param.used_hint_cond is None:
                param.used_hint_cond = param.hint_cond
                param.used_hint_cond_latent = None
                param.used_hint_inpaint_hijack = None

            # has high-res fix
            if isinstance(param.hr_hint_cond, torch.Tensor) and x.ndim == 4 and param.hint_cond.ndim == 4 and param.hr_hint_cond.ndim == 4:
                _, _, h_lr, w_lr = param.hint_cond.shape
                _, _, h_hr, w_hr = param.hr_hint_cond.shape
                _, _, h, w = x.shape
                h, w = h * 8, w * 8
                if abs(h - h_lr) < abs(h - h_hr):
                    is_in_high_res_fix = False
                    if param.used_hint_cond is not param.hint_cond:
                        param.used_hint_cond = param.hint_cond
                        param.used_hint_cond_latent = None
                        param.used_hint_inpaint_hijack = None
                else:
                    is_in_high_res_fix = True
                    if param.used_hint_cond is not param.hr_hint_cond:
                        param.used_hint_cond = param.hr_hint_cond
                        param.used_hint_cond_latent = None
                        param.used_hint_inpaint_hijack = None

        self.is_in_high_res_fix = is_in_high_res_fix
        outer.is_in_high_res_fix = is_in_high_res_fix
        no_high_res_control = is_in_high_res_fix and shared.opts.data.get("control_net_no_high_res_fix", False)

        # NOTE: hint shallow fusion, overwrite param.used_hint_cond
        for i, param in enumerate(outer.control_params):
            if interp_alpha == 0.0:     # collect hind_cond on key frames
                if len(to_hint_cond) < len(outer.control_params):
                    to_hint_cond.append(param.used_hint_cond.clone().detach().cpu())
            else:                       # interp with cached hind_cond
                param.used_hint_cond = mid_hint_cond[i].to(x.device)

        # Convert control image to latent
        for param in outer.control_params:
            if param.used_hint_cond_latent is not None:
                continue
            if param.control_model_type not in [ControlModelType.AttentionInjection] \
                    and 'colorfix' not in param.preprocessor['name'] \
                    and 'inpaint_only' not in param.preprocessor['name']:
                continue
            param.used_hint_cond_latent = outer.call_vae_using_process(process, param.used_hint_cond, batch_size=batch_size)

        # vram
        for param in outer.control_params:
            if getattr(param.control_model, 'disable_memory_management', False):
                continue

            if param.control_model is not None:
                if outer.lowvram and is_sdxl and hasattr(param.control_model, 'aggressive_lowvram'):
                    param.control_model.aggressive_lowvram()
                elif hasattr(param.control_model, 'fullvram'):
                    param.control_model.fullvram()
                elif hasattr(param.control_model, 'to'):
                    param.control_model.to(devices.get_device_for("controlnet"))

        # handle prompt token control
        for param in outer.control_params:
            if no_high_res_control:
                continue

            if param.guidance_stopped:
                continue

            if param.control_model_type not in [ControlModelType.T2I_StyleAdapter]:
                continue

            control = param.control_model(x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context)
            control = torch.cat([control.clone() for _ in range(batch_size)], dim=0)
            control *= param.weight
            control *= cond_mark[:, :, :, 0]
            context = torch.cat([context, control.clone()], dim=1)

        # handle ControlNet / T2I_Adapter
        for param_index, param in enumerate(outer.control_params):
            if no_high_res_control:
                continue

            if param.guidance_stopped:
                continue

            if param.control_model_type not in [ControlModelType.ControlNet, ControlModelType.T2I_Adapter]:
                continue

            # inpaint model workaround
            x_in = x
            control_model = param.control_model.control_model

            if param.control_model_type == ControlModelType.ControlNet:
                if x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9:
                    # inpaint_model: 4 data + 4 downscaled image + 1 mask
                    x_in = x[:, :4, ...]
                    require_inpaint_hijack = True

            assert param.used_hint_cond is not None, f"Controlnet is enabled but no input image is given"

            hint = param.used_hint_cond

            # ControlNet inpaint protocol
            if hint.shape[1] == 4:
                c = hint[:, 0:3, :, :]
                m = hint[:, 3:4, :, :]
                m = (m > 0.5).float()
                hint = c * (1 - m) - m

            control = param.control_model(x=x_in, hint=hint, timesteps=timesteps, context=context, y=y)

            if is_sdxl:
                control_scales = [param.weight] * 10
            else:
                control_scales = [param.weight] * 13

            if param.cfg_injection or param.global_average_pooling:
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    control = [torch.cat([c.clone() for _ in range(batch_size)], dim=0) for c in control]
                control = [c * cond_mark for c in control]

            high_res_fix_forced_soft_injection = False

            if is_in_high_res_fix:
                if 'canny' in param.preprocessor['name']:
                    high_res_fix_forced_soft_injection = True
                if 'mlsd' in param.preprocessor['name']:
                    high_res_fix_forced_soft_injection = True

            # if high_res_fix_forced_soft_injection:
            #     logger.info('[ControlNet] Forced soft_injection in high_res_fix in enabled.')

            if param.soft_injection or high_res_fix_forced_soft_injection:
                # important! use the soft weights with high-res fix can significantly reduce artifacts.
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    control_scales = [param.weight * x for x in (0.25, 0.62, 0.825, 1.0)]
                elif param.control_model_type == ControlModelType.ControlNet:
                    control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]

            if is_sdxl and param.control_model_type == ControlModelType.ControlNet:
                control_scales = control_scales[:10]

            if param.advanced_weighting is not None:
                control_scales = param.advanced_weighting

            control = [c * scale for c, scale in zip(control, control_scales)]
            if param.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

            for idx, item in enumerate(control):
                target = None
                if param.control_model_type == ControlModelType.ControlNet:
                    target = total_controlnet_embedding
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    target = total_t2i_adapter_embedding
                if target is not None:
                    if batch_option_uint_separate:
                        for pi, ci in enumerate(outer.current_c_indices):
                            if pi % len(outer.control_params) != param_index:
                                item[ci] = 0
                        for pi, ci in enumerate(outer.current_uc_indices):
                            if pi % len(outer.control_params) != param_index:
                                item[ci] = 0
                        target[idx] = item + target[idx]
                    else:
                        target[idx] = item + target[idx]

        # Replace x_t to support inpaint models
        for param in outer.control_params:
            if not isinstance(param.used_hint_cond, torch.Tensor):
                continue
            if param.used_hint_cond.shape[1] != 4:
                continue
            if x.shape[1] != 9:
                continue
            if param.used_hint_inpaint_hijack is None:
                mask_pixel = param.used_hint_cond[:, 3:4, :, :]
                image_pixel = param.used_hint_cond[:, 0:3, :, :]
                mask_pixel = (mask_pixel > 0.5).to(mask_pixel.dtype)
                masked_latent = outer.call_vae_using_process(process, image_pixel, batch_size, mask=mask_pixel)
                mask_latent = torch.nn.functional.max_pool2d(mask_pixel, (8, 8))
                if mask_latent.shape[0] != batch_size:
                    mask_latent = torch.cat([mask_latent.clone() for _ in range(batch_size)], dim=0)
                param.used_hint_inpaint_hijack = torch.cat([mask_latent, masked_latent], dim=1)
                param.used_hint_inpaint_hijack.to(x.dtype).to(x.device)
            x = torch.cat([x[:, :4, :, :], param.used_hint_inpaint_hijack], dim=1)

        # vram
        for param in outer.control_params:
            if param.control_model is not None:
                if outer.lowvram:
                    param.control_model.to('cpu')

        # A1111 fix for medvram.
        if shared.cmd_opts.medvram or (getattr(shared.cmd_opts, 'medvram_sdxl', False) and is_sdxl):
            try:
                # Trigger the register_forward_pre_hook
                outer.sd_ldm.model()
            except:
                pass

        # Clear attention and AdaIn cache
        for module in outer.attn_module_list:
            module.bank = []
            module.style_cfgs = []
        for module in outer.gn_module_list:
            module.mean_bank = []
            module.var_bank = []
            module.style_cfgs = []

        # Handle attention and AdaIn control
        for param in outer.control_params:
            if no_high_res_control:
                continue

            if param.guidance_stopped:
                continue

            if param.used_hint_cond_latent is None:
                continue

            if param.control_model_type not in [ControlModelType.AttentionInjection]:
                continue

            ref_xt = predict_q_sample(outer.sd_ldm, param.used_hint_cond_latent, torch.round(timesteps.float()).long())

            # Inpaint Hijack
            if x.shape[1] == 9:
                ref_xt = torch.cat([
                    ref_xt,
                    torch.zeros_like(ref_xt)[:, 0:1, :, :],
                    param.used_hint_cond_latent
                ], dim=1)

            outer.current_style_fidelity = float(param.preprocessor['threshold_a'])
            outer.current_style_fidelity = max(0.0, min(1.0, outer.current_style_fidelity))

            if is_sdxl:
                # sdxl's attention hacking is highly unstable.
                # We have no other methods but to reduce the style_fidelity a bit.
                # By default, 0.5 ** 3.0 = 0.125
                outer.current_style_fidelity = outer.current_style_fidelity ** 3.0

            if param.cfg_injection:
                outer.current_style_fidelity = 1.0
            elif param.soft_injection or is_in_high_res_fix:
                outer.current_style_fidelity = 0.0

            control_name = param.preprocessor['name']

            if control_name in ['reference_only', 'reference_adain+attn']:
                outer.attention_auto_machine = AutoMachine.Write
                outer.attention_auto_machine_weight = param.weight

            if control_name in ['reference_adain', 'reference_adain+attn']:
                outer.gn_auto_machine = AutoMachine.Write
                outer.gn_auto_machine_weight = param.weight

            if is_sdxl:
                outer.original_forward(
                    x=ref_xt.to(devices.dtype_unet),
                    timesteps=timesteps.to(devices.dtype_unet),
                    context=context.to(devices.dtype_unet),
                    y=y
                )
            else:
                outer.original_forward(
                    x=ref_xt.to(devices.dtype_unet),
                    timesteps=timesteps.to(devices.dtype_unet),
                    context=context.to(devices.dtype_unet)
                )

            outer.attention_auto_machine = AutoMachine.Read
            outer.gn_auto_machine = AutoMachine.Read

        # NOTE: hint latent fusion, overwrite control tensors
        total_control = total_controlnet_embedding
        if interp_alpha == 0.0:     # collect control tensors on key frames
            tensors: List[Tensor] = []
            for i, t in enumerate(total_control):
                if len(skip_fuse_plan) and skip_fuse_plan[i]:
                    tensors.append(None)
                else:
                    tensors.append(t.clone().detach().cpu())
            to_control_tensors.append(tensors)
        else:                       # interp with cached control tensors
            device = total_control[0].device
            for i, (ctrlA, ctrlB) in enumerate(zip(from_control_tensors[interp_ip], to_control_tensors[interp_ip])):
                if ctrlA is not None and ctrlB is not None:
                    ctrlC = weighted_sum(ctrlA.to(device), ctrlB.to(device), interp_alpha)
                    #print('  ctrl diff:', (ctrlC - total_control[i]).abs().mean().item())
                    total_control[i].data = ctrlC
            interp_ip += 1
        
        # NOTE: warn on T2I adapter
        if total_t2i_adapter_embedding[0] != 0:
            print(f'{LOG_PREFIX} warn: currently t2i_adapter is not supported. if you wanna this, put a feature request on Kahsolt/stable-diffusion-webui-prompt-travel')

        # U-Net Encoder
        hs = []
        with th.no_grad():
            t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
            emb = self.time_embed(t_emb)

            if is_sdxl:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x
            for i, module in enumerate(self.input_blocks):
                self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                h = module(h, emb, context)

                t2i_injection = [3, 5, 8] if is_sdxl else [2, 5, 8, 11]

                if i in t2i_injection:
                    h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)

                hs.append(h)

            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = self.middle_block(h, emb, context)

        # U-Net Middle Block
        h = aligned_adding(h, total_controlnet_embedding.pop(), require_inpaint_hijack)

        if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
            h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)

        # U-Net Decoder
        for i, module in enumerate(self.output_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = th.cat([h, aligned_adding(hs.pop(), total_controlnet_embedding.pop(), require_inpaint_hijack)], dim=1)
            h = module(h, emb, context)

        # U-Net Output
        h = h.type(x.dtype)
        h = self.out(h)

        # Post-processing for color fix
        for param in outer.control_params:
            if param.used_hint_cond_latent is None:
                continue
            if 'colorfix' not in param.preprocessor['name']:
                continue

            k = int(param.preprocessor['threshold_a'])
            if is_in_high_res_fix and not no_high_res_control:
                k *= 2

            # Inpaint hijack
            xt = x[:, :4, :, :]

            x0_origin = param.used_hint_cond_latent
            t = torch.round(timesteps.float()).long()
            x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
            x0 = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

            if '+sharp' in param.preprocessor['name']:
                detail_weight = float(param.preprocessor['threshold_b']) * 0.01
                neg = detail_weight * blur(x0, k) + (1 - detail_weight) * x0
                x0 = cond_mark * x0 + (1 - cond_mark) * neg

            eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

            w = max(0.0, min(1.0, float(param.weight)))
            h = eps_prd * w + h * (1 - w)

        # Post-processing for restore
        for param in outer.control_params:
            if param.used_hint_cond_latent is None:
                continue
            if 'inpaint_only' not in param.preprocessor['name']:
                continue
            if param.used_hint_cond.shape[1] != 4:
                continue

            # Inpaint hijack
            xt = x[:, :4, :, :]

            mask = param.used_hint_cond[:, 3:4, :, :]
            mask = torch.nn.functional.max_pool2d(mask, (10, 10), stride=(8, 8), padding=1)

            x0_origin = param.used_hint_cond_latent
            t = torch.round(timesteps.float()).long()
            x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
            x0 = x0_prd * mask + x0_origin * (1 - mask)
            eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

            w = max(0.0, min(1.0, float(param.weight)))
            h = eps_prd * w + h * (1 - w)

        return h

    def move_all_control_model_to_cpu():
        for param in getattr(outer, 'control_params', []) or []:
            if isinstance(param.control_model, torch.nn.Module):
                param.control_model.to("cpu")

    def forward_webui(*args, **kwargs):
        # webui will handle other compoments 
        try:
            if shared.cmd_opts.lowvram:
                lowvram.send_everything_to_cpu()
            return forward(*args, **kwargs)
        except Exception as e:
            move_all_control_model_to_cpu()
            raise e
        finally:
            if outer.lowvram:
                move_all_control_model_to_cpu()

    def hacked_basic_transformer_inner_forward(self, x, context=None):
        x_norm1 = self.norm1(x)
        self_attn1 = None
        if self.disable_self_attn:
            # Do not use self-attention
            self_attn1 = self.attn1(x_norm1, context=context)
        else:
            # Use self-attention
            self_attention_context = x_norm1
            if outer.attention_auto_machine == AutoMachine.Write:
                if outer.attention_auto_machine_weight > self.attn_weight:
                    self.bank.append(self_attention_context.detach().clone())
                    self.style_cfgs.append(outer.current_style_fidelity)
            if outer.attention_auto_machine == AutoMachine.Read:
                if len(self.bank) > 0:
                    style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                    self_attn1_uc = self.attn1(x_norm1, context=torch.cat([self_attention_context] + self.bank, dim=1))
                    self_attn1_c = self_attn1_uc.clone()
                    if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                        self_attn1_c[outer.current_uc_indices] = self.attn1(
                            x_norm1[outer.current_uc_indices],
                            context=self_attention_context[outer.current_uc_indices])
                    self_attn1 = style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                self.bank = []
                self.style_cfgs = []
            if outer.attention_auto_machine == AutoMachine.StyleAlign and not outer.is_in_high_res_fix:
                # very VRAM hungry - disable at high_res_fix

                def shared_attn1(inner_x):
                    BB, FF, CC = inner_x.shape
                    return self.attn1(inner_x.reshape(1, BB * FF, CC)).reshape(BB, FF, CC)

                uc_layer = shared_attn1(x_norm1[outer.current_uc_indices])
                c_layer = shared_attn1(x_norm1[outer.current_c_indices])
                self_attn1 = torch.zeros_like(x_norm1).to(uc_layer)
                self_attn1[outer.current_uc_indices] = uc_layer
                self_attn1[outer.current_c_indices] = c_layer
                del uc_layer, c_layer
            if self_attn1 is None:
                self_attn1 = self.attn1(x_norm1, context=self_attention_context)

        x = self_attn1.to(x.dtype) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

    def hacked_group_norm_forward(self, *args, **kwargs):
        eps = 1e-6
        x = self.original_forward_cn_hijack(*args, **kwargs)
        y = None
        if outer.gn_auto_machine == AutoMachine.Write:
            if outer.gn_auto_machine_weight > self.gn_weight:
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                self.mean_bank.append(mean)
                self.var_bank.append(var)
                self.style_cfgs.append(outer.current_style_fidelity)
        if outer.gn_auto_machine == AutoMachine.Read:
            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                var_acc = sum(self.var_bank) / float(len(self.var_bank))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                y_uc = (((x - mean) / std) * std_acc) + mean_acc
                y_c = y_uc.clone()
                if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                    y_c[outer.current_uc_indices] = x.to(y_c.dtype)[outer.current_uc_indices]
                y = style_cfg * y_c + (1.0 - style_cfg) * y_uc
            self.mean_bank = []
            self.var_bank = []
            self.style_cfgs = []
        if y is None:
            y = x
        return y.to(x.dtype)

    if getattr(process, 'sample_before_CN_hack', None) is None:
        process.sample_before_CN_hack = process.sample
    process.sample = process_sample

    model._original_forward = model.forward
    outer.original_forward = model.forward
    model.forward = forward_webui.__get__(model, UNetModel)

    if model_is_sdxl:
        register_schedule(sd_ldm)
        outer.revision_q_sampler = AbstractLowScaleModel()

    need_attention_hijack = False

    for param in outer.control_params:
        if param.control_model_type in [ControlModelType.AttentionInjection]:
            need_attention_hijack = True

    if batch_option_style_align:
        need_attention_hijack = True
        outer.attention_auto_machine = AutoMachine.StyleAlign
        outer.gn_auto_machine = AutoMachine.StyleAlign

    all_modules = torch_dfs(model)

    if need_attention_hijack:
        attn_modules = [module for module in all_modules if isinstance(module, BasicTransformerBlock) or isinstance(module, BasicTransformerBlockSGM)]
        attn_modules = sorted(attn_modules, key=lambda x: - x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            if getattr(module, '_original_inner_forward_cn_hijack', None) is None:
                module._original_inner_forward_cn_hijack = module._forward
            module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = []
            module.style_cfgs = []
            module.attn_weight = float(i) / float(len(attn_modules))

        gn_modules = [model.middle_block]
        model.middle_block.gn_weight = 0

        if model_is_sdxl:
            input_block_indices = [4, 5, 7, 8]
            output_block_indices = [0, 1, 2, 3, 4, 5]
        else:
            input_block_indices = [4, 5, 7, 8, 10, 11]
            output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]

        for w, i in enumerate(input_block_indices):
            module = model.input_blocks[i]
            module.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
            gn_modules.append(module)

        for w, i in enumerate(output_block_indices):
            module = model.output_blocks[i]
            module.gn_weight = float(w) / float(len(output_block_indices))
            gn_modules.append(module)

        for i, module in enumerate(gn_modules):
            if getattr(module, 'original_forward_cn_hijack', None) is None:
                module.original_forward_cn_hijack = module.forward
            module.forward = hacked_group_norm_forward.__get__(module, torch.nn.Module)
            module.mean_bank = []
            module.var_bank = []
            module.style_cfgs = []
            module.gn_weight *= 2

        outer.attn_module_list = attn_modules
        outer.gn_module_list = gn_modules
    else:
        for module in all_modules:
            _original_inner_forward_cn_hijack = getattr(module, '_original_inner_forward_cn_hijack', None)
            original_forward_cn_hijack = getattr(module, 'original_forward_cn_hijack', None)
            if _original_inner_forward_cn_hijack is not None:
                module._forward = _original_inner_forward_cn_hijack
            if original_forward_cn_hijack is not None:
                module.forward = original_forward_cn_hijack
        outer.attn_module_list = []
        outer.gn_module_list = []

    scripts.script_callbacks.on_cfg_denoiser(self.guidance_schedule_handler)

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
        return 'Travel from one controlnet hint condition to another in the tensor space.'

    def show(self, is_img2img):
        return controlnet_found

    def ui(self, is_img2img):
        with gr.Row(variant='compact'):
            interp_meth = gr.Dropdown(label=LABEL_INTERP_METH, value=lambda: DEFAULT_INTERP_METH, choices=CHOICES_INTERP_METH)
            steps       = gr.Text    (label=LABEL_STEPS,       value=lambda: DEFAULT_STEPS,       max_lines=1)
            
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

        with gr.Row(variant='compact', visible=DEFAULT_VIDEO) as tab_ext_video:
            video_fmt  = gr.Dropdown(label=LABEL_VIDEO_FMT,  value=lambda: DEFAULT_VIDEO_FMT, choices=CHOICES_VIDEO_FMT)
            video_fps  = gr.Number  (label=LABEL_VIDEO_FPS,  value=lambda: DEFAULT_VIDEO_FPS)
            video_pad  = gr.Number  (label=LABEL_VIDEO_PAD,  value=lambda: DEFAULT_VIDEO_PAD,  precision=0)
            video_pick = gr.Text    (label=LABEL_VIDEO_PICK, value=lambda: DEFAULT_VIDEO_PICK, max_lines=1)

        with gr.Row(variant='compact') as tab_ext:
            ext_video     = gr.Checkbox(label=LABEL_VIDEO,      value=lambda: DEFAULT_VIDEO)
            ext_skip_fuse = gr.Checkbox(label=LABEL_SKIP_FUSE,  value=lambda: DEFAULT_SKIP_FUSE)
            dbg_rife      = gr.Checkbox(label=LABEL_DEBUG_RIFE, value=lambda: DEFAULT_DEBUG_RIFE)

            ext_video    .change(gr_show, inputs=ext_video,     outputs=tab_ext_video,     show_progress=False)
            ext_skip_fuse.change(gr_show, inputs=ext_skip_fuse, outputs=tab_ext_skip_fuse, show_progress=False)

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
        return [
            interp_meth, steps, ctrlnet_ref_dir,
            video_fmt, video_fps, video_pad, video_pick,
            ext_video, ext_skip_fuse, dbg_rife,
            *skip_fuses,
        ]

    def run(self, p:Processing, 
            interp_meth:str, steps:str, ctrlnet_ref_dir:str, 
            video_fmt:str, video_fps:float, video_pad:int, video_pick:str,
            ext_video:bool, ext_skip_fuse:bool, dbg_rife:bool,
            *skip_fuses:bool,
        ):

        # Prepare ControlNet
        #self.controlnet_script: ControlNetScript = None
        self.controlnet_script = None
        try:
            for script in p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    script_args: Tuple[ControlNetUnit] = p.script_args[script.args_from:script.args_to]
                    if not any([u.enabled for u in script_args]): return Processed(p, [], p.seed, f'{CTRLNET_REPO_NAME} not enabled')
                    self.controlnet_script = script
                    break
        except ImportError:
            return Processed(p, [], p.seed, f'{CTRLNET_REPO_NAME} not installed')
        except:
            print_exc()
        if not self.controlnet_script: return Processed(p, [], p.seed, f'{CTRLNET_REPO_NAME} not loaded')

        # Enum lookup
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
        self.tmp_dp = Path(self.log_dp) / 'ctrl_cond'   # cache for rife
        self.tmp_fp = self.tmp_dp / 'tmp.png'           # cache for rife

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

        images: List[PILImage] = []
        info: str = None
        try:
            self.UnetHook_hook_original = UnetHook.hook
            UnetHook.hook = hook_hijack

            [c.clear() for c in caches]
            images, info = self.run_linear(p)
        except:
            info = format_exc()
            print(info)
        finally:
            if self.tmp_fp.exists(): os.unlink(self.tmp_fp)
            [c.clear() for c in caches]

            UnetHook.hook = self.UnetHook_hook_original

            self.controlnet_script.input_image = None
            if self.controlnet_script.latest_network:
                self.controlnet_script.latest_network: UnetHook
                self.controlnet_script.latest_network.restore(p.sd_model.model.diffusion_model)
                self.controlnet_script.latest_network = None

            reset_cuda()

        # Save video
        if ext_video: save_video(images, video_slice, video_pad, video_fps, video_fmt, os.path.join(self.log_dp, f'travel-{travel_number:05}'))

        return Processed(p, images, p.seed, info)

    def run_linear(self, p:Processing) -> RunResults:
        global from_hint_cond, to_hint_cond, from_control_tensors, to_control_tensors, interp_alpha, interp_ip

        images: List[PILImage] = []
        info: str = None
        def process_p(append:bool=True) -> Optional[List[PILImage]]:
            nonlocal p, images, info
            proc = process_images(p)
            if not info: info = proc.info
            if append: images.extend(proc.images)
            else: return proc.images

        ''' ↓↓↓ rife interp utils ↓↓↓ '''
        def save_ctrl_cond(idx:int):
            self.tmp_dp.mkdir(exist_ok=True)
            for i, x in enumerate(to_hint_cond):
                x = x[0]
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
            x = x.unsqueeze(dim=0)
            return x
        ''' ↑↑↑ rife interp utils ↑↑↑ '''

        ''' ↓↓↓ filename reorder utils ↓↓↓ '''
        iframe = 0
        def rename_image_filename(idx:int, param: ImageSaveParams):
            fn = param.filename
            stem, suffix = os.path.splitext(os.path.basename(fn))
            param.filename = os.path.join(os.path.dirname(fn), f'{idx:05d}' + suffix)
        class on_before_image_saved_wrapper:
            def __init__(self, callback_fn):
                self.callback_fn = callback_fn
            def __enter__(self):
                on_before_image_saved(self.callback_fn)
            def __exit__(self, exc_type, exc_value, exc_traceback):
                remove_callbacks_for_function(self.callback_fn)
        ''' ↑↑↑ filename reorder utils ↑↑↑ '''

        # Step 1: draw the init image
        setattr(p, 'init_images', [Image.open(self.ctrlnet_ref_fps[0])])
        interp_alpha = 0.0
        with on_before_image_saved_wrapper(partial(rename_image_filename, 0)):
            process_p()
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

            with on_before_image_saved_wrapper(partial(rename_image_filename, iframe + self.steps[i])):
                cached_images = process_p(append=False)
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
                with on_before_image_saved_wrapper(partial(rename_image_filename, iframe)):
                    process_p()
                    iframe += 1

            # adjust order
            images.extend(cached_images)
            iframe += 1

            if is_interrupted: break

        return images, info
