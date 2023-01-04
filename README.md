# stable-diffusion-webui-prompt-travel

    Travel between prompts in the latent space to make pseudo-animation, extension script for AUTOMATIC1111/stable-diffusion-webui.

----

<p align="left">
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/Kahsolt/stable-diffusion-webui-prompt-travel"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/Kahsolt/stable-diffusion-webui-prompt-travel"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Kahsolt/stable-diffusion-webui-prompt-travel"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/Kahsolt/stable-diffusion-webui-prompt-travel"></a>
  <img alt="Language" src="https://img.shields.io/github/languages/top/Kahsolt/stable-diffusion-webui-prompt-travel">
  <img alt="License" src="https://img.shields.io/github/license/Kahsolt/stable-diffusion-webui-prompt-travel">
  <br/>
</p>

![:stable-diffusion-webui-prompt-travel](https://count.getloli.com/get/@:stable-diffusion-webui-prompt-travel)

Try interpolating on the hidden vectors of conditioning prompt to make seemingly-continuous image sequence, or let's say a pseudo-animation. 😀  

⚠ 我们成立了插件反馈 QQ 群: 616795645 (赤狐屿)，欢迎出建议、意见、报告bug等 (w  
⚠ We have a QQ chat group (616795645) now, any suggestions, discussions and bug reports are highly wellllcome!!  

ℹ 实话不说，我想有可能通过这个来做ppt童话绘本<del>甚至本子</del>……  
ℹ 聪明的用法：先手工盲搜两张好看的图 (只有prompt差异)，然后再尝试在其间 travel :lolipop:  


### Change Log

⚪ Compatibility Warning

- 2023/01/04: webui's recent commit [#bd68e35de3b7cf7547ed97d8bdf60147402133cc](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/bd68e35de3b7cf7547ed97d8bdf60147402133cc) saves memory use in forward caculation, but totally ruins backward gradient caculation via `torch.autograd.grad()` which this script heavily relies on. This change is so far not pluggable but forcely applied, so we're regrettable to say, prompt-travel's grad mode and part of the replace mode will be broken henceforth. (issue #7 cannot be fixed)

⚪ Features

- 2022/12/11: work in a more 'successive' way, idea borrowed from [deforum](https://github.com/deforum-art/deforum-for-automatic1111-webui) ('genesis' option)
- 2022/11/14: walk by substituting token embedding ('replace' mode)
- 2022/11/13: walk by optimizing condition ('grad' mode)
- 2022/11/10: interpolate linearly on condition/uncondition ('linear' mode)

⚪ Fixups

- 2022/12/13: fixup no working when negative prompt is left empty (issue #6: `neg_prompts[-1] IndexError: List index out of range`)
- 2022/11/27: keep up with webui's updates (error `ImportError: FrozenCLIPEmbedderWithCustomWords`)
- 2022/11/20: keep up with webui's updates (error `AttributeError: p.all_negative_prompts[0]`)


### How it works?

- input **multiple lines** in the prompt/negative-prompt box, each line is called a **stage**
- generate images one by one, interpolating from one stage towards the next (batch configs are ignored)
- gradually change the digested inputs between prompts
  - freeze all other settings (`steps`, `sampler`, `cfg factor`, `seed`, etc.)
  - note that only the major `seed` will be forcely fixed through all processes, you can still set `subseed = -1` to allow more variances
- export a video!

⚪ Txt2Img

| sampler \ genesis | fixed | successive |
| :-: | :-: | :-: |
| Eular a | ![t2i-f-eular_a](img/t2i-f-eular_a.gif) | ![t2i-s-eular_a](img/t2i-s-eular_a.gif) |
| DDIM    | ![t2i-f-ddim](img/t2i-f-ddim.gif)       | ![t2i-s-ddim](img/t2i-s-ddim.gif)       |

⚪ Img2Img

| sampler \ genesis | fixed | successive |
| :-: | :-: | :-: |
| Eular a | ![i2i-f-eular_a](img/i2i-f-eular_a.gif) | ![i2i-s-eular_a](img/i2i-s-eular_a.gif) |
| DDIM    | ![i2i-f-ddim](img/i2i-f-ddim.gif)       | ![i2i-s-ddim](img/i2i-s-ddim.gif)       |

Reference image for img2img:

![i2i-ref](img/i2i-ref.png)

Example above run configure ('linear' mode):

```
Prompt:
(((masterpiece))), highres, ((boy)), child, cat ears, white hair, red eyes, yellow bell, red cloak, barefoot, angel, [flying], egyptian
((masterpiece)), highres, ((girl)), loli, cat ears, light blue hair, red eyes, magical wand, barefoot, [running]

Negative prompt:
(((nsfw))), ugly,duplicate,morbid,mutilated,tranny,trans,trannsexual,mutation,deformed,long neck,bad anatomy,bad proportions,extra arms,extra legs, disfigured,more than 2 nipples,malformed,mutated,hermaphrodite,out of frame,extra limbs,missing arms,missing legs,poorly drawn hands,poorty drawn face,mutation,poorly drawn,long body,multiple breasts,cloned face,gross proportions, mutated hands,bad hands,bad feet,long neck,missing limb,malformed limbs,malformed hands,fused fingers,too many fingers,extra fingers,missing fingers,extra digit,fewer digits,mutated hands and fingers,lowres,text,error,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,text font ufemale focus, poorly drawn, deformed, poorly drawn face, (extra leg:1.3), (extra fingers:1.2),out of frame

Steps: 15
CFG scale: 7
Clip skip: 1
Seed: 114514
Size: 512 x 512
Model hash: 925997e9
Hypernet: (this is my secret :)
```


### Options

- prompt: (list of strings)
- negative prompt: (list of strings)
  - input multiple lines of prompt text
  - we call each line of prompt a stage, usually you need at least 2 lines of text to starts travel (unless in 'grad' mode)
  - if len(positive_prompts) != len(negative_prompts), the shorter one's last item will be repeated to match the longer one
- mode: (categorical)
  - `linear`: interpolate linearly on condition/uncondition in latent space
  - `replace`: walk by gradually substituting word embeddings 
  - `grad`: walk by optimizing certain loss
  - NOTE: `walk` methods might not reach target stages in specified steps some times, or reached earlier than expect, in that case, manually tune `grad_alpha` and `steps`  might help a little...
- steps: (int, list of int)
  - number of images to interpolate between two stages
  - if int, constant number of travel steps
  - if list of int, length should match `len(stages)-1`, separate by comma, e.g.: `12, 24, 36`
- genesis: (categorical), the a prior for each image frame
  - `fixed`: starts from pure noise in txt2img pipeline, or from the same ref-image given in img2img pipeline
  - `successive`: starts from the last generated image (this will force txt2img turn to actually be img2img from the 2nd frame on)
- denoise_strength: (float), denoise strength in img2img pipelines when `genesis == 'successive'`
- replace_*
  - replace_order: (categorical)
    - `random`: substitute tokens randomly
    - `similiar`: substitute most similar tokens first (L1 distance of token embeddings)
    - `different`: substitute most different tokens first
    - `grad_min`: substitute tokens that causing smallest gradient first (gradient settings same as in `grad` mode)
    - `grad_max`: substitute tokens that causing largest gradient first
- grad_*
  - grad_alpha: (float), step size of a walk pace
  - grad_iter: (int), step count of walk paces
    - you can try trading `grad_alpha=0.01 grad_iter=1` for `grad_alpha=0.001 grad_iter=10`
    - might be more cautious (perhaps!), but definitely takes more time
  - grad_meth: (categorical), step function of a walk pace
    - `clip`: a triky balance between `sign` and `tanh`
    - `sign`: walk at a constant speed (often stuck into oscillation at the end)
    - `tanh`: significantly speed down when approaching (it takes infinite time to exactly reach...)
  - grad_w_latent: (float), weight factor of `loss_latent`
  - grad_w_cond: (float), weight factor of `loss_cond`
- video_*
  - fps: (float), FPS of video, set `0` to disable file saving
  - fmt: (categorical), export video file format
  - pad: (int), repeat beginning/ending frames, giving a in/out time
  - pick: (string), cherry pick frames by [python slice syntax](https://www.pythoncentral.io/how-to-slice-listsarrays-and-tuples-in-python) before padding (e.g.: set `::2` to avoid non-converging ping-pong phenomenon, set `:-1` to drop non-reaching last frame)
- debug: (bool)
  - whether show verbose debug info at console

⚠ this script will NOT probably support the schedule syntax (i.e.: `[prompt:prompt:number]`), because I don't know how to interpolate between different schedule plans :(  
⚠ max length diff for each prompts should NOT exceed `75` in token count, otherwise will only work on the first segment, cos' I also don't know how to interpolate between different-lengthed tensors 🤔  


### Installation

Easiest way to install it is to:
1. Go to the "Extensions" tab in the webui, switch to the "Install from URL" tab
2. Paste https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git into "URL for extension's git repository" and click install
3. (Optional) You will need to restart the webui for dependencies to be installed or you won't be able to generate video files

Manual install:
1. Copy this repo folder to the 'extensions' folder of https://github.com/AUTOMATIC1111/stable-diffusion-webui
2. (Optional) Restart the webui


### Experimental

⚪ grad mode

The `loss_latent` optimizes `mse_loss(current_generated_latent, target_latent)` 

  - if `grad_w_latent` is positive, minimizing
  - if `grad_w_latent` is negative, maximizing

The `loss_cond` optimizes `l1_loss(current_cond, next_stage_cond)`

  - if `grad_w_cond` is positive, walk towards the next stage (minimizing)
  - if `grad_w_cond` is negative, walk away from it (maximizing)

Grid search results: (`steps=100, grad_alpha=0.01, grad_iter=1, grad_meth='clip'`)

| w_cond\w_latent | -1 | 0 | 1 |
| :-: | :-: | :-: | :-: |
| -1 | 纹理丢失色块平滑、逆向胚胎发育，最后变成圆圈堆叠成的抽象小人 | 前几步变得精致，随后纹理丢失色块平滑，但保持作画结构，中途突然高斯模糊，旋即背景失去语义，最后变成斑点图，l_grad下降 | 走到三张别的图，画风基本一致，背景变朦胧，途中震荡，最后人物没了，变得几何重复 |
| 0 | 纹理丢失色块平滑、逆向胚胎发育，最后变成圆圈堆叠成的抽象小人，l_l1上升 | - | 走到两张别的图，画风基本一致，背景变朦胧，途中震荡，l_l1上升 |
| 1 | 纹理丢失色块平滑、逆向胚胎发育，最后变成圆形蒙版、光栅纹理 | **近似线性插值，叠加式过渡到目标，途中震荡，l_grad下降** | 走到两张别的图，画风基本一致，背景变朦胧，最后震荡 |

(*) 上表如无特殊说明，其各项 loss 变化都符合设置的优化方向  
(**) 我们似乎应当总是令 `w_latent > 0`，而 `w_cond` 的设置似乎很玄学，这里可能遭遇了对抗样本现象(神经网络的过度线性性)……  

ℹ NOTE: When 'prompt' has only single line, it will wander just **around** the initial stage, dynamically balancing `loss_latent` and `loss_cond`; this allows you to discover neighbors of your given prompt 😀

⚪ replace mode

This mode working on token embed input level, hence your can view `log.txt` to see how your input tokens are gradually changed.  
⚠ Remember that comma is a normal valid token, so you might see many commas there. However, they are different when appearing at different positions within the token sequence.  

The actual token replacing order might reveal some information of the token importance, might the listed '>> grad ascend' or '>> embed L1-distance ascend' give you some ideas to tune your input prompt (I wish so..)




### Related Projects

- deforum (2D/3D animation): [https://github.com/deforum-art/deforum-for-automatic1111-webui](https://github.com/deforum-art/deforum-for-automatic1111-webui)
- sonar (k_diffuison samplers): [https://github.com/Kahsolt/stable-diffusion-webui-sonar](https://github.com/Kahsolt/stable-diffusion-webui-sonar)

----

by Armit
2022/11/10 
