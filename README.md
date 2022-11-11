# stable-diffusion-webui-prompt-travel

    Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.

----

This is the more human-sensible version of [stable-diffusion-webui-prompt-erosion](https://github.com/Kahsolt/stable-diffusion-webui-prompt-erosion), 
now we do not modify on text char level, but do linear interpolating on the hidden embedded vectors. 😀

⚠ Though this is still not the best way to do semantics interpolate, future works will continue to explorer.
⚠ 尽管线性插值仍然不是最连续流畅的过渡方式，之后的工作将探索是否能通过探测梯度下降方向来插值（但是先摸一会儿别的东西了 :lolipop:

ℹ 实话不说，我想有可能通过这个来做ppt童话绘本<del>甚至本子</del>……
ℹ 聪明的用法：先手工盲搜两张好看的图 (只有prompt差异)，然后再尝试在其间 travel 😀

### How it works?

- generate image one by one (batch configs are ignored)
- gradually change the digested inputs between prompts
  - freeze all other settings (steps, sampler, cfg factor, rand seed, etc.)
  - force `subseed == seed, subseed_strength = 0.0`
- gather to be a video!

DDIM:
![DDIM](img/ddim.gif)

Eular a:
![eular_a](img/eular_a.gif)

ℹ 在原始的 prompt 框里输入正面/负面提示词，每一行表示一个stage
ℹ 在左下角的插件栏修改 stage之间的补帧数量 和 视频输出帧率

```
[postive prompts]
(((masterpiece))), highres, ((boy)), child, cat ears, white hair, red eyes, yellow bell, red cloak, barefoot, angel, [flying], egyptian
((masterpiece)), highres, ((girl)), loli, cat ears, light blue hair, red eyes, magical wand, barefoot, [running]

[negative prompts]
(((nsfw))), ugly,duplicate,morbid,mutilated,tranny,trans,trannsexual,mutation,deformed,long neck,bad anatomy,bad proportions,extra arms,extra legs, disfigured,more than 2 nipples,malformed,mutated,hermaphrodite,out of frame,extra limbs,missing arms,missing legs,poorly drawn hands,poorty drawn face,mutation,poorly drawn,long body,multiple breasts,cloned face,gross proportions, mutated hands,bad hands,bad feet,long neck,missing limb,malformed limbs,malformed hands,fused fingers,too many fingers,extra fingers,missing fingers,extra digit,fewer digits,mutated hands and fingers,lowres,text,error,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,text font ufemale focus, poorly drawn, deformed, poorly drawn face, (extra leg:1.3), (extra fingers:1.2),out of frame

[steps]
45
```

### Options

- postive prompts: (list of strings)
- negative prompts: (list of strings)
  - each line is a prompt stage
  - if len(postive) != len(negative), the shorter one's last item will be repeated to match the longer one
- steps: (int, list of int)
  - travel from stage1 to stage2 in n steps (即补帧数量)
  - if single int, constant number of images between two successive stages
  - if list of ints, should match `len(stages)-1`， e.g.: `12, 24, 36`

⚠ this feature does NOT support the **schedule** syntax (i.e.: `[propmt:propmt:number]`), because I don't know how to interpolate between different schedule plans :(
ℹ max length diff for each prompts should not exceed `75` in token count, cos' I also don't know how to interpolate between different-lengthed tensors :)


### Installation

Easiest way to install it is to:
1. Go to the "Extensions" tab in the webui
2. Click on the "Install from URL" tab
3. Paste https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git into "URL for extension's git repository" and click install
4. (Optional) You will need to restart the webui for dependensies to be installed or you won't be able to generate video files.

Manual install:
1. Copy the file in the scripts-folder to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui
2. Add `moviepy==1.0.3` to requirements_versions.txt

----

by Armit
2022/11/10 
