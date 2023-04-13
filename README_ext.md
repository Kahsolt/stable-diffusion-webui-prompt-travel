# stable-diffusion-webui-non-prompt-travel (extensions)

    Of course not only prompts! -- You shall also be able to travel through any other conditions. 😀

----

### ControlNet-Travel

Travel through ControlNet's control conditions like canny, depth, openpose, etc...

⚠ Memory (not VRAM) usage grows linearly with sampling steps, and fusion layers count, this is its nature 😥

Quickstart instructions:

- prepare a folder of images, might be frames from a video
- check enble `sd-webui-controlnet`, set all parameters as you want, but it's ok to **leave the ref image box empty**
  - reference images will be read from the image folder given in controlnet-travel :)
- find `ControlNet Travel` in the script dropdown, set all parameters again, specify your image folder path here
- click Generate button

Options:

- interp_meth: (categorical)
  - `linear`: linear weighted sum, better for area-based annotaions like `depth`, `seg`
  - `rife`: optical flow model (requires to install postprocess tools first), better for edge-base annotaions like `canny`, `openpose`
- skip_latent_fusion: (list of bool), experimental
  - skip some latent layers fusion for saving memory, but might get wierd results 🤔
  - ℹ in my experiences, the `mid` and `in` blocks are more safe to skip
- save_rife: (bool), save the rife interpolated condtion images


----
by Armit
2023/04/12
