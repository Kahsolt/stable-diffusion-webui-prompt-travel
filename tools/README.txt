Put your post-processing tools or linkings here.

The directory layout should be like:

tools
├── install.cmd
├── link.cmd
├── busybox.exe
├── realesrgan-ncnn-vulkan
│   ├── realesrgan-ncnn-vulkan.exe      # executable
│   └── models                          # model checkpoints
│       ├── *.bin
│       ├── *.param
│       └── *.pth
├── rife-ncnn-vulkan
│   ├── rife-ncnn-vulkan.exe            # executable
│   └── rife*                           # model checkpoints
│       ├── *.bin
│       ├── *.param
│       └── *.pth
└── ffmpeg
    └── bin
        ├── ffmpeg.exe                  # executable
        ├── ffplay.exe
        └── ffprobe.exe
