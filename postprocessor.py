#!/usr/bin/env pythonw3
# Author: Armit
# Create Time: 2023/03/31 

import sys
import os
import shutil
import psutil
from pathlib import Path
from time import time
from PIL import Image
from PIL.ImageTk import PhotoImage
import subprocess
from subprocess import Popen
from threading import Thread
from typing import Union
import gc

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from traceback import print_exc, format_exc

BASE_PATH = Path(__file__).absolute().parent
WEBUI_PATH = BASE_PATH.parent.parent
OUTPUT_PATH = WEBUI_PATH / 'outputs'
DEFAULT_OUTPUT_PATH = OUTPUT_PATH / 'txt2img-images' / 'prompt_travel'

TOOL_PATH = BASE_PATH / 'tools'
paths_ext = []
paths_ext.append(str(TOOL_PATH))
paths_ext.append(str(TOOL_PATH / 'realesrgan-ncnn-vulkan'))
paths_ext.append(str(TOOL_PATH / 'rife-ncnn-vulkan'))
paths_ext.append(str(TOOL_PATH / 'ffmpeg'))
os.environ['PATH'] += os.path.pathsep + os.path.pathsep.join(paths_ext)

RESR_MODELS = {
  'realesr-animevideov3': [2, 3, 4],
  'realesrgan-x4plus-anime': [4],
  'realesrgan-x4plus': [4],
}
RIFE_MODELS = [
  'rife-v4',
]
EXPORT_FMT = [
  'mp4',
  'gif',
  'webm',
]

def sanitize_pathname(path: Union[str, Path]) -> str:
  if isinstance(path, Path): path = str(path)
  return path.replace('\\', os.path.sep)

def startfile(path:Union[str, Path]):
  # ref: https://stackoverflow.com/questions/17317219/is-there-an-platform-independent-equivalent-of-os-startfile/17317468#17317468
  if isinstance(path, Path): path = str(path)
  if sys.platform == 'win32':
    os.startfile(path)
  else:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, path])

def run_cmd(cmd:str) -> bool:
  try:
    print(f'[exec] {cmd}')
    Popen(cmd, shell=True, encoding='utf-8').wait()
    return True
  except:
    return False

def run_resr(model:str, ratio:int, in_dp:Path, out_dp:Path) -> bool:
  if out_dp.exists(): shutil.rmtree(str(out_dp))
  out_dp.mkdir(exist_ok=True)

  if model == 'realesr-animevideov3': model = f'realesr-animevideov3-x{ratio}'
  return run_cmd(f'realesrgan-ncnn-vulkan -v -s {ratio} -n {model} -i "{sanitize_pathname(in_dp)}" -o "{sanitize_pathname(out_dp)}"')

def run_rife(model:str, interp:int, in_dp:Path, out_dp:Path) -> bool:
  if out_dp.exists(): shutil.rmtree(str(out_dp))
  out_dp.mkdir(exist_ok=True)

  if interp > 0: interp *= len(list(in_dp.iterdir()))
  return run_cmd(f'rife-ncnn-vulkan -v -n {interp} -m {model} -i "{sanitize_pathname(in_dp)}" -o "{sanitize_pathname(out_dp)}"')

def run_ffmpeg(fps:float, fmt:str, in_dp:Path, out_dp:Path) -> bool:
  out_fp = out_dp / f'synth.{fmt}'
  if out_fp.exists(): out_fp.unlink()

  if fmt == 'gif':
    return run_cmd(f'ffmpeg -y -framerate {fps} -i "{sanitize_pathname(in_dp / r"%08d.png")}" "{sanitize_pathname(out_fp)}"')
  else:
    return run_cmd(f'ffmpeg -y -framerate {fps} -i "{sanitize_pathname(in_dp / r"%08d.png")}" -crf 20 -c:v libx264 -pix_fmt yuv420p "{sanitize_pathname(out_fp)}"')


WINDOW_TITLE  = 'Postprocessor Pipeline GUI'
WINDOW_SIZE   = (700, 660)
IMAGE_SIZE    = 512
LIST_HEIGHT   = 100
COMBOX_WIDTH  = 16
COMBOX_WIDTH1 = 5
ENTRY_WIDTH   = 8
MEMINFO_REFRESH = 16    # refresh status memory info every k-image loads

HELP_INFO = '''
[Settings]
  resr: model_name, upscale_ratio
  rife: model_name, interp_ratio (NOT frame count!!)
  ffmpeg: export_format, export_fps

The check boxes are enable swicthes specifying to run or not.
'''


class App:

  def __init__(self):
    self.setup_gui()

    self.is_running = False
    self.cur_name = None  # str
    self.cache = {}       # { 'name': [Image|Path] }

    self.p = psutil.Process(os.getpid())
    self.cnt_pv_load = 0

    if DEFAULT_OUTPUT_PATH.exists():
      self.open_(DEFAULT_OUTPUT_PATH)
    self.var_status.set(self._mem_info_str())

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    #wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # menu
    menu = tk.Menu(wnd, tearoff=0)
    menu.add_command(label='Open folder...', command=self._ls_open_dir)
    menu.add_separator()
    menu.add_command(label='Memory cache clean', command=self.mem_clear)
    menu.add_command(label='Help', command=lambda: tkmsg.showinfo('Help', HELP_INFO))
    def menu_show(evt):
      try:     menu.tk_popup(evt.x_root, evt.y_root)
      finally: menu.grab_release()

    # top: travel folder
    frm1 = ttk.LabelFrame(wnd, text='Travel root folder')
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_root_dp = tk.StringVar(wnd)
      tk.Entry(frm1, textvariable=self.var_root_dp).pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
      tk.Button(frm1, text='Open..', command=self.open_).pack(side=tk.RIGHT)
    
    # bottom status
    # NOTE: do not know why the display order is messy...
    frm3 = ttk.Label(wnd)
    frm3.pack(side=tk.BOTTOM, anchor=tk.S, expand=tk.YES, fill=tk.X)
    if True:
      self.var_status = tk.StringVar(wnd)
      tk.Label(frm3, textvariable=self.var_status).pack(anchor=tk.W)

    # middel: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      # left: control
      frm21 = ttk.Frame(frm2)
      frm21.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
      if True:
        # top: action
        frm211 = ttk.Frame(frm21)
        frm211.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        if True:
          self.var_resr     = tk.BooleanVar(wnd, True)
          self.var_resr_m   = tk.StringVar(wnd, 'realesr-animevideov3')
          self.var_resr_r   = tk.IntVar(wnd, 2)
          self.var_rife     = tk.BooleanVar(wnd, True)
          self.var_rife_m   = tk.StringVar(wnd, 'rife-v4')
          self.var_rife_r   = tk.IntVar(wnd, 2)
          self.var_ffmpeg   = tk.BooleanVar(wnd, True)
          self.var_ffmpeg_r = tk.IntVar(wnd, 20)
          self.var_ffmpeg_f = tk.StringVar(wnd, 'mp4')

          frm2111 = ttk.LabelFrame(frm211, text='Real-ESRGAN')
          frm2111.pack(expand=tk.YES, fill=tk.X)
          if True:
            cb_m = ttk.Combobox(frm2111, text='model', values=list(RESR_MODELS.keys()), textvariable=self.var_resr_m, state='readonly', width=COMBOX_WIDTH)
            cb_r = ttk.Combobox(frm2111, text='ratio', values=[], textvariable=self.var_resr_r, state='readonly', width=COMBOX_WIDTH1)
            cb_m.grid(row=0, column=0, padx=2)
            cb_r.grid(row=0, column=1, padx=2)
            def _cb_r_update():
              values = RESR_MODELS[self.var_resr_m.get()]
              cb_r.config(values=values)
              if self.var_resr_r.get() not in values:
                self.var_resr_r.set(values[0])
            cb_m.bind('<<ComboboxSelected>>', lambda evt: _cb_r_update())
            _cb_r_update()

          frm2112 = ttk.LabelFrame(frm211, text='RIFE')
          frm2112.pack(expand=tk.YES, fill=tk.X)
          if True:
            cb = ttk.Combobox(frm2112, text='model', values=RIFE_MODELS, textvariable=self.var_rife_m, state='readonly', width=COMBOX_WIDTH)
            et = ttk.Entry(frm2112, text='ratio', textvariable=self.var_rife_r, width=ENTRY_WIDTH)
            cb.grid(row=0, column=0, padx=2)
            et.grid(row=0, column=1, padx=2)

          frm2113 = ttk.LabelFrame(frm211, text='FFmpeg')
          frm2113.pack(expand=tk.YES, fill=tk.X)
          if True:
            cb = ttk.Combobox(frm2113, text='format', values=EXPORT_FMT, textvariable=self.var_ffmpeg_f, state='readonly', width=COMBOX_WIDTH)
            et = ttk.Entry(frm2113, text='fps', textvariable=self.var_ffmpeg_r, width=ENTRY_WIDTH)
            cb.grid(row=0, column=0, padx=2)
            et.grid(row=0, column=1, padx=2)

          frm2114 = ttk.Frame(frm211)
          frm2114.pack(expand=tk.YES, fill=tk.X)
          if True:
            frm21141 = ttk.Frame(frm2114)
            frm21141.pack(expand=tk.YES, fill=tk.X)
            for i in range(3): frm21141.columnconfigure(i, weight=1)
            if True:
              ttk.Checkbutton(frm21141, text='resr',   variable=self.var_resr)  .grid(row=0, column=0, padx=0)
              ttk.Checkbutton(frm21141, text='rife',   variable=self.var_rife)  .grid(row=0, column=1, padx=0)
              ttk.Checkbutton(frm21141, text='ffmpeg', variable=self.var_ffmpeg).grid(row=0, column=2, padx=0)

            btn = ttk.Button(frm2114, text='Run!', command=self.run)
            btn.pack()
            self.btn = btn

        frm212 = ttk.LabelFrame(frm21, text='Travels')
        frm212.pack(expand=tk.YES, fill=tk.BOTH)
        if True:
          self.var_ls = tk.StringVar()
          sc = tk.Scrollbar(frm212, orient=tk.VERTICAL)
          ls = tk.Listbox(frm212, listvariable=self.var_ls, selectmode=tk.BROWSE, yscrollcommand=sc.set, height=LIST_HEIGHT)
          ls.bind('<<ListboxSelect>>', lambda evt: self._ls_change())
          ls.pack(expand=tk.YES, fill=tk.BOTH)
          sc.config(command=ls.yview)
          sc.pack(side=tk.RIGHT, anchor=tk.E, expand=tk.YES, fill=tk.Y)
          ls.bind('<Button-3>', menu_show)
          self.ls = ls

      # right: pv
      frm22 = ttk.LabelFrame(frm2, text='Frames')
      frm22.bind('<MouseWheel>', self._pv_change)
      frm22.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
      if True:
        # top
        if True:
          pv = ttk.Label(frm22, image=None)
          pv.bind('<MouseWheel>', self._pv_change)
          pv.bind('<Button-3>', menu_show)
          pv.pack(anchor=tk.CENTER, expand=tk.YES, fill=tk.BOTH)
          self.pv = pv

        # bottom
        if True:
          self.var_fps_ip = tk.IntVar(wnd, 0)
          sc = tk.Scale(frm22, orient=tk.HORIZONTAL, command=lambda _: self._pv_change(),
                        from_=0, to=9, tickinterval=10, resolution=1, variable=self.var_fps_ip)
          sc.bind('<MouseWheel>', self._pv_change)
          sc.pack(anchor=tk.S, expand=tk.YES, fill=tk.X)
          self.sc = sc

  def _mem_info_str(self, title='Mem'):
    mem = self.p.memory_info()
    return f'[{title}] rss: {mem.rss//2**20:.3f} MB, vms: {mem.vms//2**20:.3f} MB'

  def mem_clear(self):
    info1 = self._mem_info_str('Before')

    to_del = set(self.cache.keys()) - {self.cur_name}
    for name in to_del: del self.cache[name]
    gc.collect()

    info2 = self._mem_info_str('After')
    tkmsg.showinfo('Meminfo', info1 + '\n' + info2)

  def open_(self, root_dp:Path=None):
    if root_dp is None:
      root_dp = tkfdlg.askdirectory(initialdir=str(OUTPUT_PATH))
    if not root_dp: return
    if not Path(root_dp).exists():
      tkmsg.showerror('Error', f'invalid path: {root_dp} not exist')
      return

    self.var_root_dp.set(root_dp)

    dps = sorted([dp for dp in Path(root_dp).iterdir() if dp.is_dir()])
    if len(dps) == 0: tkmsg.showerror('Error', 'No travels found!\Your root folder should be like <root_folder>/<travel_number>/*.png')

    self.ls.selection_clear(0, tk.END)
    self.var_ls.set([dp.name for dp in dps])

    self.cache.clear() ; gc.collect()
    self.ls.select_set(len(dps) - 1)
    self.ls.yview_scroll(len(dps), 'units')
    self._ls_change()

  def _ls_change(self):
    idx: tuple = self.ls.curselection()
    if not idx: return
    name = self.ls.get(idx)
    if name is None: return

    self.cur_name = name
    if name not in self.cache:
      dp = Path(self.var_root_dp.get()) / name
      self.cache[name] = sorted([fp for fp in dp.iterdir() if fp.suffix.lower() in ['.png', '.jpg', '.jpeg'] and fp.stem != 'embryo'])

    n_imgs = len(self.cache[name])
    self.sc.config(to=n_imgs-1)
    try:    self.sc.config(tickinterval=n_imgs // (n_imgs / 10))
    except: self.sc.config(tickinterval=1)

    self.var_fps_ip.set(0)
    self._pv_change()

  def _ls_open_dir(self):
    try: startfile(Path(self.var_root_dp.get()) / self.cur_name)
    except: print_exc()

  def _pv_change(self, evt=None):
    if not self.cur_name: return

    cache = self.cache[self.cur_name]
    if not len(cache):
      tkmsg.showinfo('Info', 'This folder is empty...')
      return

    idx = self.var_fps_ip.get()
    if evt is not None:
      offset = 1 if evt.delta < 0 else -1
      idx = (idx + offset + len(cache)) % len(cache)
      self.var_fps_ip.set(idx)

    if isinstance(cache[idx], Path):
      img = Image.open(cache[idx])
      img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
      cache[idx] = PhotoImage(img)

      self.cnt_pv_load += 1
      if self.cnt_pv_load >= MEMINFO_REFRESH:
        self.cnt_pv_load = 0
        self.var_status.set(self._mem_info_str())
    
    img = cache[idx]
    self.pv.config(image=img)
    self.pv.image = img

  def run(self):
    if self.is_running:
      tkmsg.showerror('Error', 'Another task running at background, please wait before finish...')
      return

    def run_tasks(*args):
      (
        base_dp,
        var_resr, var_resr_m, var_resr_r,
        var_rife, var_rife_m, var_rife_r,
        var_ffmpeg, var_ffmpeg_r, var_ffmpeg_f
      ) = args

      if not (0 <= var_rife_r < 8):
        tkmsg.showerror('Error', f'rife_ratio is the interp ratio should be safe in range 0 ~ 4, but got {var_rife_r} :(')
        return
      if not (1 <= var_ffmpeg_r <= 60):
        tkmsg.showerror('Error', f'fps should be safe in range 1 ~ 60, but got {var_ffmpeg_r} :(')
        return

      print('[Task] start') ; t = time()
      try:
        self.is_running = True
        self.btn.config(state=tk.DISABLED, text='Running...')
        
        if var_resr:
          assert run_resr(var_resr_m, var_resr_r, base_dp, base_dp / 'resr')

          # NOTE: fix case of Embryo mode
          embryo_fp: Path = base_dp / 'resr' / 'embryo.png'
          if embryo_fp.exists(): embryo_fp.unlink()
        
        if var_rife:
          assert run_rife(var_rife_m, var_rife_r, base_dp / 'resr', base_dp / 'rife')
        
        if var_ffmpeg:
          assert run_ffmpeg(var_ffmpeg_r, var_ffmpeg_f, base_dp / 'rife', base_dp)
      
        print(f'[Task] done ({time() - t:3f}s)')
        r = tkmsg.askyesno('Ok', 'Task done! Open output folder?')
        if r: startfile(base_dp)
      except:
        e = format_exc()
        print(e)
        print(f'[Task] faild ({time() - t:3f}s)')
        tkmsg.showerror('Error', e)
      finally:
        self.is_running = False
        self.btn.config(state=tk.NORMAL, text='Run!')

    args = (
      Path(self.var_root_dp.get()) / self.cur_name,
      self.var_resr.get(),
      self.var_resr_m.get(),
      self.var_resr_r.get(),
      self.var_rife.get(),
      self.var_rife_m.get(),
      self.var_rife_r.get(),
      self.var_ffmpeg.get(),
      self.var_ffmpeg_r.get(),
      self.var_ffmpeg_f.get(),
    )
    Thread(target=run_tasks, args=args, daemon=True).start()
    print(args)


if __name__ == '__main__':
  App()
