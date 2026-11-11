# 脚本用法

先设置 ComfyUI 地址（默认本机 8188）：

```bash
export COMFY_URL=http://127.0.0.1:8188
```

示例配置在 `../examples/`。所有提交类脚本加 `--wait 输出目录` 会等任务完成并下载结果，ComfyUI 中途崩溃重启后会自动重新提交（最多 3 次）；不加则只打印 prompt_id，之后用 `wait_download.py` 取。

| 步骤 | 命令 |
|---|---|
| 出母版 | `python3 t2i_masters.py ../examples/masters.json --wait out/masters` |
| 改母版 | `python3 edit_image.py out/masters/long_00001_.png "只改头部：把头饰去掉。" --wait out/edits` |
| 设计角色声音 | `python3 voice_design.py ../examples/voices.json --wait out/voices` |
| 取原片原声 | `python3 extract_voice_refs.py 原片.mp4 segments.json out/refs` |
| 克隆各角色原声 | `python3 clone_cast.py ../examples/cast.json --wait out/cast` |
| 配音 | `python3 tts_clone.py ../examples/voice.json --wait out/voice` |
| 核对配音 | `python3 asr.py out/voice/*.flac` |
| 生成视频 | `python3 h3_ref2va.py ../examples/h3_job.json --wait out/video`（job 里加 `"turbo": true` 走 4 步加速，约快 3 倍） |
| 把一个系列做成可分享的 ComfyUI 工作流 | `python3 series_workflow.py <系列文件夹> --install --run 2`：读 `spec.json`，生成带中文分组、说明和模型下载链接的 `1_首帧.json`、`2_对话视频.json`，存进 ComfyUI「工作流 → 系列工作流」，`--run` 按工作流原样试跑一次 |
| 多机并行出片 | `python3 farm.py jobs.json out/farm`：把一批任务分给几台 ComfyUI 同时跑（机器列表 `farm_machines.json`，谁空闲谁接下一条，失败自动换机重试一次） |
| 看进度（网页） | `python3 dashboard.py` 后打开 http://localhost:8765 ：实时步数、排队、出片预览、下载工作流、给 Claude 留言 |
| 在 ComfyUI 画布里看任务 | 不用单独运行：`comfy.submit()` 会用 `comfy_ui.py` 把每个任务转成节点图，存到 ComfyUI 左侧「工作流 → Claude任务」，队列里右键任务 →「在新标签页中读取工作流」也能打开。界面中文：设置 → Comfy → 语言选中文 |
| 记录工作步骤 | `python3 board.py step "出首帧" running "4 张"`（显示在工作台右栏） |
| 看进度（命令行） | `python3 ws_progress.py 60`（工作台开着时别用，会抢走进度） |
| 取结果 | `python3 wait_download.py <prompt_id> out/video` |
| 说话人检查 | `python3 speaker_grid.py out/video/seg3.mp4 ../examples/lines.json out/qa` |
| 放大 | `python3 upscale_video.py in.mp4 hd.mp4 RealESRGAN_x2.pth 1080 1500` |
| 字幕和响度 | `python3 post_subtitle_loud.py hd.mp4 final.mp4 ../examples/subs.json` |
| 封面 | `python3 make_cover.py frame.png out "第一行" "第二行" "标签"` |

依赖：Python 3.9+、ffmpeg、Pillow；放大和封面放大另需 `torch` 和 `spandrel`；转写需 `faster-whisper`；取原声需 `demucs` 和 `soundfile`。

ComfyUI 端需要的模型和自定义节点见 `../references/ComfyUI节点与参数.md`。


## 远程机器不在同一网络时

用 RustDesk 的「TCP 隧道」把本机端口转到远程电脑的 127.0.0.1:8188（设备卡片 ⋮ → TCP 隧道），`COMFY_URL=http://127.0.0.1:18188` 即可提交；隧道窗口要保持打开。远程机器的 ComfyUI 只需 `--listen 127.0.0.1`，不用开防火墙。

## 32GB 内存机器的模型选择

`h3_ref2va.py` 默认用 nvfp4 文本编码器（`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，15.7GB，不要求 50 系显卡），比 int8 编码器（27GB）省 11GB 内存，速度持平略快。新机器先跑 `h3_model_download.py` 装基础模型，再跑 `h3_model_download_small.py` 补 nvfp4 编码器和 turbo LoRA。要换回 int8 编码器，设环境变量 `H3_CLIP=qwen3vl_32b_minimax_h3_int8_convrot.safetensors`。

2026-11-11 在公司 WJ（4090、32GB 内存）用 5 秒 640×384 20 步实测，热跑：int8 编码器 2.2 分钟，nvfp4 编码器 2.0 分钟，nvfp4 + w6a8 主模型 2.7 分钟（更慢，不用）。冷加载分别约 6.4、5.6、4.0 分钟。页面文件要调到 64GB 以上，否则加载模型时内存见底。

## Windows 新机器装 torch 卡住怎么办

pip 下载 torch 这类 2GB 以上的大包在 Windows 上会无声卡死（文件一直 0 字节）。别重跑 pip，也别删 `%TEMP%\pip-unpack-*` 里上次留下的部分文件：把它拷出来，用 `curl.exe -L -C - --retry 50 --retry-all-errors -o torch.whl <镜像上的 whl 地址>` 续传，再 `pip install torch.whl`。2026-11-11 公司 XW 用这个办法把卡了 70 分钟的安装 35 分钟收尾。

双机并行实测（同日）：farm.py 把两段 5 秒片子分给 WJ 和 XW 同时跑，各自 2 分钟（冷加载那台 4.7 分钟），成功 2 失败 0。
