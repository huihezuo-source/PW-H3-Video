# ComfyUI 节点与参数

测试环境：Windows + RTX 4090 24GB，ComfyUI 新版。下面的参数都实际跑通过。

## 角色母版：Qwen-Image 文生图

| 项 | 值 |
|---|---|
| UNETLoader | `qwen_image_fp8_e4m3fn.safetensors`，weight_dtype default |
| CLIPLoader | `qwen_2.5_vl_7b_fp8_scaled.safetensors`，type qwen_image |
| VAE | `qwen_image_vae.safetensors` |
| ModelSamplingAuraFlow | shift 3.1 |
| 尺寸 | 896×1232（竖版全身），batch 2 |
| KSampler | cfg 3，30 步，euler，simple |

提示词 = 风格前缀 + 角色描述。风格前缀示例（黑暗古装恐怖片）：

> 真实的电影剧照，阴冷压抑的古装恐怖片实拍画面，不是动画，不是插画，不是游戏CG。竖构图，人物全身站立，从头到脚完整在画面内，面向镜头。唯一的光源是从左上方照下的青白色冷光，人物一半在阴影里，大面积暗部，背景几乎全黑，飘着稀薄的雾气，地面是潮湿的旧石板。低饱和，青灰和暗金色调，胶片颗粒，真实的皮肤和毛发质感，衣物陈旧脏污、磨损潮湿，脸上有尘土，没有美化，没有修图。

负面词只放风格层面的：`卡通,动画,插画,游戏CG,3D渲染,塑料质感,美颜,磨皮,偶像,明亮,摄影棚,纯色背景,干净的衣服,鲜艳色彩,文字,水印,logo`。

## 改图：Qwen-Image-Edit-2511

| 项 | 值 |
|---|---|
| UNETLoader | `qwen_image_edit_2511_fp8mixed.safetensors`，**weight_dtype fp8_e4m3fn** |
| ModelSamplingAuraFlow | shift 3.1 |
| CFGNorm | strength 1.0 |
| 正向 | TextEncodeQwenImageEditPlus（image1 = 原图，prompt = 改动 + 保持句） |
| 负向 | TextEncodeQwenImageEditPlus，prompt 为空 |
| Latent | VAEEncode 原图 → RepeatLatentBatch 2 |
| KSampler | cfg 4，30 步，euler，simple，denoise 1.0 |

保持句：

> 其余全部保持完全不变：脸、五官、表情、发型、衣服、姿势、光线、雾气、背景和构图都不变。真实的电影剧照质感。

**已知故障：** 某次更新后，weight_dtype 选 default 加载这个模型时 ComfyUI 直接崩溃（Windows fatal exception: access violation）。文件哈希和官方一致，内存也够。改成 `fp8_e4m3fn` 就正常。

## 配音：VoxCPM2 克隆

节点：`LoadAudio` → `VoxCPM_VoiceCloning`（prompt_audio 和 reference_audio 都接参考音，prompt_text 填参考音的原文，trim_silence true）→ `VoxCPM_TTS`（model VoxCPM2，cfg 2.0，steps 10，normalize_text false）→ `SaveAudio`。

- 参考音 6–10 秒、干净、和目标风格一致最好。
- 每条换一个 seed；不满意只重出那一条。
- 显存紧张时 force_offload 设为 true。

## 角色声音：VoxCPM2 按文字描述设计

没有现成参考音时，先按描述设计出每个角色的声音，挑中一条当以后的参考音。脚本 `voice_design.py`。

节点：`VoxCPM_TTS` 不接 voice_config，`voice_design` 填描述（只在不接参考音时生效），model VoxCPM2，cfg 2.0，steps 10。

- 描述写法：身份 + 音色 + 情绪 + 语速，一句话说清，不要堆标签。中文描述可用。例：“三十多岁的男人，嗓音极低而平稳，冷漠没有情绪起伏，像在念程序指令，每个字清楚匀速”。
- 多个角色要拉开年龄、音高和语速，否则观众分不清谁在说话。出完用基频核对：librosa.yin（50–400 Hz，只取响度够的帧）比 pyin 可靠，pyin 在低沉、带气声的嗓音上会跑到上限。
- 试听文本 6–10 秒。太短（2–4 秒）做参考音不够用。
- 台词全是喊叫时，音高会被抬高，描述里写了“低沉”也听不出来。试听文本里放一句平静的话，嗓音底色才出得来。
- 带哭腔的角色描述里加“每个字都吐得清楚”，否则转写会把字听错。
- 每个角色出 2 条，两条拼成 A/B 对比（各自 loudnorm 到同一响度，中间停 1 秒）给用户试听挑选。

## 角色声音：从原片取原声再克隆

用户要“用原片里的声音”时：

1. demucs（htdemucs，Mac 上 mps 跑 157 秒约 1 分钟）把原片人声和配乐分开。脚本 `extract_voice_refs.py` 做第 1、2 步。
2. 对人声轨跑 faster-whisper 词级时间戳，按台词表找出每个角色的句子，切出来（按响度修掉首尾，30 毫秒淡入淡出），中间留 0.35 秒静音拼成一条参考音，5–10 秒最好。
3. `clone_cast.py` 一次给所有角色克隆新台词，比较两种模式：
   - continuation（参考音 + 参考音原文）：和原声的声纹相似度 0.82–0.91，明显更像。原文要和音频一字不差（以音频为准，不是字幕）。会多出一点语气词（“啊”“哼”），落在句子之间，不影响。
   - isolated（只给参考音）：0.74–0.85，有时更像别的角色。
4. 用 Resemblyzer 算声纹相似度矩阵（每条生成音 × 每个角色原声）：对角线要最高，其余低，才是像本人又分得开。
5. 试听文件：先放原声、停 1 秒、再放克隆，两段各自 loudnorm 到 -16 LUFS。

取不出来的情况：角色只有一两个字，或者台词盖在带人声的背景歌下面（demucs 会把歌声也分进人声轨），转写出来的字对不上。这种角色改用 `voice_design.py` 设计。
原片里两个角色的原声相似度很高（0.8 以上）时，多半是同一个人配的，克隆后也会像，要提前告诉用户。
克隆真人声音有法律风险（声音权），和照着真人画脸一样，要告诉用户。

## 视频：MiniMax H3 Ref2VA

| 项 | 值 |
|---|---|
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`，type minimax |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| 主节点 | MiniMaxH3ReferenceToVideo（ref_images / ref_audios 动态输入） |
| 首帧锚定 | MiniMaxH3AddGuide，frame_idx 0，image = 首帧 |
| 采样 | BasicGuider + res_multistep + simple 调度 + SamplerCustomAdvanced |
| 输出 | VAEDecode + VAEDecodeAudio → CreateVideo（24 fps）→ SaveVideo |

速度参考（4090）：

- 448×608，约 15 秒：约 8 分钟。
- 576×800：更清晰 20–40%，但约 14 分钟。
- 加 Pika H3 Flex LoRA（lcm + 7 个手动 sigma）576×800 下 10.8 秒约 3.5 分钟，适合出草稿。
- 576×800 不加速时显存溢出，可能要几个小时，先用低分辨率。

4 步加速（推荐默认，2026-11 实测）：`LoraLoaderModelOnly` 接 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`（强度 1.0）→ `MiniMaxH3SigmaShift`（video 12，audio 3）→ euler + simple，4 步。448×608 双角色对话，233 帧从 5.0 分钟降到 1.5 分钟，313 帧从 6.5 分钟降到 1.9 分钟；台词逐字正确，构图、动作、口型和 20 步版基本一致。`h3_ref2va.py` 里写 `"turbo": true`。要最高质量再回到 20 步 res_multistep。

更快的选择：开源节点 ComfyUI-Ref2VA-VSA 在 4 步之上再加 75% 视频稀疏注意力（参考图和参考音的 token 不裁，脸不漂），作者实测 4090 上 1344×768、5 秒约 72 秒，显存峰值 13.5 GB。要另装自定义节点和 3.85 GB 的 `fasth3_vsa_gate.safetensors`。**前提是 PyTorch 为 CUDA 13 版**：ComfyUI 的 comfy/quant_ops.py 在 `torch.version.cuda < 13` 时会禁用 comfy-kitchen 的 CUDA 后端，cu128 环境运行时报 `Backend 'cuda' is not available: disabled`（2026-11 实测）。

内存：128 GB 内存时，模型文件会留在系统缓存里，ComfyUI 重启后再跑同一段只要 81 秒（几乎不花加载时间）。去掉 `--disable-pinned-memory --disable-async-offload` 对 4 步出片没有可测的提速（85 秒 vs 81 秒），瓶颈在显卡计算。

换人：`h3_character_swap` LoRA 强度 1.0，20 步，源视频作参考视频并锁定音频，不用遮罩。每段最多一个硬切。

## 多镜头复刻原片：首帧锚定，不带参考视频

把一段原片（多个镜头）按原样重做成别的画风时，实测下来的做法（2026-11-05，散户动漫版测试段，10 秒 7 个镜头）：

- **不要把原片当参考视频（ref_videos）喂进去。** 参考视频会让 H3 要处理的序列长度翻倍，4090 上一步要 48 分钟，20 步要 16 小时。
- **每个镜头的切点帧用 AddGuide 钉一张改好画风的首帧**（`h3_ref2va.py` 的 `guides`，只锚定、不当参考图），再在提示词里逐镜头写清景别、动作、镜头运动和切镜时间。结果切镜时间、构图、动作（含“举手、镜头跟着往上摇”）都和原片对上，896×384、243 帧、20 步 8 分钟出片。
- 角色母版作为参考图（Picture 1–3），首帧不进参考图，省 token。
- 原片的黑边先按行亮度量出真实画面范围再裁；这次字幕全在下方黑边里，裁掉就没了。
- 画外台词挪到说话人自己的镜头里（这次挪了 0.15 秒），就不用后期配，也不会让别人对口型。

首帧（Qwen-Image-Edit 把原片帧转画风）：
- 把角色母版当图2一起输入，母版的全身站姿会把构图带跑。只给原片帧，角色的服装和面具用文字写，构图才能保住。
- 要把原片里的人换成某角色，写“把他的黄色外套改成……”，不要写“加一个……”，否则原来的人留着、又多出一个。
- 转画风时 edit_image.py 用 `--keep` 换掉默认的“真实电影剧照质感”那句。
- 同一角色跨镜头的细节（面具角的颜色）要逐张对一遍，不一致就单独改。

声音：H3 用参考音（ref_audios）只学到一部分音色，生成的台词和原声的声纹相似度 0.73，而直接用 VoxCPM 续写克隆是 0.91。要求像原声的话，考虑把克隆好的台词按时间排成音轨，用 AddGuide 的 audio 锁进去。

核对：整条音频直接转写会漏掉开头静音后的台词，要按声音段切开分别转写。

## 速度：显存溢出到内存时会慢几十倍

关掉动态显存（`--disable-dynamic-vram --cache-classic`，为了避开崩溃）以后，ComfyUI 用旧的方式估算显存，会把 H3 主模型放太多在显卡上，计算时显存不够，Windows 显卡驱动把多出来的放到内存里，结果一步要 588 秒。

判断方法：`nvidia-smi` 显示显卡占用 100%，但功耗只有 100 W 左右（正常计算 350–450 W），显存占满。

解决：`--reserve-vram 6`（原来是 1）。主模型少放一些在显卡上，每步多搬几 GB 权重，但不再溢出。同一个任务从 588 秒一步降到 19 秒一步。

## 稳定性：连续切换大模型时崩溃

ComfyUI 0.43 在 Windows 上连续跑几个任务、切换大模型（Qwen-Image、Qwen-Image-Edit、文本编码器）时，Python 进程反复崩溃。报错是 0xc0000005 内存访问冲突，崩溃位置每次都不一样，有时在给模型打补丁，有时在加载文本编码器。日志里能看到“prepared for dynamic VRAM loading”“Using RAM pressure cache”。

缓解：启动参数加上下面几项，关掉新的动态显存加载和内存压力缓存，改回旧的缓存方式。加上之后崩得少了，但长时间连续跑仍会偶尔崩（例如在文本编码器 sd1_clip 里），所以还要配合下面的自动重启和自动重交。

```
--disable-dynamic-vram --cache-classic --disable-pinned-memory --disable-async-offload
```

另外，换模型前调用一次 `comfy.free()`（POST /free 卸载模型），可以减少重复打补丁引发的崩溃。

**自动重启 + 自动重交。** 崩溃没法完全避免时，让它崩了自己起来、任务自己重跑：

1. 电脑端用循环启动脚本，ComfyUI 进程退出后 5 秒自动重启，并记一行日志（整台电脑蓝屏重启后它不会自己起来，要手动再开一次）：

```bat
@echo off
cd /d D:\ComfyUI
:loop
.venv\Scripts\python.exe main.py --listen 0.0.0.0 --port 8188 --reserve-vram 6 --disable-pinned-memory --disable-async-offload --disable-dynamic-vram --cache-classic
echo %date% %time% exit %errorlevel% >> D:\ComfyUI\user\restarts.log
timeout /t 5 /nobreak >nul
goto loop
```

2. 提交端用 `comfy.run()`（各脚本加 `--wait` 时就走它）：每 8 秒查一次 `/system_stats`，连续 4 次（约 1 分钟）都连不上才算断了；恢复后先看任务还在不在，还在就接着等，不在了才把同一个工作流重新提交，最多 3 次。只查一次会把“跑大模型时接口变慢”误判成崩溃，重复提交一份。实测一次改图任务中途崩溃，第 2 次提交自动跑完，没有人工介入。

排查思路：先用 Windows 事件日志确认崩的是不是只有 python。如果系统其他程序也在崩、还有蓝屏，先查硬件（Intel 处理器诊断工具、7-Zip 基准测试、Windows 内存诊断）和系统文件（sfc /verifyonly、DISM /ScanHealth），硬件稳定了再修系统文件。

## 放大

RealESRGAN_x2 用 spandrel 加载，Mac 上走 mps 逐帧推理，再用 ffmpeg lanczos 缩放到目标尺寸。
