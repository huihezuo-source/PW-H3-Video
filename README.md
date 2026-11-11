# H3 Video

用本地 ComfyUI 做带对白 AI 短剧的 Claude Code 技能。用到的模型：MiniMax H3（视频 + 声音）、Qwen-Image（角色母版和改图）、VoxCPM2（克隆配音）。

整套流程从定基调开始，接着写台词表、配音、出母版和首帧，然后用 H3 生成，逐句检查是谁在说话，最后放大、加字幕、调响度、出封面。每一步都来自真实片子的返工经验。

## 它解决什么问题

- **人物对着旁白对口型。** 旁白不进视频模型，一次生成只有一个发声来源。
- **说话人错位，看片才发现。** 每句台词出一张密集帧网格，只回答“谁的嘴在动”，再和台词表核对。
- **画风和故事不搭**，比如黑暗故事里出现偶像脸。动手前先写基调与风格，母版也在故事的世界里拍。
- **生成完才发现首帧有问题。** 首帧放大到原图逐项检查，确认后才生成视频。
- **一处出错整片重来。** 先判断问题出在哪一层，只修出错的那一段。

## 安装

把整个文件夹放进 Claude Code 的技能目录：

```bash
git clone https://github.com/huihezuo-source/PW-H3-Video.git ~/.claude/skills/h3-video
```

之后对 Claude 说“用 H3 把这个故事做成视频”，或者“做一条 AI 短剧”，就会按这套流程走。

推荐同时装 MiniMax 官方的 H3 提示词技能 `h3-prompt-writing`，这套流程写 H3 提示词时会用到它。

## 环境

- ComfyUI，需要 MiniMax H3、Qwen-Image、Qwen-Image-Edit-2511 和 VoxCPM 节点。模型文件名和参数见 [references/ComfyUI节点与参数.md](references/ComfyUI节点与参数.md)。
- 显卡：在 RTX 4090 24GB 上测过。
- 后期在本机做，需要 Python 3.9+、ffmpeg 和 Pillow；放大视频另需 torch 和 spandrel，核对配音需要 faster-whisper。
- ComfyUI 不在本机时，设置 `COMFY_URL`，例如 `export COMFY_URL=http://192.168.0.186:8188`。公司同事直接用现成的两台 4090，见 [references/公司4090使用说明.md](references/公司4090使用说明.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `SKILL.md` | 总流程、七条硬规则、返工原则 |
| `references/流程细节与教训.md` | 每一步的做法和踩过的坑 |
| `references/ComfyUI节点与参数.md` | 节点、模型、参数、速度、已知故障 |
| `references/开源流水线调研.md` | 9 个开源 AI 视频项目的做法对比 |
| `references/公司4090使用说明.md` | 公司两台 4090 的地址、开放步骤、同事怎么提交任务、一起用的规矩 |
| `templates/规划模板.md` | 规划文档模板：台词表、生成清单、验证表 |
| `scripts/` | 出母版、改图、配音、H3 生成、说话人检查、放大、字幕响度、封面，用法见 [scripts/README.md](scripts/README.md) |
| `examples/` | 各脚本的示例配置 |

## 许可

MIT
