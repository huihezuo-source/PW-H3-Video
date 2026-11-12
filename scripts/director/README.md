# 导演流程：先做资产，再用社区工作流出对白短片

适用场景：两人对白、多人漫剧、猫狗段子。2026-11 实测通过的做法，用的是 MiniMax H3 + 两个社区节点包：
[Songssx 时间线导演台](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) 和
[AICG3D 短剧工作台](https://github.com/JGRFW/comfyui-AICG3D)。

## 1. 先把数字资产做好

每个角色准备三类图：
- 单独的头像；
- 近景无头照；
- 远景无头照。

另外再准备背景（空场景）和道具图。

**为什么要这样做：** 脸放在全身照或远景里一起生成，远景的脸会糊。头像单独当参考，远景和近景的脸都能保持一致。实测中，中远景里人脸只占画面一小块，依然清楚。

- `make_assets.py assets.json`：用 H3 Image（`../h3_image.py`）批量出资产图，出完按中文名保存。
  - 先用文字出头像。
  - 再拿头像当 Picture 1 出近景和远景，这样脸和衣服都一致。
  - 动物从已有的角色参考出发。
- `remove_head.py in.png out.png "dog head|crown|hat"`：抠成无头照。
  - 先带头出身体照，保证衣服和姿势对，再用 SAM 3.1（`SAM3_Detect`）按文字找头。
  - 多个文字用 `|` 分开，分别检测再合并，帽子和头饰才会一起去掉。
  - 头部区域用头顶上方那块背景的颜色填。取画面边缘的颜色会因为暗角留下深色剪影。
- 导演检查：行情道具要按 A 股的红涨绿跌画，提示词里写明 Chinese stock market colour convention。

## 2. 配音和对白音轨

- 声音用 `../voice_design.py`（按描述设计）或 `../clone_cast.py`（按参考音克隆）。
- 台词核对用 Whisper medium，small 会把“涨回”听成“掌握”。
- `assemble_dialogue.py spec.json 对白.wav`：拼成一条对白。
  - 自动去掉每句前后的静音；两句合在一条录音里的，按最长的静音切开。
  - 同时输出每句的起止时间（`对白_lines.json`），写提示词和检查口型时用。
  - 结尾的笑声也要放进音轨。

## 3A. 已有定稿配音：Songssx 原音锁定（`director_job.py`）

链路：Material Planner（timeline_data JSON）→ Finite Segment Sampler → VHS 合成。音频设为 `"audioMode": "locked"`，最终音轨就是原始波形。

- **分段**：每段帧数必须是 5+17n，相邻段要有对齐的重叠（例如 0–175、136–294）。音轨要补静音到最后一段的结束帧。
- **提示词**：
  - 每句台词写精确时间点（`At 00:02.890, <Subject 1> … says: "……"`），每句后面写“只有说话的人张嘴，听的人嘴唇闭着，看着说话的人”。
  - 角色定义里写清头取自哪张图、身体取自哪张图，并说明身体图上方的空白区域不是画面内容。
- **实测**：576×1024，DMAD 8 步 LoRA 配 `lcm` 采样器 8 步。12 秒两段约 6 分钟，8 秒一段约 3 分钟。说话人分配全部正确。

## 3B. 只有音色参考：AICG3D 声音库（`aicg3d_job.py`）

链路：H3Loader → SigmaShift → SequenceGlobal（`audio_mode` 设为“生成音频”）→ SequenceSegment×N → SequenceCombine。

- **声音库**：六个座位 girl/boy/heroine/hero/old_woman/old_man 依次对应 (S1)–(S6)。每个座位设 `mode: reference`，填 1–15 秒参考音和 `picture_index`（绑定到该角色的头像），并设 `rights_confirmed: true`。
- **提示词**：用 `integrated_multimodal_description` 格式，按“0-0.6秒：…”逐段写，台词写成 `(S1) says: <d>[Chinese] 台词</d>`。
- **实测**：544×960，12 秒两段约 5 分钟。五句台词全部念对，说话人全对。每句的时间点只能大致控制。

## 3C. 打样走位：T8 画框（`visual_marker_job.py`）

借用 [T8 节点包](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) 的两个节点：`MiniMaxH3VisualMarkerPrepareEXPT8` 在参考图上画框，`MiniMaxH3VisualMarkerPromptEXPT8` 把框的含义写进提示词。后面接普通的 H3 Ref2VA，不需要新模型；5 秒约 1 分钟。

- 每个人物框都必须连一条 relation 到一个目标框。不动的角色不画框，写进场景描述。
- 目标框画在认得出的物体上（门、桌子、某个人）。画在空地上会失败：H3 会重新构图，地面区域的坐标对不上。
- 实测框没有漏进视频。

## 3D. 对照：ComfyUI 原生 InfiniteTalk（`infinitetalk_job.py`）

Wan2.1 I2V 14B + InfiniteTalk 多人补丁，每个说话人一张遮罩、一条音轨，每段 81 帧续接。

**限制**：原生节点两人模式写死为 add，必须第一个人说完、第二个人再说，不能轮流。实测第一段口型正确，第二段起嘴会放到错的人身上；每接一段，人物会被拉回起始姿势。480×832 下 6.8 秒约 6.5 分钟，比 H3 慢。对白片继续用 H3。

## 4. 检查

- `speaker_strip.py`：按对白时间每 0.25 秒截各角色的头部特写，逐句看是不是该说话的人在张嘴。
- 常见问题：
  - **道具会消失**：要写“从头到尾拿着某物”。
  - **说话的人会抬头**：H3 有这个固定倾向，提示词只能压一部分。
  - **片尾人物会转向镜头**：剪辑时在反应镜头之后、转头之前收尾。
