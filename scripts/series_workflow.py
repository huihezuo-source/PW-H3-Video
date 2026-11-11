"""把一个视频系列做成能分享的 ComfyUI 工作流（画布格式，中文分组、说明、模型下载链接都在里面）。

每个系列一个文件夹：workflows/<系列名>/
  spec.json        系列设定：角色、音色、示例提示词、镜头切点、步数（见 workflows/两只狗师徒段子/spec.json）
  素材/            示例首帧、角色参考图、音色（分享时一起打包，朋友上传到 ComfyUI 的 input 即可）
生成：
  1_首帧.json      MiniMax-H3-Image（ComfyUI 0.45+）：英文场景描述 + 角色参考图 → 400 万像素首帧候选
  2_对话视频.json  MiniMax H3 Ref2VA：首帧 + 近景 + 两个音色 + 台词提示词 → 带声音的视频（可选 2 倍放大）

用法：
  python3 series_workflow.py ../workflows/两只狗师徒段子            # 只生成 json
  python3 series_workflow.py ../workflows/两只狗师徒段子 --install  # 再上传素材，并存进 ComfyUI 左侧「工作流 → 系列工作流」
  python3 series_workflow.py ../workflows/两只狗师徒段子 --run 2    # 再按工作流原样跑一次第 2 个，验证能出片（--run 1 跑首帧；加 --no-upscale 跳过放大）
需要 COMFY_URL 指向一台装好 H3 的 ComfyUI（读节点定义用）。
"""
import json, os, sys, time, urllib.parse, urllib.request
import comfy, comfy_ui

HF = "https://huggingface.co"
SIZES = {"minimax_h3_ref2va_pruned_int8_convrot.safetensors": 21.0, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15.7,
         "minimax_h3_video_vae_int8_convrot.safetensors": 2.8, "minimax_h3_audio_vae_fp32.safetensors": 0.6,
         "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": 2.0, "qwen_image_edit_2511_fp8mixed.safetensors": 20.5,
         "qwen_2.5_vl_7b_fp8_scaled.safetensors": 9.4, "qwen_image_vae.safetensors": 0.25, "RealESRGAN_x2plus.pth": 0.07,
         "minimax_h3_image_pruned_int8_convrot.safetensors": 21.0, "minimax_h3_image_vae_fp16.safetensors": 1.2}  # GB
MODELS = {  # 文件名: (下载地址, 放到 models/ 下哪个目录)
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models"),
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders"),
    "minimax_h3_video_vae_int8_convrot.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae"),
    "minimax_h3_audio_vae_fp32.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": (f"{HF}/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras"),
    "minimax_h3_image_pruned_int8_convrot.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", "diffusion_models"),
    "minimax_h3_image_vae_fp16.safetensors": (f"{HF}/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors", "vae"),
    "qwen_image_edit_2511_fp8mixed.safetensors": (f"{HF}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "diffusion_models"),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": (f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    "qwen_image_vae.safetensors": (f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "vae"),
    "RealESRGAN_x2plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", "upscale_models"),
}
COLORS = {"模型": "#3f3f3f", "素材": "#2f4f6f", "台词": "#6f4f2f", "生成": "#3f5f3f", "输出": "#5f3f5f"}


# ---------------- API 格式的节点图（和 h3_ref2va.py / edit_image.py 同一套参数） ----------------

def video_api(s, names):
    v = s["video"]
    p, T = {}, {}
    p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}}
    T["1"] = ("模型", "H3 主模型")
    model = ["1", 0]
    if v.get("turbo", True):
        p["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}}
        p["6"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["2", 0], "shift_video": 12, "shift_audio": 3}}
        T["2"], T["6"] = ("模型", "4 步加速 LoRA"), ("模型", "加速用的噪声曲线（shift 12/3）")
        model = ["6", 0]
    p["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}}
    p["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}}
    p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}}
    T["3"], T["4"], T["5"] = ("模型", "文本编码器"), ("模型", "视频 VAE"), ("模型", "音频 VAE")
    r = {"clip": ["3", 0], "prompt": v["prompt"], "width": v["width"], "height": v["height"], "length": v["length"],
         "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0]}
    for i, im in enumerate(v["images"]):
        nid = str(10 + i)
        p[nid] = {"class_type": "LoadImage", "inputs": {"image": names[im["file"]]}}
        T[nid] = ("素材", f"图{i + 1}（<Picture {i + 1}>）：{im['label']}")
        r[f"ref_images.ref_image_{i}"] = [nid, 0]
    for i, au in enumerate(v["audios"]):
        nid = str(20 + i)
        p[nid] = {"class_type": "LoadAudio", "inputs": {"audio": names[au["file"]]}}
        T[nid] = ("素材", f"音色{i + 1}（<Audio {i + 1}>）：{au['label']}")
        r[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    p["30"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r}
    T["30"] = ("台词", "H3 提示词（台词、镜头都写在这里）")
    pos, lat = ["30", 0], ["30", 1]
    for i, g in enumerate(v.get("anchors", [])):
        nid = str(31 + i)
        src = str(10 + g["image"])
        p[nid] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": pos, "latent": lat, "frame_idx": g["frame"], "vae": ["4", 0], "image": [src, 0]}}
        T[nid] = ("台词", f"镜头锚点：第 {g['frame']} 帧（{g['frame'] / 24:.1f} 秒）= 图{g['image'] + 1}")
        pos = [nid, 0]
    p["40"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": pos}}
    p["41"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if v.get("turbo", True) else "res_multistep"}}
    p["42"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 4 if v.get("turbo", True) else 20, "denoise": 1.0}}
    p["43"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": v.get("seed", 7)}}
    p["44"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["43", 0], "guider": ["40", 0], "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": lat}}
    for k, t in (("40", "引导"), ("41", "采样器"), ("42", "步数"), ("43", "随机种子（换一个数就是重抽）"), ("44", "采样")):
        T[k] = ("生成", t)
    p["45"] = {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["4", 0]}}
    p["46"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["44", 0], "vae": ["5", 0]}}
    frames = ["45", 0]
    if v.get("upscale"):
        p["50"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x2plus.pth"}}
        p["51"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["50", 0], "image": frames}}
        T["50"], T["51"] = ("模型", "2 倍放大模型"), ("输出", "逐帧 2 倍放大")
        frames = ["51", 0]
    p["47"] = {"class_type": "CreateVideo", "inputs": {"images": frames, "audio": ["46", 0], "fps": 24}}
    p["48"] = {"class_type": "SaveVideo", "inputs": {"video": ["47", 0], "filename_prefix": v.get("prefix", "series/out"), "format": "mp4", "codec": "h264"}}
    T["45"], T["46"], T["47"], T["48"] = ("输出", "解码画面"), ("输出", "解码声音"), ("输出", "合成视频"), ("输出", "保存视频（ComfyUI/output）")
    return p, T


def frame_api(s, names):
    """首帧：MiniMax-H3-Image 多图参考出图（ComfyUI 0.45+）。图1 决定构图和尺寸参考，提示词用英文的改图结构写。"""
    f = s["first_frame"]
    w, h = f.get("size", [1728, 2368])
    p, T = {}, {}
    p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors", "weight_dtype": "default"}}
    p["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}}
    p["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"}}
    T.update({"1": ("模型", "H3 Image 主模型"), "2": ("模型", "文本编码器（和 H3 视频共用）"), "3": ("模型", "H3 Image VAE")})
    cond = {"clip": ["2", 0], "vae": ["3", 0], "prompt": f["prompt"], "width": w, "height": h}
    for i, im in enumerate(f["images"][:16]):
        lid, sid = str(20 + i * 2), str(21 + i * 2)
        p[lid] = {"class_type": "LoadImage", "inputs": {"image": names[im["file"]]}}
        T[lid] = ("素材", f"图{i + 1}（<Picture {i + 1}>）：{im['label']}")
        if i == 0:   # 低分辨率参考效果差，先放大到出图尺寸
            p[sid] = {"class_type": "ImageScale", "inputs": {"image": [lid, 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}}
        else:
            p[sid] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [lid, 0], "upscale_method": "lanczos",
                                                                          "megapixels": round(w * h / 1e6, 2), "resolution_steps": 32}}
        T[sid] = ("素材", f"图{i + 1} 放大到约 {w * h / 1e6:.0f} 百万像素")
        cond[f"images.image_{i + 1}"] = [sid, 0]
    p["11"] = {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond}
    p["12"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["11", 0]}}
    T["11"], T["12"] = ("台词", "首帧描述（英文，场景、角色、站位、眼神、光都写在这里）"), ("台词", "反向（自动置空）")
    # H3 Image 只支持一次 1 张：要几张候选，就把运行次数改成几，种子设成随机
    p["15"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["11", 1],
               "seed": f.get("seed", 9101), "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    T["15"] = ("生成", "采样（6 步；一次 1 张，换种子重抽）")
    p["16"] = {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}}
    p["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": f.get("prefix", "series/frame")}}
    T["16"], T["17"] = ("输出", "解码"), ("输出", "保存首帧候选")
    return p, T


# ---------------- 画布：分组排版、说明、模型链接 ----------------

ORDER = ["模型", "素材", "台词", "生成", "输出"]
GROUP_TITLE = {"模型": "① 模型（第一次用先下载，链接在节点属性里）", "素材": "② 素材：首帧、近景、音色（换成你的图和声音）",
               "台词": "③ 台词与镜头（每期主要改这里）", "生成": "④ 生成（一般不用动）", "输出": "⑤ 输出"}


def to_canvas(prompt, tags, notes, url):
    ui = comfy_ui.api_to_ui(prompt, url)
    byid = {str(n["id"]): n for n in ui["nodes"]}
    ids = {k: i + 1 for i, k in enumerate(prompt)}
    groups, x = [], 0
    for g in ORDER:
        members = [k for k in prompt if tags[k][0] == g]
        if not members:
            continue
        wide = g == "台词"
        w = 760 if wide else 380
        y = 60
        for k in members:
            n = byid[str(ids[k])]
            n["title"] = tags[k][1]
            if wide and prompt[k]["class_type"] == "MiniMaxH3ReferenceToVideo":
                n["size"] = [w - 40, 900]
            elif wide and prompt[k]["class_type"] == "TextEncodeQwenImageEditPlus" and prompt[k]["inputs"]["prompt"]:
                n["size"] = [w - 40, 520]
            elif prompt[k]["class_type"] == "LoadImage":
                n["size"] = [w - 40, 470]
            elif prompt[k]["class_type"] == "LoadAudio":
                n["size"] = [w - 40, 190]
            else:
                n["size"] = [w - 40, max(n["size"][1], 90)]
            n["pos"] = [x + 20, y]
            fn = next((v for v in prompt[k]["inputs"].values() if isinstance(v, str) and v in MODELS), None)
            if fn:
                link, d = MODELS[fn]
                n["properties"]["models"] = [{"name": fn, "url": link, "directory": d}]
            y += n["size"][1] + 70
        groups.append({"title": GROUP_TITLE[g], "bounding": [x, 0, w, y], "color": COLORS[g], "font_size": 22, "flags": {}})
        x += w + 40
    nid = ui["last_node_id"]          # 说明放在最左边一列，打开工作流先看到
    ny = 0
    for title, text in notes:
        nid += 1
        h = 160 + 26 * (text.count("\n") + len(text) // 34)
        ui["nodes"].append({"id": nid, "type": "Note", "pos": [-640, ny], "size": [600, h], "flags": {}, "order": nid, "mode": 0,
                            "inputs": [], "outputs": [], "title": title, "properties": {}, "widgets_values": [text],
                            "color": "#432", "bgcolor": "#653"})
        ny += h + 50
    ui["last_node_id"] = nid
    ui["groups"] = groups
    ui["extra"]["ds"] = {"scale": 0.6, "offset": [680, 40]}
    return ui


def readme(d, s, out):
    """写分享用的 使用说明.txt：要哪些模型、放哪、怎么用。"""
    L = [f"《{s['name']}》ComfyUI 工作流", "=" * 30, "",
         "需要：ComfyUI 0.45 或更新（自带 MiniMax H3 和 H3 Image 节点），显卡 24 GB（4090 实测），内存 64 GB 以上更稳。不用装任何自定义节点。PyTorch 用 CUDA 13 版（cu130）时 ComfyUI 的加速内核全开，出片快一倍。", "",
         "一、安装", "1. 把 素材/ 里的图片和音频上传到 ComfyUI（或直接拷到 ComfyUI/input/）。",
         "2. 把 1_首帧.json、2_对话视频.json 拖进 ComfyUI 网页。缺模型时会弹窗，点下载即可；也可以按下表手动下载放到对应目录。", ""]
    for key, (p, _) in out.items():
        L.append(f"{key}.json 用到的模型：")
        seen = []
        for n in p.values():
            for v in n["inputs"].values():
                if isinstance(v, str) and v in MODELS and v not in seen:
                    seen.append(v)
        for fn in seen:
            L.append(f"  - models/{MODELS[fn][1]}/{fn}  （约 {SIZES.get(fn, 0)} GB）")
            L.append(f"    {MODELS[fn][0]}")
        L.append("")
    L += ["二、每期怎么用", "工作流画布最左边一列是说明，打开先看。简单说：",
          "1. 打开 1_首帧：换英文场景描述，每次出 1 张 400 万像素候选（约 30 秒），运行次数改成 2、种子设随机就是 2 张；挑一张，在上面裁出两张近景。",
          "2. 打开 2_对话视频：换首帧和近景，改 H3 提示词里的台词和镜头，按切点设镜头锚点，运行。",
          "3. 一段最长 15 秒；更长就分段出，再在剪辑软件里拼接、加字幕。", "",
          "三、示例素材", "素材/ 里是第九期《初心》的首帧和两只狗的音色，直接运行就能复现那一段（第一段 13 秒）。"]
    open(os.path.join(d, "使用说明.txt"), "w").write("\n".join(L) + "\n")


def main():
    d = os.path.abspath(sys.argv[1])
    s = json.load(open(os.path.join(d, "spec.json")))
    tag = s["key"]
    files = {}
    for part in (s["first_frame"]["images"], s["video"]["images"], s["video"]["audios"]):
        for it in part:
            files[it["file"]] = os.path.basename(it["file"])   # 和素材文件名一致，朋友拷进 input 就能用
    if "--install" in sys.argv or "--run" in sys.argv:
        for f, n in files.items():
            comfy.upload(os.path.join(d, f), n)
    out = {}
    for key, build in (("1_首帧", frame_api), ("2_对话视频", video_api)):
        p, T = build(s, files)
        ui = to_canvas(p, T, s["notes"][key], comfy.URL)
        path = os.path.join(d, f"{key}.json")
        json.dump(ui, open(path, "w"), ensure_ascii=False, indent=1)
        json.dump(p, open(os.path.join(d, f"{key}_api.json"), "w"), ensure_ascii=False, indent=1)
        out[key] = (p, ui)
        print("写好", path)
        if "--install" in sys.argv:
            name = urllib.parse.quote(f"workflows/系列工作流/{s['name']}_{key}.json", safe="")
            urllib.request.urlopen(urllib.request.Request(f"{comfy.URL}/userdata/{name}?overwrite=true&full_info=true",
                                   json.dumps(ui, ensure_ascii=False).encode(), {"Content-Type": "application/json"}, method="POST"), timeout=15)
            print("  已存进 ComfyUI：工作流 → 系列工作流 →", f"{s['name']}_{key}")
    readme(d, s, out)
    if "--run" in sys.argv:
        k = {"1": "1_首帧", "2": "2_对话视频"}[sys.argv[sys.argv.index("--run") + 1]]
        p, ui = out[k]
        if "--no-upscale" in sys.argv and k == "2_对话视频":   # 这台机器还没有放大模型时，试跑先跳过放大
            s["video"]["upscale"] = False
            p = video_api(s, files)[0]
        body = json.dumps({"prompt": p, "client_id": comfy.CLIENT_ID, "extra_data": {"extra_pnginfo": {"workflow": ui}}}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(comfy.URL + "/prompt", body, {"Content-Type": "application/json"})))
        if r.get("node_errors"):
            raise SystemExit(f"工作流有错：{r['node_errors']}")
        t = time.time()
        rec = comfy.wait(r["prompt_id"], timeout=3 * 3600)
        print(f"跑完，用时 {(time.time() - t) / 60:.1f} 分钟")
        print(comfy.download(rec, os.path.join(d, "试跑输出")))


if __name__ == "__main__":
    main()
