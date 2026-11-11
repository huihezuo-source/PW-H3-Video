"""把 API 格式的工作流转成 ComfyUI 画布能打开的节点图（UI 格式），并按数据流自动排版。

comfy.submit() 提交时会把它放进 extra_pnginfo.workflow，于是：
- ComfyUI 网页右上「任务队列」里，右键任务 →「在新标签页中读取工作流」能看到完整节点图；
- 输出的图片/视频自带这份节点图，拖回画布就能还原。
节点定义从 /object_info 读一次后缓存。
"""
import json, urllib.request

_INFO = {}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _info(url, cls):
    if cls not in _INFO:
        _INFO[cls] = json.load(urllib.request.urlopen(f"{url}/object_info/{cls}", timeout=30)).get(cls, {})
    return _INFO[cls]


def _specs(info):
    """按顺序返回 (名字, 类型, 选项)。动态增长的输入（ref_images.ref_image_0 这种）不在这里。"""
    out = []
    order = info.get("input_order", {})
    for sec in ("required", "optional"):
        d = info.get("input", {}).get(sec, {})
        for name in order.get(sec, list(d)):
            if name not in d:
                continue
            t, opts = d[name][0], (d[name][1] if len(d[name]) > 1 else {})
            out.append((name, "COMBO" if isinstance(t, list) else t, opts or {}))
    return out


def _autogrow_type(info, name):
    base = name.split(".")[0]
    for sec in ("required", "optional"):
        spec = info.get("input", {}).get(sec, {}).get(base)
        if spec and str(spec[0]).startswith("COMFY_AUTOGROW"):
            tpl = spec[1].get("template", {}).get("input", {})
            for s in ("required", "optional"):
                for k, v in tpl.get(s, {}).items():
                    return v[0]
    return "*"


def api_to_ui(prompt, url):
    ids = {k: i + 1 for i, k in enumerate(prompt)}
    nodes, links = {}, []
    out_links = {}     # (节点, 输出口) → [link id]
    for k, n in prompt.items():
        info = _info(url, n["class_type"])
        inputs, widgets = [], []
        specs = _specs(info)
        known = {s[0] for s in specs}
        for name, t, opts in specs:
            v = n["inputs"].get(name, opts.get("default"))
            is_link = isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in prompt
            if t in WIDGET_TYPES:
                entry = {"name": name, "type": t, "widget": {"name": name}, "link": None}
                if is_link:
                    lid = len(links) + 1
                    links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), t])
                    out_links.setdefault((v[0], v[1]), []).append(lid)
                    entry["link"] = lid
                    inputs.append(entry)
                    widgets.append(None)
                else:
                    widgets.append(v)
                    if name in ("seed", "noise_seed") or opts.get("control_after_generate"):
                        widgets.append("fixed")
            elif not t.startswith("COMFY_AUTOGROW"):
                entry = {"name": name, "type": t, "link": None}
                if is_link:
                    lid = len(links) + 1
                    links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), t])
                    out_links.setdefault((v[0], v[1]), []).append(lid)
                    entry["link"] = lid
                inputs.append(entry)
        for name, v in n["inputs"].items():       # 动态增长的输入
            if name in known or not (isinstance(v, list) and len(v) == 2 and v[0] in prompt):
                continue
            t = _autogrow_type(info, name)
            lid = len(links) + 1
            links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), t])
            out_links.setdefault((v[0], v[1]), []).append(lid)
            label = name.split(".")[-1]
            for pre, zh in (("ref_video_audio_", "参考视频音轨"), ("ref_image_", "参考图"), ("ref_video_", "参考视频"), ("ref_audio_", "参考音")):
                if label.startswith(pre) and label[len(pre):].isdigit():
                    label = f"{zh} {int(label[len(pre):]) + 1}"
                    break
            inputs.append({"name": name, "type": t, "link": lid, "localized_name": label})
        outs = [{"name": nm, "type": tp, "links": [], "slot_index": i}
                for i, (nm, tp) in enumerate(zip(info.get("output_name", info.get("output", [])), info.get("output", [])))]
        nodes[k] = {"id": ids[k], "type": n["class_type"], "pos": [0, 0], "size": [340, 80 + 26 * (len(inputs) + len(widgets))],
                    "flags": {}, "order": 0, "mode": 0, "inputs": inputs, "outputs": outs,
                    "properties": {"Node name for S&R": n["class_type"]},
                    "widgets_values": [w for w in widgets if w is not None]}
        if n.get("_meta", {}).get("title"):
            nodes[k]["title"] = n["_meta"]["title"]
    for (src, slot), ls in out_links.items():
        if slot < len(nodes[src]["outputs"]):
            nodes[src]["outputs"][slot]["links"] = ls
    # 排版：按依赖深度分列，同列往下排
    depth = {}

    def d(k):
        if k not in depth:
            depth[k] = 0
            ups = [v[0] for v in prompt[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in prompt]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]
    cols = {}
    for k in prompt:
        cols.setdefault(d(k), []).append(k)
    order = 0
    for c in sorted(cols):
        y = 0
        for k in cols[c]:
            nodes[k]["pos"] = [c * 400, y]
            nodes[k]["order"] = order
            order += 1
            y += nodes[k]["size"][1] + 60
    return {"last_node_id": len(ids), "last_link_id": len(links), "nodes": list(nodes.values()), "links": links,
            "groups": [], "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [40, 40]}}, "version": 0.4}
