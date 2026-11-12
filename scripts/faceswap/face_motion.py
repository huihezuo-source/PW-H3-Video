"""量“脸会不会微微动”：MediaPipe FaceLandmarker 逐帧取 blendshapes 和头部姿态，
统计眨眼次数、眉毛起伏、脸颊/嘴角小动作、头部姿态的晃动（逐帧变化），和原片对比。用法：face_motion.py 视频...
注意：手持镜头的晃动不影响姿态角（姿态是相对相机的），但会影响人脸在画面里的位置，所以只看角度。"""
import os, sys, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
def run(path):
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.expanduser("~/models/face/face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
    cap = cv2.VideoCapture(path); fps = cap.get(5); i = 0; bs, ang = [], []
    while True:
        ok, img = cap.read()
        if not ok or i >= 340: break
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps)); i += 1
        if not r.face_blendshapes: bs.append(None); ang.append(None); continue
        bs.append({c.category_name: c.score for c in r.face_blendshapes[0]})
        R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
        ang.append([np.degrees(np.arctan2(R[2, 1], R[2, 2])), np.degrees(np.arcsin(-R[2, 0])), np.degrees(np.arctan2(R[1, 0], R[0, 0]))])
    ok_idx = [k for k, b in enumerate(bs) if b]
    get = lambda name: np.array([bs[k][name] for k in ok_idx])
    blink = (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2
    blinks = int(((blink[1:] > 0.45) & (blink[:-1] <= 0.45)).sum())
    brow = get("browInnerUp") + (get("browOuterUpLeft") + get("browOuterUpRight")) / 2
    cheek = get("cheekSquintLeft") + get("cheekSquintRight") + get("mouthSmileLeft") + get("mouthSmileRight") + get("mouthPressLeft") + get("mouthPressRight")
    a = np.array([ang[k] for k in ok_idx])
    d = np.abs(np.diff(a, axis=0)).mean(0)
    return dict(found=f"{len(ok_idx)}/{len(bs)}", blinks=blinks, brow_std=brow.std(), cheek_std=cheek.std(), jaw_std=get("jawOpen").std(),
                pitch_std=a[:, 0].std(), yaw_std=a[:, 1].std(), roll_std=a[:, 2].std(), head_move_per_frame=d.sum())
for p in sys.argv[1:]:
    r = run(p)
    print(p.split("/")[-1][:24].ljust(24), " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
