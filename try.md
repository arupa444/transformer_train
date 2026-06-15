$ cd /Users/arupanandaswain/PycharmProjects/transformer
source .venv/bin/activate
python - <<'PY' 2>&1 | grep -v -E "Ultralytics|YOLO|Speed:|image .* at |^$|WARNING|Downloading|torch|MPS|CUDA" | tail -45
import cv2, numpy as np, os, glob
from thermal.detector import YoloDetector
from thermal.colormap import build_lut, ColorToHeat
from thermal.preprocess import preprocess
from thermal.pipeline import analyze_image

td = YoloDetector("models/transformer.pt")
c2h = ColorToHeat(build_lut("inferno"))
LOCAL = "/Users/arupanandaswain/Downloads/cascade_out"; os.makedirs(LOCAL, exist_ok=True)
TMP = "/tmp/cascade_share"; os.makedirs(TMP, exist_ok=True)
SEV = {"Watch":(0,200,200),"Investigate":(0,140,255),"Critical":(0,0,255)}
T = "/Volumes/Atrisol_D2/yolo_thermal_transformer/images"

frames = [f"{T}/test/orig/JUNE 11 MEM 4_DJI_20260411175708_0004_T.jpg",
          f"{T}/train/orig/Jun 11 Mem 2 _DJI_20260407072229_0003_T.jpg"]
frames += [p for p in sorted(glob.glob(f"{T}/test/orig/*.jpg")) if p not in frames]

print(f"{'frame':44} {'tf':>2} {'findings (severity,delta)'}")
print("-"*90)
summary=[]
for p in frames:
    bgr = cv2.imread(p)
    if bgr is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tboxes = td.detect(cv2.cvtColor(preprocess(rgb), cv2.COLOR_RGB2BGR), conf=0.25)
    findings, _, calib = analyze_image(rgb, td, c2h)
    out = bgr.copy()
    for d in tboxes:
        x1,y1,x2,y2 = d.bbox; cv2.rectangle(out,(x1,y1),(x2,y2),(255,255,255),2)
    for f in findings:
        x1,y1,x2,y2 = f.bbox; c = SEV.get(f.severity,(0,0,255))
        cv2.rectangle(out,(x1,y1),(x2,y2),c,3)
        cv2.putText(out,f"{f.severity} {f.relative_delta:.2f}",(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2)
    name = os.path.basename(p)
    cv2.imwrite(f"{LOCAL}/{name}", out); cv2.imwrite(f"{TMP}/{name}.png", out)
    fs=[(f.severity,round(f.relative_delta,2)) for f in findings]
    print(f"{name[:44]:44} {len(tboxes):>2} {fs if fs else 'healthy'}")
    summary.append((name,len(tboxes),fs))
print("-"*90)
crit=sum(1 for _,_,fs in summary if any(s=='Critical' for s,_ in fs))
any_f=sum(1 for _,_,fs in summary if fs)
print(f"frames: {len(summary)}  with transformer detected: {sum(1 for _,t,_ in summary if t>0)}  with any finding: {any_f}  with Critical: {crit}")
print(f"annotated images -> {LOCAL}")
PY