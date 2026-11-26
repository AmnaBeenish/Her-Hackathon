import sys, traceback
out = open(r"c:\Users\Rana\OneDrive - MSFT\Desktop\repolymer project\rp-check.txt", "w", encoding="utf8")
out.write(f"python {sys.version}\n")
try:
    import ultralytics
    out.write(f"ultralytics {ultralytics.__version__}\n")
    import torch
    out.write(f"torch {torch.__version__}\n")
    from ultralytics import YOLO
    m = YOLO(r"c:\Users\Rana\OneDrive - MSFT\Desktop\repolymer project\yolov8n.pt")
    out.write("YOLO load OK\n")
except Exception:
    out.write("ERROR:\n" + traceback.format_exc())
out.close()
