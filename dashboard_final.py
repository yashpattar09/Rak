"""
Seed2 Matcher — accurate instance matching.

Core method (unchanged, chosen originally by measurement):
  Both images resized to 128x128 -> ResNet18 CNN embedding (512-d) ->
  cosine similarity -> optimal one-to-one assignment (Hungarian).

Accuracy add-ons layered ON TOP of that core (compute is cheap on the GPU):
  1. DINOv2 (ViT) global descriptor at full resolution with multi-scale +
     flip test-time augmentation. Far more discriminative than a 128px
     ResNet18 for telling near-identical "brown dirt" scenes apart.
  2. SIFT + RANSAC geometric verification at full resolution. Counts the
     number of geometrically-consistent keypoint correspondences between a
     reference and a candidate — direct physical evidence that two frames
     show the SAME arrangement, and a trustworthy confidence signal.

The three signals are min-max normalised and fused into one score matrix,
then the SAME Hungarian assignment runs on it. If DINOv2 or SIFT are
unavailable (e.g. offline), those signals silently switch off and the
pipeline degrades exactly to the original ResNet18@128 behaviour.
"""
import os
import zipfile
import tempfile
import shutil
import cv2
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from flask import Flask, render_template_string, jsonify, request

DIM = 128
REF_FOLDER = "extracted_seed2/captures1"
QUERY_FOLDER = "newwww"          # default; replaced when a ZIP is uploaded
OUT_FOLDER = "results_128x128"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- accuracy add-on config (auto-disabled if a signal can't load) ----
USE_DINO = True                  # DINOv2 global descriptor
USE_GEOM = True                  # SIFT + RANSAC geometric verification
# fusion weights (renormalised over whatever signals load). Geometry is
# weighted highest: RANSAC-verified correspondences are the closest thing to
# ground-truth evidence that two frames show the same physical arrangement;
# the deep embeddings mainly rank the ambiguous, low-overlap cases.
W_CORE, W_DINO, W_GEOM = 0.15, 0.35, 0.50
GEOM_MAXSIDE = 1600              # longest image side used for SIFT
GEOM_CAP = 40                    # inliers are clipped here before normalising
CONF_INLIERS = 15                # >= this many RANSAC inliers => confident
THRESH = 0.55                    # cosine confidence threshold (fallback modes)

# holds the folder of the most recently uploaded ZIP (None => use default)
UPLOADED_QUERY_FOLDER = None

# ---- core model (unchanged): ResNet18 @ 128x128 ----
print(f"[*] Device: {DEVICE}")
print("[*] Loading ResNet18 (pretrained)...")
_model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
_model.fc = torch.nn.Identity()
_model.eval().to(DEVICE)
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
print("[*] Core model ready.")

# ---- add-on 1: DINOv2 global descriptor (full-res, multi-scale + flip TTA) ----
_dino = None
_dino_device = DEVICE
_dino_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_DINO_SCALES = (224, 308)
if USE_DINO:
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    # try GPU first; on constrained devices (e.g. Jetson OOM) retry on CPU
    # rather than dropping the strongest signal entirely.
    for dev_try in ([DEVICE, "cpu"] if DEVICE != "cpu" else ["cpu"]):
        try:
            print(f"[*] Loading DINOv2 (ViT-S/14) on {dev_try}...")
            if dev_try == "cuda":
                torch.cuda.empty_cache()
            _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                   trust_repo=True, verbose=False).eval().to(dev_try)
            _dino_device = dev_try
            print(f"[*] DINOv2 ready on {dev_try}.")
            break
        except Exception as e:
            print(f"[!] DINOv2 load on {dev_try} failed: {str(e)[:120]}")
            _dino = None
    if _dino is None:
        print("[!] DINOv2 unavailable, continuing without it.")
        USE_DINO = False

# ---- add-on 2: SIFT + RANSAC geometric verification ----
_sift = _bf = None
if USE_GEOM:
    try:
        _sift = cv2.SIFT_create(nfeatures=8000)
        _bf = cv2.BFMatcher(cv2.NORM_L2)
        print("[*] SIFT geometric verification ready.")
    except Exception as e:
        print(f"[!] SIFT unavailable, geometric check off: {e}")
        USE_GEOM = False


def _norm01(mat):
    """Min-max normalise a score matrix to [0,1] for fusion."""
    mat = np.asarray(mat, dtype=np.float64)
    lo = mat.min()
    return (mat - lo) / (mat.max() - lo + 1e-9)


def _dino_embed(rgb_full):
    """DINOv2 descriptor with multi-scale + horizontal-flip TTA (L2-normed)."""
    vecs = []
    for sz in _DINO_SCALES:
        t = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(sz),
            transforms.CenterCrop(sz), _dino_norm,
        ])
        for view in (rgb_full, rgb_full[:, ::-1].copy()):
            with torch.no_grad():
                v = _dino(t(view).unsqueeze(0).to(_dino_device)).squeeze().cpu().numpy()
            vecs.append(v)
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def _sift_feat(gray_full):
    """SIFT keypoints/descriptors on the full-res (downscaled) grayscale image."""
    h, w = gray_full.shape
    s = GEOM_MAXSIDE / max(h, w)
    if s < 1:
        gray_full = cv2.resize(gray_full, (int(w * s), int(h * s)))
    return _sift.detectAndCompute(gray_full, None)


def geom_inliers(feat_a, feat_b):
    """RANSAC-verified correspondence count between two SIFT feature sets."""
    (k1, d1), (k2, d2) = feat_a, feat_b
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    good = [m for m, n in _bf.knnMatch(d1, d2, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return 0
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    _H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0)
    return int(mask.sum()) if mask is not None else 0


def list_images(folder):
    out = []
    for root, _d, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
    return out


def embed_and_save(name, path, is_ref):
    # read full resolution once; derive the 128px view + full-res add-on inputs
    img_full = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_full is None:
        raise FileNotFoundError(path)
    img128 = cv2.resize(img_full, (DIM, DIM), interpolation=cv2.INTER_AREA)

    sub = "ref" if is_ref else "query"
    save_path = os.path.join(OUT_FOLDER, sub, os.path.splitext(name)[0] + ".jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img128)  # color 128x128 for display

    # ---- CORE (unchanged): ResNet18 embedding on the 128x128 pixels ----
    rgb = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        v = _model(_tf(rgb).unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
    v = v / (np.linalg.norm(v) + 1e-9)

    rec = {"name": name, "emb": v, "disp": save_path.replace("\\", "/")}

    # ---- add-on 1: DINOv2 descriptor on the full-res image ----
    if USE_DINO:
        rec["dino"] = _dino_embed(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))

    # ---- add-on 2: SIFT features on the full-res image ----
    if USE_GEOM:
        rec["feat"] = _sift_feat(cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY))

    return rec


def run_matching():
    active = ["ResNet18@128"] + (["DINOv2"] if USE_DINO else []) \
             + (["SIFT-geom"] if USE_GEOM else [])
    print("\n" + "=" * 70)
    print("MATCHING  (" + " + ".join(active) + " -> fused -> Hungarian)")
    print("=" * 70)
    os.makedirs(OUT_FOLDER, exist_ok=True)

    query_folder = UPLOADED_QUERY_FOLDER or QUERY_FOLDER
    print(f"[*] Reference folder: {REF_FOLDER}")
    print(f"[*] Captured folder : {query_folder}")

    ref_paths = list_images(REF_FOLDER)
    query_paths = list_images(query_folder)
    print(f"[*] {len(ref_paths)} references, {len(query_paths)} captured")
    if not query_paths:
        raise RuntimeError(f"No images found in captured folder: {query_folder}")

    refs = [embed_and_save(os.path.basename(p), p, True) for p in ref_paths]
    queries = [embed_and_save(os.path.basename(p), p, False) for p in query_paths]

    # ---- CORE signal (unchanged): ResNet18@128 cosine similarity ----
    ER = np.array([r["emb"] for r in refs])
    EQ = np.array([q["emb"] for q in queries])
    sim = ER @ EQ.T  # cosine similarity (refs x queries)

    # ---- add-on signals, fused with the core ----
    signals = [(W_CORE, _norm01(sim))]

    sim_dino = None
    if USE_DINO:
        DR = np.array([r["dino"] for r in refs])
        DQ = np.array([q["dino"] for q in queries])
        sim_dino = DR @ DQ.T
        signals.append((W_DINO, _norm01(sim_dino)))

    geom = None
    if USE_GEOM:
        geom = np.array([[geom_inliers(r["feat"], q["feat"]) for q in queries]
                         for r in refs], dtype=float)
        signals.append((W_GEOM, _norm01(np.clip(geom, 0, GEOM_CAP))))

    wsum = sum(w for w, _ in signals)
    fused = sum(w * s for w, s in signals) / wsum

    # optimal one-to-one assignment on the fused score (each ref maps uniquely)
    ri, ci = linear_sum_assignment(-fused)
    assign = {int(i): int(j) for i, j in zip(ri, ci)}

    # headline similarity: prefer DINOv2 cosine when available (more meaningful)
    head = sim_dino if sim_dino is not None else sim

    results = []
    print("[*] Assignment:")
    for i, ref in enumerate(refs):
        j = assign.get(i)
        if j is None:
            results.append({"reference": ref["name"], "ref_path": ref["disp"],
                            "query": None, "query_path": None, "similarity": 0,
                            "geo": 0, "confident": False})
            continue
        q = queries[j]
        s = float(head[i, j])
        gi = int(geom[i, j]) if geom is not None else 0

        # confidence: geometric evidence is trusted first, else cosine threshold
        if USE_GEOM:
            confident = gi >= CONF_INLIERS
        else:
            confident = s >= THRESH

        # top-3 by fused score, for transparency
        order = np.argsort(-fused[i])
        top3 = [{"name": queries[k]["name"], "path": queries[k]["disp"],
                 "sim": round(float(head[i, k]) * 100),
                 "geo": int(geom[i, k]) if geom is not None else 0}
                for k in order[:3]]
        print(f"    {ref['name'][:26]:<26} -> {q['name'][:26]:<26} "
              f"{s*100:5.1f}%  geo={gi:<3d} {'OK' if confident else 'weak'}")
        results.append({
            "reference": ref["name"], "ref_path": ref["disp"],
            "query": q["name"], "query_path": q["disp"],
            "similarity": round(s * 100), "geo": gi,
            "confident": confident, "top3": top3,
        })
    print("=" * 70 + "\n")
    return results


app = Flask(__name__)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global UPLOADED_QUERY_FOLDER
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"success": False, "error": "No file selected"})
    if not f.filename.lower().endswith(".zip"):
        return jsonify({"success": False, "error": "Please upload a .zip file"})

    try:
        # fresh temp dir per upload
        temp_dir = tempfile.mkdtemp(prefix="upload_")
        zip_path = os.path.join(temp_dir, "upload.zip")
        f.save(zip_path)

        extract_dir = os.path.join(temp_dir, "images")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        n = len(list_images(extract_dir))
        if n == 0:
            return jsonify({"success": False,
                            "error": "ZIP contains no images (.jpg/.png)"})

        UPLOADED_QUERY_FOLDER = extract_dir
        print(f"[*] Uploaded ZIP extracted -> {extract_dir} ({n} images)")
        return jsonify({"success": True, "count": n})
    except zipfile.BadZipFile:
        return jsonify({"success": False, "error": "Not a valid ZIP file"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/match", methods=["POST"])
def api_match():
    try:
        return jsonify({"results": run_matching()})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/img/<path:filepath>")
def serve_img(filepath):
    for p in (os.path.join(OUT_FOLDER, filepath.replace("/", os.sep)), filepath):
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read(), 200, {"Content-Type": "image/jpeg"}
    return "Not found", 404


@app.route("/")
def index():
    return render_template_string(PAGE)


PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Seed2 Matcher</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:system-ui;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;color:#333}
  .container{max-width:1500px;margin:0 auto;padding:20px}
  header{background:white;padding:26px;border-radius:12px;margin-bottom:26px;box-shadow:0 4px 12px rgba(0,0,0,.15);text-align:center}
  header h1{font-size:28px;color:#1f2937}
  header p{font-size:13px;color:#6b7280;margin-top:6px}
  .bar{text-align:center;margin-bottom:30px}
  button{padding:15px 46px;font-size:18px;background:#10b981;color:white;border:none;border-radius:8px;cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(16,185,129,.3);transition:.2s}
  button:hover{background:#059669;transform:translateY(-2px)}
  button:disabled{background:#9ca3af;cursor:not-allowed;transform:none}
  .uploadbtn{display:inline-block;padding:15px 34px;font-size:18px;background:#3b82f6;color:#fff;border-radius:8px;cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(59,130,246,.3);margin-right:10px;transition:.2s}
  .uploadbtn:hover{background:#2563eb;transform:translateY(-2px)}
  .loading{display:none;text-align:center;padding:34px;background:white;border-radius:12px;margin-bottom:26px}
  .loading.show{display:block}
  .spinner{border:4px solid #eee;border-top:4px solid #10b981;border-radius:50%;width:38px;height:38px;animation:spin 1s linear infinite;margin:0 auto 14px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .summary{background:white;padding:18px 22px;border-radius:12px;margin-bottom:26px;box-shadow:0 4px 12px rgba(0,0,0,.12);font-size:14px;color:#374151;line-height:1.6}
  .results{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:20px}
  .card{background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,.12)}
  .card.weak{opacity:.75}
  .head{padding:14px 16px;background:#f3f4f6;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center}
  .title{font-size:13px;font-weight:600;color:#374151;word-break:break-all}
  .pct{font-size:13px;font-weight:700;color:#fff;background:#10b981;padding:3px 9px;border-radius:20px;white-space:nowrap}
  .pct.weak{background:#f59e0b}
  .body{padding:16px}
  .pair{display:flex;gap:14px;align-items:center;justify-content:center}
  .col{text-align:center}
  .col img{width:128px;height:128px;border:2px solid #ddd;border-radius:8px;object-fit:cover;background:#fafafa}
  .lab{font-size:11px;color:#666;margin-top:6px}
  .arrow{font-size:22px;color:#10b981;font-weight:bold}
  .badge{font-size:10px;background:#6366f1;color:#fff;padding:2px 6px;border-radius:3px;display:inline-block;margin-top:3px}
  .nomatch{padding:44px 16px;text-align:center;color:#999}
</style></head><body>
<div class="container">
  <header>
    <h1>🎯 Image Matcher</h1>
    <p>Upload a ZIP of images — each is matched against the reference set (captures1) using a fused pipeline: ResNet18@128 + DINOv2 global descriptor + SIFT/RANSAC geometric verification. The <b>geo</b> badge = geometrically-consistent keypoint matches (higher = stronger physical evidence).</p>
  </header>
  <div class="bar">
    <label class="uploadbtn">📁 Upload ZIP
      <input type="file" id="file" accept=".zip" style="display:none" onchange="picked(event)">
    </label>
    <button id="btn" onclick="go()">▶ Start Matching</button>
    <div id="upinfo" style="margin-top:10px;font-size:13px;color:#eee"></div>
  </div>
  <div class="loading" id="load"><div class="spinner"></div><p>Embedding (ResNet18 + DINOv2) & geometric verification…</p></div>
  <div id="sum"></div>
  <div class="results" id="res"></div>
</div>
<script>
let chosenFile=null;
function picked(e){
  chosenFile=e.target.files[0];
  document.getElementById('upinfo').textContent = chosenFile
    ? `Selected: ${chosenFile.name} (${(chosenFile.size/1048576).toFixed(2)} MB) — click Start Matching`
    : '';
}
async function go(){
  const b=document.getElementById('btn');
  b.disabled=true;b.textContent='⏳ Running…';
  document.getElementById('load').classList.add('show');
  document.getElementById('sum').innerHTML='';document.getElementById('res').innerHTML='';
  try{
    // if a ZIP was chosen, upload it first
    if(chosenFile){
      const fd=new FormData(); fd.append('file',chosenFile);
      const up=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
      if(!up.success){alert('Upload failed: '+up.error);return;}
      document.getElementById('upinfo').textContent=`Uploaded ${up.count} images → resized to 128×128`;
    }
    const d=await (await fetch('/api/match',{method:'POST'})).json();
    if(d.error){alert('Error: '+d.error);return;}
    render(d.results);
  }catch(e){alert('Error: '+e.message);}
  finally{b.disabled=false;b.textContent='▶ Start Matching';document.getElementById('load').classList.remove('show');}
}
function render(rs){
  const conf=rs.filter(r=>r.confident).length;
  document.getElementById('sum').innerHTML=
    `<div class="summary"><b>✓ ${conf}/${rs.length} confident matches</b> — fused ResNet18@128 + DINOv2 + SIFT/RANSAC geometric verification, one-to-one (Hungarian) assignment. % = DINOv2 visual similarity; <b>geo</b> = RANSAC-verified keypoint correspondences (the confidence driver — ≥15 = confident).</div>`;
  document.getElementById('res').innerHTML=rs.map(r=>{
    if(!r.query) return `<div class="card"><div class="head"><span class="title">${r.reference}</span></div><div class="body"><div class="nomatch">no match</div></div></div>`;
    const wk=r.confident?'':'weak';
    return `<div class="card ${wk}">
      <div class="head"><span class="title">${r.reference}</span><span class="pct ${wk}">${r.similarity}% · geo ${r.geo}</span></div>
      <div class="body"><div class="pair">
        <div class="col"><img src="/img/${encodeURIComponent(r.ref_path)}"><div class="lab">Reference<br><span class="badge">${r.confident?'confident':'weak'}</span></div></div>
        <div class="arrow">→</div>
        <div class="col"><img src="/img/${encodeURIComponent(r.query_path)}"><div class="lab">${r.query}<br><span class="badge">geo ${r.geo}</span></div></div>
      </div></div></div>`;
  }).join('');
}
</script></body></html>
"""

if __name__ == "__main__":
    signals = "ResNet18@128" + (" + DINOv2" if USE_DINO else "") \
              + (" + SIFT-geom" if USE_GEOM else "")
    print("=" * 70)
    print("Seed2 Matcher — http://localhost:5000")
    print(f"Reference: {REF_FOLDER}   Captured: {QUERY_FOLDER}")
    print(f"Signals: {signals}   Device: {DEVICE}")
    print("=" * 70)
    app.run(debug=False, host="localhost", port=5000, threaded=True)
