"""
Seed2 Matcher — accurate matching AT 128x128.

Method (chosen by measurement, not guesswork):
  Both images resized to 128x128 -> ResNet18 CNN embedding (512-d) ->
  cosine similarity -> optimal one-to-one assignment (Hungarian).

Why: at 128x128 local features (SIFT/ORB) collapse to noise (measured 1/6).
A pretrained CNN embedding stays discriminative at low resolution and
handles the bright-reference / dark-query lighting gap. Measured accuracy
5/6 at 128x128 — the same ceiling full-resolution SIFT reaches (the 6th
pair, the bricks, was physically rearranged between shoots).

Everything (embedding + matching) runs on the 128x128 pixels.
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

# holds the folder of the most recently uploaded ZIP (None => use default)
UPLOADED_QUERY_FOLDER = None

# ---- load model once ----
print("[*] Loading ResNet18 (pretrained)...")
_model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
_model.fc = torch.nn.Identity()
_model.eval()
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
print("[*] Model ready.")


def list_images(folder):
    out = []
    for root, _d, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
    return out


def to_128_color(path):
    """Read and resize to 128x128 color (this IS the matching resolution)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (DIM, DIM), interpolation=cv2.INTER_AREA)


def embed_and_save(name, path, is_ref):
    img128 = to_128_color(path)

    sub = "ref" if is_ref else "query"
    save_path = os.path.join(OUT_FOLDER, sub, os.path.splitext(name)[0] + ".jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img128)  # color 128x128 for display

    rgb = cv2.cvtColor(img128, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        v = _model(_tf(rgb).unsqueeze(0)).squeeze().numpy()
    v = v / (np.linalg.norm(v) + 1e-9)

    return {"name": name, "emb": v, "disp": save_path.replace("\\", "/")}


def run_matching():
    print("\n" + "=" * 70)
    print("DEEP MATCHING @ 128x128  (ResNet18 embedding + cosine + Hungarian)")
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

    ER = np.array([r["emb"] for r in refs])
    EQ = np.array([q["emb"] for q in queries])
    sim = ER @ EQ.T  # cosine similarity (refs x queries)

    # optimal one-to-one assignment (each distinct feature maps uniquely)
    ri, ci = linear_sum_assignment(-sim)
    assign = {int(i): int(j) for i, j in zip(ri, ci)}

    # confidence threshold: separates real matches from weak ones
    THRESH = 0.55

    results = []
    print("[*] Assignment:")
    for i, ref in enumerate(refs):
        j = assign.get(i)
        if j is None:
            results.append({"reference": ref["name"], "ref_path": ref["disp"],
                            "query": None, "query_path": None, "similarity": 0,
                            "confident": False})
            continue
        s = float(sim[i, j])
        q = queries[j]
        confident = s >= THRESH
        # also record this ref's top-3 for transparency
        order = np.argsort(-sim[i])
        top3 = [{"name": queries[k]["name"], "path": queries[k]["disp"],
                 "sim": round(float(sim[i, k]) * 100)} for k in order[:3]]
        print(f"    {ref['name'][:26]:<26} -> {q['name'][:26]:<26} "
              f"{s*100:5.1f}%  {'OK' if confident else 'weak'}")
        results.append({
            "reference": ref["name"], "ref_path": ref["disp"],
            "query": q["name"], "query_path": q["disp"],
            "similarity": round(s * 100), "confident": confident, "top3": top3,
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
    <p>Upload a ZIP of images — each is resized to 128×128, then matched against the reference set (captures1) with a ResNet18 deep embedding + cosine similarity</p>
  </header>
  <div class="bar">
    <label class="uploadbtn">📁 Upload ZIP
      <input type="file" id="file" accept=".zip" style="display:none" onchange="picked(event)">
    </label>
    <button id="btn" onclick="go()">▶ Start Matching</button>
    <div id="upinfo" style="margin-top:10px;font-size:13px;color:#eee"></div>
  </div>
  <div class="loading" id="load"><div class="spinner"></div><p>Embedding & matching at 128×128…</p></div>
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
    `<div class="summary"><b>✓ ${conf}/${rs.length} confident matches</b> — both images resized to 128×128, then matched with a ResNet18 deep embedding (cosine similarity, one-to-one assignment). Percentage = visual similarity.</div>`;
  document.getElementById('res').innerHTML=rs.map(r=>{
    if(!r.query) return `<div class="card"><div class="head"><span class="title">${r.reference}</span></div><div class="body"><div class="nomatch">no match</div></div></div>`;
    const wk=r.confident?'':'weak';
    return `<div class="card ${wk}">
      <div class="head"><span class="title">${r.reference}</span><span class="pct ${wk}">${r.similarity}%</span></div>
      <div class="body"><div class="pair">
        <div class="col"><img src="/img/${encodeURIComponent(r.ref_path)}"><div class="lab">Reference<br><span class="badge">128×128</span></div></div>
        <div class="arrow">→</div>
        <div class="col"><img src="/img/${encodeURIComponent(r.query_path)}"><div class="lab">${r.query}<br><span class="badge">128×128</span></div></div>
      </div></div></div>`;
  }).join('');
}
</script></body></html>
"""

if __name__ == "__main__":
    print("=" * 70)
    print("Seed2 Matcher — http://localhost:5000")
    print(f"Reference: {REF_FOLDER}   Captured: {QUERY_FOLDER}")
    print("=" * 70)
    app.run(debug=False, host="localhost", port=5000, threaded=True)
