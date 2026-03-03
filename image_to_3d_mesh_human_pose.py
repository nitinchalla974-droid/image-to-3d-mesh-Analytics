
# %cd /content
!git clone https://github.com/facebookresearch/sam-3d-body.git
# %cd /content/sam-3d-body

!pip -q install -U pip

# Core
!pip -q install "numpy>=2.0" opencv-python-headless tqdm

!pip -q install \
  pyrootutils omegaconf huggingface_hub loguru yacs \
  pytorch-lightning pyrender scikit-image einops timm dill pandas rich \
  hydra-core hydra-submitit-launcher hydra-colorlog webdataset \
  chump networkx roma joblib seaborn wandb appdirs cython jsonlines \
  pytest optree fvcore pycocotools tensorboard

!pip uninstall -y iopath
!pip install "iopath==0.1.9"

!pip -q install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Installing the deps for the need
!pip -q install black "fvcore>=0.1.5,<0.1.6" "hydra-core>=1.1" "yacs>=0.1.8"

#  iopath < 0.1.10 for Detectron2
!pip -q install "iopath==0.1.9"

import numpy as np, cv2, torch
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

from huggingface_hub import logout, login

logout()
login()

from huggingface_hub import snapshot_download
import os, shutil

HF_REPO = "facebook/sam-3d-body-dinov3"   # accepted access
LOCAL_DIR = "/content/checkpoints/sam-3d-body"

# removing partial downloads
shutil.rmtree(LOCAL_DIR, ignore_errors=True)
os.makedirs(LOCAL_DIR, exist_ok=True)

snapshot_download(
    repo_id=HF_REPO,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
)

# Verify the existence of required files
req = {
    "model.ckpt": os.path.join(LOCAL_DIR, "model.ckpt"),
    "model_config.yaml": os.path.join(LOCAL_DIR, "model_config.yaml"),
    "assets/mhr_model.pt": os.path.join(LOCAL_DIR, "assets", "mhr_model.pt"),
}

print("=== Required files ===")
for k, p in req.items():
    print(k, "->", p, "exists:", os.path.exists(p))

missing = [k for k, p in req.items() if not os.path.exists(p)]
if missing:
    raise RuntimeError(f"Missing required files: {missing}.")

print("\n=== Found ckpt/yaml/pt files ===")
os.system(f'find {LOCAL_DIR} -maxdepth 3 -type f \\( -name "*.ckpt" -o -name "*.yaml" -o -name "*.pt" \\) -print')

from google.colab import files
import os, shutil

os.makedirs("/content/images", exist_ok=True)

uploaded = files.upload()
for fn in uploaded.keys():
    shutil.move(fn, f"/content/images/{fn}")

print("Images:", os.listdir("/content/images"))

!pip -q install trimesh

import os, inspect, importlib, pkgutil, shutil
import numpy as np
import torch
import trimesh

import sam_3d_body
from sam_3d_body.build_models import load_sam_3d_body

# -------------------------
# Paths
# -------------------------
IMG_DIR = "/content/images"
CKPT_DIR = "/content/checkpoints/sam-3d-body"

img_path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
checkpoint_path = os.path.join(CKPT_DIR, "model.ckpt")
config_path = os.path.join(CKPT_DIR, "model_config.yaml")
mhr_path = os.path.join(CKPT_DIR, "assets", "mhr_model.pt")

assert os.path.exists(img_path), f"Missing image: {img_path}"
assert os.path.exists(checkpoint_path), f"Missing ckpt: {checkpoint_path}"
assert os.path.exists(config_path), f"Missing config: {config_path}"
assert os.path.exists(mhr_path), f"Missing mhr: {mhr_path}"


# 1) Find SAM3DBodyEstimator

SAM3DBodyEstimator = None
found_in = None

for m in pkgutil.walk_packages(sam_3d_body.__path__, prefix=sam_3d_body.__name__ + "."):
    try:
        mod = importlib.import_module(m.name)
    except Exception:
        continue
    if hasattr(mod, "SAM3DBodyEstimator"):
        SAM3DBodyEstimator = getattr(mod, "SAM3DBodyEstimator")
        found_in = m.name
        break

if SAM3DBodyEstimator is None:
    raise ModuleNotFoundError("Could not find SAM3DBodyEstimator in sam_3d_body package.")

print("✅ Found SAM3DBodyEstimator in:", found_in)


# 2) Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

sig = inspect.signature(load_sam_3d_body)
params = sig.parameters

# Base kwargs we *might* pass
kwargs = {}
if "device" in params:
    kwargs["device"] = device
if "mhr_path" in params:
    kwargs["mhr_path"] = mhr_path

# Trying to pass the config via any supported arg name; otherwise copy to default expected location
config_arg_candidates = ["model_cfg", "model_cfg_file", "config", "config_path", "config_file",
                         "model_config", "model_config_path", "cfg", "cfg_file"]
used_config_arg = None
for cand in config_arg_candidates:
    if cand in params:
        kwargs[cand] = config_path
        used_config_arg = cand
        break

if used_config_arg:
    print("✅ Passing config via:", used_config_arg)
else:
    # Many versions expect it at a default path; your earlier error showed /content/checkpoints/model_config.yaml
    os.makedirs("/content/checkpoints", exist_ok=True)
    default_expected = "/content/checkpoints/model_config.yaml"
    shutil.copy(config_path, default_expected)
    print("✅ load_sam_3d_body has no config arg; copied config to:", default_expected)

# Call load_sam_3d_body with either checkpoint_path kwarg or positional
try:
    if "checkpoint_path" in params:
        kwargs["checkpoint_path"] = checkpoint_path
        model, model_cfg = load_sam_3d_body(**kwargs)
    else:
        model, model_cfg = load_sam_3d_body(checkpoint_path, **kwargs)
    print("✅ Model loaded")
except TypeError as e:
    print("❌ Failed calling load_sam_3d_body.")
    print("Signature:", sig)
    print("Tried kwargs:", kwargs)
    raise

# 3) Build estimator

init_sig = inspect.signature(SAM3DBodyEstimator.__init__)
init_params = init_sig.parameters

est_kwargs = {}
# fill only supported names
for name in init_params.keys():
    if name == "self":
        continue
    if name in ("sam_3d_body_model", "model"):
        est_kwargs[name] = model
    elif name in ("model_cfg", "cfg"):
        est_kwargs[name] = model_cfg

estimator = SAM3DBodyEstimator(**est_kwargs)
print("✅ Estimator ready")

import os
import numpy as np
import trimesh

# 4) Run inference
outputs = estimator.process_one_image(img_path)
print("✅ Inference done. Output type:", type(outputs))
print("Number of people detected:", len(outputs) if isinstance(outputs, list) else 1)

# 5) Extract verts + faces

if isinstance(outputs, list):
    if len(outputs) == 0:
        raise RuntimeError("No person detected in the image!")
    result = outputs[0]
else:
    result = outputs

# Correct key for vertices
verts = result['pred_vertices']

# Faces come from the estimator (standard for this model)
if hasattr(estimator, 'faces'):
    faces = estimator.faces
elif hasattr(estimator, 'mesh_faces'):
    faces = estimator.mesh_faces
else:
    raise AttributeError("Could not find faces on estimator. Check estimator.__dict__.keys()")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

verts = to_numpy(verts)
faces = to_numpy(faces)

print("Vertices shape:", verts.shape)
print("Faces shape:", faces.shape)


# 6) Export OBJ

os.makedirs("/content/output_mesh", exist_ok=True)
obj_path = "/content/output_mesh/avatar.obj"

mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.export(obj_path)

print("✅ Mesh successfully saved:", obj_path)
print("Download it with:")
from google.colab import files
files.download(obj_path)
