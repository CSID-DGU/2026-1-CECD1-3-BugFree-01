import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import time
import psutil

# CPU 강제
device = torch.device("cpu")

print("📦 Loading ViT-B/16 model... (heavy model)")

weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.eval()
model.to(device)

# preprocessing
preprocess = weights.transforms()

# dummy image
img = Image.new("RGB", (224, 224), color="white")
x = preprocess(img).unsqueeze(0)

print("🚀 Start inference benchmark")

# warmup
with torch.no_grad():
    _ = model(x)

# benchmark loop
for i in range(100):
    start = time.time()

    with torch.no_grad():
        out = model(x)

    end = time.time()

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent

    print(f"Iter {i:03d} | time: {end-start:.3f}s | CPU: {cpu}% | RAM: {mem}%")

print("✅ Done")