import torch
import torchvision
import time
import argparse
import yaml
import importlib
import random
import numpy as np



seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)



# Kiểm tra xem GPU có khả dụng không
device = "cuda"
print("Device:", device)

# Đường dẫn đến tệp YAML
yaml_file = "/home/bigdata/Documents/TND_Modeling/config.yaml"

# Đọc tệp YAML
with open(yaml_file, "r") as file:
    yaml_data = yaml.safe_load(file)

# Chuyển đổi dữ liệu YAML thành đối tượng namespace
args = argparse.Namespace(**yaml_data)


# Chọn mô hình
model_mapping = {
    "Attentionunet": "Models.Attentionunet",
    "Doubleunet": "Models.Doubleunet",
    "Fcn": "Models.Fcn",
    "Unext": "Models.Unext",
    "Unet": "Models.Unet",
    "Vapenet": "Models.Vapenet",
}

model_module = importlib.import_module(model_mapping[args.model_name["name"]])
model = getattr(model_module, args.model_name["version"])()


model.to(device)
model.eval()

# Tạo dữ liệu ngẫu nhiên để đưa qua mô hình
input_size = (1, 3, 256, 256)
input_data = torch.randn(*input_size).to(device)



# Đo FPS
num_frames = 500  # Số lượng khung hình để đo FPS
total_time = 0.0

with torch.no_grad():
    for i in range(num_frames):
        start_time = time.time()   # Ghi lại thời điểm bắt đầu

        # Chạy mô hình trên GPU
        output = model(input_data)

        end_time = time.time()  # Ghi lại thời điểm kết thúc
        
        # Tính thời gian và cộng dồn để tính tổng thời gian thực hiện
        elapsed_time = end_time - start_time
        total_time += elapsed_time

# Tính FPS
average_time = total_time / num_frames
fps = 1.0 / average_time  # Chuyển từ milliseconds sang seconds

print("Average FPS: {:.2f}".format(fps))