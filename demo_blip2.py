# 模型保存路径：C:\Users\<你的用户名>\.cache\huggingface\    未来可以删除


import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

# 设置设备：优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和预处理器
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",  # 使用带 T5 的模型
    model_type="pretrain_flant5xl",  # 零样本模型
    is_eval=True,
    device=device
)

# 载入并预处理图片
raw_image = Image.open("./img/BlueUp2.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 构造 prompt 并生成描述
prompt = "Question: What color is the block below? ? Answer:"
output = model.generate({"image": image, "prompt": prompt})
print("模型输出：", output)
