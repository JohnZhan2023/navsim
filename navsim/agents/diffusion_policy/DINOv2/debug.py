from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

# 加载图像
url = 'https://th.bing.com/th/id/R.4f326e2967cea1af408cf6ee5f512446?rik=9N4iTV9fzcnlyg&riu=http%3a%2f%2fpic1.bbzhi.com%2fdongwubizhi%2fjiayouyouquan-keaixiaogoubizhi%2fanimal_sz237_lovely_puppy_31132_5.jpg&ehk=CAu1WJ8wFqH0oecq45Q7fof84doNuLOC8zOIkHs2zsY%3d&risl=&pid=ImgRaw&r=0'
image = Image.open(requests.get(url, stream=True).raw)

# 加载图像处理器和模型
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# 处理图像并获取模型输出
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 提取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state

# 打印输出
print(last_hidden_states.shape)
