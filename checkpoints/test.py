import torch

# 加载整个模块对象（你原先保存的模型）
mapping_layer = torch.load("mapping_layer.pt", weights_only=False)
reprogramming_layer = torch.load("reprogramming_layer.pt", weights_only=False)
classifier = torch.load("classifier.pt", weights_only=False)

# 提取 state_dict 并重新保存
torch.save(mapping_layer.state_dict(), "mapping_layer.pt")
torch.save(reprogramming_layer.state_dict(), "reprogramming_layer.pt")
torch.save(classifier.state_dict(), "classifier.pt")
