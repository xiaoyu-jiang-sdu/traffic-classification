import json

from transformers import AutoTokenizer

with open('./mapper/USTC-TFC.json', 'r') as jsonData:
    mapper = json.load(jsonData)

label_names = list(mapper.keys())
aim = f'given the categories  predict the flow category'

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    # "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True,
    local_files_only=True,
)

class A:
    def __init__(self):
        self.label_names = list(mapper.keys())
        self.tokenizer = tokenizer
        self.aim = f'given the categories {"".join(self.label_names)}, predict the flow category'
        encoded = self.tokenizer(self.aim, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        self.n_seq = encoded.input_ids.shape[1]

a = A()
print(a.n_seq)