import json
from collections import OrderedDict

import torch

index_file_path = "./merge_ckpt/pytorch_model.bin.index.json"

with open(index_file_path, "r") as f:
    index_data = json.load(f)

shard_files = index_data["weight_map"].values()
state_dict = OrderedDict()
for shard_file in set(shard_files):
    shard_path = f"./merge_ckpt/{shard_file}"
    shard_state_dict = torch.load(shard_path, map_location="cpu")
    state_dict.update(shard_state_dict)

embed_keys = [k for k in state_dict.keys() if "embedding" in k]
embed_state_dict = {k: state_dict[k] for k in embed_keys}

print(embed_state_dict)