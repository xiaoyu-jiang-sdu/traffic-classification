import torch.nn as nn
import torch
from utils.llm_utils import get_llm_model
from utils.statistics_utils import get_link_name


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_llm = configs.llm_dim
        self.num_labels = configs.num_labels
        self.label_names = configs.label_names
        self.categories_desc = configs.categories_desc
        assert self.num_labels == len(self.label_names)
        self.llm_model, self.tokenizer = get_llm_model(configs.llm_model, configs.llm_layers)
        for p in self.llm_model.parameters():
            p.requires_grad = False
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = ('The USTC-TFC dataset contains various network traffic flows,'
                                'including both benign and malicious patterns.')
        self.aim = 'predict the flow category'

    def forward(self, flow, packet_num, link_type, duration):
        B, _ = flow.shape
        device = flow.device
        with torch.no_grad():
            flow_embeddings = self.llm_model.get_input_embeddings()(flow.to(device))

            prompt = []
            aim = []
            for ii in range(B):
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: forecast the flow category in the given {self.num_labels} categories"
                    f"given the following 5 packet data truncated from a session flow; "
                    "Input statistics: "
                    f"total packet num: {packet_num}, "
                    f"link type: {get_link_name(link_type[ii].item())}"
                    f"flow duration: {duration[ii].item()} seconds <|<end_prompt>|>"
                )
                prompt.append(prompt_)
                aim.append(self.aim)
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    max_length=2048).input_ids
            aim = self.tokenizer(aim, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            # B * L * N
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(device))
            aim_embeddings = self.llm_model.get_input_embeddings()(aim.to(device))
            x = torch.cat([prompt_embeddings, flow_embeddings, aim_embeddings], dim=1)
            x = self.llm_model(inputs_embeds=x).last_hidden_state[:, -1, :]
            return x
