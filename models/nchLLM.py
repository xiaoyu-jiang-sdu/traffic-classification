import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.llm_utils import get_llm_model
from layers.embedding import Embedding
from layers.reprogramming import ReprogrammingLayer
from utils.statistics_utils import get_link_name


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.num_labels = configs.num_labels
        self.label_names = configs.label_names
        self.categories_desc = configs.categories_desc
        assert self.num_labels == len(self.label_names)
        self.llm_model, self.tokenizer = get_llm_model(configs.llm_model, configs.llm_layers)

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = ('The USTC-TFC dataset contains various network traffic flows,'
                                'including both benign and malicious patterns.')

        self.categories_embed = None  # label_num * d_llm

        self.embedding = Embedding(
            d_model=configs.d_model, dropout=configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1024
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

    def get_label_cls(self):
        categories = []
        for i in range(self.num_labels):
            prompt_input = (
                f"<|start_prompt|>Dataset description: {self.description};"
                f"Task description: forecast the flow category in the given {self.num_labels} categories"
                f"<|<end_prompt>|>."
                f"distinguish the {i}th flow category {self.label_names[i]} feature "
            )
            categories.append(prompt_input)
        categories = self.tokenizer(categories, return_tensors="pt", padding=True,
                                    truncation=True, max_length=5000).input_ids
        categories_embeddings = self.llm_model.get_input_embeddings()(categories.to(self.llm_model.device))
        x = self.llm_model(inputs_embeds=categories_embeddings).last_hidden_state
        categories_embeddings = x[:, -1, :]
        categories_embeddings = F.normalize(categories_embeddings, p=2, dim=1)
        return categories_embeddings

    def update_categories(self):
        self.categories_embed = self.get_label_cls()

    def save_categories_embeddings(self, embedding_pt):
        torch.save(self.categories_embed, embedding_pt)

    def load_categories_embeddings(self, categories_embeddings):
        self.categories_embed = categories_embeddings

    def forward(self, headers, payloads, packet_num, link_type, duration):
        if self.categories_embed is None:
            self.update_categories()

        assert headers.device == payloads.device
        device = headers.device
        B, P, T = headers.size()
        _, _, L = payloads.size()
        headers, payloads = self.embedding(headers, payloads)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        headers = headers.view(B, P * T, -1)
        payloads = payloads.view(B, P * L, -1)
        headers = self.reprogramming_layer(headers, source_embeddings, source_embeddings)
        payloads = self.reprogramming_layer(payloads, source_embeddings, source_embeddings)
        headers = headers.view(B, P, T, -1)
        payloads = payloads.view(B, P, L, -1)

        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        hdr_token = torch.full((B, P, 1), pad_token_id, dtype=torch.long).to(device)
        pld_token = torch.full((B, P, 1), pad_token_id, dtype=torch.long).to(device)

        hdr_embeddings = self.llm_model.get_input_embeddings()(hdr_token).squeeze(2)  # B * P * N
        pld_embeddings = self.llm_model.get_input_embeddings()(pld_token).squeeze(2)  # B * P * N

        headers = torch.cat([headers, hdr_embeddings.unsqueeze(2)], dim=2)  # B * P * (T+1) * N
        payloads = torch.cat([payloads, pld_embeddings.unsqueeze(2)], dim=2)

        flow = []
        for i in range(P):
            flow.append(headers[:, i])  # B * (T+1) * N
            flow.append(payloads[:, i])  # B * (L+1) * N
        flow = torch.cat(flow, dim=1)  # B * (P*(T+L+2)) * N
        flow_token_num = (P * T + P) + (P * L + P) + 1

        prompt = []
        aim = []
        for ii in range(B):
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the flow category in the given {self.num_labels} categories"
                f"given {flow_token_num} tokens flow data; "
                "Input statistics: "
                f"total packet num: {packet_num}, "
                f"link type: {get_link_name(link_type[ii].item())}"
                f"flow duration: {duration[ii].item()} seconds <|<end_prompt>|>"
            )
            aim_ = 'predict the flow category'
            prompt.append(prompt_)
            aim.append(aim_)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        aim = self.tokenizer(aim, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # B * L * N
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(device))
        aim_embeddings = self.llm_model.get_input_embeddings()(aim.to(device))
        x = torch.cat([prompt_embeddings, flow, aim_embeddings], dim=1)
        x = self.llm_model(inputs_embeds=x).last_hidden_state
        flow_cls = x[:, -1, :]  # batch_size * d_llm

        scores = torch.matmul(flow_cls, self.categories_embed.T)

        return scores
