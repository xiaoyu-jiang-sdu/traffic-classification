from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoTokenizer, AutoModel


def get_llm_model(llm_model, llm_layers, cache_dir=None):
    if llm_model == 'LLAMA':
        # llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        llama_config.num_hidden_layers = llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        try:
            llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=llama_config,
                cache_dir=cache_dir
                # load_in_4bit=True
            )
        except EnvironmentError:  # downloads models from HF is not already done
            print("Local models files not found. Attempting to download...")
            llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
                config=llama_config,
                cache_dir=cache_dir
                # load_in_4bit=True
            )
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.models",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.models",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=cache_dir
            )
    elif llm_model == 'GPT2':
        gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

        gpt2_config.num_hidden_layers = llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        try:
            llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=gpt2_config,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # downloads models from HF is not already done
            print("Local models files not found. Attempting to download...")
            llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=gpt2_config,
                cache_dir=cache_dir
            )

        try:
            tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=cache_dir
            )
    elif llm_model == 'BERT':
        bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

        bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        try:
            llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=True,
                config=bert_config,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # downloads models from HF is not already done
            print("Local models files not found. Attempting to download...")
            llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=bert_config,
                cache_dir=cache_dir
            )

        try:
            tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=cache_dir
            )
    elif llm_model == 'DEEPSEEK':
        # 使用模型名称加载配置
        # .cache/huggingface/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        config.num_hidden_layers = llm_layers
        config.output_attentions = True
        config.output_hidden_states = True

        try:
            # 加载模型
            llm_model = AutoModel.from_pretrained(
                # "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=True,
                config=config,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # 如果本地没有模型文件，则尝试下载
            print("Local model files not found. Attempting to download...")
            llm_model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=False,
                config=config,
                cache_dir=cache_dir
            )

        try:
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                # "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
        except EnvironmentError:  # 如果本地没有 tokenizer 文件，则尝试下载
            print("Local tokenizer files not found. Attempting to download...")
            tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=False,
                cache_dir=cache_dir
            )
    else:
        raise Exception('LLM models is not defined')

    return llm_model, tokenizer
