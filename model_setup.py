# model_setup.py
import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig, LlamaForCausalLM, LlamaTokenizer

def setup_time_series_transformer():
    config = TimeSeriesTransformerConfig(
        prediction_length=10,
        context_length=50,
        # Other parameters as needed
    )
    return TimeSeriesTransformerModel(config)

def setup_llama():
    model = LlamaForCausalLM.from_pretrained("path_to_llama_3_weights")
    tokenizer = LlamaTokenizer.from_pretrained("path_to_llama_3_tokenizer")
    return model, tokenizer

ts_model = setup_time_series_transformer()
llama_model, llama_tokenizer = setup_llama()
