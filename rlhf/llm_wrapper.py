# llm_wrapper.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMWrapper(torch.nn.Module):
    def __init__(self, model_name, save_path="./model_cache"):
        super().__init__()
        self.model_name = model_name
        self.save_path = save_path
        self.tokenizer, self.model = self.load_model()

    def load_model(self):
        if not os.path.exists(self.save_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="bfloat16", _attn_implementation="sdpa"
            ).to("cuda")
            tokenizer.save_pretrained(self.save_path)
            model.save_pretrained(self.save_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.save_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.save_path, torch_dtype="bfloat16", _attn_implementation="sdpa"
            ).to("cuda")
        return tokenizer, model

    def generate(self, prompts, max_new_tokens=512, temperature=0.7, top_p=0.9, num_return_sequences=1):
        tip_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **tip_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=num_return_sequences
            )
        return outputs

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
