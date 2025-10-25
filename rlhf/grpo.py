# grpo.py
import torch
from torch import nn
from utils import get_per_token_logps

class GRPO(nn.Module):
    def __init__(self, llm, clip_param=0.2, kl_coef=0.4):
        super().__init__()
        self.clip_param = clip_param
        self.engine = llm.model
        self.tokenizer = llm.tokenizer
        self.grpo_kl_coefficient = kl_coef

    def forward(self, batch):
        prompt_length = batch['plen']
        inputs = batch['inputs'].to(self.engine.device)
        advantages = batch['rewards'].to(self.engine.device).unsqueeze(1)

        logits = self.engine(inputs).logits[:, :-1, :]
        input_ids = inputs[:, 1:]
        per_token_logps = get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length-1:]
        ref_per_token_logps = batch['ref'].to(per_token_logps.device)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completion_mask = (inputs[:, prompt_length:] != self.tokenizer.pad_token_id).int()

        if 'gen_logps' in batch:
            ratio = torch.exp(per_token_logps - batch['gen_logps'].to(self.engine.device))
            clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
            per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages

        per_token_loss = -(per_token_loss - self.grpo_kl_coefficient * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss
