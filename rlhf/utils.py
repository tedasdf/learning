# utils.py
import torch
import re
from math_verify import parse, verify, ExprExtractionConfig

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def reward_correct(item, answer):
    pattern_num = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern_num, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

def reward_format(answer):
    pattern_format = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern_format, answer, re.DOTALL | re.VERBOSE) else -1

def combined_reward(item, answer):
    return reward_correct(item, answer) + reward_format(answer)
