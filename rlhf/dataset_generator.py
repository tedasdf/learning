# dataset_generator.py
import random
import torch
from datasets import load_dataset
from utils import combined_reward, get_per_token_logps

class DatasetGenerator:
    def __init__(self, Q_batch_size=2, num_pre_Q=2, max_prompt_length=400, compute_gen_logps=True):
        self.Q_batch_size = Q_batch_size
        self.num_pre_Q = num_pre_Q
        self.max_prompt_length = max_prompt_length
        self.compute_gen_logps = compute_gen_logps
        self.system_prompt = "..."  # keep your system prompt here

        self.dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir="./dataset")
        self.QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(self.dataset['question'], self.dataset['answer'])]

    def gen_answers(self, prompts, tokenizer, gen_model, max_new_tokens=512):
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": x}],
                tokenize=False, add_generation_prompt=True
            ))

        tip_inputs = tokenizer(
            tip_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )
        prompt_length = tip_inputs["input_ids"].shape[1]
        if prompt_length > self.max_prompt_length:
            return []

        tip_inputs = {k: v.to(gen_model.device) for k, v in tip_inputs.items()}

        # Hugging Face generate
        with torch.inference_mode():
            tip_completion_ids = gen_model.generate(
                **tip_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=self.num_pre_Q
            )

        # Slice off prompt tokens
        completion_ids = tip_completion_ids[:, prompt_length:]
        
        answers = [tokenizer.decode(x, skip_special_tokens=True).replace('<|endoftext|>', '') for x in completion_ids]
        return answers

    def generate_samples(self, inputs, tokenizer, gen_model):
        prompts = [x["Q"] for x in inputs]
        answers = self.gen_answers(prompts, tokenizer, gen_model)
        if len(answers) == 0:
            return None, None, None, None

        # Compute rewards
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*self.num_pre_Q:(i+1)*self.num_pre_Q]:
                rewards.append(self.rew_func(inp, a))

        # Tokenize prompts and answers
        prompts_text = [tokenizer.apply_chat_template([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": x}],
            tokenize=False, add_generation_prompt=True) for x in prompts]

        prompt_inputs = tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )["input_ids"]

        output_ids = tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False
        )["input_ids"]

        return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers




    def generate_dataset(self, inputs, tokenizer, model ,ref_model , num= 10 , rank=0):
        data_list = []
        # if rank == 0: print(f'enter generate mode , {num}')
        tic = time.time()
        for ii in range(num):

            inputs = random.sample(self.QAs, self.Q_batch_size)
            

            prompt_inputs, output_ids , rewards , answer = self.generate_samples(inputs, tokenizer, model)
            
            if prompt_inputs is None: continue
            # if rank == 0: 
            #     print('rewards:', rewards)
            #     if ii == 5: print('answers:', answers[0])

            if (rewards.max() - rewards.min()).item() < 0.01: continue
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            rep = output_ids.shape[0] // prompt_inputs.shape[0]
            prompt_length = prompt_inputs.shape[1]
            Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            data = {
                'plen':prompt_length, 
                'inputs':merged_ids, 
                'rewards':rewards
            }       

            
            with torch.inference_mode():
                mids = merged_ids.to(model.device)
                if self.compute_gen_logps: 
                    gen_logps = get_per_token_logps(model(mids).logits[:, :-1, :], mids[:, 1:])
                ref_gen_logps = get_per_token_logps(ref_model(mids).logits[:, :-1, :], mids[:, 1:])
            data['ref'] = ref_gen_logps[:, prompt_length-1:]
            if self.compute_gen_logps:
                data['gen_logps'] = gen_logps[:, prompt_length-1:]
            
            data_list.append(data)
       
        return data_list   
