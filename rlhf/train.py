# train.py
import torch, random, os
import wandb
from llm_wrapper import LLMWrapper
from dataset_generator import DatasetGenerator
from grpo import GRPO

NUM_EPOCHS = 10
SAVE_EVERY = 1
LR = 1e-6
MODEL_NAME = "Qwen/Qwen3-0.6B"
SAVE_DIR = "./checkpoint"

wandb.init(project="grpo_rlhf", name="qwen3_training", config={"model": MODEL_NAME, "batch_size": 1, "lr": LR, "num_epochs": NUM_EPOCHS})

gen_model = LLMWrapper(MODEL_NAME)
ref_model = LLMWrapper(MODEL_NAME)
dataset_gen = DatasetGenerator()
trainer = GRPO(gen_model)
optimizer = torch.optim.AdamW(gen_model.model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    inputs = random.sample(dataset_gen.QAs, dataset_gen.Q_batch_size)
    data_list = dataset_gen.generate_dataset(inputs, gen_model.tokenizer, gen_model.model, ref_model.model, num=2)

    total_loss = 0
    for i, data in enumerate(data_list):
        optimizer.zero_grad()
        loss = trainer(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        print(f"[Epoch {epoch}] step {i} | loss = {loss.item():.4f}")

    avg_loss = total_loss / len(data_list)
    print(f"Epoch {epoch} complete | avg_loss = {avg_loss:.4f}")
    wandb.log({"avg_epoch_loss": avg_loss, "epoch": epoch})

    if (epoch + 1) % SAVE_EVERY == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        gen_model.model.save_pretrained(SAVE_DIR)
        gen_model.tokenizer.save_pretrained(SAVE_DIR)
        wandb.save(os.path.join(SAVE_DIR, "*"))
