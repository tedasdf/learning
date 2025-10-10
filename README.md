link:
https://arxiv.org/pdf/2505.09388
https://arxiv.org/pdf/2305.13245
https://arxiv.org/pdf/2412.15115
https://arxiv.org/pdf/2505.09388
https://arxiv.org/pdf/2505.09388
https://arxiv.org/pdf/2104.09864
https://arxiv.org/abs/2407.06483
https://github.com/orgs/EleutherAI/repositories
https://www.alphaxiv.org/abs/2501.12948v1


sota-ai/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/                           # training configs (YAML)
│   ├── llm/
│   │   ├── gpt2_base.yaml
│   │   ├── llama2.yaml
│   │   └── qwen.yaml
│   ├── vision/
│   │   ├── vit.yaml
│   │   ├── sam.yaml
│   │   └── sd.yaml
│   └── trainer.yaml
│
├── src/
│   ├── llm/
│   │   ├── architectures/
│   │   │   ├── transformer.py
│   │   │   ├── gpt2.py
│   │   │   ├── llama2.py
│   │   │   └── qwen.py
│   │   ├── training/
│   │   │   ├── dataset.py
│   │   │   ├── optimizer.py
│   │   │   ├── scheduler.py
│   │   │   ├── trainer.py
│   │   │   └── loss_functions.py
│   │   ├── reasoning/
│   │   │   ├── react.py
│   │   │   ├── reflexion.py
│   │   │   └── lats.py
│   │   └── utils/
│   │       ├── tokenization.py
│   │       ├── model_utils.py
│   │       └── config_parser.py
│   │
│   ├── vision/
│   │   ├── architectures/
│   │   │   ├── vit.py
│   │   │   ├── convnext.py
│   │   │   ├── sam.py
│   │   │   └── unet.py
│   │   ├── training/
│   │   │   ├── dataset.py
│   │   │   ├── augmentations.py
│   │   │   ├── loss_functions.py
│   │   │   ├── optimizer.py
│   │   │   └── trainer.py
│   │   └── utils/
│   │       ├── evaluation.py
│   │       ├── visualization.py
│   │       └── config_parser.py
│   │
│   ├── applications/                  # optional later
│   │   ├── autoencoder/
│   │   ├── diffusion/
│   │   └── multimodal/
│   │
│   └── core/
│       ├── base_model.py              # abstract Model class
│       ├── base_trainer.py            # abstract Trainer class
│       └── registry.py                # for registering models
│
├── experiments/
│   ├── llm_pretraining/
│   │   ├── run_gpt2.py
│   │   ├── run_llama2.py
│   │   └── run_qwen.py
│   ├── vision_training/
│   │   ├── run_vit.py
│   │   ├── run_sam.py
│   │   └── run_sd.py
│   └── logs/
│
├── notebooks/
│   ├── visualize_attention.ipynb
│   ├── train_gpt2_from_scratch.ipynb
│   └── train_vit_on_cifar10.ipynb
│
├── tests/
│   ├── test_transformer.py
│   ├── test_trainer.py
│   └── test_loss.py
│
└── .github/
    └── workflows/
        ├── test.yml
        ├── lint.yml
        └── docs.yml
