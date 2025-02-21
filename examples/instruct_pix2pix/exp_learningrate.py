import subprocess

exp1 = [
    "accelerate", "launch", "--mixed_precision=fp16", "./train_instruct_pix2pix.py",
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
    "--dataset_name=eve25yan/cup_openMV_dirty",
    "--enable_xformers_memory_efficient_attention",
    "--resolution=256", "--random_flip", "--train_batch_size=2",
    "--gradient_accumulation_steps=8", "--gradient_checkpointing",
    "--max_train_steps=500", "--checkpointing_steps=25",
    "--checkpoints_total_limit=1", "--learning_rate=5e-06",
    "--max_grad_norm=1", "--lr_warmup_steps=20",
    "--conditioning_dropout_prob=0.1", "--mixed_precision=fp16",
    "--seed=42", "--report_to=tensorboard",
    "--output_dir=C:/Users/mobil/Desktop/25spring/stylePalm/diffusers/dataset_cup/experiments/exp_prompts/dirty",
    "--original_image_column=before", "--edit_prompt=prompt",
    "--edited_image=after"
]

# exp2 = [
#     "accelerate", "launch", "--mixed_precision=fp16", "./train_instruct_pix2pix.py",
#     "--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
#     "--dataset_name=eve25yan/cup_openMV_prompts",
#     "--enable_xformers_memory_efficient_attention",
#     "--resolution=256", "--random_flip", "--train_batch_size=2",
#     "--gradient_accumulation_steps=8", "--gradient_checkpointing",
#     "--max_train_steps=1000", "--checkpointing_steps=25",
#     "--checkpoints_total_limit=1", "--learning_rate=5e-06",
#     "--max_grad_norm=1", "--lr_warmup_steps=20",
#     "--conditioning_dropout_prob=0.1", "--mixed_precision=fp16",
#     "--seed=42", "--report_to=tensorboard",
#     "--output_dir=C:/Users/mobil/Desktop/25spring/stylePalm/diffusers/dataset_cup/experiments/exp_prompts/v1",
#     "--original_image_column=before", "--edit_prompt=prompt",
#     "--edited_image=after"
# ]

# exp3 = [
#     "accelerate", "launch", "--mixed_precision=fp16", "./train_instruct_pix2pix.py",
#     "--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
#     "--dataset_name=eve25yan/cup_openMV_prompts_none",
#     "--enable_xformers_memory_efficient_attention",
#     "--resolution=256", "--random_flip", "--train_batch_size=2",
#     "--gradient_accumulation_steps=4", "--gradient_checkpointing",
#     "--max_train_steps=300", "--checkpointing_steps=25",
#     "--checkpoints_total_limit=1", "--learning_rate=5e-06",
#     "--max_grad_norm=1", "--lr_warmup_steps=20",
#     "--conditioning_dropout_prob=0.1", "--mixed_precision=fp16",
#     "--seed=42", "--report_to=tensorboard",
#     "--output_dir=C:/Users/mobil/Desktop/25spring/stylePalm/diffusers/exp_lr/5e-6",
#     "--original_image_column=before", "--edit_prompt=prompt",
#     "--edited_image=after"
# ]

# Run commands sequentially
subprocess.run(exp1, check=True)
# subprocess.run(exp2, check=True)
# subprocess.run(exp3, check=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            