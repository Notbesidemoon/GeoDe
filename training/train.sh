# Requirements:
# please use python=3.10/3.11, cuda12.*
# sh requirements/install_all.sh
# pip install the following packages:
# torch 2.8.0
# pip install "vllm>=0.5.1,<0.11.1" -U
# pip install "transformers<4.58" "trl<0.25" peft -U
# pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
# pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
# pip install timm "deepspeed<0.18" -U
# pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases



# Arguments:
# - model_path: path to the model
# - input_data_path: path to the input data
# - output_dir: path to the output directory

 

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model model_path \
    --train_type full \
    --dataset input_data_path \
    --output_dir output_dir \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --save_strategy epoch \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 1 \
    --save_only_model true \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --report_to  wandb \