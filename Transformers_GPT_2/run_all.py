# This code was used to train the models and generate samples from Transformers GPT-2 models used in our paper:
# 'Training GPT-2 to represent two Romantic-era authors challenges, evaluations and pitfalls'.

import os
import shutil

# seed_start and seed_end set the range of samples being generated from your models:
seed_start = 1
seed_end = 3

# ckpt sets the first checkpoint where the model will be saved and the samples will be generated from:
ckpt = 100

#steps_increment sets the increment of steps between checkpoints that are saved and generated from:
steps_increment = 100

# sample_length sets the length of the samples generated (maximum is 1024, but if you are using prompt, 
# you will have to deduct the prompt size from this number):
sample_length = 100

# model_descr can be anything, it is used only to name your samples: 
model_descr = "Shelley"

## model_size has can be:
# GPT-2 Small:
model_size = "gpt2"
## GPT-2 Medium:
# model_size = "gpt2-medium"
## GPT-2 Large (to fine-tune GPT-2 Large you will need T4 or P100 GPU):
# model_size = "gpt2-large"

# Dataset for training without validation:
input_name = "input_data/Shelley.txt"

## Datasets for training with validation:
input_train = "input_data/Shelley_train.txt"
input_val = "input_data/Shelley_train.txt"

prompt =  'The eternal sky '

prompt = prompt.replace(" ", "$$$$$")

for i in range (1,11):

    # FINE-TUNING THE MODELS
    if i == 1:

        os.system('python run_lm_finetuning.py --output_dir=output  --model_type=gpt2 --model_name_or_path={model_size} --do_train --train_data_file {input_name} --overwrite_output_dir --block_size=200 --per_gpu_train_batch_size=1 --save_steps {ckpt} --max_steps {ckpt} --save_total_limit 1 --num_train_epochs=50000 --logging_steps=50'.format(model_size=model_size, input_name= input_name, ckpt=ckpt))

        path = 'input_data/output'
        if not os.path.exists(path):
            os.makedirs(path)
            
        source_file = "input_data/gpt2_cached_lm_200_Shelley.txt"
        target_file = "input_data/output/checkpoint-100_gpt2_cached_lm_200_Shelley.txt"

        shutil.copy(source_file, target_file)
        print("source_file: ", source_file)
        print("target_file: ", target_file)
        #####################################################################

    else:
    
        ckpt = ckpt + steps_increment
        last_ckpt = ckpt - steps_increment

        os.system('python run_lm_finetuning.py --output_dir=output  --model_type=gpt2 --model_name_or_path=output/checkpoint-{last_ckpt} --do_train --train_data_file {input_name} --overwrite_output_dir --block_size=200 --per_gpu_train_batch_size=1 --save_steps {ckpt} --max_steps {ckpt} --save_total_limit 100 --num_train_epochs=50000 --logging_steps=50'.format(last_ckpt=last_ckpt, input_name= input_name, ckpt=ckpt))
        
    for item in os.listdir("output"):
        to_remove  = "output/" + item    
        if not os.path.isdir(to_remove):
            os.remove(to_remove)
            
    # GENERATING THE SAMPLES:
    for s in range(seed_start, seed_end):
        prompt = prompt.replace(" ", "£££££") 

        os.system('python generate_conditional_samples_to_file.py --model_type gpt2 --temperature 1.0 --top_k 50 --top_p 1.0 --model_name_or_path output/checkpoint-{ckpt} --length {sample_length}  --prompt {prompt} --model_descr {model_descr} --seed {s} --num_samples 1'.format(ckpt=ckpt, sample_length=sample_length, prompt=prompt, model_descr=model_descr,s=s)) 

    # This code will remove the fine-tuned models that finished generation:
    # (comment out if you want to keep them):
    if i > 1:
        shutil.rmtree('output/checkpoint-{}'.format(str(last_ckpt)))






