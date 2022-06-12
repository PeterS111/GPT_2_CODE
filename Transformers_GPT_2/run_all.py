# This code was used to train the models and generate samples from Transformers GPT-2 models used in our paper:
# 'Training GPT-2 to represent two Romantic-era authors challenges, evaluations and pitfalls'.

import os
import shutil

# seed_start and seed_end set the range of samples being generated from your models:
seed_start = 1
seed_end = 11

# ckpt sets the first checkpoint where the model will be saved and the samples will be generated from:
ckpt = 1000

#steps_increment sets the increment of steps between checkpoints that are saved and generated from:
steps_increment = 1000

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


if model_size =="gpt2":
    source = "input_data/gpt2_cached_lm_1024_"
if model_size =="gpt2-medium":
    source = "input_data/gpt2-medium_cached_lm_1024_"
if model_size =="gpt2-large":
    source = "input_data/gpt2-large_cached_lm_1024_"

# Dataset for training without validation:
input_name = "input_data/Shelley.txt"

## Datasets for training with validation:
input_train = "input_data/Shelley_train.txt"
input_val = "input_data/Shelley_train.txt"

prompt =  'The eternal sky '

prompt = prompt.replace(" ", "£££££")

validate = True

for i in range (1,11):

    # FINE-TUNING THE MODELS
    if i == 1:

        
        if validate:
        
            # Fine-tuning with validation dataset:
            os.system('python run_lm_finetuning.py --output_dir=output --model_name_or_path {model_size} --do_train --train_data_file {input_train} --do_eval --eval_data_file {input_val} --overwrite_output_dir --max_steps {ckpt} --save_steps {ckpt}'.format(model_size=model_size, input_train=input_train, input_val=input_val, ckpt=ckpt))
    
            path = 'input_data/output'
            if not os.path.exists(path):
                os.makedirs(path)
                
            source_file = source +  input_train.split("/")[1]
            target_file = "input_data/output/checkpoint-" + str(ckpt) + "_cached_lm_1024_" + input_train.split("/")[1]
     
        else:
        
            # Fine-tuning without validation dataset

            os.system('python run_lm_finetuning.py --output_dir=output --model_name_or_path {model_size} --do_train --train_data_file {input_name} --overwrite_output_dir --save_steps {ckpt} --max_steps {ckpt}'.format(model_size=model_size, input_name= input_name, ckpt=ckpt))
            
            source_file = source +  input_name.split("/")[1]
            target_file = "input_data/output/checkpoint-" + str(ckpt) + "_cached_lm_1024_" + input_name.split("/")[1]
            
        shutil.copy(source_file, target_file)
        print("source_file: ", source_file)
        print("target_file: ", target_file)
        

    else:
    
        ckpt = ckpt + steps_increment
        last_ckpt = ckpt - steps_increment

        if validate:
            # Fine-tuning with validation dataset:
            os.system('python run_lm_finetuning.py --output_dir=output --model_name_or_path=output/checkpoint-{last_ckpt} --do_train --train_data_file {input_train} --do_eval --eval_data_file {input_val} --overwrite_output_dir --save_steps {ckpt} --max_steps {ckpt}'.format(last_ckpt=last_ckpt, input_train=input_train, input_val=input_val, ckpt=ckpt))
            
        else:
        
            # Fine-tuning without validation dataset:
            os.system('python run_lm_finetuning.py --output_dir=output --model_name_or_path=output/checkpoint-{last_ckpt} --do_train --train_data_file {input_name} --overwrite_output_dir --save_steps {ckpt} --max_steps {ckpt}'.format(last_ckpt=last_ckpt, input_name= input_name, ckpt=ckpt))
        
        
    for item in os.listdir("output"):
        to_remove  = "output/" + item
        
        if not os.path.isdir(to_remove):
            os.remove(to_remove)
            
    # GENERATING THE SAMPLES:
    for s in range(seed_start, seed_end):
        prompt = prompt.replace(" ", "£££££") 

        os.system('python generate_conditional_samples_to_file.py --temperature 1.0 --top_k 50 --top_p 1.0 --model_name_or_path output/checkpoint-{ckpt} --length {sample_length}  --prompt {prompt} --model_descr {model_descr} --seed {s}'.format(ckpt=ckpt, sample_length=sample_length, prompt=prompt, model_descr=model_descr,s=s)) 

    # This code will remove the fine-tuned models that finished generation:
    # (comment out if you want to keep them):
    if i > 1:
        shutil.rmtree('output/checkpoint-{}'.format(str(last_ckpt)))
