## This code was used to train the models and generate samples from OpenAI GPT-2 models used in our paper:
## 'Training GPT-2 to represent two Romantic-era authors challenges, evaluations and pitfalls'.

import os
import shutil
import argparse


## model_size is passed as an argument from the colab notebook:
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="124M")
args = parser.parse_args()
model_size = args.model_size

## seed_start and seed_end set the range of samples being generated from your models:
seed_start = 1
seed_end = 1001

## ckpt sets the first checkpoint where the model will be saved and the samples will be generated from:
ckpt = 1000

## steps_increment sets the increment of steps between checkpoints that are saved and generated from:
steps_increment = 1000

## sample_length sets the length of the samples generated (maximum is 1024, but if you are using prompt, 
## you will have to deduct the prompt size from this number):
sample_length = 600

## model_descr can be anything, it is used only to name your samples: 
model_descr = "Shelley"

##This directory_path is valid only for Colab notebooks working with our repository:
directory_path = "/content/GPT_2_CODE/Open_AI_GPT_2/"

####################################
# Text prompt for the samples
prompt = "The eternal sky "
####################################
prompt = prompt.replace(" ", "£££££")

## Dataset for training without validation:
input_name = "input_data/Shelley.txt"

## Datasets for training with validation:
input_train = "input_data/Shelley_train.txt"
input_val = "input_data/Shelley_train.txt"


# Main fine-tuning and generation loop:
for i in range(1,11):

    # You can choose to fine-tune the model with or without validation. Comment out as required:
    
    # Fine-tuning the model without validatation:
    os.system('python train.py --dataset {} --model_name {} --top_k 50 --top_p 1.0 --max_steps {} --save_every {}'.format(input_name, model_size, ckpt, steps_increment))
    
    # Fine-tuning the model with validatation:
    # os.system('python train.py --dataset {} --val_dataset {} --val_every 50 --model_name {} --top_k 50 --top_p 1.0 --max_steps {} --save_every {}'.format(input_train, input_val, model_size, ckpt, ckpt))
 
    # This line calls the "mover.py" script to copy the fine-tuned model into the "models" folder
    # and to copy from the original model the files required to run the fine-tuned model:
    os.system('python mover.py --model_steps {} --model_size {} --directory_path {}'.format(ckpt, model_size, directory_path))       
        
    # Generation loop:
    for s in range(seed_start, seed_end):
        # Change the --raw_text, --top_k, --top_p and --temperature arguments as required:
        os.system('python generate_conditional_samples_to_file.py --raw_text {} --model_descr {} --model_name model_{} --length {} --seed {} --temperature 1.0 --top_k 50 --top_p 1.0'.format(prompt, model_descr, ckpt, sample_length, s))

    # This code will remove the fine-tuned models that finished generation:
    # (comment out if you want to keep them):
    shutil.rmtree('models/model_{}'.format(str(ckpt)))
    previous_model = "model-" + str(ckpt - steps_increment)
    
    for item in os.listdir("checkpoint/run1"):
        if previous_model in z:
            to_remove  = "checkpoint/run1/" + item
            os.remove(to_remove)
        
    ckpt += steps_increment
