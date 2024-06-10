# UROP-3100-project
## Overview
This repository is dedicated to the reproduction of the StructLM model, with the aim of enhancing structured data understanding using large language models. The project leverages the training data and techniques from the StructLM project by Zhuang, A. (2023).

## Model and Dataset
### Base Model
The base model used for this project is the Mistral-7B model. Mistral-7B is known for its robust language understanding capabilities, making it a suitable candidate for fine-tuning on structured data tasks.

### Training Dataset
The dataset used for fine-tuning is the StructLM dataset provided by TIGER-Lab/SKGInstruct. This dataset is specifically designed for structured data understanding, which is crucial for the objectives of this project.

## Fine-Tuning Process
### Tool
The fine-tuning process was carried out using LMFlow, a flexible and efficient tool for language model fine-tuning. LMFlow is available at [OptimalScale/LMFlow](https://github.com/OptimalScale/LMFlow).

### Technique
To achieve efficient and scalable fine-tuning, the LoRA (Low-Rank Adaptation) technique was employed. LoRA allows for significant reductions in the number of trainable parameters by decomposing weight matrices into lower-rank representations, which speeds up the fine-tuning process and reduces resource requirements.

## Project Structure
- **data/**: Contains the StructLM dataset and any other related data files.
- **scripts/**: Includes scripts for preprocessing data, running fine-tuning processes, and evaluating the model.
- **models/**: Directory where the fine-tuned model checkpoints are saved.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and experiments.
- **README.md**: Project documentation.

## Getting Started
To get started with this project, follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/YangHaolin0526/UROP-3100-project.git
    cd UROP-3100-project
    ```

2. **Install Dependencies**
    Ensure you have Python and the required packages installed. 
    For Fine-tuning the model, you can install the dependencies using:
    ```bash
    pip install -r fine-tune_requirements.txt
    ```

4. **Prepare Data**
    Download and preprocess the StructLM dataset:
    ```bash
    huggingface-cli download --repo-type=dataset --local-dir=data/processed/ TIGER-Lab/SKGInstruct ./skginstruct_test_file_7b.json
    python Format_t.py
    ```

5. **Fine-Tune the Model**
    Run the fine-tuning script using LMFlow and LoRA:
    ```bash
    deepspeed --include localhost:8,9 ./examples/finetune.py --model_name_or_path mistralai/Mistral-7B-v0.1 --dataset_path data/Struct/train --output_dir output_models/finetuned_Mistral_StructLM --overwrite_output_dir --num_train_epochs 0.01 --learning_rate 1e-4 --block_size 512 --per_device_train_batch_size 1 --use_lora 1 --lora_r 8 --save_aggregated_lora 1 --deepspeed ./configs/ds_config_zero2.json  --fp16 —run_name
finetune_with_lora --validation_split_percentage 0 --logging_steps 20 --do_train     
-- ddp_timeout 72000 --save_steps 5000 --dataloader_num_workers 1
    ```

6. **Evaluate the Model**
    After fine-tuning, evaluate the model performance on the validation set:
    ```bash
    python scripts/evaluate.py
    ```

## Contributing
Contributions to the project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
