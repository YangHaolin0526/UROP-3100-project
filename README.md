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
    git clone https://github.com/yourusername/UROP-3100-project.git
    cd UROP-3100-project
    ```

2. **Install Dependencies**
    Ensure you have Python and the required packages installed. You can install the dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**
    Download and preprocess the StructLM dataset:
    ```bash
    python scripts/preprocess_data.py
    ```

4. **Fine-Tune the Model**
    Run the fine-tuning script using LMFlow and LoRA:
    ```bash
    python scripts/fine_tune.py
    ```

5. **Evaluate the Model**
    After fine-tuning, evaluate the model performance on the validation set:
    ```bash
    python scripts/evaluate.py
    ```

## Contributing
Contributions to the project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
