# Fine-tuning BERT for Text Classification

This repository contains a Jupyter notebook for fine-tuning the BERT model to classify consumer complaints from a dataset of financial consumer complaints. The model is trained to categorize complaints based on their narrative descriptions using the Hugging Face `transformers` library and PyTorch.

## Project Overview
This notebook demonstrates the process of fine-tuning the BERT model for multi-class text classification, using a dataset of consumer complaints related to various financial products. The BERT model is fine-tuned to predict the product category (e.g., credit card, mortgage) from complaint narratives. The dataset includes around 66,806 entries, but for the sake of training efficiency, a sample of 20,000 entries is used.

## Features:
- **Data Loading & Cleaning**: The dataset is loaded, and rows with missing complaint narratives are removed. A sample of 20,000 entries is used for model training.
- **Text Preprocessing**: Complaint narratives are tokenized using the BERT tokenizer, which processes the text into a format suitable for the BERT model.
- **BERT Model Fine-Tuning**: The BERT model is fine-tuned on the training dataset for classification tasks. The output layer is modified to predict 11 distinct product categories.
- **Training & Evaluation**: The model is trained using the AdamW optimizer and a learning rate scheduler. It is evaluated using accuracy and ROC-AUC scores.

## Requirements:
To run this notebook, you'll need to install the following libraries:
- `transformers`
- `torch`
- `sklearn`
- `pandas`
- `numpy`
- `google.colab` (for Colab users)

## Setup Instructions:
1. **Google Colab Integration**: The notebook is set up to run in Google Colab. It automatically mounts your Google Drive to access the dataset. If you are not using Colab, you can adjust the paths accordingly.
2. **Data Preprocessing**: After loading the dataset, the script cleans it by removing entries with missing complaint narratives and samples 20,000 entries.
3. **Fine-Tuning**: Fine-tune the BERT model using the provided setup. The model is optimized using AdamW with a learning rate scheduler for better convergence.

## How to Use:
1. Clone this repository to your local machine or open it in Google Colab.
2. Ensure that the dataset `consumer_complaints.csv` is uploaded to your Google Drive and the correct path is set in the notebook.
3. Execute each cell step-by-step to load the data, preprocess it, and fine-tune the BERT model.

## Evaluation:
The model's performance is evaluated using:
- **Accuracy**: The percentage of correct predictions.
- **ROC-AUC Score**: A metric used for multi-class classification to assess how well the model distinguishes between classes.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
