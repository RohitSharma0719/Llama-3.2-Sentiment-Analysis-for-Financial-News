# Llama-3.2-Sentiment-Analysis-for-Financial-News

Overview

This repository contains a sentiment analysis project on financial news headlines using a fine-tuned LLaMA-3.2-1B model. The goal is to classify news headlines into three sentiment categories: positive, neutral, or negative. The project includes data preprocessing, model fine-tuning using LoRA (Low-Rank Adaptation), and evaluation.

Dataset

The dataset used for this project is from Kaggle: Sentiment Analysis for Financial News.

File: all-data.csv

Columns:

sentiment: Label (positive, neutral, negative)

text: News headline

Model

Base Model: LLaMA-3.2-1B

Fine-tuning: LoRA-based parameter-efficient fine-tuning

Quantization: 4-bit with bitsandbytes

Installation & Dependencies

To run this project, install the required libraries:

pip install numpy pandas tqdm bitsandbytes torch transformers datasets peft scikit-learn

Data Preprocessing

The dataset is split into training (300 samples per class), testing (300 samples per class), and evaluation (50 samples per class) sets. Prompts are generated to frame the sentiment classification task.

Training

The model is fine-tuned using LoRA with the following configurations:

Epochs: 3

Batch Size: 1

Gradient Accumulation: 8

Optimizer: Paged AdamW (32-bit)

Learning Rate: 2e-4 (cosine scheduler)

Evaluation

Before fine-tuning, the model achieves an accuracy of 35.3%. After fine-tuning, the accuracy improves to 78.2%, with significant performance gains in the negative and positive classes.

Classification Report (After Fine-Tuning)

Accuracy: 0.782
Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.95      0.93       300
           1       0.89      0.45      0.59       300
           2       0.65      0.95      0.78       300

Future Improvements

Experiment with larger LLaMA models

Implement dataset augmentation for better generalization

Explore few-shot learning with in-context examples
