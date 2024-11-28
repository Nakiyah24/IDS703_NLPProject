# ECE 684 / IDS 703 Natural Language Processing Final Project 

## NaN-Tastic Fact Checkers: Uncovering Truth in Headlines with Advanced Classification Techniques
### By: Ashley Hong, Nakiyah Dhariwala, Nruta Choudhari

### Table of Contents:
1. Introduction
2. Data Overview
3. Data Preprocessing
4. Model / Implementation
5. Evaluation Metrics
6. Analysis Plots
7. Conclusion

### 1. Introduction

### 2. Data Overview
We made use of the [LIAR 2 dataset](https://paperswithcode.com/dataset/liar2), curated from PolitiFact, which is an extension of the original LIAR dataset and contains over 12,000 short statements that are labeled across six levels of truthfulness: true, mostly true, half true, mostly false, false, and pants on fire (indicating a blatantly false statement). The dataset also provides rich metadata for each statement, including details such as the speaker, the context of the statement, and the subject matter, making it highly suitable for short-text classification tasks.
What sets LIAR 2 apart is its incorporation of article context, where each statement is paired with the full context of the article it is derived from. This additional layer of information allows models to go beyond superficial text analysis and leverage deeper contextual insights, enhancing the accuracy of truthfulness classification. The dataset is particularly valuable for applications like fact-checking and misinformation detection, where understanding both the statement and its context is crucial.
It comes pre-divided into training, validation and testing files. We made use of the training dataset as a basis to train the model; the testing dataset to evaluate our results and chose our model; and finally, validation dataset to judge the final model.

|      Dataset      | Number of Rows (Input Rows) |
|:-----------------:|:---------------------------:|
| Training Dataset  |           18,369           |
| Testing Dataset   |           2,296            |
| Validation Dataset|           2,297            |

The columns of the dataset are as follows:
| **Column Name**        | **Description**                                                                 |
|:-----------------------:|:-------------------------------------------------------------------------------:|
| `id`                   | Unique identifier for each statement.                                           |
| `label`                | Ground truth label for the statement (e.g., true, false, half-true, etc.).      |
| `statement`            | The text of the statement being evaluated.                                      |
| `date`                 | The date when the statement was made or reported.                               |
| `subject`              | The main topic(s) or subject(s) of the statement.                               |
| `speaker`              | The individual or entity who made the statement.                                |
| `speaker_description`  | A brief description or title of the speaker (e.g., politician, public figure).  |
| `state_info`           | The U.S. state associated with the speaker (if applicable).                    |
| `true_counts`          | The count of statements made by the speaker that were classified as true.       |
| `mostly_true_counts`   | The count of statements made by the speaker that were classified as mostly true.|
| `half_true_counts`     | The count of statements made by the speaker that were classified as half-true.  |
| `mostly_false_counts`  | The count of statements made by the speaker that were classified as mostly false.|
| `false_counts`         | The count of statements made by the speaker that were classified as false.      |
| `pants_on_fire_counts` | The count of statements made by the speaker that were classified as "pants on fire" (extremely false). |
| `context`              | Additional context or setting where the statement was made.                     |
| `justification`        | Explanation or reasoning provided for the truthfulness evaluation.              |


### 3. Model 1
**FakeNewsClassifier**
FakeNewsClassifier is a feedforward neural network designed for multi-class classification of textual data, specifically tailored to classify statements into categories such as “True,” “False,” “Mostly True,” etc. The model efficiently handles high-dimensional vectorized features, making it suitable for large-scale classification tasks.

For data preprocessing, the model converts textual data into numerical features using one-hot encoding and tokenization. Columns like statement, justification, and speaker_description undergo tokenization, where words are split and punctuation is removed. For other columns like subject and state_info, text is parsed based on semicolon delimiters. To handle out-of-vocabulary words, an `<UNK>` token is introduced. The high-dimensional vectorized data is then reduced using Sparse PCA, which reduces the dimensionality to 500 components, making the model computationally efficient and less prone to overfitting.

The core architecture of the model consists of an input layer that accepts feature vectors of size 74,529. This is followed by two hidden layers with 500 and 20 neurons, each utilizing Batch Normalization and ReLU activation functions. The output layer contains six neurons corresponding to the six target classes, with a Softmax activation function for multi-class classification. The model is trained using Cross-Entropy Loss, which is well-suited for categorical tasks, and the Adam optimizer with a learning rate of 0.001 and weight decay of 1e-3 to prevent overfitting.

Training occurs over 10 epochs with a batch size of 128, and the model leverages a GPU for accelerated computations. During training, a custom SentimentDataset class organizes the preprocessed data into tensors, and PyTorch’s DataLoader is used for efficient batch processing, shuffling, and parallel loading during training, validation, and testing phases. This setup ensures that the model can handle large datasets efficiently while avoiding overfitting through randomized data shuffling.

### 4. Model 2
**Enhanced Classification with BERT and Refined Architecture**

In our second model, we integrated BERT embeddings for improved feature extraction, particularly for textual columns like “statement,” “justification,” and “speaker description.” BERT’s tokenizer was used to convert these columns into context-aware embeddings, capturing the relationships between words within a maximum sequence length of 256 tokens. This approach replaced the one-hot encoding used in Model 1, which treated words independently. For categorical columns such as “subject” and “state_info,” specialized vocabularies were used, and columns like “speaker” and “context” were processed with one-hot encoding. Sparse PCA was again applied to reduce dimensionality, optimizing the retained components to reduce computational overhead and prevent overfitting.

The SentimentDataset class was enhanced to handle BERT tokenization and preprocessing, ensuring smooth integration with PyTorch. We used DataLoader for efficient batch processing, shuffling, and parallel data loading during training, validation, and testing.

The updated FakeNewsClassifier architecture in Model 2 processed both BERT embeddings and vectorized features through four hidden layers. The first hidden layer contained 512 neurons, followed by 256, 128, and 64 neurons in subsequent layers, with Batch Normalization and ReLU activations. The output layer had six neurons for the truthfulness categories, using Softmax for class probabilities.

### 5. Evaluation Metrics

### 6. Analysis Plots
**Model 1:**  
![main_confusion_matrix](notebooks/main_confusion_matrix_white_background.png)

**Model 2:**  
![test_confusion_matrix](notebooks/test_confusion_matrix_white_background.png)

### 7. Conclusion