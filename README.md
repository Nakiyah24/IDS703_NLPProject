# ECE 684 / 1DS 703 Natural Language Processing Final Project 

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


### 3. Data Preprocessing
### 4. Model / Implementation
### 5. Evaluation Metrics
### 6. Analysis Plots
### 7. Conclusion