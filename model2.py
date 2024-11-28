# Import necessary libraries
import pandas as pd
import numpy as np
from transformers import BertTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay


# Function to build a one-hot vocabulary for given text data
def build_one_hot_vocab(input_text):
    """
    Builds a one-hot encoded vocabulary from input text data.

    Args:
        input_text (pd.Series): A pandas Series containing text data.

    Returns:
        dict: A dictionary mapping tokens to their unique integer index.
    """
    vocab = set()
    input_text = input_text.str.lower()
    for word in input_text:
        vocab.add(word)
    vocab.add("<UNK>")
    return {token: i for i, token in enumerate(vocab)}

def one_hot_encode(input_text, vocab):
    """
    Encodes text using one-hot representation based on a vocabulary.

    Args:
        input_text (str): The text to encode.
        vocab (dict): The one-hot vocabulary.

    Returns:
        np.array: A one-hot encoded vector.
    """
    vectorized_text = np.zeros(len(vocab))
    for word in input_text:
        if word in vocab:
            vectorized_text[vocab[word]] += 1
        else:
            vectorized_text[vocab["<UNK>"]] += 1
    return vectorized_text

def build_specialized_vocab(input_text):
    """
    Builds a specialized vocabulary for semicolon-separated text data.

    Args:
        input_text (pd.Series): A pandas Series containing semicolon-separated text.

    Returns:
        dict: A dictionary mapping tokens to their unique integer index.
    """
    vocab = set()
    vocab.add("<UNK>")
    input_text = input_text.str.lower().astype(str)

    # Build vocabulary
    for text in input_text:
        for word in text.split(";"):
            word = word.strip()  # Remove extra spaces
            if word:
                vocab.add(word)

    return {token: i for i, token in enumerate(vocab)}

def vectorize_text(input_text, vocab):
    """
    Converts semicolon-separated text into a vector using a specialized vocabulary.

    Args:
        input_text (str): The text to encode.
        vocab (dict): The vocabulary to use for vectorization.

    Returns:
        np.array: A vectorized representation of the input text.
    """
    # Ensure the input is a string
    vectorized_text = np.zeros(len(vocab))
    for word in input_text.split(";"):
        if word in vocab:
            vectorized_text[vocab[word]] += 1
        else:
            vectorized_text[vocab["<UNK>"]] += 1
    return vectorized_text

# Converting csv to a PyTorch Dataset Object
# Function to preprocess data and convert to PyTorch-compatible format
def process_data(df, vocabs):
    """
    Processes a dataset, encoding categorical variables and tokenizing text.

    Args:
        df (pd.DataFrame): The input dataset.
        vocabs (dict): A dictionary of pre-built vocabularies.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    print("processing data")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Drop irrelevant columns
    dropped_columns = ["id", "date"]
    df = df.drop(dropped_columns, axis=1)
    
    # Tokenize textual data columns
    for col in ["statement", "justification", "speaker_description"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: tokenizer.encode(x, padding='max_length', truncation=True, max_length=256))
        
    # Vectorize categorical columns using specialized vocabularies
    for col in ["subject", "state_info"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: vectorize_text(x, vocabs[col]))
        
    # One-hot encode certain columns
    for col in ["speaker", "context"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: one_hot_encode(x, vocabs[col]))

    return df

# Function to create vocabularies for categorical columns
def create_vocabs():
    """
    Creates vocabularies for the dataset's categorical columns.

    Returns:
        dict: A dictionary containing vocabularies for various columns.
    """
    df = pd.read_csv("data/train.csv")
    dropped_columns = ["id", "date"]
    df = df.drop(dropped_columns, axis=1)
    
    vocabs = {}
        
    for col in ["subject", "state_info"]:
        df[col] = df[col].fillna("None").astype(str)
        vocabs[col] = build_specialized_vocab(df[col])
        
    for col in ["speaker", "context"]:
        df[col] = df[col].fillna("None").astype(str)
        vocabs[col] = build_one_hot_vocab(df[col])
    return vocabs
    
class SentimentDataset(Dataset):
    def __init__(self, path, vocabs, transform=None):
        self.sentiment = pd.read_csv(path)
        self.sentiment = process_data(self.sentiment, vocabs)
        self.transform = transform
        
        
    def __len__(self):
        return len(self.sentiment)
    
    def __getitem__(self, idx):
        data = self.sentiment.iloc[idx]
        label = data["label"]
        data = data.drop("label")
        
        max_length = 0
        for col in data.index:
            value = data[col]
            if isinstance(value, (np.ndarray, list)):
                max_length = max(max_length, len(value))
        
        feature_vectors = []
        for col in data.index:
            value = data[col]
            if isinstance(value, (np.ndarray, list)):
                feature_vectors.append(np.array(value))
            else:
                feature_vectors.append(np.array([value], dtype=np.float32))

        feature_vectors = np.concatenate(feature_vectors)

        if self.transform:
            feature_vectors = self.transform(feature_vectors)
            
        return torch.tensor(feature_vectors, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        
t = transform.Compose([transform.ToTensor()])
vocabs = create_vocabs()
train_dataset = SentimentDataset(path="data/train.csv", vocabs=vocabs)
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
iterator = iter(dataloader)
data, label = next(iterator)

print(data.shape, label.shape)

class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim=9638, num_classes=6):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vocabs = create_vocabs()
train_dataset = SentimentDataset(path="data/train.csv", vocabs=vocabs)
test_dataset = SentimentDataset(path="data/test.csv", vocabs=vocabs)
val_dataset = SentimentDataset(path="data/valid.csv", vocabs=vocabs)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

model = FakeNewsClassifier().to(device)
optimizer = Adam(model.parameters(), lr=.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

epochs = 10
training_losses = []
val_losses = []
for epoch in range(1, epochs + 1):
    train_loss = 0
    val_loss = 0
    train_correct_predictions = 0
    train_total_samples = 0
    val_correct_predictions = 0
    val_total_samples = 0
    
    model.train()
    for features, labels in tqdm(train_loader, desc="Training", unit="its"):
        features = features.to(device).float()
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(features)
        t_loss = criterion(outputs, labels)
        t_loss.backward()
        optimizer.step()
        
        train_loss += t_loss.item()
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        train_correct_predictions += (predicted == labels).sum().item()
        train_total_samples += labels.size(0)
        
    model.eval()
    for features, labels in tqdm(val_loader, desc="Validating", unit="its"):
        features = features.to(device).float()
        labels = labels.to(device).long()
        with torch.no_grad():
            outputs = model(features)
            v_loss = criterion(outputs, labels)
        
        val_loss += v_loss.item()
        _, predicted = torch.max(outputs, 1)
        val_correct_predictions += (predicted == labels).sum().item()
        val_total_samples += labels.size(0)
        
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    training_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracy = train_correct_predictions / train_total_samples * 100
    val_accuracy = val_correct_predictions / val_total_samples * 100
    print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss: .4f}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy: .2f}, Val Accuracy: {val_accuracy: .2f}")
    
torch.save(model.state_dict(), "trained_model.pth")

plt.plot(np.arange(epochs), training_losses, label="Train")
plt.plot(np.arange(epochs), val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of loss during training")
plt.legend()
plt.tight_layout()
plt.show()

test_model = FakeNewsClassifier().to(device)
test_model.load_state_dict(torch.load("trained_model.pth"))
test_model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        features = features.to(device).float()
        labels = labels.to(device).long()
        
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="weighted")
recall = recall_score(all_labels, all_predictions, average="weighted")
f1 = f1_score(all_labels, all_predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions))

corresponding_labels = ["Pants on Fire", "False", "Barely True", "Half True", "Mostly True", "True"]
ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions, display_labels=corresponding_labels)
plt.title("Confusion Matrix")
plt.savefig("results/test_confusion_matrix.png")
plt.show()