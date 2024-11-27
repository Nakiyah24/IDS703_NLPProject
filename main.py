import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# statement, speaker_description, justification


def build_descriptive_text_vocab_ashley(input_text):
    vocab = set()
    vocab.add("<UNK>")
    for text in input_text:
        for word in text.split():
            word = remove_punctuation_ashley(word)
            if word:
                vocab.add(word)
    return {token: i for i, token in enumerate(vocab)}


def vectorize_descriptive_text_ashley(input_text, vocab):
    vectorized_text = np.zeros(len(vocab))
    for word in input_text.split():
        word = remove_punctuation_ashley(word)
        if word in vocab:
            vectorized_text[vocab[word]] += 1
        else:
            vectorized_text[vocab["<UNK>"]] += 1
    return vectorized_text


def remove_punctuation_ashley(word):
    punctuation = set([".", "(", ")", ",", ";", "?", "!", '"', ":", "'"])
    while word and word[0] in punctuation:
        word = word[1:]
    while word and word[-1] in punctuation:
        word = word[:-1]
    return word.lower()

# one hot encoding of subjects and context


# for the subject and context columns we only need to add each row into the vocab list and see if there are any repetitions
def build_descriptive_text_vocab_nruta(input_text):
    vocab = set()
    input_text = input_text.str.lower()
    for word in input_text:
        vocab.add(word)
    vocab.add("<UNK>")
    return {token: i for i, token in enumerate(vocab)}


def vectorize_descriptive_text_nruta(input_text, vocab):
    vectorized_text = np.zeros(len(vocab))
    for word in input_text:
        if word in vocab:
            vectorized_text[vocab[word]] += 1
        else:
            vectorized_text[vocab["<UNK>"]] += 1
    return vectorized_text

def build_descriptive_text_vocab_subject_stateInfo_nakiyah(input_text):
    vocab = set()
    vocab.add("<UNK>")
    input_text = input_text.str.lower()
    input_text = input_text.astype(str)

    # Build vocabulary
    for text in input_text:
        for word in text.split(";"):
            word = word.strip()  # Remove extra spaces
            if word:
                vocab.add(word)

    return {token: i for i, token in enumerate(vocab)}


def vectorize_descriptive_text_subject_nakiyah(input_text, vocab):
    # Ensure the input is a string
    if isinstance(input_text, list):
        input_text = ";".join(input_text)  # Join list into a string
    vectorized_text = np.zeros(len(vocab))
    for word in input_text.split(";"):
        if word in vocab:
            vectorized_text[vocab[word]] += 1
        else:
            vectorized_text[vocab["<UNK>"]] += 1
    return vectorized_text

### Converting csv to a PyTorch Dataset Object

def process_data(df, vocabs):
    """
    Processes a dataset file to encode categorical variables and convert data into PyTorch tensors.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: A tuple containing:
        - features_tensor (torch.Tensor) : Tensor of features
        - labels_tensor (torch.Tensor) : Tensor of labels.
        - label_encoders (dict): Dictionary of LabelEncoders for categorical columns
    """
    print("processing data")
    # drop useless data
    dropped_columns = ["id", "date"]
    df = df.drop(dropped_columns, axis=1)
    
    for col in ["statement", "justification", "speaker_description"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_ashley(x, vocabs[col]))
        
    for col in ["subject", "state_info"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_subject_nakiyah(x, vocabs[col]))
        
    for col in ["speaker", "context"]:
        df[col] = df[col].fillna("None").astype(str)
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_nruta(x, vocabs[col]))

    return df

def create_vocabs():
    df = pd.read_csv("data/train.csv")
    dropped_columns = ["id", "date"]
    df = df.drop(dropped_columns, axis=1)
    
    vocabs = {}
    for col in ["statement", "justification", "speaker_description"]:
        df[col] = df[col].fillna("None").astype(str)
        V = build_descriptive_text_vocab_ashley(df[col])
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_ashley(x, V))
        vocabs[col] = V
        
    for col in ["subject", "state_info"]:
        df[col] = df[col].fillna("None").astype(str)
        V = build_descriptive_text_vocab_subject_stateInfo_nakiyah(df[col])
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_subject_nakiyah(x, V))
        vocabs[col] = V
        
    for col in ["speaker", "context"]:
        df[col] = df[col].fillna("None").astype(str)
        V = build_descriptive_text_vocab_nruta(df[col])
        df[col] = df[col].apply(lambda x: vectorize_descriptive_text_nruta(x, V))
        vocabs[col] = V
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


"""Training the Neural Network"""


class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim=85446, num_classes=6):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
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
optimizer = Adam(model.parameters(), lr=.001, weight_decay=1e-3)
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
plt.savefig("results/confusion_matrix.png")
plt.show()