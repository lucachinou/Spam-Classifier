import json
import sys

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ngrams.ngrams import generate_ngrams, sentence_to_ngram_bow
from arch import Model

torch.manual_seed(42)

df = pd.read_csv("data.csv")

sentences = df['content'].tolist()
labels = df['label'].tolist()

perm = torch.randperm(len(sentences))
split = int(0.8 * len(sentences))

eval_sentence = [sentences[i] for i in perm[split:]]
eval_label = [labels[i] for i in perm[split:]]

train_sentence = [sentences[i] for i in perm[:split]]
train_label = [labels[i] for i in perm[:split]]

vocab = list(set(" ".join(sentences).split()))
word_to_idx = {w.lower(): i for i, w in enumerate(vocab)}

all_ngrams = set()
for s in train_sentence:
    all_ngrams.update(generate_ngrams(s))

ngram_to_idx = {ng: i for i, ng in enumerate(all_ngrams)}
with open("ngrams/ngram_training.json", "w") as f:
    json.dump(ngram_to_idx, f)

model = Model(len(ngram_to_idx), 32, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_eval = torch.stack([sentence_to_ngram_bow(sentence, ngram_to_idx) for sentence in eval_sentence])
Y_eval = torch.tensor(eval_label, dtype=torch.long)

X = torch.stack([sentence_to_ngram_bow(s, ngram_to_idx) for s in train_sentence])
y = torch.tensor(train_label, dtype=torch.long)
model.train()

for i in range(450):
    optimizer.zero_grad()
    output = model(X)

    loss = criterion(output, y)

    loss.backward()
    optimizer.step()

    pred = torch.argmax(output, dim=1)
    accuracy = (pred == y).float().mean()

    print(f"Model accuracy: {accuracy}")

    print("epoch: "+ str(i))
    print(f"loss: {loss}")

torch.save(model.state_dict(), "model.pth")

model.eval()

model_result = []
for sentence, label in zip(X_eval, Y_eval):
    with torch.no_grad():
        out = model(sentence.unsqueeze(0))

        pred = torch.argmax(out, dim=1)
        probs = torch.softmax(out, dim=1)

        loss = criterion(out, label.unsqueeze(0))
        accuracy = (pred == label).float().mean()
        model_result.append((sentence, label, loss, out, pred, accuracy, probs.max().item()))

result_df = pd.DataFrame(model_result, columns=["sentence", "label", "loss", "response", "pred", "accuracy", "confidence"])
print(f"## Model evaluation accuracy (mean): {result_df['accuracy'].mean()}\n## Model evaluation prediction (mean): {result_df['pred'].mean()}\n## Model evaluation confidence (mean): {result_df['confidence'].mean()}\n### Model eval loss: {result_df['loss'].mean()}")

while True:
    user_input = input("> ")

    vector = sentence_to_ngram_bow(user_input, ngram_to_idx).unsqueeze(0)
    with torch.no_grad():
        response = model(vector)
        pred = torch.argmax(response, dim=1)
        prob = torch.softmax(response, dim=1)
        print(f"Model: {prob[0][1]}")

        print(f"Predicted: {pred}")

        if prob[0][1] >= 0.7:
            print("Scam detected!")