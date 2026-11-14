import torch
import torch.nn as nn
import pathlib
import json

from ngrams.ngrams import generate_ngrams, sentence_to_ngram_bow
import arch

with open(pathlib.Path(__file__).parent / "ngrams" / "ngram_training.json") as f:
    all_ngrams = json.load(f)
ngrams_idx = {ng: i for i, ng in enumerate(all_ngrams)}

model = arch.Model(len(ngrams_idx), 32, 2)
model.load_state_dict(torch.load(pathlib.Path(__file__).parent / "model.pth", map_location="cpu"))

msg = "Hello"
vec = sentence_to_ngram_bow(msg, all_ngrams).unsqueeze(0)
output = model(vec)

pred = torch.argmax(output, dim=1)
prob = torch.softmax(output, dim=1)
print(f"Prediction: {pred}\nProbability: {prob[0][1]}")