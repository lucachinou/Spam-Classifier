# Spam Classifier
This project is a spam classifier that uses n-grams to classify text messages as either spam or not spam. The classifier is trained on a dataset of text messages and their corresponding labels (spam or not spam). The n-grams are generated from the text messages and used as features for the classifier.
You can try easily by running inference.py file.

Dataset: By me
Pre Processing: Generating n-grams from the text messages and converting them into a bag of words representation.
Architecture: A simple neural network with 1 hidden layers and ReLU activation function.
Training Method: Supervised training with CrossEntropyLoss loss function and Adam optimizer.
Used metrics: mostly Accuracy
Limits: *Dataset is small and the model don't classify images.*

You can also train the model by running training.py file. The model will be saved as "model.pth" and can be used in inference.py file.
I will add more features to this project in the future. If you have any suggestions, please feel free to open an issue or a pull request.
