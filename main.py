import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer import Encoder, Decoder
from torch import nn
import os, sys

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16  # Number of independent sequences
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # Number of maximum iterations
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


n_input = 64  # Input size for the classifier
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name.
    """
    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  # don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader."""
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y)
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity

    decoderLMmodel.train()
    return perplexity

def main():
    torch.manual_seed(seed)
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # Create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
  
    with open("train_LM.txt", 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    with open("test_hbush.txt", 'r', encoding='utf-8') as f:
        hbushText = f.read()
    with open("test_obama.txt", 'r', encoding='utf-8') as f:
        obamaText = f.read()
    with open("test_wbush.txt", 'r', encoding='utf-8') as f:
        wbushText = f.read()

    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, hbushText, block_size)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=False)
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, obamaText, block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=False)
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, wbushText, block_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=False)

    if sys.argv[1] == "encoder":
        model = Encoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, n_input, n_hidden, n_output)
        model = model.to(device)
        print(sum(p.numel() for p in model.parameters()), 'parameters')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        # Training loop
        for epoch in range(epochs_CLS):
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred, _ = model(xb)
                loss = loss_fn(pred, yb)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_accuracy = compute_classifier_accuracy(model, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, test accuracy {test_accuracy:.3f}')


    elif sys.argv[1] == "decoder":
        model = Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer)
        model = model.to(device)
        print(sum(p.numel() for p in model.parameters()), 'parameters')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Training loop
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            if i == 0 or i % eval_interval == eval_interval-1:
                train_perplexity = compute_perplexity(model, train_LM_loader, eval_iters)
                test_LM_hbush_perplexity = compute_perplexity(model, test_LM_hbush_loader, eval_iters)
                test_LM_obama_perplexity = compute_perplexity(model, test_LM_obama_loader, eval_iters)
                test_LM_wbush_perplexity = compute_perplexity(model, test_LM_wbush_loader, eval_iters)
                print(f"step {i+1}: train perplexity {train_perplexity}, test_hbush perplexity {test_LM_hbush_perplexity}, test_obama perplexity {test_LM_obama_perplexity}, test_wbush perplexity {test_LM_wbush_perplexity}")
            
            xb, yb = xb.to(device), yb.to(device)
            loss, _ = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
