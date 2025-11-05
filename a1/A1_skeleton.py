# %%
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
# %%
###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    vocabulary = {
        pad_token: 0,
        unk_token: 1,
        bos_token: 2,
        eos_token: 3,
    }
    token_counts = Counter()
    with open(train_file, "r", errors='ignore') as file:

        for line in file:
            line = line.strip()
            if not line:
                continue
            token_counts.update(tokenize_fun(line))

    remaining = max_voc_size - len(vocabulary) if max_voc_size else None
    
    most_common = token_counts.most_common(remaining) if remaining else token_counts.most_common()
    total_counts = token_counts.total()
    cut_off = 20

    for word, count in most_common:
        if count < cut_off:
            break
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    
    print(f"Total token counts = {total_counts}")
    print(f"Vocabulary length = {len(vocabulary)} \t Total unique tokens = {len(token_counts.items())}")
    print(f"Least common tokens in vocabulary: \n{list(vocabulary.keys())[-200:]}")
    print(f"Least common tokens in text: \n{token_counts.most_common()[:-200-1:-1]}")

    inv_vocabulary = {i: tok for tok, i in vocabulary.items()}

    tokenizer = A1Tokenizer(pad_token, unk_token, bos_token, eos_token, model_max_length, vocabulary, inv_vocabulary)


train_file = "train.txt"
build_tokenizer(train_file=train_file)
# %%

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, pad_token, unk_token, bos_token, eos_token, model_max_length, vocabulary, inv_vocabulary):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = vocabulary[pad_token]     # Compulsory attribute.
        self.unk_token_id = vocabulary[unk_token]
        self.bos_token_id = vocabulary[bos_token]
        self.eos_token_id = vocabulary[eos_token]
        self.model_max_length = model_max_length # Needed for truncation.
        self.vocabulary = vocabulary
        self.inv_vocabulary = inv_vocabulary

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.



        return BatchEncoding({'input_ids': ...})

    def __len__(self):
        """Return the size of the vocabulary."""
        return ...
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

# %%
class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(
            input_size=config.embedding_size, 
            hidden_size=config.hidden_size, 
            batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)        # (B, N, E)
        rnn_out, _ = self.rnn(embedded)     # (B, N, H)
        out = self.unembedding(rnn_out)     # (B, N, V)
        return out

config = A1RNNModelConfig(
    vocab_size=1000,
    embedding_size=128,
    hidden_size=256
)
model = A1RNNModel(config)

# Test case
X = torch.tensor([[1, 33, 54, 99]])
out = model(X)
print(out.shape)
# %%

###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size:
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.

        Args:
          model:          The model to train.
          args:           The training parameters stored in a TrainingArguments object.
          train_dataset:  The dataset containing the training documents.
          eval_dataset:   The dataset containing the validation documents.
          eval_dataset:   The dataset containing the validation documents.
          tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert args.optim == "adamw_torch"
        assert args.eval_strategy == "epoch"

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device("cpu")
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print("Device:", device)
        self.model.to(device)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            params=self.model.parameters(),
        )

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            self.eval_dataset, batch_size=args.per_device_eval_batch_size
        )

        for epoch in range(args.num_train_epochs):
            for batch in train_loader:
                #       PREPROCESSING AND FORWARD PASS:
                #       input_ids = apply your tokenizer to B
                input_ids = self.tokenizer(
                    batch["text"], truncation=True, padding=True, return_tensors="pt"
                )["input_ids"]
                #       X = all columns in input_ids except the last one
                #       Y = all columns in input_ids except the first one
                #       put X and Y onto the GPU (or whatever device you use)
                X = input_ids[:, :-1].to(device)
                Y = input_ids[:, 1:].to(device)
                #       apply the model to X
                outputs = self.model(X)
                #       compute the loss for the model output and Y
                targets = Y.view(-1)  # 2-dimensional -> 1-dimensional
                logits = outputs.view(
                    -1, outputs.shape[-1]
                )  # 3-dimensional -> 2-dimensional
                loss = loss_func(logits, targets)
                #       BACKWARD PASS AND MODEL UPDATE:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluation using perplexity
                if args.eval_strategy == "epoch":
                    # compute mean cross entropy loss on validation set
                    self.model.eval()
                    total_loss = 0.0
                    total_batches = 0

                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids = self.tokenizer(
                                batch["text"],
                                truncation=True,
                                padding=True,
                                return_tensors="pt",
                            )["input_ids"]
                            X = input_ids[:, :-1].to(device)
                            Y = input_ids[:, 1:].to(device)

                            outputs = self.model(X)

                            targets = Y.view(-1)  # 2-dimensional -> 1-dimensional
                            logits = outputs.view(
                                -1, outputs.shape[-1]
                            )  # 3-dimensional -> 2-dimensional
                            loss = loss_func(logits, targets)
                            total_loss += loss.item()
                            total_batches += 1

                    avg_loss = total_loss / total_batches if total_batches > 0 else 0
                    perplexity = np.exp(avg_loss)

                    print(
                        f"Epoch {epoch} Loss: {loss.item()} Val Loss: {avg_loss} Perplexity: {perplexity}"
                    )

        print(f"Saving to {args.output_dir}.")
        self.model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
    print("Tokenizer loaded successfully.")

    config = A1RNNModelConfig(
        vocab_size=len(tokenizer), embedding_size=128, hidden_size=256
    )
    model = A1RNNModel(config)
    print("Model initialized successfully.")

    TRAIN_FILE = "train.txt"
    VAL_FILE = "valid.txt"
    from datasets import load_dataset

    dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "val": VAL_FILE})
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    # TODO: remove for full data
    from torch.utils.data import Subset

    for sec in ["train", "val"]:
        dataset[sec] = Subset(dataset[sec], range(1000))

    print("Datasets loaded successfully.")

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./a1_rnn_model",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir="./logs",
        use_cpu=False,
    )

    trainer = A1Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
    )
    trainer.train()

    # Save the trained model
    model.save_pretrained("a1_rnn_model_trained")
    print("Trained model saved successfully.")
