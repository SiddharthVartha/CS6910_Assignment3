
# Seq2Seq Model for English to Marathi Transliteration

This repository contains a PyTorch implementation of a Seq2Seq model with optional attention mechanism for transliterating English words to Marathi words. The implementation supports various RNN cell types including RNN, LSTM, and GRU, and allows for configurable model parameters such as the number of layers, embedding dimension, hidden size, and dropout rates.


## Introduction

The Seq2Seq model is designed to perform sequence-to-sequence transliteration tasks. In this project, the task is to transliterate English words into Marathi words. The model is implemented using PyTorch and supports various configurations through command-line arguments.

## Dataset

The dataset consists of parallel corpora of English and Marathi words. The data is split into training, validation, and test sets.

- `train_file`: Path to the training data CSV file.
- `test_file`: Path to the test data CSV file.
- `val_file`: Path to the validation data CSV file.

## Model Architecture

The model consists of an Encoder and a Decoder. The Encoder encodes the input sequence into a context vector, which is then decoded by the Decoder to generate the output sequence.

### Encoder

The Encoder is an RNN-based model (RNN, LSTM, or GRU) that processes the input sequence and generates hidden states.

### Decoder

The Decoder is also an RNN-based model that takes the Encoder's context vector and generates the output sequence. The Decoder can optionally use an attention mechanism to focus on different parts of the input sequence during decoding.

### Seq2SeqModel

The Seq2SeqModel class integrates the Encoder and Decoder, handling the forward pass and the sequence generation process.

## Training

The model is trained using cross-entropy loss and the Adam optimizer. The training loop involves feeding batches of data to the model, calculating the loss, and updating the model parameters.

### Hyperparameters

| Hyperparameter  | Values         | Explanation                                                                                         |
|-----------------|----------------|-----------------------------------------------------------------------------------------------------|
| epochs          | 10, 15, 20     | Number of training epochs, i.e., the number of times the model will iterate over the entire dataset.|
| learning_rate   | 1e-3, 1e-4     | Learning rate for the optimizer, controlling the size of the steps taken during optimization.      |
| cell_type       | RNN, LSTM, GRU| Type of recurrent neural network cell used in the model.                                            |
| bidirectional   | True, False    | Whether to use bidirectional RNN layers, which process the input sequence both forwards and backwards.|
| enc_layers      | 1, 2, 3, 4, 5 | Number of layers in the encoder (input) RNN.                                                         |
| dec_layers      | 1, 2, 3, 4, 5 | Number of layers in the decoder (output) RNN.                                                        |
| batch_size      | 128, 256, 512 | Number of training examples processed simultaneously by the model.                                   |
| embedding_dim   | 256, 384, 512 | Dimensionality of the word embeddings, representing words as dense vectors in a continuous space.    |
| hidden_size     | 256, 384, 512 | Size of the hidden state in the RNN cells, determining the model's capacity to learn complex patterns.|
| enc_dropout     | 0, 0.1, 0.2   | Dropout probability applied to the encoder RNN, reducing overfitting by randomly dropping units.      |
| dec_dropout     | 0, 0.1, 0.2   | Dropout probability applied to the decoder RNN, reducing overfitting by randomly dropping units.      |
| attention       | True, False          | Whether to use an attention mechanism in the decoder, allowing the model to focus on relevant parts of the input sequence. |


## Usage
```bash
python train.py --epochs 15 --learning_rate 1e-3 --cell_type LSTM --bidirectional True --enc_layers 1 --dec_layers 1 --batch_size 256 --embedding_dim 512 --hidden_size 384 --enc_dropout 0.2 --dec_dropout 0.1 --attention True
```

## Evaluation

The model is evaluated based on the accuracy of the transliteration on the test set.

### Accuracy Calculation

Accuracy is calculated by comparing the predicted sequences with the actual target sequences.

### Attention Visualization

Attention heatmaps can be generated to visualize the focus of the model on different parts of the input sequence.

## Results

After training the model, the performance on the test set is reported.<br>
Wandb Report Link:-https://wandb.ai/cs23m063/deep_learn_assignment_3/reports/CS6910-Assignment-3--Vmlldzo3OTUwNjE0

