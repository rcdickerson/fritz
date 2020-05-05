from typing import Iterator, Dict, List, Union

import torch
import torch.optim as optim

from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.activations import Activation
from allennlp.training.trainer import Trainer

from allennlp.models.decomposable_attention import DecomposableAttention

TRAIN_SET = '../../multinli_1.0/multinli_1.0_train.jsonl'
DEV_SET_MATCHED = '../../multinli_1.0/multinli_1.0_dev_matched.jsonl'
DEV_SET_MISMATCHED = '../../multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
MODEL_SAVE_PATH = '../da.th'
VOCAB_SAVE_PATH = '../da_vocab'
EMBEDDING_DIM = 8

snli_reader = SnliReader()
train_dataset = snli_reader.read(TRAIN_SET)
dev_matched = snli_reader.read(DEV_SET_MATCHED)
dev_mismatched = snli_reader.read(DEV_SET_MISMATCHED)
validation_dataset = dev_matched + dev_mismatched

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
#vocab.save_to_files(VOCAB_SAVE_PATH)
#vocab = Vocabulary.from_files(VOCAB_SAVE_PATH)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

ATTEND_LAYERS = 1
ATTEND_HIDDEN_DIM = 8
COMPARE_LAYERS = 1
COMPARE_HIDDEN_DIM = 8
AGGREGATE_LAYERS = 1
OUTPUT_DIM = 3

relu = Activation.by_name('relu')()
attend_feedforward = FeedForward(EMBEDDING_DIM, ATTEND_LAYERS, ATTEND_HIDDEN_DIM, relu)
similarity_function = DotProductSimilarity()
compare_feedforward = FeedForward(2 * ATTEND_HIDDEN_DIM, COMPARE_LAYERS, COMPARE_HIDDEN_DIM, relu)
aggregate_feedforward = FeedForward(2 * COMPARE_HIDDEN_DIM, AGGREGATE_LAYERS, OUTPUT_DIM, relu)

model = DecomposableAttention(vocab, word_embeddings, attend_feedforward, similarity_function, compare_feedforward, aggregate_feedforward)

if torch.cuda.is_available():
    print('CUDA device is available')
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    print('No CUDA device available')
    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[('premise', 'num_tokens')])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=300,
                  cuda_device=cuda_device)
trainer.train()

# Save the model.
with open(MODEL_SAVE_PATH, 'wb') as f:
    torch.save(model.state_dict(), f)
