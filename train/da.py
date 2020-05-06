from typing import Iterator, Dict, List, Union

import torch
import torch.optim as optim

from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.activations import Activation
from allennlp.training.trainer import Trainer

from allennlp.models.decomposable_attention import DecomposableAttention

# Based on configuration found at:
# https://github.com/ZhaofengWu/allennlp/blob/master/training_config/decomposable_attention.jsonnet

TRAIN_SET = '../../multinli_1.0/multinli_1.0_train.jsonl'
DEV_SET_MATCHED = '../../multinli_1.0/multinli_1.0_dev_matched.jsonl'
DEV_SET_MISMATCHED = '../../multinli_1.0/multinli_1.0_dev_mismatched.jsonl'

MODEL_SAVE_PATH = '../da.th'
VOCAB_SAVE_PATH = '../da_vocab'

#TRAIN_SET = '../../snli_1.0/snli_1.0_train.jsonl'
#TEST_SET = '../../snli_1.0/snli_1.0_dev.jsonl'

tokenizer = WordTokenizer(end_tokens=['@@NULL@@'])
token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
snli_reader = SnliReader(tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

train_dataset = snli_reader.read(TRAIN_SET)
dev_matched = snli_reader.read(DEV_SET_MATCHED)
dev_mismatched = snli_reader.read(DEV_SET_MISMATCHED)
validation_dataset = dev_matched + dev_mismatched

#train_dataset = snli_reader.read(TRAIN_SET)
#validation_dataset = snli_reader.read(TEST_SET)

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
#vocab.save_to_files(VOCAB_SAVE_PATH)
#vocab = Vocabulary.from_files(VOCAB_SAVE_PATH)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            projection_dim=200,
                            embedding_dim=300,
                            trainable=False,
                            pretrained_file='https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz')
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

relu = Activation.by_name('relu')()
linear = Activation.by_name('linear')()
attend_feedforward = FeedForward(input_dim=200,
                                 num_layers=2,
                                 hidden_dims=200,
                                 activations=relu,
                                 dropout=0.2)
similarity_function = DotProductSimilarity()
compare_feedforward = FeedForward(input_dim=400,
                                  num_layers=2,
                                  hidden_dims=200,
                                  activations=relu,
                                  dropout=0.2)
aggregate_feedforward = FeedForward(input_dim=400,
                                    num_layers=2,
                                    hidden_dims=[200, 3],
                                    activations=[relu, linear],
                                    dropout=[0.2, 0.0])
model = DecomposableAttention(vocab,
                              word_embeddings,
                              attend_feedforward,
                              similarity_function,
                              compare_feedforward,
                              aggregate_feedforward)

if False: #torch.cuda.is_available():
    print('CUDA device is available')
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    print('No CUDA device available')
    cuda_device = -1

iterator = BucketIterator(sorting_keys=[('premise', 'num_tokens'), ('hypothesis', 'num_tokens')],
                          batch_size=64)
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optim.Adagrad(model.parameters()),
                  validation_metric='+accuracy',
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=20,
                  num_epochs=140,
                  cuda_device=cuda_device,
                  grad_clipping=5.0)
trainer.train()

# Save the model.
with open(MODEL_SAVE_PATH, 'wb') as f:
    torch.save(model.state_dict(), f)
