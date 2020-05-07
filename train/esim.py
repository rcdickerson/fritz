from typing import Iterator, Dict, List, Union

import torch
import torch.optim as optim
from torch.nn import LSTM

from allennlp.common.params import Params
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.trainer import Trainer

from allennlp.models.esim import ESIM

# Based on configuration found at:
# https://github.com/ZhaofengWu/allennlp/blob/master/training_config/esim.jsonnet

TRAIN_SET = '../../multinli_1.0/multinli_1.0_train.jsonl'
DEV_SET_MATCHED = '../../multinli_1.0/multinli_1.0_dev_matched.jsonl'
DEV_SET_MISMATCHED = '../../multinli_1.0/multinli_1.0_dev_mismatched.jsonl'

MODEL_SAVE_PATH = '../esim.th'
VOCAB_SAVE_PATH = '../esim_vocab'

tokenizer = WordTokenizer()
token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
snli_reader = SnliReader(tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

#train_dataset = snli_reader.read(TRAIN_SET)
dev_matched = snli_reader.read(DEV_SET_MATCHED)
train_dataset = dev_matched
dev_mismatched = snli_reader.read(DEV_SET_MISMATCHED)
validation_dataset = dev_matched + dev_mismatched
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=300,
                            trainable=True,
                            pretrained_file='https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz')
text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

relu = Activation.by_name('relu')()
linear = Activation.by_name('linear')()

encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=300,
                                              hidden_size=300,
                                              num_layers=1,
                                              bidirectional=True,
                                              batch_first=True))

similarity_function = DotProductSimilarity()

projection_feedforward = FeedForward(input_dim=2400,
                                     num_layers=1,
                                     hidden_dims=300,
                                     activations=relu)

inference_encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=300,
                                                        hidden_size=300,
                                                        num_layers=1,
                                                        bidirectional=True,
                                                        batch_first=True))

output_feedforward = FeedForward(input_dim=2400,
                                 num_layers=1,
                                 hidden_dims=300,
                                 activations=relu,
                                 dropout=0.5)

output_logit = FeedForward(input_dim=300,
                           num_layers=1,
                           hidden_dims=3,
                           activations=linear)

initializer = InitializerApplicator.from_params(
    [[".*linear_layers.*weight", "xavier_uniform"],
     [".*linear_layers.*bias", "zero"],
     [".*weight_ih.*", "xavier_uniform"],
     [".*weight_hh.*", "orthogonal"],
     [".*bias_ih.*", "zero"],
     [".*bias_hh.*", "lstm_hidden_bias"]])

model = ESIM(vocab,
             text_field_embedder,
             encoder,
             similarity_function,
             projection_feedforward,
             inference_encoder,
             output_feedforward,
             output_logit,
             initializer=initializer)


if torch.cuda.is_available():
    print('CUDA device is available')
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    print('No CUDA device available')
    cuda_device = -1

iterator = BucketIterator(sorting_keys=[('premise', 'num_tokens'), ('hypothesis', 'num_tokens')],
                          batch_size=64)
iterator.index_with(vocab)

optimizer = optim.Adam(model.parameters(), lr=0.0004)
lr_scheduler = LearningRateScheduler.from_params(optimizer, Params(
                                                {'type': 'reduce_on_plateau',
                                                 'factor': 0.5,
                                                 'mode': 'max',
                                                 'patience': 0}))

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  validation_metric='+accuracy',
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=5,
                  num_epochs=75,
                  grad_norm=10.0,
                  cuda_device=cuda_device,
                  learning_rate_scheduler=lr_scheduler)
trainer.train()

# Save the model.
vocab.save_to_files(VOCAB_SAVE_PATH)
with open(MODEL_SAVE_PATH, 'wb') as f:
    torch.save(model.state_dict(), f)
