// This configuration is based on [1] using hyperparameter values from [2].

// [1] https://github.com/allenai/show-your-work/blob/056b3a591f43f9126976893ccde0a8c1bbcbf23f/training_config/esim.jsonnet
// [2] https://github.com/ZhaofengWu/allennlp/blob/4749fc3671a40ca3cd6eafc65e2d95bf3dead82c/training_config/esim.jsonnet

{
   "numpy_seed": 0,
   "pytorch_seed": 0,
   "random_seed": 0,
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
  "train_data_path": "../multinli_1.0/multinli_1.0_train.jsonl",
  "validation_data_path": "../multinli_1.0/multinli_1.0_dev.jsonl",
    "model": {
        "type": "esim",
        "dropout": 0.2,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": 2400,
            "hidden_dims": std.makeArray(1, function(i) 300),
            "num_layers": 1,
            "activations": std.makeArray(1, function(i) "relu")
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": 2400,
            "num_layers": 1,
            "hidden_dims": std.makeArray(1, function(i) 300),
            "activations": std.makeArray(1, function(i) "relu"),
            "dropout": 0
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "max_instances_in_memory": 500,
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 75,
        "grad_norm": 10,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}
