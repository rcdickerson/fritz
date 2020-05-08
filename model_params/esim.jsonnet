// This configuration file is from [1].
// Using hyperparameter values from [2], a training run could look like:
// SEED=0 BATCH_SIZE=32 DROPOUT=0.2 ENCODER_HIDDEN_SIZE=300 ENCODER_NUM_LAYERS=1 INFERENCE_ENCODER_HIDDEN_SIZE=300 PROJECTION_FEEDFORWARD_HIDDEN_DIM=300 INFERENCE_ENCODER_NUM_LAYERS=1 OUTPUT_FEEDFORWARD_NUM_LAYERS=1 OUTPUT_FEEDFORWARD_ACTIVATION=relu OUTPUT_FEEDFORWARD_DROPOUT=0 OUTPUT_FEEDFORWARD_HIDDEN_DIM=300 PROJECTION_FEEDFORWARD_NUM_LAYERS=1 PROJECTION_FEEDFORWARD_ACTIVATION=relu GRAD_NORM=10 LEARNING_RATE=0.0004 allennlp train -s ./esim esim.jsonnet

// [1] https://github.com/allenai/show-your-work/blob/056b3a591f43f9126976893ccde0a8c1bbcbf23f/training_config/esim.jsonnet
// [2] https://github.com/ZhaofengWu/allennlp/blob/4749fc3671a40ca3cd6eafc65e2d95bf3dead82c/training_config/esim.jsonnet

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
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
        "dropout": std.extVar("DROPOUT"),
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
            "hidden_size": std.parseInt(std.extVar("ENCODER_HIDDEN_SIZE")),
            "num_layers": std.parseInt(std.extVar("ENCODER_NUM_LAYERS")),
            "bidirectional": true
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": std.parseInt(std.extVar("ENCODER_HIDDEN_SIZE")) * 8,
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("PROJECTION_FEEDFORWARD_NUM_LAYERS")), function(i) std.parseInt(std.extVar("PROJECTION_FEEDFORWARD_HIDDEN_DIM"))),
            "num_layers": std.parseInt(std.extVar("PROJECTION_FEEDFORWARD_NUM_LAYERS")),
            "activations": std.makeArray(std.parseInt(std.extVar("PROJECTION_FEEDFORWARD_NUM_LAYERS")), function(i) std.extVar("PROJECTION_FEEDFORWARD_ACTIVATION"))
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": std.parseInt(std.extVar("PROJECTION_FEEDFORWARD_HIDDEN_DIM")),
            "hidden_size": std.parseInt(std.extVar("INFERENCE_ENCODER_HIDDEN_SIZE")),
            "num_layers": std.parseInt(std.extVar("INFERENCE_ENCODER_NUM_LAYERS")),
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": std.parseInt(std.extVar("INFERENCE_ENCODER_HIDDEN_SIZE")) * 8,
            "num_layers": std.parseInt(std.extVar("OUTPUT_FEEDFORWARD_NUM_LAYERS")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("OUTPUT_FEEDFORWARD_NUM_LAYERS")), function(i) std.parseInt(std.extVar("OUTPUT_FEEDFORWARD_HIDDEN_DIM"))),
            "activations": std.makeArray(std.parseInt(std.extVar("OUTPUT_FEEDFORWARD_NUM_LAYERS")), function(i) std.extVar("OUTPUT_FEEDFORWARD_ACTIVATION")),
            "dropout": std.extVar("OUTPUT_FEEDFORWARD_DROPOUT")
        },
        "output_logit": {
            "input_dim": std.parseInt(std.extVar("OUTPUT_FEEDFORWARD_HIDDEN_DIM")),
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
        "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": std.extVar("LEARNING_RATE")
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 75,
        "grad_norm": std.extVar("GRAD_NORM"),
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
