"""
This file generates the json configuration that allennlp uses to build and train the model
"""

import json
import os
from pathlib import Path

from anlp_a2.config import DATASET_DIR

BATCH_SIZE = 512 // 4


def dataset_reader():
    return {
        "type": "sharded",
        "base_reader": {
            "type": "simple_language_modeling",
            "tokenizer": {"type": "just_spaces"},
            "token_indexers": {
                "tokens": {"type": "single_id"},
                "token_characters": {"type": "elmo_characters"},
            },
            "max_sequence_length": 400,
            "start_tokens": ["<S>"],
            "end_tokens": ["</S>"],
        },
    }


def vocabulary():
    return {
        "type": "from_files",
        "directory": str(DATASET_DIR / "vocab"),
    }


def contextualizer():
    return {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1,
    }


def model():
    return {
        "type": "language_model",
        "bidirectional": True,
        "num_samples": 8192,
        "sparse_embeddings": False,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {"type": "empty"},
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {"num_embeddings": 262, "embedding_dim": 16},
                    "encoder": {
                        "type": "cnn-highway",
                        "activation": "relu",
                        "embedding_dim": 16,
                        "filters": [
                            [1, 32],
                            [2, 32],
                            [3, 64],
                            [4, 128],
                            [5, 256],
                            [6, 512],
                            [7, 1024],
                        ],
                        "num_highway": 2,
                        "projection_dim": 512,
                        "projection_location": "after_highway",
                        "do_layer_norm": True,
                    },
                },
            }
        },
        "dropout": 0.1,
        "contextualizer": contextualizer(),
    }


def data_loader():
    return {
        "max_instances_in_memory": BATCH_SIZE * 100,
        "batch_sampler": {
            "type": "bucket",
            "batch_size": BATCH_SIZE,
        },
    }


def trainer():
    return {
        "num_epochs": 30,
        "optimizer": {"type": "dense_sparse_adam"},
        "learning_rate_scheduler": {
            "type": "noam",
            "model_size": 512,
            "warmup_steps": 6000,
        },
        "num_gradient_accumulation_steps": 4,
        "use_amp": True,
    }


def build_model():
    return {
        "dataset_reader": dataset_reader(),
        "train_data_path": f"{(DATASET_DIR / 'train')}/*",
        "vocabulary": vocabulary(),
        "model": model(),
        "data_loader": data_loader(),
        "distributed": {
            "cuda_devices": list(map(int, os.getenv("SLURM_STEP_GPUS", "0").split(",")))
        },
        "trainer": trainer(),
    }


if __name__ == "__main__":
    json.dump(build_model(), Path("model.json").open("w+"))
