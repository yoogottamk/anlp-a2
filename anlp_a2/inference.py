import numpy as np
import torch
from allennlp_models.lm.modules.token_embedders.bidirectional_lm import (
    BidirectionalLanguageModelTokenEmbedder as BLMTEncoder,
)
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoTokenCharactersIndexer as ETCIndexer,
)
from allennlp.data.tokenizers.token_class import Token

from anlp_a2.config import DATASET_DIR

model_file = str(DATASET_DIR.parent / "model" / "model.tar.gz")
ENCODER = BLMTEncoder(archive_file=model_file, bos_eos_tokens=None)
INDEXER = ETCIndexer()
vocab = ENCODER._lm.vocab


def get_embeddings(sentence: str, word: str):
    idx = sentence.split().index(word)
    sentence = f"<S> {sentence} </S>"
    tokens = [Token(w) for w in sentence.split()]
    character_indices = INDEXER.tokens_to_indices(tokens, vocab)["elmo_tokens"]
    indices_tensor = torch.LongTensor([character_indices])

    emb = ENCODER(indices_tensor).numpy()
    return emb[0][idx]


if __name__ == "__main__":
    print(get_embeddings("i am wearing a big ring", "ring").shape)
    print(get_embeddings("i have a diamond ring", "ring").shape)
    print(get_embeddings("my phone rings", "rings").shape)
