from itertools import combinations

import numpy as np
import torch
from allennlp_models.lm.modules.token_embedders.bidirectional_lm import (
    BidirectionalLanguageModelTokenEmbedder as BLMTEncoder,
)
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoTokenCharactersIndexer as ETCIndexer,
)
from allennlp.data.tokenizers.token_class import Token
from scipy.spatial.distance import cosine, euclidean

from anlp_a2.config import DATASET_DIR

model_file = str(DATASET_DIR.parent / "model" / "model.tar.gz")
ENCODER = BLMTEncoder(archive_file=model_file, bos_eos_tokens=None)
ENCODER.eval()
INDEXER = ETCIndexer()
vocab = ENCODER._lm.vocab


def get_embeddings(sentence: str, word: str):
    idx = sentence.split().index(word)
    sentence = f"<S> {sentence} </S>"
    tokens = [Token(w) for w in sentence.split()]
    character_indices = INDEXER.tokens_to_indices(tokens, vocab)["elmo_tokens"]
    indices_tensor = torch.LongTensor([character_indices])

    emb = ENCODER(indices_tensor).detach().numpy()
    return emb[0][idx]

if __name__ == "__main__":
    N = int(input("#sentences: "))
    words = []
    sentences = []
    embeddings = []

    idx_list = list(combinations(range(N), 2))

    for _ in range(N):
        sentence = input("Enter sentence: ").strip()
        word = input("Enter word: ").strip()
        sentences.append(sentence)
        words.append(word)
        embeddings.append(get_embeddings(sentence, word))

    print()
    for w1, w2 in idx_list:
        print("===" * 20)
        print(sentences[w1])
        print(sentences[w2])
        print("---" * 20)
        print(f"{words[w1]}[{w1}] <-> {words[w2]}[{w2}]")
        print(f"Euclidean Distance: {euclidean(embeddings[w1], embeddings[w2])}")
        print(f"   Cosine Distance: {cosine(embeddings[w1], embeddings[w2])}")
        print("===" * 20)
