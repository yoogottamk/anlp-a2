import torch
from allennlp_models.lm.modules.token_embedders.bidirectional_lm import (
    BidirectionalLanguageModelTokenEmbedder as BLMTEncoder,
)
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoTokenCharactersIndexer as ETCIndexer,
)
from allennlp.data.tokenizers.token_class import Token

from anlp_a2.config import DATASET_DIR

if __name__ == "__main__":
    model_file = str(DATASET_DIR.parent / "model" / "model.tar.gz")
    sentence = "this is a test sentence"
    tokens = [Token(word) for word in sentence.split()]

    encoder = BLMTEncoder(archive_file=model_file, bos_eos_tokens=None)

    indexer = ETCIndexer()
    vocab = encoder._lm.vocab
    character_indices = indexer.tokens_to_indices(tokens, vocab)["elmo_tokens"]

    indices_tensor = torch.LongTensor([character_indices])

    embeddings = encoder(indices_tensor)[0]
    print(embeddings)
