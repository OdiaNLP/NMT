import torch
import sentencepiece as spm


def load_tokenizers(src_tok_path, trg_tok_path):
    sp_bpe_src = spm.SentencePieceProcessor()
    sp_bpe_trg = spm.SentencePieceProcessor()
    # load src tokenizer
    sp_bpe_src.load(f"{src_tok_path}")
    # load trg tokenizer
    sp_bpe_trg.load(f"{trg_tok_path}")
    return sp_bpe_src, sp_bpe_trg


def load_vocab(src_vocab_path, trg_vocab_path):
    # load src vocab
    with open(src_vocab_path, "rb") as f:
        src_vocab = torch.load(f)
    # load trg vocab
    with open(trg_vocab_path, "rb") as f:
        trg_vocab = torch.load(f)
    return src_vocab, trg_vocab


if __name__ == "__main__":
    # load tokenizers
    _sp_bpe_src, _sp_bpe_trg = load_tokenizers(
        "models/bpe_en.model", "models/bpe_od.model"
    )
    _src_vocab, _trg_vocab = load_vocab("models/SRC_vocab.pkl", "models/TRG_vocab.pkl")
    print(f"src vocab size: {len(_src_vocab)}")
    print(f"trg vocab size: {len(_trg_vocab)}")
