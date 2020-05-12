import os
from datetime import datetime
from flask import Flask, render_template, request

from .form_model import InputForm
from .data_utils import load_tokenizers, load_vocab
from .model_utils import load_model
from .translate_utils import (
    translate_sentence,
    tokenize_src,
    detokenize_trg,
)

app = Flask(__name__)
template_name = "my_view"
responses_path = "responses/logs.txt"
os.makedirs("responses", exist_ok=True)
sp_bpe_src, sp_bpe_trg = load_tokenizers(
    "nmt/models/bpe_en.model", "nmt/models/bpe_od.model"
)
# load vocab
SRC_vocab, TRG_vocab = load_vocab(
    "nmt/models/SRC_vocab.pkl", "nmt/models/TRG_vocab.pkl"
)
# load model
model = load_model("nmt/models/model.pt", SRC_vocab, TRG_vocab)
if responses_path is not None:
    with open(responses_path, "a", encoding="utf-8") as f:
        f.write(
            f"\nstarting app.. "
            f'[{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'
            f"\n"
        )


@app.route("/translate", methods=["GET", "POST"])
def index():
    form = InputForm(request.form)
    if request.method == "POST" and form.validate():
        # tokenize
        sentence = tokenize_src(form.src_text.data, sp_bpe_src)

        # translate
        translation, _ = translate_sentence(
            sentence, SRC_vocab, TRG_vocab, model, model.device
        )

        result = detokenize_trg(translation, sp_bpe_trg)

    else:
        result = None

    if result is not None:
        result = f"Odia Translation: {result}"

        if responses_path is not None:
            with open(responses_path, "a", encoding="utf-8") as _f:
                _f.write(
                    f"\n\tNEW REQUEST 🤩 @"
                    f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                    f"\tEnglish Text: {form.src_text.data}\n"
                    f"\t{result}\n"
                )

    return render_template(template_name + ".html", form=form, result=result)
