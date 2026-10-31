# Transcribes the single-word name recordings of PLD FIL with the original pretrained phoneme model
# (facebook/wav2vec2-xlsr-53-espeak-cv-ft), as acoustic evidence for choosing between name readings in
# notebooks/EDA/PLD_FIL_EDA_G2P.ipynb (section 7.8). Needs a GPU and the dataset pushed by PLD_FIL.py.
# Usage: python ./scripts/data_loaders/PLD_FIL_names_asr.py <output.jsonl>
#   e.g. on Colab: colab exec -s <session> -f scripts/data_loaders/PLD_FIL_names_asr.py (writes /content/FIL_names_pretrained_asr.jsonl)
#   then copy it to .data/MDC/UP-DSP-PLD/PLD/FIL_names_pretrained_asr.jsonl
import io
import re
import sys
import json

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
NAME_FILES = re.compile(r"male|female|Name|urname|ities|andmark|otel|ompan|uisine|irline", re.I)

DATASET_ID = "jomszxcvb/PLD-FIL"


def main(args):
    out_path = args[0] if args else "/content/FIL_names_pretrained_asr.jsonl"
    ds = load_dataset(DATASET_ID)
    rows = []
    for split in ds:
        d = ds[split].cast_column("audio", Audio(decode=False))
        keep = [i for i, (p, t) in enumerate(zip(d["prompt_file"], d["text"])) if NAME_FILES.search(p) and " " not in t.strip()]
        rows.append(d.select(keep))
    print("name utterances:", sum(len(r) for r in rows))

    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cuda").eval()
    vocab = json.load(open(hf_hub_download(MODEL_ID, "vocab.json")))
    id2tok = {i: t for t, i in vocab.items()}
    blank = vocab["<pad>"]

    with open(out_path, "w") as out:
        for d in rows:
            for start in range(0, len(d), 32):
                batch = d[start : start + 32]
                audio = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in batch["audio"]]
                inputs = fe(audio, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt").to("cuda")
                with torch.no_grad(), torch.autocast("cuda"):
                    ids = model(inputs.input_values, attention_mask=inputs.attention_mask).logits.argmax(-1).cpu().tolist()
                for i, seq in enumerate(ids):
                    toks, prev = [], None
                    for t in seq:
                        if t != prev and t != blank and not id2tok[t].startswith("<"):
                            toks.append(id2tok[t])
                        prev = t
                    out.write(json.dumps({"id": batch["id"][i], "text": batch["text"][i], "speaker_id": batch["speaker_id"][i], "tokens": toks}, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
