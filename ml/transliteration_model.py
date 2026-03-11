# ml/transliteration.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

# -------------------------
# Paths and device
# -------------------------

MODEL_PATH = "models/transliteration_model"
DEVICE = "cpu"

# -------------------------
# Load tokenizer and model
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).to(DEVICE)
model.eval()

# -------------------------
# Load Indic processor
# -------------------------

ip = IndicProcessor(inference=True)

# -------------------------
# Transliteration function
# -------------------------

def transliterate(text):
    """
    Converts English / Roman text to Devanagari Nepali.
    """

    if isinstance(text, str):
        text = [text]

    # define source and target languages
    src_lang = "eng_Latn"
    tgt_lang = "npi_Deva"

    # preprocess
    batch = ip.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
        )

    # decode and postprocess
    outputs = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    outputs = ip.postprocess_batch(outputs, lang=tgt_lang)

    return outputs[0]