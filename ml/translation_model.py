import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from ml.base import BaseMLModel

MODEL_PATH = "models/transliteration_model"
DEVICE     = "cpu"

class TransliterationModel(BaseMLModel):

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def predict(self, text: str):
        if isinstance(text, str):
            text = [text]

        src_lang = "eng_Latn"
        tgt_lang = "npi_Deva"

        batch = self.ip.preprocess_batch(
            text, src_lang=src_lang, tgt_lang=tgt_lang
        )

        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=1,
            )

        outputs = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        outputs = self.ip.postprocess_batch(outputs, lang=tgt_lang)
        return outputs[0]