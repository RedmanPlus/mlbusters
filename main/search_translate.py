from dataclasses import dataclass

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class OpusTranslatorModel:
    _model: AutoModelForSeq2SeqLM | None = None
    _tokenizer: AutoTokenizer | None = None

    _model_name: str = "Helsinki-NLP/opus-mt-ru-en"
    _device: str = "cpu"

    def __post_init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir="./model_cache"
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name,
            cache_dir="./model_cache"
        )


    def __call__(self, search_query: str) -> str:
        input_ids = self._tokenizer.encode(search_query, return_tensors="pt")
        output_ids = self._model.generate(input_ids.to(self._device), max_new_tokens=100)
        en_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return en_text

