from dataclasses import dataclass

from transformers import T5ForConditionalGeneration, T5Tokenizer


@dataclass
class T5TranslatorModel:
    _model: T5ForConditionalGeneration | None = None
    _tokenizer: T5Tokenizer | None = None

    _model_name: str = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
    _device: str = 'cpu'
    _prefix: str = 'translate to en: '


    def __post_init__(self):
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._model_name,
            cache_dir="./model_cache"
        )
        self._model.to(self._device)
        self._tokenizer = T5Tokenizer.from_pretrained(
            self._model_name,
            cache_dir="./model_cache"
        )

    def __call__(self, search_query: str) -> str:
        input_ids = self._tokenizer(search_query, return_tensors="pt")
        generated_tokens = self._model.generate(**input_ids.to(self._device))
        result = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return result[0]
