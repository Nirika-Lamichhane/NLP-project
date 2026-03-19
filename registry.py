import logging
from ml.devanagari import DevanagariModel
from ml.transliteration_model import TransliterationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def initialize(self):
        if self._ready:
            return

        logger.info("Loading transliteration model...")
        self.transliterator = TransliterationModel()
        self.transliterator.load()

        logger.info("Loading devanagari + sentiment model...")
        self.devanagari = DevanagariModel()
        self.devanagari.load()

        self._ready = True
        logger.info("All models ready.")

    def is_ready(self):
        return self._ready

registry = ModelRegistry()