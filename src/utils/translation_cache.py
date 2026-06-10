import json
import os
from typing import Dict, Optional
from src.utils.logger import logger
from src.config import SUBTITLES_DIR

class TranslationCache:
    """
    Cache hệ thống dịch từ tiếng Anh → Tiếng Việt (đa ngành).
    - Lưu các thuật ngữ ngành cần giữ nguyên (word_type='tech') để tránh dịch lại
    - Lưu các từ phổ thông đã dịch qua LLM (word_type='word')
    """
    
    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_file = os.path.join(SUBTITLES_DIR, "translation_cache.json")
        
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache từ file, nếu không có thì tạo mới."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded translation cache với {len(data)} entries")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Creating new cache.")
        
        return {}
    
    def save_cache(self):
        """Save cache xuống file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved translation cache ({len(self.cache)} entries)")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get(self, english_word: str, word_type: str = "tech") -> Optional[str]:
        """
        Lấy bản dịch từ cache.
        word_type: "tech" (thuật ngữ ngành giữ nguyên) hoặc "word" (từ phổ thông đã dịch)
        """
        key = english_word.lower().strip()
        if key in self.cache:
            entry = self.cache[key]
            if entry.get("type") == word_type:
                return entry.get("translation")
        return None
    
    def set(self, english_word: str, vietnamese_translation: str, word_type: str = "tech"):
        """
        Lưu bản dịch vào cache.
        word_type: "tech" (thuật ngữ ngành giữ nguyên) hoặc "word" (từ phổ thông đã dịch)
        """
        key = english_word.lower().strip()
        self.cache[key] = {
            "type": word_type,
            "translation": vietnamese_translation.strip(),
            "english": english_word
        }
    
    def get_batch(self, english_words: list, word_type: str = "tech") -> Dict[str, Optional[str]]:
        """Lấy batch dịch từ cache."""
        return {
            word: self.get(word, word_type)
            for word in english_words
        }
    
    def set_batch(self, translations: Dict[str, str], word_type: str = "tech"):
        """Lưu batch dịch vào cache."""
        for english, vietnamese in translations.items():
            self.set(english, vietnamese, word_type)
        self.save_cache()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        tech_count = sum(1 for v in self.cache.values() if v.get("type") == "tech")
        word_count = sum(1 for v in self.cache.values() if v.get("type") == "word")
        return {
            "total": len(self.cache),
            "tech_terms": tech_count,
            "words": word_count
        }
