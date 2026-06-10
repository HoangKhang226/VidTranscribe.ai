import re
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.utils.logger import logger
from src.config import OLLAMA_MODEL_NAME
from src.utils.llm_factory import make_chat_ollama
from src.utils.translation_cache import TranslationCache

class WordLevelTranslator:
    """
    Dịch từng từ tiếng Anh một cách độc lập, có kĩ xảo ngữ cảnh.
    - Sử dụng cache để tránh re-translate
    - Gọi LLM cho từ mới chưa có trong cache
    """
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, cache: TranslationCache = None):
        self.model_name = model_name
        self.cache = cache or TranslationCache()
        self.llm = make_chat_ollama(model_name=model_name, temperature=0.3)
    
    def translate_word(self, english_word: str, context_sentence: str = "") -> str:
        """
        Dịch một từ tiếng Anh duy nhất.
        - Kiểm tra cache trước
        - Nếu không có, gọi LLM với ngữ cảnh
        """
        # Check cache
        cached = self.cache.get(english_word, word_type="word")
        if cached:
            logger.debug(f"Cache hit: {english_word} -> {cached}")
            return cached
        
        # Not in cache, ask LLM
        logger.debug(f"Cache miss, translating: {english_word}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Bạn là chuyên gia dịch ĐA NGÀNH Anh-Việt (công nghệ, y tế, tài chính, pháp lý, giáo dục, "
             "khoa học, marketing, ẩm thực, thể thao, nghệ thuật, lịch sử...). "
             "Dịch DUY NHẤT từ này, không dịch ngữ cảnh. "
             "Trả về CHỈ bản dịch (1-3 từ), không giải thích. "
             "Nếu là thuật ngữ chuyên ngành, tên sản phẩm/thương hiệu, danh từ riêng, hoặc từ viết tắt thì giữ nguyên tiếng Anh."),
            ("human", 
             f"Dịch từ: '{english_word}'\n"
             f"Ngữ cảnh: {context_sentence}\n"
             f"Trả về chỉ bản dịch:")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({})
            vietnamese = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Clean up response
            vietnamese = re.sub(r'["\']', '', vietnamese).strip()
            
            # Save to cache
            self.cache.set(english_word, vietnamese, word_type="word")
            logger.debug(f"LLM translation: {english_word} -> {vietnamese}")
            
            return vietnamese
        except Exception as e:
            logger.error(f"Failed to translate word '{english_word}': {e}")
            return english_word  # Fallback to original
    
    def translate_words_batch(self, english_words: List[str], context_sentence: str = "") -> Dict[str, str]:
        """Dịch batch các từ."""
        results = {}
        
        # Check cache first
        cached_results = self.cache.get_batch(english_words, word_type="word")
        
        # Translate only words not in cache
        uncached_words = [w for w in english_words if cached_results.get(w) is None]
        
        if uncached_words:
            logger.info(f"Translating {len(uncached_words)} uncached words...")
            for word in uncached_words:
                results[word] = self.translate_word(word, context_sentence)
        
        # Combine cached + newly translated
        for word in english_words:
            if word in results:
                cached_results[word] = results[word]
        
        return cached_results
    
    def save_cache(self):
        """Save cache to disk."""
        self.cache.save_cache()
