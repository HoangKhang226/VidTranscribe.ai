import json
import os
from typing import Dict, Optional, Set
from src.utils.logger import logger
from src.utils.g2p_helper import transliterate_batch
from src.config import PHONETIC_CACHE_DIR

class PhoneticCacheManager:
    """
    Quản lý phonetic cache cho mỗi domain.
    - Load domain-specific cache
    - Add new tech terms với G2P
    - Update cache file
    - Multi-domain aware (hoạt động cho mọi domain)
    """
    
    def __init__(self, domain_name: str = "General"):
        self.domain_name = domain_name
        self.cache_file = os.path.join(
            PHONETIC_CACHE_DIR, 
            f"domain_{domain_name}.json"
        )
        self.cache: Dict[str, str] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load domain-specific phonetic cache từ file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    logger.info(f"Loaded phonetic cache for domain '{self.domain_name}': {len(cache)} entries")
                    return cache
            except Exception as e:
                logger.warning(f"Failed to load cache {self.cache_file}: {e}")
        
        # Fallback: load general cache
        if self.domain_name != "General":
            general_cache_file = os.path.join(PHONETIC_CACHE_DIR, "domain_general.json")
            if os.path.exists(general_cache_file):
                try:
                    with open(general_cache_file, 'r', encoding='utf-8') as f:
                        logger.info(f"Using general phonetic cache as fallback")
                        return json.load(f)
                except:
                    pass
        
        logger.info(f"Starting fresh phonetic cache for domain '{self.domain_name}'")
        return {}
    
    def _save_cache(self):
        """Save cache xuống file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved phonetic cache for '{self.domain_name}': {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get(self, term: str) -> Optional[str]:
        """Lấy phonetic reading từ cache."""
        key = term.lower().strip()
        return self.cache.get(key)
    
    def has(self, term: str) -> bool:
        """Check xem term có trong cache chưa."""
        key = term.lower().strip()
        return key in self.cache
    
    def add_or_skip(self, terms: Set[str]) -> Dict[str, str]:
        """
        Thêm terms vào cache nếu chưa có.
        - Nếu đã có → skip
        - Nếu chưa có → generate G2P + add
        
        Return: {"added": {...}, "skipped": [...]}
        """
        added = {}
        skipped = []
        
        # Filter: chỉ lấy terms chưa có trong cache
        new_terms = []
        for term in terms:
            clean_term = term.lower().strip()
            if not self.has(clean_term):
                new_terms.append(clean_term)
            else:
                skipped.append(clean_term)
        
        if not new_terms:
            logger.debug(f"No new terms to add. Skipped {len(skipped)} terms already in cache.")
            return {"added": {}, "skipped": skipped}
        
        # Generate G2P cho các terms mới
        logger.info(f"Generating G2P for {len(new_terms)} new terms: {new_terms[:5]}...")
        try:
            g2p_map = transliterate_batch(new_terms, domain=self.domain_name)
            
            # Add to cache
            for term, phonetic in g2p_map.items():
                clean_term = term.lower().strip()
                self.cache[clean_term] = phonetic
                added[clean_term] = phonetic
            
            logger.info(f"Added {len(added)} new terms to phonetic cache")
            return {"added": added, "skipped": skipped}
        
        except Exception as e:
            logger.error(f"Failed to generate G2P: {e}")
            return {"added": {}, "skipped": skipped + new_terms}
    
    def add_batch(self, terms: Set[str]) -> Dict:
        """Add batch of terms, save to cache."""
        result = self.add_or_skip(terms)
        self._save_cache()
        return result
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "domain": self.domain_name,
            "total_entries": len(self.cache),
            "cache_file": self.cache_file
        }
    
    @staticmethod
    def get_domain_name_from_context(context: Dict) -> str:
        """Extract domain name từ context."""
        return context.get("topic", "General")
