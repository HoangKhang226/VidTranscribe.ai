import os
import json
import shutil
from datetime import datetime
from src.utils.logger import logger


def _now_str() -> str:
    """Trả về timestamp dạng chuỗi (YYYY-MM-DD HH:MM)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# Schema chuẩn cho một mục từ điển ngành (đa ngành):
#   {
#     "phonetic":     "tiếng việt bồi",   # cách đọc bồi
#     "explanation":  "giải thích sơ qua",# có thể rỗng
#     "date_added":   "YYYY-MM-DD HH:MM",
#     "added_by":     "AI" | "Human",
#     "last_updated": "YYYY-MM-DD HH:MM"
#   }
DICT_FIELDS = ("phonetic", "explanation", "date_added", "added_by", "last_updated")


def normalize_dict_entry(details, *, default_added_by: str = "Human") -> dict:
    """
    Chuẩn hóa một mục từ điển về schema mới, tương thích ngược với schema cũ {vi, pho}.
    """
    now = _now_str()
    if not isinstance(details, dict):
        # Trường hợp giá trị chỉ là chuỗi phonetic
        details = {"phonetic": str(details)}

    phonetic = details.get("phonetic", details.get("pho", "")) or ""
    explanation = details.get("explanation", details.get("vi", "")) or ""
    return {
        "phonetic": str(phonetic).strip(),
        "explanation": str(explanation).strip(),
        "date_added": details.get("date_added") or now,
        "added_by": details.get("added_by") or default_added_by,
        "last_updated": details.get("last_updated") or details.get("date_added") or now,
    }


class DatabaseManager:
    """Quản lý các thao tác đọc/ghi cơ sở dữ liệu (từ điển và cache)."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.db_dir = os.path.join(os.path.dirname(__file__))
        self.dict_dir = os.path.join(self.db_dir, "dictionaries")
        self.cache_dir = os.path.join(self.db_dir, "phonetic_caches")
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.dict_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Khởi tạo Global Cache nếu chưa có
        self.global_cache_path = os.path.join(self.cache_dir, "global_cache.json")
        if not os.path.exists(self.global_cache_path):
            with open(self.global_cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
                
        self.migrate_old_cache()

    def get_dictionary_path(self, domain: str) -> str:
        domain_clean = domain.lower().replace(" ", "_")
        return os.path.join(self.dict_dir, f"{domain_clean}.json")

    def get_domain_cache_path(self, domain: str) -> str:
        domain_clean = domain.lower().replace(" ", "_")
        return os.path.join(self.cache_dir, f"domain_{domain_clean}.json")

    @staticmethod
    def _term_lookup_key(term: str) -> str:
        """Chuẩn hóa key lookup nhưng vẫn giữ schema file dictionary thân thiện với người đọc."""
        return (term or "").strip().lower()

    def load_dictionary_phonetics(self, domain: str = None) -> dict:
        """
        Tải curated phonetic lexicon từ src/db/dictionaries.

        Đây là tầng ưu tiên cao hơn phonetic cache:
        - dictionary: human/AI curated, có metadata và giải thích
        - cache: kết quả G2P sinh tự động để tăng tốc
        """
        if not domain or str(domain).lower() == "auto detect":
            return {}

        dictionary = self.load_dictionary(domain)
        phonetics = {}
        for term, details in dictionary.items():
            if not isinstance(details, dict):
                continue
            phonetic = (details.get("phonetic") or "").strip()
            if phonetic:
                phonetics[self._term_lookup_key(term)] = phonetic
        return phonetics

    def load_dictionary(self, domain: str) -> dict:
        path = self.get_dictionary_path(domain)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Chuẩn hóa mọi mục về schema mới (tương thích ngược schema cũ {vi, pho})
            return {eng: normalize_dict_entry(details) for eng, details in raw.items()}
        except Exception as e:
            logger.error(f"Lỗi đọc từ điển {domain}: {e}")
            return {}

    def save_dictionary(self, domain: str, data: dict):
        path = self.get_dictionary_path(domain)
        # Chuẩn hóa trước khi lưu để file luôn đúng schema
        normalized = {eng: normalize_dict_entry(details) for eng, details in data.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=4)

    def upsert_term(self, domain: str, english: str, phonetic: str = "",
                    explanation: str = "", added_by: str = "Human") -> dict:
        """
        Thêm mới hoặc cập nhật một thuật ngữ trong từ điển ngành.
        - Nếu chưa có: tạo mới với date_added = now, added_by như truyền vào.
        - Nếu đã có: giữ nguyên date_added & added_by gốc (không ghi đè công sức người dùng),
          chỉ cập nhật phonetic/explanation nếu có giá trị mới, và đặt last_updated = now.
        Trả về mục từ điển sau khi cập nhật.
        """
        english = (english or "").strip()
        if not english:
            return {}
        now = _now_str()
        data = self.load_dictionary(domain)
        key = english

        if key in data:
            entry = data[key]
            changed = False
            if phonetic and phonetic.strip() and phonetic.strip() != entry.get("phonetic"):
                entry["phonetic"] = phonetic.strip()
                changed = True
            if explanation and explanation.strip() and explanation.strip() != entry.get("explanation"):
                entry["explanation"] = explanation.strip()
                changed = True
            if changed:
                entry["last_updated"] = now
            data[key] = entry
        else:
            data[key] = {
                "phonetic": (phonetic or "").strip(),
                "explanation": (explanation or "").strip(),
                "date_added": now,
                "added_by": added_by or "Human",
                "last_updated": now,
            }

        self.save_dictionary(domain, data)
        return data[key]
            
    def load_phonetic_cache(self, domain: str = None) -> dict:
        """Tải cache theo dạng Hierarchical (Local Domain ưu tiên hơn Global)."""
        cache = {}
        # 1. Tải Global
        try:
            with open(self.global_cache_path, "r", encoding="utf-8") as f:
                cache.update(json.load(f))
        except Exception:
            pass
            
        # 2. Ghi đè bằng Domain Cache (nếu có)
        if domain and domain.lower() != "auto detect":
            domain_path = self.get_domain_cache_path(domain)
            if os.path.exists(domain_path):
                try:
                    with open(domain_path, "r", encoding="utf-8") as f:
                        domain_data = json.load(f)
                        cache.update(domain_data)
                except Exception:
                    pass
        return {self._term_lookup_key(term): phonetic for term, phonetic in cache.items() if str(phonetic).strip()}

    def load_pronunciation_overrides(self, domain: str = None) -> dict:
        """
        Tải pronunciation overrides theo thứ tự ưu tiên chuẩn:
        1. Global/domain phonetic cache
        2. Domain dictionary phonetic ghi đè cache

        Hàm này là API chính cho G2P helper để tránh trùng trách nhiệm giữa
        dictionaries/ và phonetic_caches/.
        """
        overrides = self.load_phonetic_cache(domain)
        overrides.update(self.load_dictionary_phonetics(domain))
        return overrides

    def save_to_domain_cache(self, domain: str, term: str, phonetic: str):
        """Lưu một từ mới học được vào Domain Cache."""
        if not domain or domain.lower() == "auto detect":
            path = self.global_cache_path
        else:
            path = self.get_domain_cache_path(domain)
            
        cache = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        cache[term] = phonetic
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

    def migrate_old_cache(self):
        """Chuyển dữ liệu từ phonetic_cache.json cũ sang global_cache.json"""
        old_cache_path = os.path.join(self.db_dir, "..", "..", "output", "subtitles", "phonetic_cache.json")
        if os.path.exists(old_cache_path):
            try:
                with open(old_cache_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                
                with open(self.global_cache_path, "r", encoding="utf-8") as f:
                    global_data = json.load(f)
                    
                global_data.update(old_data)
                with open(self.global_cache_path, "w", encoding="utf-8") as f:
                    json.dump(global_data, f, ensure_ascii=False, indent=4)
                    
                logger.info("Đã di chuyển dữ liệu từ phonetic_cache cũ sang db/phonetic_caches/global_cache.json")
                # Xóa file cũ để không migrate lại
                os.remove(old_cache_path)
            except Exception as e:
                logger.error(f"Lỗi migrate cache: {e}")

db = DatabaseManager()
