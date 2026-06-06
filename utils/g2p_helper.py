# -*- coding: utf-8 -*-
import os
import re
import json
from g2p_en import G2p
from utils.logger import logger

VOWEL_MAP = {
    'AA': 'a', 'AE': 'a', 'AH': 'ơ', 'AO': 'o', 'AW': 'ao', 'AY': 'ai',
    'EH': 'e', 'ER': 'ơ', 'EY': 'ây', 'IH': 'i', 'IY': 'i', 'OW': 'ô',
    'OY': 'oi', 'UH': 'u', 'UW': 'u'
}

CONSONANT_START_MAP = {
    'B': 'b', 'CH': 'ch', 'D': 'đ', 'DH': 'đ', 'F': 'ph', 'G': 'g',
    'HH': 'h', 'JH': 'gi', 'K': 'c', 'L': 'l', 'M': 'm', 'N': 'n',
    'NG': 'ng', 'P': 'p', 'R': 'r', 'S': 'x', 'SH': 'x', 'T': 't',
    'TH': 'th', 'V': 'v', 'W': 'u', 'Y': 'i', 'Z': 'd', 'ZH': 'd'
}

CONSONANT_END_MAP = {
    'K': 'c', 'L': 'o', 'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'T': 't'
}
ENGLISH_LETTERS_MAP = {
    'a': 'e', 'b': 'bi', 'c': 'xi', 'd': 'đi', 'e': 'i', 'f': 'ép',
    'g': 'gi', 'h': 'ếch', 'i': 'ai', 'j': 'giê', 'k': 'kei', 'l': 'eo',
    'm': 'em', 'n': 'en', 'o': 'ô', 'p': 'pi', 'q': 'qui', 'r': 'a',
    's': 'ét', 't': 'ti', 'u': 'iu', 'v': 'vi', 'w': 'đáp liu', 'x': 'ích',
    'y': 'quai', 'z': 'dét'
}

CACHE_FILE_PATH = os.path.join("output", "subtitles", "phonetic_cache.json")
_g2p_instance = None
_cache = None

def _get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        # Lazy initialization
        import nltk
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict', quiet=True)
            
        _g2p_instance = G2p()
    return _g2p_instance

def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
        
    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            logger.info(f"Loaded {len(_cache)} entries from phonetic cache: {CACHE_FILE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load phonetic cache: {e}. Starting fresh.")
            _cache = {}
    else:
        _cache = {}
    return _cache

def _save_cache():
    global _cache
    if _cache is None:
        return
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save phonetic cache: {e}")

def clean_phoneme(ph: str) -> tuple:
    """Bóc tách stress digit (0, 1, 2) ra khỏi phoneme."""
    match = re.match(r'^([A-Z]+)([0-2])?$', ph)
    if not match:
        return ph, False
    base, stress = match.groups()
    return base, stress is not None

def arpabet_to_vietnamese(phonemes: list) -> str:
    syllables = []
    current_onset = []
    
    i = 0
    n = len(phonemes)
    while i < n:
        base, is_vowel = clean_phoneme(phonemes[i])
        if not is_vowel:
            current_onset.append(base)
            i += 1
        else:
            vowel = base
            coda = []
            i += 1
            
            # Gom tất cả phụ âm đứng sau nguyên âm hiện tại
            while i < n:
                next_base, next_is_vowel = clean_phoneme(phonemes[i])
                if next_is_vowel:
                    break
                coda.append(next_base)
                i += 1
            
            # Phân chia Coda và Onset kế tiếp dựa trên Maximal Onset Principle
            if i < n: # Nếu vẫn còn nguyên âm phía sau
                if len(coda) == 0:
                    next_onset_start = []
                elif len(coda) == 1:
                    # Có 1 phụ âm ở giữa -> Nhường làm âm đầu cho từ sau
                    next_onset_start = [coda[0]]
                    coda = []
                else:
                    # Có từ 2 phụ âm trở lên -> Giữ 1 cho từ trước, phần còn lại chuyển cho từ sau
                    next_onset_start = coda[1:]
                    coda = [coda[0]]
            else:
                # Nếu đã là âm tiết cuối cùng của từ -> Ôm hết làm Coda
                next_onset_start = []
                
            syllables.append((current_onset, vowel, coda))
            current_onset = next_onset_start

    # 2. Ánh xạ sang tiếng Việt bồi
    vi_words = []
    for onset, vowel, coda in syllables:
        vi_onset = "".join([CONSONANT_START_MAP.get(c, "") for c in onset])
        vi_vowel = VOWEL_MAP.get(vowel, "")
        
        # Áp dụng bộ lọc nuốt âm cuối (Chỉ giữ lại phụ âm kết thúc hợp lệ)
        vi_coda = ""
        for c in coda:
            if c in CONSONANT_END_MAP:
                vi_coda = CONSONANT_END_MAP[c]
                break 
                
        word = f"{vi_onset}{vi_vowel}{vi_coda}"
        if word:
            vi_words.append(word)
            
    return " ".join(vi_words)

def split_word_parts(word: str) -> list:
    """Tách từ dựa trên CamelCase, ký tự đặc biệt, gạch nối hoặc số."""
    raw_parts = re.split(r'[-_]', word)
    final_parts = []
    for part in raw_parts:
        if not part:
            continue
        # Tách CamelCase, ví dụ: ChatGPT -> Chat, GPT; pH -> p, H
        subparts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+|[\W]+', part)
        if subparts:
            final_parts.extend(subparts)
        else:
            final_parts.append(part)
    return final_parts

def is_abbreviation(word: str) -> bool:
    """Kiểm tra xem một từ có phải là từ viết tắt/acronym không (không phân biệt ngành nghề)."""
    # Nếu là 1 ký tự duy nhất
    if len(word) == 1:
        return word.lower() in ENGLISH_LETTERS_MAP
        
    # Nếu là chữ hoa hoàn toàn (ví dụ: GPT, AI, CRM, DNA)
    if word.isupper():
        return True
        
    # Nếu không chứa bất kỳ nguyên âm tiếng Anh nào
    vowels = set("aeiouy")
    clean = word.lower()
    if not any(char in vowels for char in clean):
        return True
        
    # Một số cụm viết tắt ngắn phổ biến ít nguyên âm
    if len(clean) <= 3 and clean in ["ai", "it", "ip", "ui", "ux", "io", "os", "db", "ph"]:
        return True
        
    return False

def spell_out_word(word: str) -> str:
    """Đánh vần từng chữ cái của từ viết tắt sang tiếng Việt bồi."""
    letters = []
    for char in word.lower():
        if char in ENGLISH_LETTERS_MAP:
            letters.append(ENGLISH_LETTERS_MAP[char])
        else:
            letters.append(char)
    return " ".join(letters)

def transliterate_word(word: str) -> str:
    """Phiên âm một từ đơn hoặc từ ghép tiếng Anh sang tiếng Việt bồi một cách tổng quát."""
    clean_w = word.strip()
    if not clean_w:
        return ""
        
    # 1. Kiểm tra cache trước để tránh phân tích lại
    cache = _load_cache()
    if clean_w.lower() in cache:
        return cache[clean_w.lower()]
        
    # 2. Tách từ ghép / CamelCase / ký tự đặc biệt
    parts = split_word_parts(clean_w)
    
    vi_parts = []
    for part in parts:
        # Nếu là chữ số hoặc ký tự đặc biệt, giữ nguyên
        if re.match(r'^[0-9\W]+$', part):
            vi_parts.append(part)
            continue
            
        # Nếu là từ viết tắt / acronym
        if is_abbreviation(part):
            vi_parts.append(spell_out_word(part))
            continue
            
        # Sử dụng G2P cho từ thông thường
        try:
            g2p = _get_g2p()
            phonemes = g2p(part.lower())
            # Loại bỏ ký tự đặc biệt trong phoneme
            phonemes = [p for p in phonemes if re.match(r'^[A-Z0-9]+$', p)]
            if not phonemes:
                vi_parts.append(part)
                continue
                
            vi_phonetic = arpabet_to_vietnamese(phonemes)
            if vi_phonetic:
                vi_parts.append(vi_phonetic)
            else:
                vi_parts.append(part)
        except Exception as e:
            logger.error(f"Error transliterating part '{part}': {e}")
            vi_parts.append(part)
            
    result = " ".join(vi_parts)
    # Gom khoảng trắng thừa
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Lưu vào cache
    cache[clean_w.lower()] = result
    _save_cache()
    return result

def transliterate_batch(words: list) -> dict:
    """Phiên âm hàng loạt từ tiếng Anh sang tiếng Việt bồi."""
    result = {}
    for word in words:
        clean_w = word.strip()
        if clean_w:
            result[clean_w.lower()] = transliterate_word(clean_w)
    return result
