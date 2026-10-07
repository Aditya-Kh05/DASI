"""
Text Preprocessor — Hindi/English/Code-Mixed text normalization.

Handles the linguistic reality of Indian digital arrest scams, which frequently
use Hindi-English code-mixing: "Aapka Aadhaar card money laundering mein use hua hai."

Preprocessing steps:
  1. Unicode normalization (Devanagari + Latin)
  2. Phone number / Aadhaar number masking
  3. Profanity / filler word handling
  4. Transliteration normalization (Hinglish → standard)
  5. Tokenization-ready output
"""

import re
from typing import List, Optional


class TextPreprocessor:
    """Normalize and clean text from scam communications."""
    
    # Common Hindi transliterations used in scams
    HINGLISH_NORMALIZATION = {
        "aadhar": "aadhaar", "adhar": "aadhaar", "aadhaar": "aadhaar",
        "paisa": "money", "rupaye": "rupees", "lakh": "lakh",
        "crore": "crore", "giraftaar": "arrest", "giraftar": "arrest",
        "warrant": "warrant", "kachehri": "court", "adalat": "court",
        "thana": "police station", "daroga": "inspector",
        "fir": "fir", "FIR": "FIR",
    }
    
    # Patterns indicating sensitive data to mask
    AADHAAR_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b')
    PAN_PATTERN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
    ACCOUNT_PATTERN = re.compile(r'\b\d{9,18}\b')
    
    # Scam-specific authority keywords (for feature extraction)
    AUTHORITY_TERMS = {
        "cbi", "police", "cyber cell", "cyber crime", "rbi",
        "enforcement directorate", "ed", "narcotics", "ncb",
        "income tax", "customs", "court", "judge", "magistrate",
        "supreme court", "high court", "commissioner", "dcp",
        "sp", "ips", "ias", "officer", "inspector", "sub inspector",
        "trai", "telecom", "doi", "interpol", "headquarters",
        "pmla", "money laundering", "digital arrest", "fir",
    }
    
    URGENCY_TERMS = {
        "immediately", "urgent", "right now", "turant", "abhi",
        "arrest", "giraftaar", "warrant", "jail", "cancel",
        "block", "freeze", "frozen", "suspend", "terminate", "seize",
        "raid", "legal action", "case registered", "fir filed",
        "non-bailable", "life imprisonment", "punishment", "aadhaar",
        "section 420", "uapa", "digital arrest", "investigation",
    }
    
    ISOLATION_TERMS = {
        "don't tell anyone", "kisi ko mat batana", "secret",
        "confidential", "classified", "national security",
        "gag order", "court order", "media blackout",
        "family ko mat batao", "don't disconnect",
        "stay on call", "video call pe raho", "stay on this line",
        "do not tell", "do not disconnect",
    }
    
    FINANCIAL_TERMS = {
        "transfer", "send money", "safe account", "rbi account",
        "verification amount", "refundable deposit", "fine",
        "penalty", "settlement", "upi", "neft", "rtgs", "imps",
        "bank account", "google pay", "phonepe", "paytm",
        "bitcoin", "crypto", "gift card", "verification account",
        "security deposit", "refundable", "funds", "deposit",
    }
    
    def __init__(self):
        self._compiled_patterns = {
            "aadhaar": self.AADHAAR_PATTERN,
            "phone": self.PHONE_PATTERN,
            "pan": self.PAN_PATTERN,
            "account": self.ACCOUNT_PATTERN,
        }
    
    def preprocess(self, text: str, mask_pii: bool = True) -> str:
        """Full preprocessing pipeline.
        
        Args:
            text: Raw input text (Hindi, English, or code-mixed)
            mask_pii: Whether to mask PII (Aadhaar, phone numbers, etc.)
            
        Returns:
            Cleaned, normalized text ready for model input
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: Basic cleaning
        text = self._clean_basic(text)
        
        # Step 2: Mask sensitive data
        if mask_pii:
            text = self._mask_pii(text)
        
        # Step 3: Normalize Hinglish
        text = self._normalize_hinglish(text)
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keyword_features(self, text: str) -> dict:
        """Extract rule-based keyword features from text.
        
        These complement the neural model's semantic understanding with
        explicit keyword matching for high-precision scam indicators.
        
        Returns:
            Dictionary with counts/scores for each category
        """
        text_lower = text.lower()
        
        authority_hits = sum(1 for term in self.AUTHORITY_TERMS if term in text_lower)
        urgency_hits = sum(1 for term in self.URGENCY_TERMS if term in text_lower)
        isolation_hits = sum(1 for term in self.ISOLATION_TERMS if term in text_lower)
        financial_hits = sum(1 for term in self.FINANCIAL_TERMS if term in text_lower)
        
        # Normalize by category size for fair comparison
        return {
            "authority_keyword_density": authority_hits / len(self.AUTHORITY_TERMS),
            "urgency_keyword_density": urgency_hits / len(self.URGENCY_TERMS),
            "isolation_keyword_density": isolation_hits / len(self.ISOLATION_TERMS),
            "financial_keyword_density": financial_hits / len(self.FINANCIAL_TERMS),
            "total_scam_keywords": authority_hits + urgency_hits + isolation_hits + financial_hits,
            "authority_hits": authority_hits,
            "urgency_hits": urgency_hits,
            "isolation_hits": isolation_hits,
            "financial_hits": financial_hits,
        }
    
    def _clean_basic(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        # Normalize unicode quotes and dashes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2014', '-').replace('\u2013', '-')
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        return text
    
    def _mask_pii(self, text: str) -> str:
        """Mask personally identifiable information."""
        text = self.AADHAAR_PATTERN.sub('[AADHAAR]', text)
        text = self.PHONE_PATTERN.sub('[PHONE]', text)
        text = self.PAN_PATTERN.sub('[PAN]', text)
        # Be more selective with account numbers — only mask in financial context
        if any(term in text.lower() for term in ["account", "a/c", "transfer"]):
            text = self.ACCOUNT_PATTERN.sub('[ACCOUNT_NUM]', text)
        return text
    
    def _normalize_hinglish(self, text: str) -> str:
        """Normalize common Hinglish transliterations."""
        words = text.split()
        normalized = []
        for word in words:
            lower = word.lower()
            if lower in self.HINGLISH_NORMALIZATION:
                normalized.append(self.HINGLISH_NORMALIZATION[lower])
            else:
                normalized.append(word)
        return ' '.join(normalized)
    
    def get_language_mix(self, text: str) -> dict:
        """Detect the language composition of the text.
        
        Returns approximate ratio of Hindi (Devanagari) vs Latin script.
        Useful as a feature — scam calls often have distinctive code-mixing patterns.
        """
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total = devanagari_chars + latin_chars
        
        if total == 0:
            return {"hindi_ratio": 0.0, "english_ratio": 0.0, "script": "unknown"}
        
        hindi_ratio = devanagari_chars / total
        english_ratio = latin_chars / total
        
        if hindi_ratio > 0.7:
            script = "hindi"
        elif english_ratio > 0.7:
            script = "english"
        else:
            script = "code_mixed"
        
        return {
            "hindi_ratio": round(hindi_ratio, 3),
            "english_ratio": round(english_ratio, 3),
            "script": script,
        }
