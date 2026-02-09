"""
Utility functions for cross-lingual explanation consistency paper.
Includes: cyber/domain cue patterns, text cleaning, CTAM metrics, overlap metrics.
"""

import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ====================
# Cyber / Domain Cues
# ====================

CYBER_PATTERNS = [
    # CVE pattern
    r'\bcve[-_]?\d{4}[-_]?\d+\b',
    # Common cyber keywords (case-insensitive via flags later)
    r'\bransomware\b', r'\bphishing\b', r'\bddos\b', r'\bmalware\b',
    r'\bexploit\b', r'\bzero[- ]?day\b', r'\bbreach\b', r'\bhack\b',
    r'\bsql[- ]?injection\b', r'\bfirewall\b', r'\bsoc\b',
    r'\bvuln(erability)?\b', r'\bthreat\b', r'\battack\b',
    r'\bbotnet\b', r'\btrojan\b', r'\bbackdoor\b', r'\brootkit\b',
    r'\bspyware\b', r'\badware\b', r'\bcrypto[- ]?jacking\b',
]

# Compile patterns (case-insensitive)
CYBER_REGEX = [re.compile(p, re.IGNORECASE) for p in CYBER_PATTERNS]

# Multilingual cyber keywords for ES/FR (minimal, expand if needed)
CYBER_MULTILANG = {
    'en': [
        'ransomware', 'phishing', 'ddos', 'malware', 'exploit', 'breach',
        'hack', 'firewall', 'vuln', 'threat', 'attack', 'botnet'
    ],
    'es': [
        'ransomware', 'phishing', 'ddos', 'malware', 'exploit', 
        'brecha', 'hackeo', 'cortafuegos', 'vulnerabilidad', 'amenaza', 'ataque'
    ],
    'fr': [
        'ransomware', 'hameçonnage', 'phishing', 'ddos', 'malware', 'exploit',
        'fuite', 'piratage', 'pare-feu', 'vulnérabilité', 'menace', 'attaque'
    ],
}


def has_cyber_cue(text: str, lang: str = 'en') -> bool:
    """
    Check if text contains any cyber cue (pattern or keyword).
    Uses CYBER_REGEX for EN, and keyword list for ES/FR.
    """
    text_lower = text.lower()
    
    # Check regex patterns (work for all languages)
    for regex in CYBER_REGEX:
        if regex.search(text):
            return True
    
    # Check multilingual keywords
    keywords = CYBER_MULTILANG.get(lang, CYBER_MULTILANG['en'])
    for kw in keywords:
        if kw.lower() in text_lower:
            return True
    
    return False


def extract_cyber_tokens(text: str, lang: str = 'en') -> List[str]:
    """
    Extract cyber-related tokens from text.
    Returns list of matched cyber cues (lowercased).
    """
    matches = []
    text_lower = text.lower()
    
    # Regex matches
    for regex in CYBER_REGEX:
        found = regex.findall(text)
        matches.extend([m.lower() for m in found])
    
    # Keyword matches
    keywords = CYBER_MULTILANG.get(lang, CYBER_MULTILANG['en'])
    for kw in keywords:
        if kw.lower() in text_lower:
            matches.append(kw.lower())
    
    return list(set(matches))  # deduplicate


# ====================
# Text Cleaning (minimal)
# ====================

def minimal_clean(text: str) -> str:
    """
    Minimal cleaning: strip extra whitespace, keep hashtags/mentions/URLs.
    """
    if not isinstance(text, str):
        return ""
    # Remove duplicate whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ====================
# XAI Metrics: CTAM (CyberTerm Attribution Mass)
# ====================

def compute_ctam(
    tokens: List[str],
    attributions: np.ndarray,
    lang: str = 'en',
    normalize: bool = True
) -> float:
    """
    Compute CyberTerm Attribution Mass (CTAM):
    
    CTAM = sum(attributions of cyber tokens) / sum(positive attributions)
    
    Parameters:
    - tokens: list of tokens (strings)
    - attributions: array of attribution scores (same length as tokens)
    - lang: language code ('en', 'es', 'fr')
    - normalize: if True, divide by total positive attribution (default)
    
    Returns:
    - CTAM score (0 to 1 if normalized)
    """
    if len(tokens) != len(attributions):
        raise ValueError(f"tokens and attributions must have same length: {len(tokens)} vs {len(attributions)}")
    
    # Identify cyber tokens
    cyber_mask = np.array([has_cyber_cue(tok, lang) for tok in tokens])
    
    # Sum attributions
    cyber_attr_sum = attributions[cyber_mask].sum()
    
    if normalize:
        # Normalize by total positive attribution
        pos_attr_sum = attributions[attributions > 0].sum()
        if pos_attr_sum == 0:
            return 0.0
        return cyber_attr_sum / pos_attr_sum
    else:
        return float(cyber_attr_sum)


# ====================
# XAI Metrics: Top-k Cyber Token Overlap
# ====================

def top_k_cyber_overlap(
    tokens_en: List[str],
    attributions_en: np.ndarray,
    tokens_tgt: List[str],
    attributions_tgt: np.ndarray,
    k: int = 10,
    lang_tgt: str = 'es'
) -> Dict[str, float]:
    """
    Compute top-k cyber token overlap between EN and target language.
    
    Returns:
    - jaccard: Jaccard similarity of cyber tokens in top-k
    - overlap_count: number of overlapping cyber tokens
    """
    # Get top-k tokens by attribution (descending)
    en_topk_idx = np.argsort(attributions_en)[-k:]
    tgt_topk_idx = np.argsort(attributions_tgt)[-k:]
    
    en_topk_tokens = [tokens_en[i] for i in en_topk_idx]
    tgt_topk_tokens = [tokens_tgt[i] for i in tgt_topk_idx]
    
    # Extract cyber tokens
    en_cyber = set(extract_cyber_tokens(" ".join(en_topk_tokens), lang='en'))
    tgt_cyber = set(extract_cyber_tokens(" ".join(tgt_topk_tokens), lang=lang_tgt))
    
    # Jaccard
    if len(en_cyber) == 0 and len(tgt_cyber) == 0:
        jaccard = 1.0
    elif len(en_cyber.union(tgt_cyber)) == 0:
        jaccard = 0.0
    else:
        jaccard = len(en_cyber.intersection(tgt_cyber)) / len(en_cyber.union(tgt_cyber))
    
    return {
        'jaccard': jaccard,
        'overlap_count': len(en_cyber.intersection(tgt_cyber)),
        'en_cyber_count': len(en_cyber),
        'tgt_cyber_count': len(tgt_cyber),
    }


# ====================
# Helper: Aggregate WordPiece attributions to word-level
# ====================

def aggregate_subword_attributions(
    tokens: List[str],
    attributions: np.ndarray,
    special_tokens: List[str] = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate subword tokens (e.g., WordPiece/BPE) to word-level.
    
    Rules:
    - Merge tokens starting with '##' (BERT WordPiece) or 'Ġ' (GPT BPE) to previous token.
    - Skip special tokens.
    
    Returns:
    - words: list of aggregated words
    - word_attributions: aggregated attribution per word
    """
    words = []
    word_attrs = []
    
    current_word = []
    current_attr = []
    
    for tok, attr in zip(tokens, attributions):
        # Skip special tokens
        if tok in special_tokens:
            continue
        
        # Check if subword continuation
        is_continuation = tok.startswith('##') or tok.startswith('Ġ')
        
        if is_continuation:
            # Merge with previous word
            if current_word:
                current_word.append(tok.replace('##', '').replace('Ġ', ''))
                current_attr.append(attr)
        else:
            # Start new word
            if current_word:
                # Flush previous word
                words.append(''.join(current_word))
                word_attrs.append(np.sum(current_attr))
            current_word = [tok]
            current_attr = [attr]
    
    # Flush last word
    if current_word:
        words.append(''.join(current_word))
        word_attrs.append(np.sum(current_attr))
    
    return words, np.array(word_attrs)


# ====================
# Dataset utility: stratified sampling
# ====================

def stratified_sample_from_csv(
    csv_path: str,
    label_col: str,
    text_col: str,
    target_per_label: int = 20000,
    chunksize: int = 200000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Stratified sampling from large CSV (memory-efficient).
    
    Returns DataFrame with balanced labels (target_per_label per label).
    """
    from collections import Counter
    
    label_counts = Counter()
    rows = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.dropna(subset=[text_col, label_col])
        
        for lbl, grp in chunk.groupby(label_col):
            need = target_per_label - label_counts[lbl]
            if need <= 0:
                continue
            
            take = grp.sample(min(need, len(grp)), random_state=random_state)
            rows.append(take[[text_col, label_col]])
            label_counts[lbl] += len(take)
        
        # Early stop if all labels filled
        if all(label_counts[l] >= target_per_label for l in label_counts):
            break
    
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=random_state)
    return df


if __name__ == '__main__':
    # Quick test
    test_text = "CVE-2024-1234 ransomware attack detected via phishing email"
    print("Has cyber cue:", has_cyber_cue(test_text))
    print("Cyber tokens:", extract_cyber_tokens(test_text))
    
    # Test CTAM
    tokens = ['the', 'ransomware', 'attack', 'was', 'severe']
    attrs = np.array([0.1, 0.8, 0.5, 0.05, 0.2])
    ctam = compute_ctam(tokens, attrs, lang='en')
    print(f"CTAM: {ctam:.3f}")
