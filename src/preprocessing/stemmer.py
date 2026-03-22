"""
preprocessing/stemmer.py

Pure-Python implementation of the Porter stemming algorithm (1980).
No external dependencies required.

Reference: M.F. Porter, "An algorithm for suffix stripping",
           Program, 14(3), pp. 130-137, 1980.
"""

import re


def _has_vowel(stem: str) -> bool:
    return bool(re.search(r"[aeiou]", stem))


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and word[-1] not in "aeiou"
    )


def _cvc(word: str) -> bool:
    """True if word ends consonant-vowel-consonant and last consonant is not w/x/y."""
    if len(word) < 3:
        return False
    c, v, c2 = word[-3], word[-2], word[-1]
    return (
        c  not in "aeiou"
        and v  in "aeiou"
        and c2 not in "aeiouwxy"
    )


def _measure(stem: str) -> int:
    """Counts VC sequences (measure m) in a stem."""
    s = re.sub(r"^[^aeiou]+", "", stem)
    s = re.sub(r"[aeiou]+",   "V", s)
    s = re.sub(r"[^V]+",      "C", s)
    return s.count("VC")


def _step1a(word: str) -> str:
    if word.endswith("sses"): return word[:-2]
    if word.endswith("ies"):  return word[:-2]
    if word.endswith("ss"):   return word
    if word.endswith("s"):    return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word

    changed = False
    if word.endswith("ed"):
        stem = word[:-2]
        if _has_vowel(stem):
            word, changed = stem, True
    elif word.endswith("ing"):
        stem = word[:-3]
        if _has_vowel(stem):
            word, changed = stem, True

    if changed:
        if word.endswith(("at", "bl", "iz")):
            return word + "e"
        if _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            return word[:-1]
        if _measure(word) == 1 and _cvc(word):
            return word + "e"
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step2(word: str) -> str:
    suffixes = {
        "ational": "ate",  "tional": "tion", "enci": "ence",
        "anci":    "ance", "izer":   "ize",  "abli": "able",
        "alli":    "al",   "entli":  "ent",  "eli":  "e",
        "ousli":   "ous",  "ization":"ize",  "ation":"ate",
        "ator":    "ate",  "alism":  "al",   "iveness":"ive",
        "fulness": "ful",  "ousness":"ous",  "aliti":"al",
        "iviti":   "ive",  "biliti": "ble",
    }
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix) and _measure(word[:-len(suffix)]) > 0:
            return word[:-len(suffix)] + replacement
    return word


def _step3(word: str) -> str:
    suffixes = {
        "icate": "ic", "ative": "", "alize": "al",
        "iciti": "ic", "ical":  "ic", "ful": "", "ness": "",
    }
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix) and _measure(word[:-len(suffix)]) > 0:
            return word[:-len(suffix)] + replacement
    return word


def _step4(word: str) -> str:
    for suffix in [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize",
    ]:
        stem = word[:-len(suffix)]
        if word.endswith(suffix) and _measure(stem) > 1:
            if suffix == "ion" and stem and stem[-1] in "st":
                return stem
            elif suffix != "ion":
                return stem
    return word


def _step5a(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        if _measure(stem) > 1:
            return stem
        if _measure(stem) == 1 and not _cvc(stem):
            return stem
    return word


def _step5b(word: str) -> str:
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        return word[:-1]
    return word


def stem(word: str) -> str:
    """
    Returns the Porter stem of a lowercase word.

    Args:
        word: A single lowercase token.

    Returns:
        The stemmed form of the word.

    Example:
        >>> stem("aerodynamics")
        'aerodynam'
    """
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5a(word)
    word = _step5b(word)
    return word