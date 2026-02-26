from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import jellyfish
from rapidfuzz import fuzz
from unidecode import unidecode

LEET_MAP = str.maketrans("0134578@!", "oieastbba")
DISALLOWED_WORDS = {"police", "crime", "corruption", "cbi", "cid", "army"}
PREFIXES_SUFFIXES = {"the", "india", "samachar", "news"}
ADVANCED_PERIODICITY = {
    "daily",
    "weekly",
    "monthly",
    "fortnightly",
    "annual",
    "dainik",
    "saptahik",
    "masik",
    "varshik",
    "pratidin",
    "rozana",
}

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


@lru_cache(maxsize=200000)
def _sanitize_cached(title: str) -> str:
    romanized_title = unidecode(title)
    de_leeted = romanized_title.lower().translate(LEET_MAP)
    cleaned = NON_ALNUM_RE.sub(" ", de_leeted)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


@lru_cache(maxsize=200000)
def _metaphone_cached(text: str) -> str:
    return jellyfish.metaphone(text)


@lru_cache(maxsize=200000)
def _compact_ngrams_cached(compact: str, n: int) -> FrozenSet[str]:
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset({compact})
    return frozenset(compact[i : i + n] for i in range(len(compact) - n + 1))


def sanitize_input(title: str) -> str:
    return _sanitize_cached(title or "")


def char_ngrams(text: str, n: int = 3) -> FrozenSet[str]:
    compact = text.replace(" ", "")
    return _compact_ngrams_cached(compact, n)


def make_acronym(words: List[str]) -> str:
    return "".join(word[0] for word in words if word)


@dataclass
class TitleIndex:
    existing_titles: Set[str] = field(default_factory=set)
    canonical_titles: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    phonetic_map: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    sorted_titles_map: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    acronym_map: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    token_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    trigram_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    first_char_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def clear(self) -> None:
        self.existing_titles.clear()
        self.canonical_titles.clear()
        self.phonetic_map.clear()
        self.sorted_titles_map.clear()
        self.acronym_map.clear()
        self.token_index.clear()
        self.trigram_index.clear()
        self.first_char_index.clear()

    def add_title(self, raw_title: str) -> None:
        normalized = sanitize_input(raw_title)
        if not normalized:
            return

        words = normalized.split()
        self.existing_titles.add(normalized)
        self.canonical_titles[normalized].add(raw_title.strip())

        metaphone = _metaphone_cached(normalized)
        if metaphone:
            self.phonetic_map[metaphone].add(normalized)

        if len(words) > 1:
            sorted_key = " ".join(sorted(words))
            self.sorted_titles_map[sorted_key].add(normalized)
            acronym = make_acronym(words)
            if acronym:
                self.acronym_map[acronym].add(normalized)

        for token in set(words):
            self.token_index[token].add(normalized)

        for gram in char_ngrams(normalized):
            self.trigram_index[gram].add(normalized)

        if normalized:
            self.first_char_index[normalized[0]].add(normalized)

    def extend(self, titles: Iterable[str]) -> None:
        for title in titles:
            self.add_title(title)

    def display_title(self, normalized_title: str) -> str:
        candidates = self.canonical_titles.get(normalized_title)
        if not candidates:
            return normalized_title.title()
        return sorted(candidates)[0]

    def _candidate_titles(self, clean_title: str, max_candidates: int = 700) -> Set[str]:
        score_counter: Counter[str] = Counter()
        words = clean_title.split()

        for token in set(words):
            for candidate in self.token_index.get(token, ()):
                score_counter[candidate] += 3

        for gram in char_ngrams(clean_title):
            for candidate in self.trigram_index.get(gram, ()):
                score_counter[candidate] += 1

        if clean_title:
            for candidate in self.first_char_index.get(clean_title[0], ()):
                if abs(len(candidate) - len(clean_title)) <= 8:
                    score_counter[candidate] += 1

        if not score_counter:
            return set()

        return {candidate for candidate, _ in score_counter.most_common(max_candidates)}

    def _detect_combination(self, clean_title: str) -> Optional[List[str]]:
        words = clean_title.split()
        total_words = len(words)
        if total_words < 2:
            return None

        @lru_cache(maxsize=None)
        def segment(start: int) -> List[List[str]]:
            combos: List[List[str]] = []
            for end in range(start + 1, total_words + 1):
                phrase = " ".join(words[start:end])
                if phrase not in self.existing_titles or phrase == clean_title:
                    continue

                if end == total_words:
                    combos.append([phrase])
                else:
                    for tail in segment(end):
                        combos.append([phrase] + tail)
            return combos

        for grouping in segment(0):
            if len(grouping) >= 2:
                return [self.display_title(item) for item in grouping]
        return None

    def _detect_periodicity_extension(self, clean_title: str) -> Optional[str]:
        words = clean_title.split()
        if len(words) <= 1:
            return None

        stripped_words = [word for word in words if word not in ADVANCED_PERIODICITY]
        if len(stripped_words) == len(words):
            return None

        base_title = " ".join(stripped_words)
        if base_title and base_title in self.existing_titles and base_title != clean_title:
            return self.display_title(base_title)
        return None

    def detect_lexical_conflicts(
        self, raw_title: str, *, precleaned: bool = False
    ) -> Tuple[List[str], float]:
        clean_title = raw_title if precleaned else sanitize_input(raw_title)
        if not clean_title:
            return ["Title cannot be empty after normalization."], 100.0

        words = clean_title.split()

        if clean_title in self.existing_titles:
            exact_reason = (
                f"Exact match found with existing title "
                f"'{self.display_title(clean_title)}'."
            )
            return [exact_reason], 100.0

        if len(words) > 1:
            sorted_key = " ".join(sorted(words))
            sorted_matches = self.sorted_titles_map.get(sorted_key, set())
            if sorted_matches:
                matched = next(iter(sorted_matches))
                sorted_reason = (
                    "Word-order variation matches existing title "
                    f"'{self.display_title(matched)}'."
                )
                return [sorted_reason], 99.0

        if len(clean_title) <= 8 and clean_title.isalpha():
            acronym_matches = self.acronym_map.get(clean_title, set())
            if acronym_matches:
                matched = next(iter(acronym_matches))
                acronym_reason = (
                    f"Acronym collision with existing title "
                    f"'{self.display_title(matched)}'."
                )
                return [acronym_reason], 98.0

        metaphone = _metaphone_cached(clean_title)
        if metaphone:
            phonetic_matches = self.phonetic_map.get(metaphone, set())
            for matched in phonetic_matches:
                if matched == clean_title:
                    continue
                ratio = float(fuzz.ratio(clean_title, matched))
                if ratio >= 60.0:
                    return [
                        (
                            f"Phonetic conflict with '{self.display_title(matched)}' "
                            f"(lexical similarity {ratio:.1f}%)."
                        )
                    ], max(92.0, ratio)

        periodic_base = self._detect_periodicity_extension(clean_title)
        if periodic_base:
            return [f"Periodicity modifier added to existing title '{periodic_base}'."], 96.0

        combination = self._detect_combination(clean_title)
        if combination:
            joined = " + ".join(combination)
            return [f"Title appears to combine existing titles: {joined}."], 94.0

        best_score = 0.0
        best_match = None
        for candidate in self._candidate_titles(clean_title):
            if candidate == clean_title:
                continue
            score = float(fuzz.ratio(clean_title, candidate))
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_match and best_score >= 80.0:
            return [
                (
                    "Spelling/transliteration variation too close to existing title "
                    f"'{self.display_title(best_match)}' ({best_score:.1f}% lexical match)."
                )
            ], best_score

        return [], best_score


def enforce_guidelines(
    title: str, index: Optional[TitleIndex] = None, *, precleaned: bool = False
) -> List[str]:
    clean_title = title if precleaned else sanitize_input(title)
    words = clean_title.split()
    if not words:
        return ["Title cannot be empty."]

    reasons: List[str] = []
    words_set = set(words)
    disallowed = sorted(words_set.intersection(DISALLOWED_WORDS))
    if disallowed:
        reasons.append(f"Contains disallowed words: {', '.join(disallowed).upper()}.")

    periodicity_words = words_set.intersection(ADVANCED_PERIODICITY)
    if periodicity_words and index:
        stripped = [word for word in words if word not in ADVANCED_PERIODICITY]
        base = " ".join(stripped)
        if base and base in index.existing_titles and base != clean_title:
            reasons.append(
                "Uses periodicity term to modify an existing title "
                f"('{index.display_title(base)}')."
            )

    if words[0] in PREFIXES_SUFFIXES and index:
        base = " ".join(words[1:])
        if base and base in index.existing_titles:
            reasons.append(
                f"Disallowed prefix '{words[0]}' creates conflict with existing title "
                f"'{index.display_title(base)}'."
            )

    if words[-1] in PREFIXES_SUFFIXES and index:
        base = " ".join(words[:-1])
        if base and base in index.existing_titles:
            reasons.append(
                f"Disallowed suffix '{words[-1]}' creates conflict with existing title "
                f"'{index.display_title(base)}'."
            )

    return reasons
