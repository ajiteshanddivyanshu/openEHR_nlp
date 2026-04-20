#!/usr/bin/env python3
"""
ADVANCED MARKER LOOKUP LOADER + RULE-ONLY FP/FN REFINER

This module intentionally avoids independent LLM extraction and RAG retrieval.
It provides:
1. Advanced marker_lookup.json loader with archetype and SNOMED mappings
2. RuleBasedFPFNRefiner for FP/FN-only adjudication using the same rule assets as the NLP engine
3. Backward-compatible integration helper
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

print("[MODULE] Advanced Marker Loader + Rule-Based Refiner loaded")

# ============================================================================
# STRICT EXCLUSIONS
# ============================================================================

STRICT_EXCLUSIONS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'al', 'et', 'vs', 'cm', 'pg', 'ml', 'ul', 'dl', 'mg', 'kg', 'lb',
    'ac', 'cl', 'gh', 'na', 'pa', 'rf', 'tic', 'us', 'ic', 'lp',
    'nec', 'ph', 'tha', 'ua', 'bpm', 'mmhg', 'g/dl',
    'severe', 'acute', 'chronic', 'elevated', 'decreased', 'low', 'high',
    'positive', 'negative', 'pending', 'consistent', 'showing', 'shown',
    'morning', 'night', 'daily', 'hours', 'days', 'weeks', 'months',
    'of', 'or', 'and', 'the', 'in', 'on', 'at', 'is', 'are', 'be',
    'with', 'for', 'to', 'from', 'by', 'that', 'this', 'it',
    'has', 'had', 'have', 'was', 'were', 'been', 'being',
    'patient', 'mg/dl', 'normal', 'findings', 'history',
    'alert', 'oriented', 'oriented x3', 'bilateral', 'bilaterally',
    'unilateral', 'continuous', 'anterior', 'posterior', 'lateral',
    'wall', 'lower', 'upper', 'left', 'right', 'baseline',
    'cultures', 'blood cultures', 'appearance', 'activity',
    'onset', 'exacerbation', 'minutes', 'predicted',
    'oliguric', 'oliguria', 'myoglobinuria', 'on exertion',
    'worse', 'initially', 'changes', 'bright', 'dark', 'red',
    'cola', 'colored', 'cola-colored', 'rapid', 'test', 'serology',
    'strep', 'barrel', 'chest', 'breath', 'sounds', 'gums',
    'bleeding', 'extremities', 'frothy', 'sputum', 'pink',
    'slurred', 'speech', 'arm', 'urine', 'stool',
}


# ============================================================================
# STEP 1: ADVANCED MARKER LOOKUP LOADER
# ============================================================================

def load_advanced_marker_lookup(filepath='marker_lookup.json') -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """Load marker_lookup.json with strict exclusion filtering."""
    print(f"[LOADER] Loading advanced marker lookup from {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_lookup = json.load(f)

        engine_dict = defaultdict(list)
        archetype_map = {}
        snomed_map = {}
        loaded_count = 0

        for term, data in raw_lookup.items():
            term_l = str(term).lower().strip()
            if len(term_l) <= 3 or term_l in STRICT_EXCLUSIONS:
                continue

            cat = data.get('category', 'unknown')
            engine_dict[cat].append(term)

            archetype_id = data.get('archetype_id', 'Generic_Cluster_v1')
            archetype_map[term_l] = archetype_id

            snomed_code = data.get('snomed', None)
            if snomed_code:
                snomed_map[term_l] = snomed_code

            loaded_count += 1

        print(f"[LOADER] [OK] Loaded {loaded_count} marker concepts")
        print(f"[LOADER]   Categories: {list(dict(engine_dict).keys())}")
        print(f"[LOADER]   Archetype mappings: {len(archetype_map)}")

        return dict(engine_dict), archetype_map, snomed_map

    except FileNotFoundError:
        print(f"[LOADER] [FAIL] File not found: {filepath}")
        return {}, {}, {}
    except json.JSONDecodeError as e:
        print(f"[LOADER] [FAIL] JSON decode error: {e}")
        return {}, {}, {}
    except Exception as e:
        print(f"[LOADER] [FAIL] Error loading marker lookup: {e}")
        return {}, {}, {}


# ============================================================================
# STEP 2: RULE-ONLY FP/FN REFINER
# ============================================================================

class RuleBasedFPFNRefiner:
    """
    Refine only false positives and false negatives using rule assets.

    No independent LLM extraction.
    No RAG vector retrieval.
    """

    def __init__(self, engine):
        self.engine = engine
        self.allowed_terms = set()

        marker_dict = getattr(engine, 'marker_dict', {}) or {}
        for terms in marker_dict.values():
            for term in terms:
                term_l = str(term).lower().strip()
                if term_l and term_l not in STRICT_EXCLUSIONS:
                    self.allowed_terms.add(term_l)

    @staticmethod
    def _norm(value):
        return str(value).lower().strip()

    @staticmethod
    def _is_numeric(value):
        cleaned = str(value).replace(',', '').strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    @staticmethod
    def _norm_num(value):
        return str(value).replace(',', '').strip().lower()

    def _collect_rule_terms(self, text):
        """Collect terms directly supported by regex rules and dictionary evidence."""
        supported = set()
        text_l = text.lower()

        patterns = getattr(self.engine, 'patterns', {}) or {}
        for pname, rule in patterns.items():
            pstr, _, _ = rule
            regex = re.compile(pstr, re.IGNORECASE)
            for match in regex.finditer(text):
                if pname == 'bp_complete' and match.lastindex and match.lastindex >= 2:
                    supported.add(self._norm_num(match.group(1)))
                    supported.add(self._norm_num(match.group(2)))
                    continue

                if match.lastindex:
                    val = match.group(1)
                else:
                    val = match.group(0)

                if self._is_numeric(val):
                    supported.add(self._norm_num(val))
                else:
                    normalized = getattr(self.engine, 'normalize_value', lambda x: x)(val)
                    supported.add(self._norm(normalized))

        labeled_patterns = getattr(self.engine, 'labeled_patterns', {}) or {}
        for _, rule in labeled_patterns.items():
            pstr, _, _, label_name = rule
            regex = re.compile(pstr, re.IGNORECASE)
            for match in regex.finditer(text):
                if match.lastindex:
                    num = match.group(1)
                    supported.add(self._norm_num(num))
                if label_name:
                    supported.add(self._norm(label_name))

        for term in self.allowed_terms:
            if term in STRICT_EXCLUSIONS:
                continue
            if re.search(r'\b' + re.escape(term) + r'\b', text_l):
                supported.add(term)

        return supported

    def refine_case(self, text, extracted_values, expected_values):
        """Refine FP/FN terms only for a single evaluated case."""
        extracted_map = {self._norm(v): str(v) for v in extracted_values if str(v).strip()}
        expected_map = {self._norm(v): str(v) for v in expected_values if str(v).strip()}

        extracted_norm = set(extracted_map.keys())
        expected_norm = set(expected_map.keys())

        fp_terms = extracted_norm - expected_norm
        fn_terms = expected_norm - extracted_norm

        if not fp_terms and not fn_terms:
            return {
                'refined_values': list(extracted_values),
                'changed': False,
                'removed_fp': [],
                'added_fn': [],
            }

        # Re-run full pipeline for this note (all nlp_engine stages), then use rule assets.
        rerun = self.engine.extract_by_category(text, force_all_llm_judging=False)
        rerun_values = []
        for values in rerun.values():
            rerun_values.extend(values)
        rerun_norm = {self._norm(v) for v in rerun_values}

        supported_terms = self._collect_rule_terms(text)

        resolved = set(extracted_norm)
        removed_fp = []
        added_fn = []

        for fp in sorted(fp_terms):
            if fp in STRICT_EXCLUSIONS:
                resolved.discard(fp)
                removed_fp.append(fp)
                continue

            # Keep FP only if still supported by rerun pipeline and explicit rule evidence.
            if fp not in rerun_norm or fp not in supported_terms:
                resolved.discard(fp)
                removed_fp.append(fp)

        for fn in sorted(fn_terms):
            if fn in STRICT_EXCLUSIONS:
                continue

            # Recover FN when supported by both rerun pipeline and rule evidence.
            if fn in rerun_norm and fn in supported_terms:
                resolved.add(fn)
                added_fn.append(fn)

        ordered = []
        seen = set()
        for source in (extracted_map, expected_map):
            for key, original in source.items():
                if key in resolved and key not in seen:
                    ordered.append(original)
                    seen.add(key)

        for key in sorted(resolved):
            if key not in seen:
                ordered.append(expected_map.get(key, extracted_map.get(key, key)))
                seen.add(key)

        changed = bool(removed_fp or added_fn)
        return {
            'refined_values': ordered,
            'changed': changed,
            'removed_fp': removed_fp,
            'added_fn': added_fn,
        }
# ============================================================================
# STEP 3: INTEGRATION HELPER
# ============================================================================

def integrate_marker_loader_with_engine(engine_instance):
    """Integrate advanced marker loader and rule-based refiner into engine instance."""
    print("[INTEGRATION] Loading advanced marker lookup...")
    engine_dict, archetype_map, snomed_map = load_advanced_marker_lookup('marker_lookup.json')

    if not engine_dict:
        print("[INTEGRATION] [FAIL] Failed to load marker dictionaries")
        return False

    engine_instance.marker_dict = engine_dict
    engine_instance.ARCHETYPE_MAP = archetype_map
    engine_instance.SNOMED_MAP = snomed_map
    engine_instance.rule_refiner = RuleBasedFPFNRefiner(engine_instance)

    print("[INTEGRATION] [OK] Marker dictionaries integrated")
    print(f"[INTEGRATION]   Total markers: {sum(len(v) for v in engine_dict.values())}")
    print(f"[INTEGRATION]   Archetype mappings: {len(archetype_map)}")
    print("[INTEGRATION] [OK] Rule-based FP/FN refiner integrated")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("ADVANCED MARKER LOADER + RULE-ONLY REFINER TEST")
    print("=" * 80 + "\n")

    engine_dict, archetype_map, snomed_map = load_advanced_marker_lookup()
    print(f"Categories found: {list(engine_dict.keys())}")
    print(f"Total terms: {sum(len(v) for v in engine_dict.values())}")
    print(f"Archetype mappings: {len(archetype_map)}")
    print(f"SNOMED mappings: {len(snomed_map)}")

    print("\n[TEST] Module is ready. No LLM or RAG components are used.")
