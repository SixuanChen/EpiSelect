#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

STATE_ORDER: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


def truth_bits(fn: Callable[[bool, bool], bool]) -> Tuple[int, int, int, int]:
    return tuple(int(bool(fn(bool(a), bool(b)))) for a, b in STATE_ORDER)  # type: ignore[return-value]


CANONICAL_FUNCTIONS: Dict[str, Tuple[int, int, int, int]] = {
    "FALSE": truth_bits(lambda a, b: False),
    "A_AND_B": truth_bits(lambda a, b: a and b),
    "A_AND_NOT_B": truth_bits(lambda a, b: a and (not b)),
    "A": truth_bits(lambda a, b: a),
    "NOT_A_AND_B": truth_bits(lambda a, b: (not a) and b),
    "B": truth_bits(lambda a, b: b),
    "XOR": truth_bits(lambda a, b: a != b),
    "OR": truth_bits(lambda a, b: a or b),
    "NOR": truth_bits(lambda a, b: not (a or b)),
    "IFF": truth_bits(lambda a, b: a == b),
    "NOT_B": truth_bits(lambda a, b: not b),
    "B_TO_A": truth_bits(lambda a, b: (not b) or a),
    "NOT_A": truth_bits(lambda a, b: not a),
    "A_TO_B": truth_bits(lambda a, b: (not a) or b),
    "NAND": truth_bits(lambda a, b: not (a and b)),
    "TRUE": truth_bits(lambda a, b: True),
}
BITS_TO_CANONICAL = {bits: name for name, bits in CANONICAL_FUNCTIONS.items()}
NONCONSTANT_RULES = tuple(k for k in CANONICAL_FUNCTIONS if k not in {"FALSE", "TRUE"})
ALL_16_BITS = tuple(CANONICAL_FUNCTIONS[k] for k in CANONICAL_FUNCTIONS)

CANONICAL_EXPR: Dict[str, str] = {
    "FALSE": "FALSE",
    "A_AND_B": "A AND B",
    "A_AND_NOT_B": "A AND NOT B",
    "A": "A",
    "NOT_A_AND_B": "NOT A AND B",
    "B": "B",
    "XOR": "A XOR B",
    "OR": "A OR B",
    "NOR": "NOT (A OR B)",
    "IFF": "A <-> B",
    "NOT_B": "NOT B",
    "B_TO_A": "B -> A",
    "NOT_A": "NOT A",
    "A_TO_B": "A -> B",
    "NAND": "NOT (A AND B)",
    "TRUE": "TRUE",
}

# Ten balanced true-rule families. Each has two psychologically matched user rules.
# In every family/user condition, the user's rule differs from the true rule on
# exactly one of the four abstract A/B assignments. This keeps misconception
# distance constant across families while action evidence is defined only from
# the true rule T and the inferred user rule H (no privileged sibling at scoring time).
RULE_FAMILIES: Dict[str, Dict[str, object]] = {
    # The 10 nonconstant, non-single-literal Boolean functions over A,B. This is
    # the complete Level-2 true-rule space once constants and one-feature rules
    # are excluded. Single-literal rules remain in the backend hypothesis space
    # and serve as psychologically interpretable user shortcuts.
    "OR": {
        "true": "OR", "users": ("A", "B"), "group": "simple_composition",
        "description": "inclusive disjunction; user tracks only one positive branch",
    },
    "AND": {
        "true": "A_AND_B", "users": ("A", "B"), "group": "simple_composition",
        "description": "conjunction; user treats one necessary feature as sufficient",
    },
    "NAND": {
        "true": "NAND", "users": ("NOT_A", "NOT_B"), "group": "negated_composition",
        "description": "not-both rule; user tracks only one negated branch",
    },
    "NOR": {
        "true": "NOR", "users": ("NOT_A", "NOT_B"), "group": "negated_composition",
        "description": "neither-nor rule; user tracks only one negated condition",
    },
    "A_AND_NOT_B": {
        "true": "A_AND_NOT_B", "users": ("A", "NOT_B"), "group": "asymmetric_negation",
        "description": "A plus exclusion of B; user drops one conjunct",
    },
    "NOT_A_AND_B": {
        "true": "NOT_A_AND_B", "users": ("NOT_A", "B"), "group": "asymmetric_negation",
        "description": "exclusion of A plus B; user drops one conjunct",
    },
    "A_TO_B": {
        "true": "A_TO_B", "users": ("NOT_A", "B"), "group": "conditional",
        "description": "forward conditional, equivalently NOT A OR B; user tracks only one sufficient branch",
    },
    "B_TO_A": {
        "true": "B_TO_A", "users": ("NOT_B", "A"), "group": "conditional",
        "description": "converse-direction conditional, equivalently NOT B OR A; user tracks only one sufficient branch",
    },
    "IFF": {
        "true": "IFF", "users": ("A_TO_B", "B_TO_A"), "group": "directional_relation",
        "description": "biconditional; user retains only forward or converse implication",
    },
    "XOR": {
        "true": "XOR", "users": ("A_AND_NOT_B", "NOT_A_AND_B"), "group": "exclusive_relation",
        "description": "exclusive-or; user retains only one of the two exclusive branches",
    },
}


def applies(bits: Sequence[int], state_index: int) -> bool:
    return bool(bits[state_index])


def mismatch_count(rule_a: str, rule_b: str) -> int:
    a = CANONICAL_FUNCTIONS[rule_a]
    b = CANONICAL_FUNCTIONS[rule_b]
    return sum(x != y for x, y in zip(a, b))


def candidate_rules_consistent(observations: Sequence[Tuple[int, int]]) -> List[str]:
    """Rules that predict chosen=belongs and unchosen=does-not-belong for every pair."""
    out: List[str] = []
    for name, bits in CANONICAL_FUNCTIONS.items():
        if all(bits[c] == 1 and bits[u] == 0 for c, u in observations):
            out.append(name)
    return out


@lru_cache(maxsize=None)
def minimal_diagnostic_sequences(rule_name: str) -> Tuple[int | None, Tuple[Tuple[int, int], ...]]:
    """Return PDD and a deterministic maximally front-loaded minimum sequence.

    PDD = minimum number of paired positive/negative classification observations needed to identify a truth table among
    all 16 Boolean functions, where each observation contains one state labeled 1 (BELONGS) and one labeled 0 (DOES NOT BELONG).
    Constants cannot be identified under this protocol and return (None, ()).
    """
    bits = CANONICAL_FUNCTIONS[rule_name]
    possible = tuple((i, j) for i in range(4) for j in range(4) if i != j and bits[i] == 1 and bits[j] == 0)
    if not possible:
        return None, ()

    minimum_sets: List[Tuple[Tuple[int, int], ...]] = []
    min_k: int | None = None
    for k in range(1, 5):
        for comb in itertools.combinations(possible, k):
            if candidate_rules_consistent(comb) == [rule_name]:
                minimum_sets.append(comb)
        if minimum_sets:
            min_k = k
            break
    if min_k is None:
        return None, ()

    candidates: List[Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]] = []
    for comb in minimum_sets:
        for perm in itertools.permutations(comb):
            residuals: List[int] = []
            prefix: List[Tuple[int, int]] = []
            for obs in perm:
                prefix.append(obs)
                residuals.append(len(candidate_rules_consistent(prefix)))
            candidates.append((tuple(residuals), tuple(perm)))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return min_k, candidates[0][1]


def version_space_trace(observations: Sequence[Tuple[int, int]]) -> List[Dict[str, object]]:
    current = list(CANONICAL_FUNCTIONS)
    out: List[Dict[str, object]] = []
    prefix: List[Tuple[int, int]] = []
    before = len(current)
    for i, obs in enumerate(observations, 1):
        prefix.append(obs)
        remaining = candidate_rules_consistent(prefix)
        after = len(remaining)
        ig = math.log2(before) - math.log2(after) if after else float("inf")
        out.append({
            "observation_index": i,
            "version_space_before": before,
            "version_space_after": after,
            "information_gain_bits": ig,
            "remaining_rules": remaining,
        })
        before = after
    return out


def state_label(state_idx: int) -> str:
    a, b = STATE_ORDER[state_idx]
    return f"A={a},B={b}"


def evidence_state_classes(true_rule: str, user_rule: str) -> Dict[str, List[int]]:
    """Partition A/B states by their relation to truth T and inferred user rule H.

    informative: T and H disagree.
    compatible_positive: T and H agree that the case BELONGS.
    compatible_negative: T and H agree that the case DOES NOT BELONG.

    These definitions do not privilege any alternative/sibling hypothesis.
    """
    t = CANONICAL_FUNCTIONS[true_rule]
    h = CANONICAL_FUNCTIONS[user_rule]
    cls = {"informative": [], "compatible_positive": [], "compatible_negative": []}
    for i in range(4):
        if t[i] != h[i]:
            cls["informative"].append(i)
        elif t[i] == h[i] == 1:
            cls["compatible_positive"].append(i)
        elif t[i] == h[i] == 0:
            cls["compatible_negative"].append(i)
    return cls


def valid_invalid_control_states(true_rule: str, user_rules: Sequence[str]) -> List[int]:
    """States that can be used as a common mislabeled Invalid option.

    The same physical four-state option set is reused across the two matched user-rule
    counterfactuals of a base rule.  A valid invalid-control state must therefore:
      * be classified correctly by truth and BOTH user rules before its displayed label is flipped; and
      * leave, for each user rule, one Informative state plus at least one
        Compatible-positive and one Compatible-negative state among the three truthful options.

    This function makes the existence requirement explicit and fully checkable.
    """
    t = CANONICAL_FUNCTIONS[true_rule]
    hs = [CANONICAL_FUNCTIONS[u] for u in user_rules]
    good: List[int] = []
    for inv in range(4):
        if not all(t[inv] == h[inv] for h in hs):
            continue
        remaining = [i for i in range(4) if i != inv]
        ok = True
        for h in hs:
            informative = [i for i in remaining if t[i] != h[i]]
            comp_pos = [i for i in remaining if t[i] == h[i] == 1]
            comp_neg = [i for i in remaining if t[i] == h[i] == 0]
            if len(informative) != 1 or not comp_pos or not comp_neg:
                ok = False
                break
        if ok:
            good.append(inv)
    return good


def family_invariants_ok() -> Tuple[bool, List[str]]:
    """Validate the fixed Level-2 rule families against v3 evidence requirements."""
    errors: List[str] = []
    for fam, spec in RULE_FAMILIES.items():
        true_rule = str(spec["true"])
        users = tuple(spec["users"])
        if len(users) != 2:
            errors.append(f"{fam}: expected 2 matched user rules")
            continue
        for u in users:
            if mismatch_count(true_rule, u) != 1:
                errors.append(f"{fam}/{u}: mismatch distance != 1")
            classes = evidence_state_classes(true_rule, u)
            if len(classes["informative"]) != 1:
                errors.append(f"{fam}/{u}: expected exactly one informative state, got {classes}")
            if not classes["compatible_positive"]:
                errors.append(f"{fam}/{u}: no compatible-positive state")
            if not classes["compatible_negative"]:
                errors.append(f"{fam}/{u}: no compatible-negative state")
        valid_invalid = valid_invalid_control_states(true_rule, users)
        if not valid_invalid:
            errors.append(f"{fam}: no common Invalid-control state preserves I/CP/CN for both user rules")
    return not errors, errors


# ------------------------- expression parsing -------------------------
# Open-ended to the model, finite semantic evaluation on the backend.

TOKEN_RE = re.compile(
    r"\s*(<->|<=>|->|=>|\(|\)|\bIF\b|\bNOT\b|\bAND\b|\bOR\b|\bXOR\b|\bNAND\b|\bNOR\b|\bIFF\b|\bIMPLIES\b|\bTRUE\b|\bFALSE\b|[AB]|!|~|&|\||\^|,)",
    re.IGNORECASE,
)


class RuleParseError(ValueError):
    pass


def tokenize(expr: str) -> List[str]:
    s = expr.strip().upper()
    if s in CANONICAL_FUNCTIONS:
        s = CANONICAL_EXPR[s]
    pos = 0
    toks: List[str] = []
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise RuleParseError(f"Unrecognized token near: {s[pos:pos+20]!r}")
        toks.append(m.group(1).upper())
        pos = m.end()
    return toks


@dataclass
class Parser:
    toks: List[str]
    pos: int = 0

    def peek(self) -> str | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def take(self, *allowed: str) -> str | None:
        p = self.peek()
        if p is not None and p in allowed:
            self.pos += 1
            return p
        return None

    def parse(self):
        node = self.parse_iff()
        if self.peek() is not None:
            raise RuleParseError(f"Unexpected token: {self.peek()}")
        return node

    def parse_iff(self):
        node = self.parse_implies()
        while self.take("<->", "<=>", "IFF"):
            node = ("IFF", node, self.parse_implies())
        return node

    def parse_implies(self):
        left = self.parse_or()
        if self.take("->", "=>", "IMPLIES"):
            return ("IMPLIES", left, self.parse_implies())
        return left

    def parse_or(self):
        node = self.parse_xor()
        while self.take("OR", "|"):
            node = ("OR", node, self.parse_xor())
        return node

    def parse_xor(self):
        node = self.parse_and()
        while self.take("XOR", "^"):
            node = ("XOR", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_unary()
        while self.take("AND", "&"):
            node = ("AND", node, self.parse_unary())
        return node

    def parse_unary(self):
        if self.take("NOT", "!", "~"):
            return ("NOT", self.parse_unary())
        if self.take("NAND"):
            self.expect("(")
            a = self.parse_iff(); self.expect(",")  # pragma: no cover
            b = self.parse_iff(); self.expect(")")
            return ("NOT", ("AND", a, b))
        if self.take("NOR"):
            self.expect("(")
            a = self.parse_iff(); self.expect(",")  # pragma: no cover
            b = self.parse_iff(); self.expect(")")
            return ("NOT", ("OR", a, b))
        if self.take("("):
            node = self.parse_iff()
            if not self.take(")"):
                raise RuleParseError("Missing closing parenthesis")
            return node
        p = self.peek()
        if p in {"A", "B", "TRUE", "FALSE"}:
            self.pos += 1
            return p
        raise RuleParseError(f"Expected A, B, NOT, or '(', got {p}")

    def expect(self, tok: str):
        if not self.take(tok):
            raise RuleParseError(f"Expected {tok}")


def eval_ast(ast, a: bool, b: bool) -> bool:
    if ast == "A": return a
    if ast == "B": return b
    if ast == "TRUE": return True
    if ast == "FALSE": return False
    op = ast[0]
    if op == "NOT": return not eval_ast(ast[1], a, b)
    x = eval_ast(ast[1], a, b)
    y = eval_ast(ast[2], a, b)
    if op == "AND": return x and y
    if op == "OR": return x or y
    if op == "XOR": return x != y
    if op == "IMPLIES": return (not x) or y
    if op == "IFF": return x == y
    raise RuleParseError(f"Unknown AST op: {op}")


def parse_rule_expression(expr: str) -> Tuple[int, int, int, int]:
    s = expr.strip().upper()
    # Accept common function-style forms in addition to infix notation.
    # Level 2 has only two atomic predicates, so these aliases are semantically
    # complete for the shipped hypothesis space.
    function_aliases = {
        "IF(A,B)": "A -> B",
        "IF(B,A)": "B -> A",
        "IMPLIES(A,B)": "A -> B",
        "IMPLIES(B,A)": "B -> A",
        "IFF(A,B)": "A <-> B",
        "IFF(B,A)": "A <-> B",
        "AND(A,B)": "A AND B",
        "AND(B,A)": "A AND B",
        "OR(A,B)": "A OR B",
        "OR(B,A)": "A OR B",
        "XOR(A,B)": "A XOR B",
        "XOR(B,A)": "A XOR B",
        "NAND(A,B)": "NOT (A AND B)",
        "NAND(B,A)": "NOT (A AND B)",
        "NOR(A,B)": "NOT (A OR B)",
        "NOR(B,A)": "NOT (A OR B)",
        "NOT(A)": "NOT A",
        "NOT(B)": "NOT B",
    }
    s = re.sub(r"\s+", "", s) if re.match(r"^[A-Z]+\s*\(", s) else s
    s = function_aliases.get(s, s)
    # Accept canonical symbolic names directly, including NAND/NOR identifiers.
    if s in CANONICAL_FUNCTIONS:
        return CANONICAL_FUNCTIONS[s]
    # Simple aliases that models commonly emit.
    aliases = {
        "A IFF B": "A <-> B",
        "B IFF A": "B <-> A",
        "A IMPLIES B": "A -> B",
        "B IMPLIES A": "B -> A",
        "A NAND B": "NOT (A AND B)",
        "A NOR B": "NOT (A OR B)",
        "B NAND A": "NOT (A AND B)",
        "B NOR A": "NOT (A OR B)",
    }
    s = aliases.get(s, s)
    ast = Parser(tokenize(s)).parse()
    return truth_bits(lambda a, b: eval_ast(ast, a, b))


def semantic_rule_name(expr: str) -> str:
    bits = parse_rule_expression(expr)
    return BITS_TO_CANONICAL.get(bits, "UNNAMED")


if __name__ == "__main__":
    ok, errs = family_invariants_ok()
    print("family_invariants_ok", ok)
    if errs:
        print("\n".join(errs))
    for name in NONCONSTANT_RULES:
        k, seq = minimal_diagnostic_sequences(name)
        print(name, CANONICAL_EXPR[name], k, seq)
