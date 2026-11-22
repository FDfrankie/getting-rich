"""Single-Source-of-Truth Canonical FastExpr AST Parser & Edit Motif Diff Engine.

Unifies expression normalization, exact/normalized/structural hashing,
operator/field multiset extraction, and parent->child AST diffing across
alpha_miner, alpha_novelty, alpha_experiment_store, alpha_repository,
two_stage_funnel, and forbidden_edit_motif.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

AST_SCHEMA_VERSION = "fastexpr-ast-v1"

GROUP_IDENTIFIERS: frozenset[str] = frozenset({
    "market",
    "sector",
    "industry",
    "subindustry",
    "country",
    "exchange",
    "currency",
    "densify",
})

KEYWORD_CONSTANTS: frozenset[str] = frozenset({
    "true",
    "false",
    "nan",
    "inf",
    "none",
})

_TOKEN_SPEC: list[tuple[str, str]] = [
    ("WS", r"\s+"),
    ("NUMBER", r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"),
    ("STRING", r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\""),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP2", r"<=|>=|==|!=|&&|\|\|"),
    ("OP1", r"[+\-*/^<>!&|?:=;]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


@dataclass(frozen=True)
class ASTNode:
    kind: str  # "call", "binop", "unary", "ternary", "kwarg", "assign", "seq", "ident", "number", "string"
    value: str
    children: tuple["ASTNode", ...] = ()

    def to_canonical(self) -> str:
        if self.kind == "number":
            return self.value
        if self.kind == "string":
            return self.value
        if self.kind == "ident":
            return self.value.lower()
        if self.kind == "unary":
            child = self.children[0]
            if child.kind in ("binop", "ternary"):
                return f"{self.value}({child.to_canonical()})"
            return f"{self.value}{child.to_canonical()}"
        if self.kind == "binop":
            left, right = self.children[0].to_canonical(), self.children[1].to_canonical()
            if self.value in {"+", "*"} and left > right:
                left, right = right, left
            return f"({left}{self.value}{right})"
        if self.kind == "ternary":
            c, t, f_node = (x.to_canonical() for x in self.children)
            return f"({c}?{t}:{f_node})"
        if self.kind == "kwarg":
            return f"{self.value.lower()}={self.children[0].to_canonical()}"
        if self.kind == "assign":
            return f"{self.value.lower()}={self.children[0].to_canonical()}"
        if self.kind == "seq":
            return ";".join(c.to_canonical() for c in self.children)
        if self.kind == "call":
            op_low = self.value.lower()
            rendered = [c.to_canonical() for c in self.children]
            if op_low in {"add", "multiply", "min", "max"} and all(c.kind != "kwarg" for c in self.children):
                rendered = sorted(rendered)
            args = ",".join(rendered)
            return f"{op_low}({args})"
        return self.value

    def to_normalized(self) -> str:
        """Replace numeric literals with '#' while preserving full AST structure."""
        if self.kind == "number":
            return "#"
        if self.kind == "string":
            return self.value
        if self.kind == "ident":
            return self.value.lower()
        if self.kind == "unary":
            child = self.children[0]
            if child.kind in ("binop", "ternary"):
                return f"{self.value}({child.to_normalized()})"
            return f"{self.value}{child.to_normalized()}"
        if self.kind == "binop":
            left, right = self.children[0].to_normalized(), self.children[1].to_normalized()
            if self.value in {"+", "*"} and left > right:
                left, right = right, left
            return f"({left}{self.value}{right})"
        if self.kind == "ternary":
            c, t, f_node = (x.to_normalized() for x in self.children)
            return f"({c}?{t}:{f_node})"
        if self.kind == "kwarg":
            return f"{self.value.lower()}={self.children[0].to_normalized()}"
        if self.kind == "assign":
            return f"{self.value.lower()}={self.children[0].to_normalized()}"
        if self.kind == "seq":
            return ";".join(c.to_normalized() for c in self.children)
        if self.kind == "call":
            op_low = self.value.lower()
            rendered = [c.to_normalized() for c in self.children]
            if op_low in {"add", "multiply", "min", "max"} and all(c.kind != "kwarg" for c in self.children):
                rendered = sorted(rendered)
            args = ",".join(rendered)
            return f"{op_low}({args})"
        return self.value

    def to_structural(self) -> str:
        """Strip outer decay wrappers and normalize numeric literals."""
        node = self
        while node.kind == "call" and node.value.lower() in ("ts_decay_linear", "decay_linear") and node.children:
            node = node.children[0]
        return node.to_normalized()

    def to_signature(self) -> str:
        """Abstract parent AST signature using FIELD / WINDOW / GROUP placeholders."""
        if self.kind == "number":
            return "WINDOW"
        if self.kind == "string":
            return "STR"
        if self.kind == "ident":
            low = self.value.lower()
            if low in GROUP_IDENTIFIERS:
                return "GROUP"
            if low in KEYWORD_CONSTANTS:
                return low.upper()
            return "FIELD"
        if self.kind == "unary":
            op = "neg" if self.value == "-" else self.value
            return f"{op}({self.children[0].to_signature()})"
        if self.kind == "binop":
            if self.value == "*" and self.children[0].kind == "unary" and self.children[0].value == "-" and self.children[0].children[0].kind == "number" and self.children[0].children[0].value in ("1", "1.0"):
                return f"neg({self.children[1].to_signature()})"
            return f"({self.children[0].to_signature()}{self.value}{self.children[1].to_signature()})"
        if self.kind == "ternary":
            c, t, f_node = (x.to_signature() for x in self.children)
            return f"({c}?{t}:{f_node})"
        if self.kind == "kwarg":
            return f"{self.value.lower()}={self.children[0].to_signature()}"
        if self.kind == "assign":
            return f"VAR={self.children[0].to_signature()}"
        if self.kind == "seq":
            return ";".join(c.to_signature() for c in self.children)
        if self.kind == "call":
            args = ",".join(c.to_signature() for c in self.children)
            return f"{self.value.lower()}({args})"
        return self.value

    def subtree_hash(self) -> str:
        return hashlib.sha256(self.to_canonical().encode("utf-8")).hexdigest()[:16]


class _FastExprParser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token | None:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_kind: str | None = None, expected_val: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"Unexpected end of expression (expected {expected_kind or expected_val})")
        if expected_kind is not None and tok.kind != expected_kind:
            raise ValueError(f"Expected {expected_kind}, got {tok.kind}({tok.value!r}) at pos {tok.pos}")
        if expected_val is not None and tok.value != expected_val:
            raise ValueError(f"Expected {expected_val!r}, got {tok.value!r} at pos {tok.pos}")
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        stmts = [self.parse_statement()]
        while self.peek() and self.peek().value == ";":
            self.consume(expected_val=";")
            if self.peek() is None:
                break
            stmts.append(self.parse_statement())
        if self.peek() is not None:
            tok = self.peek()
            raise ValueError(f"Unexpected trailing token {tok.value!r} at pos {tok.pos}")
        if len(stmts) == 1:
            return stmts[0]
        return ASTNode("seq", ";", tuple(stmts))

    def parse_statement(self) -> ASTNode:
        if (
            self.peek() is not None
            and self.peek().kind == "IDENT"
            and self.peek(1) is not None
            and self.peek(1).value == "="
        ):
            name = self.consume("IDENT").value
            self.consume(expected_val="=")
            rhs = self.parse_ternary()
            return ASTNode("assign", name, (rhs,))
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        cond = self.parse_or()
        if self.peek() and self.peek().value == "?":
            self.consume(expected_val="?")
            t_branch = self.parse_ternary()
            self.consume(expected_val=":")
            f_branch = self.parse_ternary()
            return ASTNode("ternary", "?:", (cond, t_branch, f_branch))
        return cond

    def parse_or(self) -> ASTNode:
        node = self.parse_and()
        while self.peek() and self.peek().value in ("||", "|"):
            op = self.consume().value
            right = self.parse_and()
            node = ASTNode("binop", op, (node, right))
        return node

    def parse_and(self) -> ASTNode:
        node = self.parse_comparison()
        while self.peek() and self.peek().value in ("&&", "&"):
            op = self.consume().value
            right = self.parse_comparison()
            node = ASTNode("binop", op, (node, right))
        return node

    def parse_comparison(self) -> ASTNode:
        node = self.parse_additive()
        while self.peek() and self.peek().value in ("==", "!=", "<", "<=", ">", ">="):
            op = self.consume().value
            right = self.parse_additive()
            node = ASTNode("binop", op, (node, right))
        return node

    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        while self.peek() and self.peek().value in ("+", "-"):
            op = self.consume().value
            right = self.parse_multiplicative()
            node = ASTNode("binop", op, (node, right))
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_power()
        while self.peek() and self.peek().value in ("*", "/"):
            op = self.consume().value
            right = self.parse_power()
            node = ASTNode("binop", op, (node, right))
        return node

    def parse_power(self) -> ASTNode:
        node = self.parse_unary()
        if self.peek() and self.peek().value == "^":
            op = self.consume().value
            right = self.parse_power()
            return ASTNode("binop", op, (node, right))
        return node

    def parse_unary(self) -> ASTNode:
        if self.peek() and self.peek().value in ("-", "+", "!", "~"):
            op = self.consume().value
            operand = self.parse_unary()
            return ASTNode("unary", op, (operand,))
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if tok.kind == "NUMBER":
            self.consume("NUMBER")
            return ASTNode("number", tok.value)
        if tok.kind == "STRING":
            self.consume("STRING")
            return ASTNode("string", tok.value)
        if tok.kind == "LPAREN":
            self.consume("LPAREN")
            node = self.parse_ternary()
            self.consume("RPAREN")
            return node
        if tok.kind == "IDENT":
            ident = self.consume("IDENT").value
            if self.peek() and self.peek().kind == "LPAREN":
                self.consume("LPAREN")
                args: list[ASTNode] = []
                if self.peek() and self.peek().kind != "RPAREN":
                    while True:
                        if (
                            self.peek() is not None
                            and self.peek().kind == "IDENT"
                            and self.peek(1) is not None
                            and self.peek(1).value == "="
                        ):
                            k = self.consume("IDENT").value
                            self.consume(expected_val="=")
                            v = self.parse_ternary()
                            args.append(ASTNode("kwarg", k, (v,)))
                        else:
                            args.append(self.parse_ternary())
                        if self.peek() and self.peek().kind == "COMMA":
                            self.consume("COMMA")
                        else:
                            break
                self.consume("RPAREN")
                return ASTNode("call", ident, tuple(args))
            return ASTNode("ident", ident)
        raise ValueError(f"Unexpected token {tok.value!r} at pos {tok.pos}")


def tokenize_fastexpr(expression: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(expression or ""):
        kind = m.lastgroup
        if kind == "WS":
            continue
        if kind == "MISMATCH":
            raise ValueError(f"Illegal character {m.group()!r} at pos {m.start()}")
        tokens.append(Token(kind=str(kind), value=m.group(), pos=m.start()))
    return tokens


@dataclass(frozen=True)
class CanonicalASTResult:
    raw_expression: str
    canonical_expression: str
    normalized_expression: str
    structural_expression: str
    ast_signature: str
    exact_hash: str
    normalized_hash: str
    structural_hash: str
    ast_family_hash: str
    operator_multiset: dict[str, int] = field(default_factory=dict)
    field_multiset: dict[str, int] = field(default_factory=dict)
    numeric_literals: tuple[str, ...] = ()
    ast_schema_version: str = AST_SCHEMA_VERSION
    parse_ok: bool = True
    parse_error: str | None = None
    root: ASTNode | None = None

    @property
    def operators(self) -> tuple[str, ...]:
        return tuple(sorted(self.operator_multiset.keys()))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted(self.field_multiset.keys()))


def _collect_multisets(
    node: ASTNode,
    ops: Counter[str],
    fields: Counter[str],
    numbers: list[str],
    assigned_vars: set[str],
) -> None:
    if node.kind == "assign":
        assigned_vars.add(node.value.lower())
        for c in node.children:
            _collect_multisets(c, ops, fields, numbers, assigned_vars)
        return
    if node.kind == "call":
        ops[node.value.lower()] += 1
        for c in node.children:
            _collect_multisets(c, ops, fields, numbers, assigned_vars)
        return
    if node.kind == "kwarg":
        for c in node.children:
            _collect_multisets(c, ops, fields, numbers, assigned_vars)
        return
    if node.kind == "ident":
        low = node.value.lower()
        if low not in GROUP_IDENTIFIERS and low not in KEYWORD_CONSTANTS and low not in assigned_vars:
            fields[low] += 1
        return
    if node.kind == "number":
        numbers.append(node.value)
        return
    for c in node.children:
        _collect_multisets(c, ops, fields, numbers, assigned_vars)


def parse_fastexpr(expression: str) -> CanonicalASTResult:
    raw = str(expression or "").strip()
    if not raw:
        empty_hash = hashlib.sha256(b"").hexdigest()
        return CanonicalASTResult(
            raw_expression="",
            canonical_expression="",
            normalized_expression="",
            structural_expression="",
            ast_signature="",
            exact_hash=empty_hash,
            normalized_hash=empty_hash,
            structural_hash=empty_hash,
            ast_family_hash=empty_hash[:16],
            parse_ok=False,
            parse_error="Empty expression",
            root=None,
        )
    try:
        tokens = tokenize_fastexpr(raw)
        root = _FastExprParser(tokens).parse()
        canonical = root.to_canonical()
        normalized = root.to_normalized()
        structural = root.to_structural()
        signature = root.to_signature()
        ops: Counter[str] = Counter()
        flds: Counter[str] = Counter()
        nums: list[str] = []
        _collect_multisets(root, ops, flds, nums, set())
        exact_h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        norm_h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        struct_h = hashlib.sha256(structural.encode("utf-8")).hexdigest()
        return CanonicalASTResult(
            raw_expression=raw,
            canonical_expression=canonical,
            normalized_expression=normalized,
            structural_expression=structural,
            ast_signature=signature,
            exact_hash=exact_h,
            normalized_hash=norm_h,
            structural_hash=struct_h,
            ast_family_hash=norm_h[:16],
            operator_multiset=dict(sorted(ops.items())),
            field_multiset=dict(sorted(flds.items())),
            numeric_literals=tuple(nums),
            parse_ok=True,
            parse_error=None,
            root=root,
        )
    except Exception as exc:
        compact = re.sub(r"\s+", "", raw).lower()
        norm = re.sub(r"\b\d+(?:\.\d+)?\b", "#", compact)
        exact_h = hashlib.sha256(compact.encode("utf-8")).hexdigest()
        norm_h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return CanonicalASTResult(
            raw_expression=raw,
            canonical_expression=compact,
            normalized_expression=norm,
            structural_expression=norm,
            ast_signature=norm,
            exact_hash=exact_h,
            normalized_hash=norm_h,
            structural_hash=norm_h,
            ast_family_hash=norm_h[:16],
            parse_ok=False,
            parse_error=str(exc),
            root=None,
        )


def _find_first_diff_path(a: ASTNode | None, b: ASTNode | None, path: str = "$") -> tuple[str, ASTNode | None, ASTNode | None]:
    if a is None or b is None:
        return path, a, b
    if a.kind != b.kind or a.value.lower() != b.value.lower() or len(a.children) != len(b.children):
        return path, a, b
    for idx, (ca, cb) in enumerate(zip(a.children, b.children)):
        if ca.to_canonical() != cb.to_canonical():
            return _find_first_diff_path(ca, cb, f"{path}.children[{idx}]")
    return path, a, b


def diff_ast_edit(
    parent_expression: str,
    child_expression: str,
    parent_settings: Mapping[str, Any] | None = None,
    child_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a structured parent->child AST edit diff for Forbidden Edit Motif tracking."""
    p_ast = parse_fastexpr(parent_expression)
    c_ast = parse_fastexpr(child_expression)

    path, before_node, after_node = _find_first_diff_path(p_ast.root, c_ast.root)

    p_ops = Counter(p_ast.operator_multiset)
    c_ops = Counter(c_ast.operator_multiset)
    added_ops = tuple(sorted((c_ops - p_ops).elements()))
    removed_ops = tuple(sorted((p_ops - c_ops).elements()))

    p_flds = set(p_ast.field_multiset.keys())
    c_flds = set(c_ast.field_multiset.keys())
    changed_fields = tuple(sorted((p_flds ^ c_flds)))

    changed_literals: list[str] = []
    if p_ast.numeric_literals != c_ast.numeric_literals:
        for i in range(max(len(p_ast.numeric_literals), len(c_ast.numeric_literals))):
            lv = p_ast.numeric_literals[i] if i < len(p_ast.numeric_literals) else None
            rv = c_ast.numeric_literals[i] if i < len(c_ast.numeric_literals) else None
            if lv != rv:
                changed_literals.append(f"{lv}->{rv}")

    setting_changes: dict[str, dict[str, Any]] = {}
    ps = dict(parent_settings or {})
    cs = dict(child_settings or {})
    for key in sorted(set(ps.keys()) | set(cs.keys())):
        if ps.get(key) != cs.get(key):
            setting_changes[str(key)] = {"before": ps.get(key), "after": cs.get(key)}

    structural_change = (p_ast.normalized_hash != c_ast.normalized_hash)
    parameter_only = (not structural_change) and (bool(changed_literals) or bool(setting_changes))

    # Infer motif_type
    motif_type = "CUSTOM_STRUCTURAL_EDIT"
    if not p_ast.parse_ok or not c_ast.parse_ok:
        motif_type = "PARSE_FAILURE"
    elif parameter_only:
        if setting_changes and not changed_literals:
            if "neutralization" in setting_changes:
                motif_type = "CHANGE_NEUTRALIZATION"
            elif "decay" in setting_changes:
                motif_type = "CHANGE_DECAY"
            elif "truncation" in setting_changes:
                motif_type = "CHANGE_TRUNCATION"
            else:
                motif_type = "CHANGE_SETTINGS"
        else:
            motif_type = "WINDOW_RESIZE"
    elif "rank" in added_ops:
        motif_type = "ADD_RANK"
    elif "rank" in removed_ops:
        motif_type = "REMOVE_RANK"
    elif "hump" in added_ops:
        motif_type = "ADD_HUMP"
    elif "winsorize" in added_ops:
        motif_type = "ADD_WINSORIZE"
    elif "trade_when" in added_ops:
        motif_type = "ADD_TRADE_WHEN"
    elif "ts_backfill" in added_ops:
        motif_type = "ADD_BACKFILL"
    elif "ts_decay_linear" in added_ops or "decay_linear" in added_ops:
        motif_type = "ADD_DECAY"
    elif "group_neutralize" in added_ops:
        motif_type = "ADD_GROUP_NEUTRALIZE"
    elif changed_fields:
        motif_type = "REPLACE_FIELD"

    return {
        "motif_type": motif_type,
        "target_ast_path": path,
        "before_subtree_hash": f"hash:{before_node.to_canonical()}" if before_node else "hash:none",
        "after_subtree_hash": f"hash:{after_node.to_canonical()}" if after_node else "hash:none",
        "added_operators": added_ops,
        "removed_operators": removed_ops,
        "changed_fields": changed_fields,
        "changed_literals": tuple(changed_literals),
        "setting_changes": setting_changes,
        "structural_change": structural_change,
        "parameter_only": parameter_only,
        "ast_schema_version": AST_SCHEMA_VERSION,
        "parent_ast": p_ast,
        "child_ast": c_ast,
    }
