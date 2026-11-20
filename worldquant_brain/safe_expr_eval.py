"""Strict AST-whitelisted expression evaluator replacing unsafe eval() calls."""
from __future__ import annotations

import ast
import operator
from typing import Any, Mapping

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

MAX_AST_NODES = 512


def safe_eval_expr(expr: str, env: Mapping[str, Any]) -> Any:
    """Safely evaluate a FastExpr/math expression using a strict AST whitelist.

    Disallows attribute access (__globals__, __class__, __subclasses__), subscripts,
    lambdas, comprehensions, and any name not explicitly provided in `env`.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Expression must be a non-empty string")
    tree = ast.parse(expr, mode="eval")
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise ValueError(f"Expression AST exceeds limit ({node_count} > {MAX_AST_NODES})")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                return node.value
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id not in env:
                raise ValueError(f"Disallowed or unknown identifier: {node.id}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")
            return op_fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"Disallowed comparison operator: {type(op).__name__}")
                right = _eval(comp)
                cur = op_fn(left, right)
                result = cur if result is None else (result & cur)
                left = right
            return result
        if isinstance(node, ast.IfExp):
            return _eval(node.body) if _eval(node.test) else _eval(node.orelse)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct whitelisted function calls are allowed")
            fn_name = node.func.id
            if fn_name.startswith("__") or fn_name not in env:
                raise ValueError(f"Disallowed function call: {fn_name}")
            fn = env[fn_name]
            if not callable(fn):
                raise ValueError(f"Identifier is not callable: {fn_name}")
            args = [_eval(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None or kw.arg.startswith("__"):
                    raise ValueError("Starred or dunder kwargs are not allowed")
                kwargs[kw.arg] = _eval(kw.value)
            return fn(*args, **kwargs)
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    return _eval(tree)
