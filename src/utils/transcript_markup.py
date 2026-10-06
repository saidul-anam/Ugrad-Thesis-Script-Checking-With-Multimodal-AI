"""
Well-formedness of transcript markup.

Every extraction stage emits plain text with inline tags: `[struck: ...]`, `[unclear: a | b]`,
`[illegible]` and `[truncated]`. VLM output and string-level patching can leave these tags
unbalanced or nested (`[struck: a [struck: b] c`, an opener with no closer, a stray `]`), and every
downstream consumer (CER metric, Stage 3, Stage 4, segmenter) then interprets the text differently.

This module parses a transcript into tokens that carry a `struck` flag and serialises it back with
exactly one balanced `[struck: ...]` tag per maximal run of struck tokens on a line. The repair is
purely structural (bracket matching); it never looks at what the words are.

  * Brackets are matched with a stack across the whole text, so a tag that legitimately spans
    several lines (`[struck: line one\\nline two]`) strikes every line it covers.
  * `[struck:]` nested inside `[struck:]` is flattened.
  * An opener without a matching closer is closed at the end of its own line.
  * A closer without a matching opener is dropped.
  * `[unclear: ...]`, `[illegible]` and `[truncated]` are atomic and kept verbatim.
  * A student-written literal `[` ... `]` pair is kept as text.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

_LEX = re.compile(
    r"(?P<atom>\[unclear:[^\]\n]*\]|\[illegible\]|\[truncated\])"
    r"|(?P<open>\[struck:)"
    r"|(?P<lit>\[)"
    r"|(?P<close>\])"
    r"|(?P<nl>\n)"
    r"|(?P<ws>[ \t\r\f\v]+)"
    r"|(?P<text>[^\[\]\s]+)",
    re.IGNORECASE,
)


@dataclass
class MarkupToken:
    kind: str      # text | atom | ws | nl
    value: str
    struck: bool = False


def parse_markup(text: str) -> List[MarkupToken]:
    """Tokenise `text` and resolve which content tokens are inside a [struck: ...] span."""
    raw: List[Tuple[str, str]] = [(m.lastgroup, m.group(0)) for m in _LEX.finditer(text or "")]

    # Pair brackets with a stack. A ']' closes the innermost open bracket of either kind.
    partner = {}
    stack: List[int] = []
    for i, (kind, _) in enumerate(raw):
        if kind in ("open", "lit"):
            stack.append(i)
        elif kind == "close" and stack:
            j = stack.pop()
            partner[i] = j
            partner[j] = i

    # Struck depth per token: matched openers span to their closer; unmatched openers span to the
    # end of their own line.
    tokens: List[MarkupToken] = []
    depth = 0
    line_scoped = 0   # unmatched openers active on the current line
    for i, (kind, value) in enumerate(raw):
        if kind == "ws" and (
            (i > 0 and raw[i - 1][0] == "open")
            or (i + 1 < len(raw) and raw[i + 1][0] == "close" and raw[partner.get(i + 1, i + 1)][0] == "open")
        ):
            continue   # padding inside the tag syntax: "[struck: x ]"
        if kind == "nl":
            line_scoped = 0
            tokens.append(MarkupToken("nl", value))
            continue
        if kind == "open":
            if i in partner:
                depth += 1
            else:
                line_scoped += 1
            continue
        if kind == "close":
            if i in partner:
                if raw[partner[i]][0] == "open":
                    depth -= 1
                else:
                    tokens.append(MarkupToken("text", value, depth > 0 or line_scoped > 0))
            continue
        if kind == "lit":
            tokens.append(MarkupToken("text", value, depth > 0 or line_scoped > 0))
            continue
        tokens.append(MarkupToken(kind, value, (depth > 0 or line_scoped > 0) and kind != "ws"))
    return tokens


def serialize_markup(tokens: List[MarkupToken]) -> str:
    """Write tokens back as text, one [struck: ...] per maximal struck run within a line."""
    out: List[str] = []
    run: List[str] = []       # content of the open struck run
    pending_ws: List[str] = []  # whitespace seen after a struck token, not yet assigned

    def close_run() -> None:
        if run and "".join(run).strip():
            out.append("[struck: " + "".join(run).strip() + "]")
        run.clear()

    for tok in tokens:
        if tok.kind == "ws":
            pending_ws.append(tok.value)
            continue
        if tok.kind == "nl":
            close_run()
            out.extend(pending_ws)
            pending_ws.clear()
            out.append(tok.value)
            continue
        if tok.struck:
            if run:
                run.extend(pending_ws)
            else:
                out.extend(pending_ws)
            pending_ws.clear()
            run.append(tok.value)
        else:
            close_run()
            out.extend(pending_ws)
            pending_ws.clear()
            out.append(tok.value)
    close_run()
    out.extend(pending_ws)
    return "".join(out)


def normalize_markup(text: str) -> str:
    """Return `text` with balanced, non-nested, line-scoped [struck: ...] tags."""
    if not text or "[" not in text and "]" not in text:
        return text
    return serialize_markup(parse_markup(text))


def struck_word_spans(text: str) -> List[Tuple[str, bool]]:
    """(word, struck) pairs in reading order; convenience for evaluation and alignment."""
    return [(t.value, t.struck) for t in parse_markup(text) if t.kind in ("text", "atom")]
