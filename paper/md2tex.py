#!/usr/bin/env python3
"""Minimal, targeted Markdown -> LaTeX converter for the paper/ section files.

Handles exactly the features these drafts use: ATX headers (#/##/###), pipe tables
(booktabs), bold/italic/inline-code, unordered + ordered lists, horizontal rules,
and the unicode symbols present in the text. Emits a \\input-able body fragment
(no preamble). main.tex supplies the preamble + \\input order.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# md stem -> tex filename. paper_abstract.md is handled separately
# (extract_abstract_ieee) -- IEEE format wants a bare abstract paragraph in
# \begin{abstract}, not the full draft file with revision commentary and
# alternate-length variants as a numbered section.
FILES = [
    ("paper_intro",      "intro.tex"),
    ("paper_method",     "method.tex"),
    ("paper_results",    "results.tex"),
    ("paper_discussion", "discussion.tex"),
    ("paper_conclusion", "conclusion.tex"),
]

UNICODE = {
    "≈": r"$\approx$", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "±": r"$\pm$", "×": r"$\times$", "−": r"$-$", "–": "--", "—": "---",
    "≤": r"$\le$", "≥": r"$\ge$", "≠": r"$\neq$", "·": r"$\cdot$",
    "²": r"\textsuperscript{2}", "°": r"$^\circ$", "√": r"$\sqrt{}$",
    "α": r"$\alpha$", "β": r"$\beta$", "λ": r"$\lambda$", "μ": r"$\mu$",
    "τ": r"$\tau$", "σ": r"$\sigma$", "Δ": r"$\Delta$", "…": r"\dots{}",
    "“": "``", "”": "''", "‘": "`", "’": "'", "•": r"$\bullet$",
    "∈": r"$\in$", "∞": r"$\infty$", "∉": r"$\notin$", "∑": r"$\sum$",
    "∀": r"$\forall$", "∩": r"$\cap$", "∪": r"$\cup$", "∝": r"$\propto$",
    "→": r"$\rightarrow$", "§": r"\S{}",
}
SPECIAL = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "\\": r"\textbackslash{}"}


def esc(t):
    """Char-level escape: unicode -> LaTeX (inserted verbatim, NOT re-escaped),
    then LaTeX specials. Iterating original chars avoids double-escaping the
     dollar signs / backslashes introduced by the unicode mapping."""
    out = []
    for ch in t:
        if ch in UNICODE:
            out.append(UNICODE[ch])
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline(t, store=None):
    """Convert inline md (code, bold, italic, links). Tokens share one store across
    recursion; only the top-level call restores, so nested markup (italic spanning
    bold, etc.) resolves correctly."""
    top = store is None
    if top:
        store = []
    def stash(s):
        store.append(s); return f"\x00{len(store)-1}\x00"
    # inline code first (\texttt, escape inside). \x00 excluded so a code span
    # can't start/end inside an already-stashed token.
    t = re.sub(r"`([^`\x00]+)`", lambda m: stash(r"\texttt{" + esc(m.group(1)) + "}"), t)
    t = re.sub(r"\*\*([^*]+?)\*\*", lambda m: stash(r"\textbf{" + inline(m.group(1), store) + "}"), t)
    t = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", lambda m: stash(r"\textit{" + inline(m.group(1), store) + "}"), t)
    t = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", lambda m: stash(r"\textit{" + inline(m.group(1), store) + "}"), t)
    # [[wikilinks]] -> plain text
    t = re.sub(r"\[\[([^\]]+?)\]\]", lambda m: stash(esc(m.group(1))), t)
    # markdown links [text](url) -> text
    t = re.sub(r"\[([^\]]+?)\]\([^)]+\)", lambda m: stash(inline(m.group(1), store)), t)
    t = esc(t)
    if top:
        # iterative restore: a token's replacement may itself contain tokens
        while re.search(r"\x00\d+\x00", t):
            t = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], t)
    return t


def col_spec(sep_cells):
    spec = []
    for c in sep_cells:
        c = c.strip()
        if c.startswith(":") and c.endswith(":"): spec.append("c")
        elif c.endswith(":"): spec.append("r")
        else: spec.append("l")
    return "".join(spec)


def split_row(line):
    line = line.strip()
    if line.startswith("|"): line = line[1:]
    if line.endswith("|"): line = line[:-1]
    return [c.strip() for c in line.split("|")]


# IEEE double-column text width is too narrow for tables with many columns
# or long text cells; tables at/above this column count span both columns
# via a table* float instead of sitting inline in one column.
WIDE_TABLE_COLS = 5


def convert_table(lines, caption=None):
    header = split_row(lines[0])
    sep = split_row(lines[1])
    spec = col_spec(sep)
    body = [split_row(l) for l in lines[2:]]
    tbl = [r"\small", r"\begin{tabular}{" + spec + "}", r"\toprule"]
    tbl.append(" & ".join(inline(c) for c in header) + r" \\")
    tbl.append(r"\midrule")
    for row in body:
        # pad/truncate to header width
        row = (row + [""] * len(header))[:len(header)]
        tbl.append(" & ".join(inline(c) for c in row) + r" \\")
    tbl += [r"\bottomrule", r"\end{tabular}"]

    if len(header) >= WIDE_TABLE_COLS:
        # spans both columns -- standard IEEE handling for wide tables
        out = [r"\begin{table*}[!t]", r"\centering"]
        if caption:
            out.append(r"\caption{" + inline(caption) + "}")
        out += tbl
        out.append(r"\end{table*}")
    else:
        out = [r"\begin{center}"] + tbl + [r"\end{center}"]
    return out


def convert(md_text):
    lines = md_text.split("\n")
    out = []
    i = 0
    in_list = None  # 'ul' | 'ol' | None
    last_header = None  # nearest preceding header title, used as a table* caption
    def close_list():
        nonlocal in_list
        if in_list == "ul": out.append(r"\end{itemize}")
        elif in_list == "ol": out.append(r"\end{enumerate}")
        in_list = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # table block
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:|-]+\|", lines[i+1].strip()):
            close_list()
            tbl = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                tbl.append(lines[j]); j += 1
            out += convert_table(tbl, caption=last_header); out.append("")
            i = j; continue
        # headers
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_list()
            lvl = len(m.group(1)); title = m.group(2).strip()
            title = re.sub(r"\s*\(draft\)\s*$", "", title, flags=re.I)
            last_header = title
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(lvl, "paragraph")
            out.append("\\%s{%s}" % (cmd, inline(title)))
            i += 1; continue
        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            close_list()
            i += 1; continue  # drop (section breaks handled by headers)
        # unordered list
        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            if in_list != "ul": close_list(); out.append(r"\begin{itemize}"); in_list = "ul"
            out.append(r"  \item " + inline(m.group(1))); i += 1; continue
        # ordered list
        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            if in_list != "ol": close_list(); out.append(r"\begin{enumerate}"); in_list = "ol"
            out.append(r"  \item " + inline(m.group(1))); i += 1; continue
        # blank line
        if stripped == "":
            close_list(); out.append(""); i += 1; continue
        # paragraph
        close_list()
        out.append(inline(stripped)); i += 1
    close_list()
    # collapse 3+ blank lines to 1
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def extract_abstract_ieee(md_text):
    """IEEE format needs a bare abstract paragraph (no header, no draft/revision
    commentary, no alternate-length variants) inside \\begin{abstract}...\\end{abstract}
    in main.tex. Pull just the '## Full abstract' section's body text."""
    m = re.search(r"^##\s+Full abstract\s*$(.*?)(?=^---\s*$|^##\s+|\Z)",
                  md_text, flags=re.M | re.S)
    if not m:
        raise ValueError("paper_abstract.md: no '## Full abstract' section found")
    para = m.group(1).strip()
    return inline(para) + "\n"


def main():
    for stem, texname in FILES:
        src = HERE / f"{stem}.md"
        if not src.exists():
            print(f"skip missing {src}"); continue
        body = convert(src.read_text(encoding="utf-8"))
        (HERE / texname).write_text(body, encoding="utf-8")
        print(f"{stem}.md -> {texname}  ({len(body)} chars)")

    abs_src = HERE / "paper_abstract.md"
    if abs_src.exists():
        abs_ieee = extract_abstract_ieee(abs_src.read_text(encoding="utf-8"))
        (HERE / "abstract_ieee.tex").write_text(abs_ieee, encoding="utf-8")
        print(f"paper_abstract.md -> abstract_ieee.tex (Full abstract only, {len(abs_ieee)} chars)")


if __name__ == "__main__":
    main()
