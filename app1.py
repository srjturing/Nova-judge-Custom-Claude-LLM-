# app.py — Meta-Evaluator UI with criteria parser & boolean normalization
import base64
import os
import re
from typing import Optional, List, Dict, Any

import streamlit as st
import anthropic

st.set_page_config(page_title="Claude - Meta-Evaluator (Nova Judge - Criteria based)", page_icon="🧪", layout="centered")
st.title("🧪 Claude - Meta-Evaluator (Nova Judge - Criteria based)")
# ======================
# Config / limits
# ======================
SYSTEM_PROMPT_PATH = "system_prompt.txt"   # your full meta-evaluator rubric
CHECK_PDF_NAME = "check.pdf"               # optional reference bundle
MIN_CRITERIA = 4
MAX_CRITERIA = 10
WEIGHT_SUM_TOL = 1e-3  # acceptable tolerance for sum(weights)≈1.0

# ======================
# Client setup (API key from Streamlit secrets)
# ======================
try:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
except Exception:
    st.error("Missing ANTHROPIC_API_KEY in .streamlit/secrets.toml")
    st.stop()

# ======================
# Helpers
# ======================
@st.cache_data(show_spinner=False)
def load_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def file_to_base64(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

CRIT_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<crit>.+?)                           # criterion text (greedy, minimal)
    \s+                                     # space
    (?P<ctype>boolean|non-boolean)\s+       # type token
    (?P<a>-?\d+(?:\.\d+)?)\s+               # score A
    (?P<b>-?\d+(?:\.\d+)?)\s+               # score B
    (?P<w>\d+(?:\.\d+)?)\s+                 # weight
    (?P<just>.+?)\s*                        # justification (rest of line)
    $
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_criteria_block(text: str) -> List[Dict[str, Any]]:
    """
    Parse criteria provided in single-line format:
    <criterion>  <boolean|non-boolean>  <scoreA>  <scoreB>  <weight>  <justification>
    Returns list of dicts.
    """
    items: List[Dict[str, Any]] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = CRIT_LINE_RE.match(line)
        if not m:
            raise ValueError(
                f"Line {i} is not in the expected format.\n"
                "Expected: <criterion>  <boolean|non-boolean>  <scoreA>  <scoreB>  <weight>  <justification>\n"
                f"Got: {raw}"
            )
        ctype = m.group("ctype").lower()
        score_a = float(m.group("a"))
        score_b = float(m.group("b"))
        weight = float(m.group("w"))
        if weight < 0 or weight > 1:
            raise ValueError(f"Line {i}: weight must be in [0,1]. Got {weight}")
        items.append({
            "line_idx": i,
            "criterion": m.group("crit").strip(),
            "type": ctype,  # "boolean" | "non-boolean"
            "score_a_raw": score_a,
            "score_b_raw": score_b,
            "weight": weight,
            "justification": m.group("just").strip(),
        })
    return items

def normalize_scores(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    - For boolean criteria: coerce scores to {1,5} using rule:
        score < 3 -> 1; score >= 3 -> 5
    - For non-boolean: clamp to [1,5] (no rounding).
    Adds normalized scores and flags.
    """
    normed = []
    for it in items:
        a, b = it["score_a_raw"], it["score_b_raw"]
        changed = {"a": False, "b": False}
        if it["type"] == "boolean":
            a_n = 1.0 if a < 3 else 5.0
            b_n = 1.0 if b < 3 else 5.0
            changed["a"] = (a_n != a)
            changed["b"] = (b_n != b)
        else:
            a_n = min(5.0, max(1.0, a))
            b_n = min(5.0, max(1.0, b))
            changed["a"] = (a_n != a)
            changed["b"] = (b_n != b)
        normed.append({
            **it,
            "score_a": a_n,
            "score_b": b_n,
            "changed_a": changed["a"],
            "changed_b": changed["b"],
        })
    return normed

def build_normalized_block(items: List[Dict[str, Any]]) -> str:
    """
    Create a normalized text block to send to the model, mirroring the single-line format.
    """
    lines = []
    for it in items:
        lines.append(
            f"{it['criterion']}  {it['type']}  {int(it['score_a']) if it['score_a'].is_integer() else it['score_a']}  "
            f"{int(it['score_b']) if it['score_b'].is_integer() else it['score_b']}  "
            f"{it['weight']}  {it['justification']}"
        )
    return "\n".join(lines)

# ======================
# UI form
# ======================
with st.form("llm_form", clear_on_submit=False):
    user_prompt = st.text_area(
        "1) PROMPT",
        height=140,
        placeholder="Paste the original user prompt…"
    )
    response_a = st.text_area(
        "2) RESPONSE 01 (Model A)",
        height=220,
        placeholder="Paste Model A's response…"
    )
    response_b = st.text_area(
        "3) RESPONSE 02 (Model B)",
        height=220,
        placeholder="Paste Model B's response…"
    )
    criteria_block = st.text_area(
        "4) EVALUATION CRITERIA (single-line entries; 4–10 criteria)",
        height=260,
        help=(
            "Format per line:\n"
            "<criterion>  <boolean|non-boolean>  <scoreA>  <scoreB>  <weight 0–1>  <combined justification>\n\n"
            "Boolean scores will be normalized to {1,5} (1 = False, 5 = True).\n"
            f"Minimum {MIN_CRITERIA}, maximum {MAX_CRITERIA} criteria."
        ),
        placeholder=(
            "Does the response fulfill the prompt's goal?  boolean  5  5  0.20  "
            "Both responses meet the story requirement clearly and directly.\n"
            "Is the required format strictly followed?  boolean  5  1  0.15  "
            "A follows the requested format; B deviates from section order.\n"
            "Does the solution show correct reasoning steps?  non-boolean  4  3  0.25  "
            "A cites formulas and applies them correctly; B skips intermediate justification.\n"
            "Is the final answer concise and precise?  non-boolean  4  4  0.40  "
            "Both are concise; A is slightly clearer in numerical presentation."
        )
    )

    cols = st.columns(2)
    with cols[0]:
        reported_score_a = st.number_input(
            "5) REPORTED FINAL SCORE — Model A",
            min_value=0.0, max_value=1000.0, value=0.0, step=0.01,
            help="Enter the score the evaluator reported for Model A (supports decimals)."
        )
    with cols[1]:
        reported_score_b = st.number_input(
            "5) REPORTED FINAL SCORE — Model B",
            min_value=0.0, max_value=1000.0, value=0.0, step=0.01,
            help="Enter the score the evaluator reported for Model B (supports decimals)."
        )

    overall_judgment = st.text_area(
        "6) OVERALL JUDGMENT & SUMMARY",
        height=160,
        placeholder="State the decision (A > B, B > A, or Tie) and a concise written summary (10–60 words)…"
    )

    model_name = st.text_input(
        "Model name",
        value="claude-opus-4-1-20250805",
        help="Use a model your account has access to and that supports long text inputs."
    )
    temperature = st.slider(
        "Temperature (0–1)",
        0.0, 1.0, 0.1, 0.1,  # default = 0.1
        help="Lower values = more focused and deterministic outputs."
    )
    max_tokens = st.number_input("Max tokens (max 8192)", min_value=128, max_value=8192, value=6000, step=64)

    submitted = st.form_submit_button("Run")

# ======================
# Submission handler
# ======================
if submitted:
    # Require all fields
    missing = []
    if not user_prompt: missing.append("PROMPT")
    if not response_a: missing.append("RESPONSE 01 (Model A)")
    if not response_b: missing.append("RESPONSE 02 (Model B)")
    if not criteria_block: missing.append("EVALUATION CRITERIA")
    if overall_judgment.strip() == "": missing.append("OVERALL JUDGMENT & SUMMARY")

    if missing:
        st.warning("Please provide: " + ", ".join(missing))
        st.stop()

    # Load system prompt (required)
    system_prompt = load_text_file(SYSTEM_PROMPT_PATH)
    if system_prompt is None:
        st.error(f"System prompt file '{SYSTEM_PROMPT_PATH}' is missing. Place it next to app.py and try again.")
        st.stop()

    # Parse & validate criteria
    try:
        parsed = parse_criteria_block(criteria_block)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_crit = len(parsed)
    if n_crit < MIN_CRITERIA or n_crit > MAX_CRITERIA:
        st.error(f"Criteria count must be between {MIN_CRITERIA} and {MAX_CRITERIA}. You provided {n_crit}.")
        st.stop()

    # Normalize scores (boolean -> {1,5}; non-boolean clamp [1,5])
    normed = normalize_scores(parsed)
    weight_sum = sum(it["weight"] for it in normed)

    # Preview table + warnings
    if any(it["changed_a"] or it["changed_b"] for it in normed):
        st.info("Some scores were normalized: boolean scores coerced to {1,5}; non-boolean scores clamped to [1,5].")

    table_rows = []
    for it in normed:
        table_rows.append({
            "Line": it["line_idx"],
            "Criterion": it["criterion"],
            "Type": it["type"],
            "A (raw)": it["score_a_raw"],
            "A (norm)": it["score_a"],
            "B (raw)": it["score_b_raw"],
            "B (norm)": it["score_b"],
            "Weight": it["weight"],
        })
    st.subheader("Parsed Criteria (normalized)")
    st.table(table_rows)

    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
        st.warning(f"Sum of weights is {weight_sum:.3f}, which is not ~1.000. The meta-evaluator will still run and verify.")

    # Build message content (attach check.pdf silently if present, then text)
    content = []

    # Attach check.pdf silently if present
    check_b64 = file_to_base64(CHECK_PDF_NAME)
    if check_b64:
        content.append({
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": check_b64},
            "cache_control": {"type": "ephemeral"},
        })

    # Assemble the full payload with both RAW and NORMALIZED blocks (for transparency)
    normalized_block = build_normalized_block(normed)
    text_parts = [
        "### PROMPT",
        user_prompt.strip(),
        "### RESPONSE 01 (Model A)",
        response_a.strip(),
        "### RESPONSE 02 (Model B)",
        response_b.strip(),
        "### EVALUATION CRITERIA (RAW)",
        criteria_block.strip(),
        "### EVALUATION CRITERIA (NORMALIZED FOR BOOLEAN/BOUNDED SCORES)",
        normalized_block,
        "### REPORTED FINAL SCORES",
        f"Model A: {reported_score_a}",
        f"Model B: {reported_score_b}",
        "### OVERALL JUDGMENT & SUMMARY",
        overall_judgment.strip(),
        "### NOTES FOR REVIEWER",
        (
            f"- Criteria count: {n_crit}\n"
            f"- Sum of weights (user-provided): {weight_sum:.6f}\n"
            "- Boolean criteria were normalized to {1,5} (1=False, 5=True). "
            "Non-boolean scores were clamped to [1,5] if out of range."
        ),
    ]
    content.append({"type": "text", "text": "\n".join(text_parts)})

    # Call Anthropic
    with st.spinner("Asking Claude…"):
        try:
            resp = client.messages.create(
                model=model_name,
                system=system_prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages=[{"role": "user", "content": content}],
            )

            # Extract text content
            output_text = ""
            for block in resp.content:
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if btype == "text":
                    output_text += getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else "")

            st.subheader("LLM Response")
            st.markdown(output_text.strip() or "_(Empty)_")

            # Token usage (if available)
            usage = getattr(resp, "usage", None)
            if usage:
                in_tok = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
                out_tok = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
                if in_tok is not None and out_tok is not None:
                    st.caption(f"Tokens — input: {in_tok}, output: {out_tok}")

        except anthropic.APIStatusError as e:
            detail = getattr(e, "message", str(e))
            st.error(f"Anthropic API error ({e.status_code}): {detail}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
