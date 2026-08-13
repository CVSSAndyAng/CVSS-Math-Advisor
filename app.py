from __future__ import annotations

import ast
import base64
import hashlib
import math
import json
import os
import re
import secrets
from io import BytesIO
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from gemini_service import (
    DEFAULT_MODEL,
    GeminiAnalysis,
    GeminiTutorError,
    PracticeEvaluation,
    QuestionDetectionResult,
    QuestionFeasibilityResult,
    TargetedPracticeQuestion,
    VisualExplanationResult,
    UploadedAsset,
    analyze_submission,
    assess_question_feasibility,
    detect_questions_in_assets,
    evaluate_practice_attempt,
    generate_followup_practice_question,
    generate_visual_explanation,
    get_api_key,
    required_parts_for_question,
)
from offline_engine import (
    TRACKS,
    AttemptResult,
    Question,
    analyze_own_algebra_question,
    evaluate_attempt,
    generate_question,
    generate_similar,
    official_topic_code,
    topics_for_track,
)

st.set_page_config(
    page_title="Singapore O/N-Level Math Tutor — Gemini + Offline",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
<style>
:root {
  --omt-ink: #172033;
  --omt-muted: #667085;
  --omt-border: #e4e7ec;
  --omt-surface: #ffffff;
  --omt-soft: #f7f9fc;
  --omt-brand: #3b5ccc;
  --omt-brand-soft: #eef2ff;
  --omt-success: #12805c;
  --omt-success-soft: #ecfdf3;
  --omt-warn: #b25e09;
  --omt-warn-soft: #fff7ed;
  --omt-danger: #c2414b;
  --omt-danger-soft: #fff1f2;
  --omt-radius: 18px;
  --omt-shadow: 0 10px 30px rgba(23, 32, 51, .07);
}

html, body, [class*="css"] { color: var(--omt-ink); }
[data-testid="stAppViewContainer"] { background: #f6f8fc; }
[data-testid="stHeader"] { background: rgba(246,248,252,.86); backdrop-filter: blur(10px); }
.block-container { padding-top: 1.25rem; padding-bottom: 4rem; max-width: 1240px; }

h1, h2, h3, h4 { letter-spacing: -.025em; color: var(--omt-ink); }
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 740 !important; }
p, li { line-height: 1.58; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #f9fbff 0%, #f4f6fb 100%); border-right: 1px solid var(--omt-border); }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.omt-hero {
  background: radial-gradient(circle at 92% 12%, rgba(112, 132, 255, .18), transparent 28%),
              linear-gradient(135deg, #ffffff 0%, #f7f8ff 100%);
  border: 1px solid #e1e6f4;
  border-radius: 24px;
  padding: 1.55rem 1.7rem;
  box-shadow: var(--omt-shadow);
  margin: .25rem 0 1.25rem;
}
.omt-eyebrow { color: var(--omt-brand); font-weight: 750; font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .35rem; }
.omt-hero h1 { margin: 0 0 .45rem; font-size: clamp(1.8rem, 3.3vw, 2.75rem); line-height: 1.08; }
.omt-hero p { color: var(--omt-muted); font-size: 1.02rem; margin: 0; max-width: 780px; }
.omt-chip-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
.omt-chip { display:inline-flex; align-items:center; gap:.35rem; background:#fff; border:1px solid var(--omt-border); border-radius:999px; padding:.38rem .68rem; font-size:.82rem; color:#475467; }

.omt-side-brand { padding:.45rem .15rem .85rem; }
.omt-side-brand .title { font-size:1.2rem; font-weight:800; letter-spacing:-.02em; }
.omt-side-brand .sub { color:var(--omt-muted); font-size:.86rem; line-height:1.45; margin-top:.25rem; }
.omt-status-pill { display:flex; align-items:center; gap:.45rem; border-radius:12px; padding:.62rem .72rem; font-size:.84rem; margin:.35rem 0; }
.omt-status-pill.good { background:var(--omt-success-soft); color:#116149; border:1px solid #c9f1df; }
.omt-status-pill.neutral { background:var(--omt-brand-soft); color:#4353a3; border:1px solid #dfe4ff; }

.omt-section-kicker { color:var(--omt-brand); font-weight:750; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
.omt-section-title { font-size:1.45rem; font-weight:780; letter-spacing:-.025em; margin:.08rem 0 .25rem; }
.omt-section-copy { color:var(--omt-muted); margin-bottom:.9rem; }

.omt-focus-card {
  background: linear-gradient(145deg, #ffffff, #fbfcff);
  border: 1px solid #dfe4ee;
  border-radius: 20px;
  box-shadow: 0 7px 22px rgba(23,32,51,.055);
  padding: 1rem 1.1rem;
}
.omt-focus-title { font-size:.8rem; text-transform:uppercase; letter-spacing:.07em; color:var(--omt-brand); font-weight:760; margin-bottom:.4rem; }
.omt-key-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.55rem; margin:.7rem 0; }
.omt-key-item { background:#f8faff; border:1px solid #e6eaf5; border-radius:12px; padding:.62rem .72rem; }

.omt-stage-row { display:grid; grid-template-columns:repeat(3,1fr); gap:.65rem; margin:.7rem 0 1.1rem; }
.omt-stage { border:1px solid var(--omt-border); background:#fff; border-radius:14px; padding:.72rem .8rem; min-height:66px; }
.omt-stage .name { font-weight:720; font-size:.92rem; }
.omt-stage .detail { color:var(--omt-muted); font-size:.78rem; margin-top:.18rem; }
.omt-stage.current { border-color:#aab8ff; background:#f4f6ff; box-shadow:0 0 0 2px rgba(59,92,204,.06); }
.omt-stage.done { border-color:#bfe8d8; background:#f1fbf6; }
.omt-stage.locked { opacity:.72; }

.omt-logic-break { background:var(--omt-warn-soft); border:1px solid #fed7aa; border-left:5px solid #f59e0b; border-radius:14px; padding:.8rem 1rem; margin:.7rem 0 1rem; }
.omt-success-card { background:var(--omt-success-soft); border:1px solid #cceedd; border-left:5px solid #1ca878; border-radius:14px; padding:.8rem 1rem; margin:.7rem 0 1rem; }

/* Streamlit cards and widgets */
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--omt-border) !important; border-radius: var(--omt-radius) !important; background: var(--omt-surface); box-shadow: 0 4px 18px rgba(23,32,51,.035); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--omt-border); border-radius:15px; padding:.75rem .85rem; }
[data-testid="stMetricValue"] { font-weight:780; letter-spacing:-.03em; }
[data-testid="stExpander"] { border:1px solid var(--omt-border) !important; border-radius:14px !important; background:#fff; overflow:hidden; }
[data-testid="stFileUploader"] { border-radius:16px; }
.stTextArea textarea, .stTextInput input { border-radius:12px !important; }

.stButton > button { border-radius:12px; font-weight:680; border-color:#d8deea; transition:transform .12s ease, box-shadow .12s ease; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(23,32,51,.08); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#425fd6,#5b6fe8); border:none; color:white; box-shadow:0 7px 16px rgba(66,95,214,.20); }

/* Tabs */
button[data-baseweb="tab"] { border-radius:12px 12px 0 0; padding:.7rem .9rem !important; font-weight:650; }
button[data-baseweb="tab"][aria-selected="true"] { background:#eef2ff; color:#334bb3; }

/* Alerts */
[data-testid="stAlert"] { border-radius:14px; }

@media (max-width: 1100px) {
  .block-container { max-width:100%; padding-left:1rem; padding-right:1rem; }
}
@media (max-width: 720px) {
  .omt-hero { padding:1.15rem 1.05rem; border-radius:18px; }
  .omt-stage-row { grid-template-columns:1fr; }
}
@media (pointer: coarse) {
  button, [role="button"] { min-height:46px; }
  input, textarea, select { font-size:16px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024


_MATHIO_RENDER_SEQ = 0
_mathio_display_component = None
_mathio_rich_component = None


def _strip_math_transport_delimiters(text: str) -> str:
    """Remove model transport delimiters before sending maths to the MathIO view."""
    if not text:
        return ""
    value = str(text).strip()
    pairs = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))
    for left, right in pairs:
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left): len(value) - len(right)]
            break
    return value.strip()


def _next_mathio_key(text: str) -> str:
    global _MATHIO_RENDER_SEQ
    _MATHIO_RENDER_SEQ += 1
    digest = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"mathio_view_{_MATHIO_RENDER_SEQ}_{digest}"


def render_mathio(text: str) -> None:
    """Render mathematics with the read-only MathIO/MathLive view; never expose source notation."""
    value = _strip_math_transport_delimiters(text)
    if not value:
        return
    if _mathio_display_component is None:
        st.info("Equation view is temporarily unavailable. Reload the page to restore the maths display.")
        return
    _mathio_display_component(
        data={"math": value},
        default={},
        key=_next_mathio_key(value),
        width="stretch",
        height="content",
    )


_MATHIO_MIXED_PATTERN = re.compile(
    r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)"
)


def _contains_raw_math_source(text: str) -> bool:
    """Detect source-style maths commands that should never be shown directly to students."""
    return bool(re.search(r"\\(?:frac|sqrt|times|div|cdot|theta|alpha|beta|gamma|pi|sin|cos|tan|log|ln|leq|geq|neq|pm|text|overline|bar|angle|circ)\b|\^\{|_\{", text or ""))


def render_mathio_mixed(text: str) -> None:
    r"""Render prose and mathematics together in one natural, inline MathIO view.

    This avoids the old stacked layout where every symbol/formula became a separate
    Streamlit row. New Gemini responses delimit maths with \(...\) or \[...\].
    """
    if not text:
        return
    value = str(text).strip()
    if not value:
        return

    # Preferred path: one browser component lays out prose + inline mathematics as a
    # single paragraph/card, so expressions such as AB, 40° and tan(theta) stay inline.
    if _mathio_rich_component is not None and _MATHIO_MIXED_PATTERN.search(value):
        _mathio_rich_component(
            data={"text": value},
            default={},
            key=_next_mathio_key("mixed:" + value),
            width="stretch",
            height="content",
        )
        return

    # Plain prose remains native Streamlit text. A source-heavy compatibility value from
    # an older session is rendered as mathematics rather than exposing raw commands.
    if _contains_raw_math_source(value):
        render_mathio(value)
    else:
        st.markdown(value)


def render_math_text(text: str) -> None:
    """Compatibility wrapper: all student-facing maths now goes through MathIO."""
    render_mathio_mixed(text)


MATHLIVE_VERSION = "0.110.0"  # Patched MathLive release used by the visual equation editor.

_EQUATION_EDITOR_HTML = """
<div class="omt-math-editor">
  <div class="omt-editor-label"></div>
  <div class="omt-editor-help">Type directly into each maths box. Use the keyboard icon for fractions, roots, powers, trig and symbols.</div>
  <div class="omt-editor-rows"></div>
  <div class="omt-editor-actions">
    <button type="button" class="omt-add-step">＋ Add step</button>
  </div>
  <div class="omt-editor-status" aria-live="polite"></div>
</div>
"""

_EQUATION_EDITOR_CSS = """
.omt-math-editor { width: 100%; font-family: var(--st-font, sans-serif); }
.omt-editor-label { font-weight: 600; margin-bottom: .3rem; }
.omt-editor-help { color: var(--st-text-color); opacity: .72; font-size: .86rem; margin-bottom: .65rem; }
.omt-editor-row { display: grid; grid-template-columns: 3.2rem minmax(0,1fr) 2.5rem; align-items: center; gap: .45rem; margin: .45rem 0; }
.omt-step-label { font-size: .83rem; opacity: .75; }
.omt-editor-row math-field { width: 100%; min-height: 3.1rem; box-sizing: border-box; border: 1px solid rgba(128,128,128,.45); border-radius: .55rem; padding: .45rem .6rem; font-size: 1.12rem; background: var(--st-background-color, white); color: var(--st-text-color, #222); --caret-color: var(--st-primary-color, #ff4b4b); --selection-background-color: color-mix(in srgb, var(--st-primary-color, #ff4b4b) 20%, transparent); }
.omt-editor-row math-field:focus-within { outline: 2px solid color-mix(in srgb, var(--st-primary-color, #ff4b4b) 45%, transparent); outline-offset: 1px; }
.omt-remove-step { border: 0; background: transparent; cursor: pointer; font-size: 1.05rem; opacity: .65; padding: .3rem; }
.omt-remove-step:hover { opacity: 1; }
.omt-editor-actions { margin-top: .5rem; }
.omt-add-step { border: 1px solid rgba(128,128,128,.38); border-radius: .45rem; background: transparent; color: var(--st-text-color, #222); padding: .4rem .7rem; cursor: pointer; }
.omt-add-step:hover { border-color: var(--st-primary-color, #ff4b4b); }
.omt-editor-status { font-size: .78rem; opacity: .7; margin-top: .4rem; min-height: 1rem; }
@media (max-width: 640px) {
  .omt-editor-row { grid-template-columns: 2.7rem minmax(0,1fr) 2.2rem; }
  .omt-editor-row math-field { font-size: 1.05rem; }
}
@media (pointer: coarse) {
  .omt-editor-row math-field { min-height: 4rem; font-size: 1.2rem; padding: .7rem .75rem; }
  .omt-add-step, .omt-remove-step { min-height: 44px; min-width: 44px; }
}
"""

_EQUATION_EDITOR_JS = f"""
const MATHLIVE_URL = 'https://cdn.jsdelivr.net/npm/mathlive@{MATHLIVE_VERSION}/+esm';

async function ensureMathLive() {{
  if (!customElements.get('math-field')) {{
    if (!globalThis.__omtMathLivePromise) {{
      globalThis.__omtMathLivePromise = import(MATHLIVE_URL);
    }}
    await globalThis.__omtMathLivePromise;
  }}
}}

function normalizedPayload(raw) {{
  const latex = Array.isArray(raw?.latex) ? raw.latex.map(x => String(x ?? '')) : [''];
  const ascii = Array.isArray(raw?.ascii) ? raw.ascii.map(x => String(x ?? '')) : latex.map(() => '');
  if (latex.length === 0) latex.push('');
  while (ascii.length < latex.length) ascii.push('');
  return {{ latex: latex.slice(0, 20), ascii: ascii.slice(0, 20) }};
}}

export default async function(component) {{
  const {{ parentElement, data, setStateValue }} = component;
  const label = parentElement.querySelector('.omt-editor-label');
  const rows = parentElement.querySelector('.omt-editor-rows');
  const addButton = parentElement.querySelector('.omt-add-step');
  const status = parentElement.querySelector('.omt-editor-status');
  label.textContent = data?.label || 'Student working';

  try {{
    await ensureMathLive();
  }} catch (err) {{
    status.textContent = 'Equation editor could not load. Check the browser connection and reload the page.';
    return;
  }}

  const incoming = normalizedPayload(data?.payload);
  const state = parentElement.__omtState || {{ payload: incoming, timer: null }};
  parentElement.__omtState = state;

  // Python session state is authoritative after a Streamlit rerun.
  state.payload = incoming;

  const emit = () => {{
    const editors = Array.from(rows.querySelectorAll('math-field'));
    state.payload = {{
      latex: editors.map(mf => mf.value || ''),
      ascii: editors.map(mf => {{
        try {{ return mf.getValue('ascii-math') || ''; }} catch (_) {{ return ''; }}
      }}),
    }};
    setStateValue('payload', state.payload);
    status.textContent = 'Working saved';
  }};

  const scheduleEmit = () => {{
    status.textContent = 'Editing…';
    if (state.timer) clearTimeout(state.timer);
    state.timer = setTimeout(emit, 700);
  }};

  const renderRows = () => {{
    rows.replaceChildren();
    state.payload.latex.forEach((value, index) => {{
      const row = document.createElement('div');
      row.className = 'omt-editor-row';

      const step = document.createElement('span');
      step.className = 'omt-step-label';
      step.textContent = `Step ${{index + 1}}`;

      const mf = document.createElement('math-field');
      mf.value = value || '';
      mf.setAttribute('virtual-keyboard-mode', 'auto');
      mf.setAttribute('smart-fence', '');
      mf.setAttribute('aria-label', `Mathematics working step ${{index + 1}}`);
      mf.addEventListener('input', scheduleEmit);
      mf.addEventListener('change', emit);
      mf.addEventListener('blur', emit);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'omt-remove-step';
      remove.textContent = '✕';
      remove.title = 'Remove this step';
      remove.setAttribute('aria-label', `Remove step ${{index + 1}}`);
      remove.disabled = state.payload.latex.length <= 1;
      remove.onclick = () => {{
        if (state.payload.latex.length <= 1) return;
        state.payload.latex.splice(index, 1);
        state.payload.ascii.splice(index, 1);
        renderRows();
        emit();
      }};

      row.append(step, mf, remove);
      rows.appendChild(row);
    }});
  }};

  addButton.onclick = () => {{
    if (state.payload.latex.length >= 20) {{
      status.textContent = 'Maximum 20 working steps.';
      return;
    }}
    state.payload.latex.push('');
    state.payload.ascii.push('');
    renderRows();
    emit();
    const editors = rows.querySelectorAll('math-field');
    editors[editors.length - 1]?.focus();
  }};

  renderRows();
}}
"""

try:
    _equation_editor_component = st.components.v2.component(
        "omt_math_working_editor",
        html=_EQUATION_EDITOR_HTML,
        css=_EQUATION_EDITOR_CSS,
        js=_EQUATION_EDITOR_JS,
        isolate_styles=False,
    )
except Exception:
    _equation_editor_component = None


_MATHIO_DISPLAY_HTML = """
<div class="omt-mathio-display" aria-live="polite"></div>
"""

_MATHIO_DISPLAY_CSS = """
.omt-mathio-display { width: 100%; overflow-x: auto; padding: .15rem 0 .25rem 0; }
.omt-mathio-display math-field {
  display: inline-block;
  width: auto;
  min-width: 0;
  border: 0 !important;
  outline: 0 !important;
  background: transparent !important;
  color: var(--st-text-color, #222);
  padding: 0;
  margin: 0;
  font-size: 1.08rem;
  pointer-events: none;
  --caret-color: transparent;
  --selection-background-color: transparent;
}
@media (pointer: coarse) { .omt-mathio-display math-field { font-size: 1.16rem; } }
"""

_MATHIO_DISPLAY_JS = f"""
const MATHLIVE_URL = 'https://cdn.jsdelivr.net/npm/mathlive@{MATHLIVE_VERSION}/+esm';

async function ensureMathLiveForDisplay() {{
  if (!customElements.get('math-field')) {{
    if (!globalThis.__omtMathLivePromise) {{
      globalThis.__omtMathLivePromise = import(MATHLIVE_URL);
    }}
    await globalThis.__omtMathLivePromise;
  }}
}}

export default async function(component) {{
  const {{ parentElement, data }} = component;
  const root = parentElement.querySelector('.omt-mathio-display');
  root.replaceChildren();
  try {{
    await ensureMathLiveForDisplay();
    const mf = document.createElement('math-field');
    mf.value = String(data?.math || '');
    mf.readOnly = true;
    mf.setAttribute('read-only', '');
    mf.setAttribute('virtual-keyboard-mode', 'off');
    mf.setAttribute('aria-label', 'Rendered mathematics');
    mf.tabIndex = -1;
    root.appendChild(mf);
  }} catch (err) {{
    const msg = document.createElement('span');
    msg.textContent = 'Equation view could not load. Reload the page to restore the maths display.';
    msg.style.opacity = '.72';
    root.appendChild(msg);
  }}
}}
"""

try:
    _mathio_display_component = st.components.v2.component(
        "omt_mathio_display",
        html=_MATHIO_DISPLAY_HTML,
        css=_MATHIO_DISPLAY_CSS,
        js=_MATHIO_DISPLAY_JS,
        isolate_styles=False,
    )
except Exception:
    _mathio_display_component = None


_MATHIO_RICH_HTML = """
<div class="omt-rich-math" aria-live="polite"></div>
"""

_MATHIO_RICH_CSS = """
.omt-rich-math {
  width: 100%; color: var(--st-text-color, #172033); line-height: 1.68;
  font-size: .98rem; overflow-wrap: anywhere;
}
.omt-rich-math .omt-rich-paragraph { margin: .15rem 0 .45rem; }
.omt-rich-math strong { font-weight: 720; }
.omt-rich-math math-field {
  display: inline-block; width: auto; min-width: 0; border: 0 !important;
  outline: 0 !important; background: transparent !important; padding: 0 .03rem;
  margin: 0 .03rem; color: var(--st-text-color, #172033); font-size: 1.03em;
  pointer-events: none; vertical-align: -0.12em; --caret-color: transparent;
  --selection-background-color: transparent;
}
.omt-rich-math math-field.omt-display-math {
  display: block; width: fit-content; max-width: 100%; margin: .42rem 0 .55rem;
  font-size: 1.12em; overflow-x: auto; vertical-align: baseline;
}
@media (pointer: coarse) { .omt-rich-math { font-size: 1rem; } }
"""

_MATHIO_RICH_JS = f"""
const MATHLIVE_URL = 'https://cdn.jsdelivr.net/npm/mathlive@{MATHLIVE_VERSION}/+esm';

async function ensureMathLiveForRich() {{
  if (!customElements.get('math-field')) {{
    if (!globalThis.__omtMathLivePromise) globalThis.__omtMathLivePromise = import(MATHLIVE_URL);
    await globalThis.__omtMathLivePromise;
  }}
}}

function appendTextWithBold(root, text) {{
  const bits = String(text || '').split(/(\\*\\*[^*]+\\*\\*)/g);
  for (const bit of bits) {{
    if (!bit) continue;
    if (bit.startsWith('**') && bit.endsWith('**')) {{
      const strong = document.createElement('strong');
      strong.textContent = bit.slice(2, -2);
      root.appendChild(strong);
    }} else {{
      const lines = bit.split('\\n');
      lines.forEach((line, index) => {{
        if (index) root.appendChild(document.createElement('br'));
        root.appendChild(document.createTextNode(line));
      }});
    }}
  }}
}}

function unwrapMath(token) {{
  if (token.startsWith('\\\\[') && token.endsWith('\\\\]')) return [token.slice(2,-2), true];
  if (token.startsWith('\\\\(') && token.endsWith('\\\\)')) return [token.slice(2,-2), false];
  if (token.startsWith('$$') && token.endsWith('$$')) return [token.slice(2,-2), true];
  if (token.startsWith('$') && token.endsWith('$')) return [token.slice(1,-1), false];
  return [token, false];
}}

export default async function(component) {{
  const {{ parentElement, data }} = component;
  const root = parentElement.querySelector('.omt-rich-math');
  root.replaceChildren();
  try {{
    await ensureMathLiveForRich();
    const raw = String(data?.text || '');
    const pattern = /(\\\\\\[[\\s\\S]*?\\\\\\]|\\\\\\([\\s\\S]*?\\\\\\)|\\$\\$[\\s\\S]*?\\$\\$|\\$[^$\\n]+?\\$)/g;
    let last = 0;
    for (const match of raw.matchAll(pattern)) {{
      appendTextWithBold(root, raw.slice(last, match.index));
      const [latex, display] = unwrapMath(match[0]);
      const mf = document.createElement('math-field');
      mf.value = latex.trim();
      mf.readOnly = true;
      mf.setAttribute('read-only', '');
      mf.setAttribute('virtual-keyboard-mode', 'off');
      mf.tabIndex = -1;
      if (display) mf.classList.add('omt-display-math');
      root.appendChild(mf);
      last = match.index + match[0].length;
    }}
    appendTextWithBold(root, raw.slice(last));
  }} catch (err) {{
    const msg = document.createElement('span');
    msg.textContent = 'Rich equation view could not load. Reload the page to restore the maths display.';
    msg.style.opacity = '.72'; root.appendChild(msg);
  }}
}}
"""

try:
    _mathio_rich_component = st.components.v2.component(
        "omt_rich_math_text",
        html=_MATHIO_RICH_HTML,
        css=_MATHIO_RICH_CSS,
        js=_MATHIO_RICH_JS,
        isolate_styles=False,
    )
except Exception:
    _mathio_rich_component = None


def equation_working_editor(label: str, *, key: str) -> tuple[list[str], list[str]]:
    """Render a visual multi-step MathLive editor and return LaTeX + ASCIIMath lines."""
    if _equation_editor_component is None:
        fallback = st.text_area(
            label,
            key=f"{key}_fallback",
            height=150,
            placeholder="Fallback: type one mathematical step per line, e.g. m=(4-1)/(-2-7)",
        )
        lines = [line.strip() for line in fallback.splitlines() if line.strip()]
        return lines, lines

    prior = st.session_state.get(key, {})
    payload = prior.get("payload", {"latex": [""], "ascii": [""]}) if isinstance(prior, dict) else {"latex": [""], "ascii": [""]}
    if not isinstance(payload, dict):
        payload = {"latex": [""], "ascii": [""]}
    payload.setdefault("latex", [""])
    payload.setdefault("ascii", [""])

    result = _equation_editor_component(
        data={"label": label, "payload": payload},
        default={"payload": payload},
        key=key,
        on_payload_change=lambda: None,
        width="stretch",
        height="content",
    )
    output = getattr(result, "payload", None) or payload
    latex = [str(x).strip() for x in output.get("latex", [])]
    ascii_values = [str(x).strip() for x in output.get("ascii", [])]
    while len(ascii_values) < len(latex):
        ascii_values.append("")
    return latex, ascii_values


def working_input(
    label: str,
    *,
    text_key: str,
    format_key: str,
    height: int = 170,
    plain_placeholder: str = "Show the important reasoning steps, not only the final answer.",
) -> tuple[str, str, str]:
    mode = st.radio(
        "Working input method",
        ["Equation editor", "Text working"],
        horizontal=True,
        key=format_key,
        help="Equation editor gives a visual maths keyboard; Text working is useful for sentences and explanations.",
    )

    if mode == "Equation editor":
        latex_lines, ascii_lines = equation_working_editor(label, key=f"{text_key}_equation")
        explanation = st.text_area(
            "Optional explanation in words",
            key=f"{text_key}_explanation",
            height=90,
            placeholder="Example: I expanded the bracket first, then collected like terms.",
        )
        used_latex = [line for line in latex_lines if line]
        used_ascii = [line for line in ascii_lines if line]
        working_lines = [f"Step {i}: \\({line}\\)" for i, line in enumerate(used_latex, 1)]
        if explanation.strip():
            working_lines.append(f"Student explanation: {explanation.strip()}")
        working_for_gemini = "\n".join(working_lines)
        offline_text = "\n".join(used_ascii)
        st.caption("The equation editor stores the mathematical structure automatically; students do not need to type equation code.")
        return working_for_gemini, mode, offline_text

    value = st.text_area(label, key=text_key, height=height, placeholder=plain_placeholder)
    return value, mode, value



_HANDWRITING_HTML = """
<div class="omt-handwriting-pad">
  <div class="omt-handwriting-help">Write with Apple Pencil, stylus, or finger. Nothing is sent to Streamlit while you are writing, so the pad will not refresh after every stroke.</div>
  <div class="omt-handwriting-toolbar">
    <button type="button" class="omt-undo-pad">Undo</button>
    <button type="button" class="omt-clear-pad">Clear</button>
    <button type="button" class="omt-save-pad omt-primary-pad">Save handwriting</button>
  </div>
  <canvas class="omt-handwriting-canvas" aria-label="Handwritten mathematics working area"></canvas>
  <div class="omt-handwriting-status" aria-live="polite">Write first, then tap Save handwriting before checking your answer.</div>
</div>
"""

_HANDWRITING_CSS = """
.omt-handwriting-pad { width:100%; font-family:var(--st-font,sans-serif); overscroll-behavior:contain; }
.omt-handwriting-help { opacity:.78; font-size:.9rem; margin:0 0 .55rem 0; }
.omt-handwriting-toolbar { display:flex; justify-content:flex-end; gap:.45rem; flex-wrap:wrap; margin-bottom:.5rem; }
.omt-handwriting-toolbar button { min-height:44px; padding:.5rem .85rem; border:1px solid rgba(128,128,128,.42); border-radius:.55rem; background:transparent; color:var(--st-text-color,#222); font-weight:600; }
.omt-handwriting-toolbar .omt-primary-pad { background:#ff4b4b; color:#fff; border-color:#ff4b4b; }
.omt-handwriting-canvas { width:100%; height:430px; display:block; background:#fff; border:1px solid rgba(128,128,128,.48); border-radius:.7rem; touch-action:none; user-select:none; -webkit-user-select:none; -webkit-touch-callout:none; overscroll-behavior:contain; box-sizing:border-box; }
.omt-handwriting-status { min-height:1.2rem; margin-top:.4rem; font-size:.8rem; opacity:.76; }
@media (max-width:900px) { .omt-handwriting-canvas { height:390px; } }
@media (pointer:coarse) { .omt-handwriting-canvas { height:48vh; min-height:340px; max-height:600px; } .omt-handwriting-toolbar button { min-height:48px; font-size:1rem; } }
"""

_HANDWRITING_JS = r"""
function validPng(value) {
  return typeof value === 'string' && value.startsWith('data:image/png;base64,') && value.length > 100;
}

export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const canvas = parentElement.querySelector('.omt-handwriting-canvas');
  const undoButton = parentElement.querySelector('.omt-undo-pad');
  const clearButton = parentElement.querySelector('.omt-clear-pad');
  const saveButton = parentElement.querySelector('.omt-save-pad');
  const status = parentElement.querySelector('.omt-handwriting-status');
  const ctx = canvas.getContext('2d', { alpha:false, desynchronized:true });

  // Important: setStateValue() causes a Streamlit rerun. Therefore it is only
  // called when the student taps Save handwriting, never at pointer-up.
  let drawing = false;
  let hasInk = false;
  let dirty = false;
  let lastX = 0;
  let lastY = 0;
  let history = [];
  const restoreData = validPng(data?.image_data_url) ? data.image_data_url : '';

  const cssSize = () => {
    const r = canvas.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  };

  const paintWhite = () => {
    const { w, h } = cssSize();
    ctx.save();
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.restore();
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,w,h);
  };

  const configureCanvasOnce = () => {
    const { w, h } = cssSize();
    const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    canvas.width = Math.max(1, Math.round(w * ratio));
    canvas.height = Math.max(1, Math.round(h * ratio));
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 2.6;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,w,h);

    if (restoreData) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img,0,0,w,h);
        hasInk = true;
        dirty = false;
        status.textContent = 'Saved handwriting restored. Continue writing or tap Save handwriting again after changes.';
      };
      img.src = restoreData;
    }
  };

  const snapshot = () => {
    try {
      history.push(canvas.toDataURL('image/png'));
      if (history.length > 24) history.shift();
    } catch (_) {}
  };

  const restoreSnapshot = (url) => {
    if (!validPng(url)) return;
    const img = new Image();
    img.onload = () => {
      const { w, h } = cssSize();
      ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);
      ctx.drawImage(img,0,0,w,h);
      hasInk = true; dirty = true;
      status.textContent = 'Undo applied. Tap Save handwriting when finished.';
    };
    img.src = url;
  };

  const point = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [ev.clientX-r.left, ev.clientY-r.top];
  };

  const drawEvent = (ev) => {
    const events = ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev];
    for (const e of events) {
      const [x,y] = point(e);
      const pressure = e.pressure && e.pressure > 0 ? e.pressure : 0.5;
      ctx.lineWidth = 1.9 + pressure * 2.7;
      ctx.lineTo(x,y);
      ctx.stroke();
      lastX=x; lastY=y;
    }
  };

  const start = (ev) => {
    if (ev.pointerType === 'touch' && ev.isPrimary === false) return;
    ev.preventDefault();
    snapshot();
    drawing = true; hasInk = true; dirty = true;
    canvas.setPointerCapture?.(ev.pointerId);
    [lastX,lastY] = point(ev);
    ctx.beginPath(); ctx.moveTo(lastX,lastY);
    status.textContent = 'Writing… tap Save handwriting when the page is complete.';
  };

  const move = (ev) => {
    if (!drawing) return;
    ev.preventDefault();
    drawEvent(ev);
  };

  const end = (ev) => {
    if (!drawing) return;
    ev.preventDefault();
    drawing = false;
    try { canvas.releasePointerCapture?.(ev.pointerId); } catch (_) {}
    ctx.closePath();
    status.textContent = 'Unsaved handwriting. Tap Save handwriting before checking your answer.';
  };

  canvas.onpointerdown=start;
  canvas.onpointermove=move;
  canvas.onpointerup=end;
  canvas.onpointercancel=end;

  undoButton.onclick = () => {
    const previous = history.pop();
    if (previous) restoreSnapshot(previous);
  };

  clearButton.onclick = () => {
    snapshot();
    const { w,h }=cssSize();
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);
    hasInk=false; dirty=true;
    status.textContent='Canvas cleared locally. Tap Save handwriting to save the blank page, or Undo to restore.';
  };

  saveButton.onclick = () => {
    // This is the only normal path that sends canvas state to Python and reruns Streamlit.
    const url = hasInk ? canvas.toDataURL('image/png') : '';
    dirty=false;
    status.textContent = hasInk ? 'Saving handwriting…' : 'Saving blank canvas…';
    setStateValue('image_data_url', url);
  };

  configureCanvasOnce();

  // Prevent Safari gestures/scrolling from stealing Pencil strokes while inside the pad.
  canvas.addEventListener('touchstart', e => e.preventDefault(), { passive:false });
  canvas.addEventListener('touchmove', e => e.preventDefault(), { passive:false });
}
"""

try:
    _handwriting_component = st.components.v2.component(
        "omt_handwriting_pad",
        html=_HANDWRITING_HTML,
        css=_HANDWRITING_CSS,
        js=_HANDWRITING_JS,
        isolate_styles=False,
    )
except Exception:
    _handwriting_component = None


def handwriting_pad(*, key: str) -> UploadedAsset | None:
    """Return a PNG UploadedAsset from a touch/Pencil handwriting canvas."""
    if _handwriting_component is None:
        st.info("The on-screen handwriting pad is unavailable in this browser. Use camera or file upload below.")
        return None
    prior = st.session_state.get(key, {})
    prior_url = prior.get("image_data_url", "") if isinstance(prior, dict) else ""
    result = _handwriting_component(
        data={"image_data_url": prior_url},
        default={"image_data_url": prior_url},
        key=key,
        on_image_data_url_change=lambda: None,
        width="stretch",
        height="content",
    )
    data_url = getattr(result, "image_data_url", "") or prior_url
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except Exception:
        return None
    if len(raw) < 200:
        return None
    return UploadedAsset(name="ipad-handwritten-working.png", mime_type="image/png", data=raw)





def _student_table_tool(*, key_base: str) -> str:
    """Optional fillable working table whose contents are submitted with the attempt."""
    st.markdown("#### Working tools")
    use_table = st.toggle(
        "Insert a table",
        key=f"{key_base}_use_table",
        help="Useful for value tables, frequency tables, coordinates, sequences, and organised calculations.",
    )
    if not use_table:
        return ""

    c1, c2 = st.columns([1, 2])
    with c1:
        rows = int(st.number_input("Starting rows", min_value=1, max_value=15, value=4, step=1, key=f"{key_base}_table_rows"))
    with c2:
        headers_text = st.text_input(
            "Column headings",
            value="x, y",
            key=f"{key_base}_table_headers",
            help="Separate headings with commas, e.g. x, y or Class interval, Frequency.",
        )
    headers = [h.strip() for h in headers_text.split(",") if h.strip()][:8]
    if not headers:
        headers = ["Column 1", "Column 2"]

    seed_key = f"{key_base}_table_seed_{'|'.join(headers)}_{rows}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = pd.DataFrame([[""] * len(headers) for _ in range(rows)], columns=headers)

    edited = st.data_editor(
        st.session_state[seed_key],
        key=f"{key_base}_table_editor_{hashlib.sha1(seed_key.encode()).hexdigest()[:8]}",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        height=min(460, 78 + max(rows, 3) * 38),
    )
    st.caption("You can add or delete rows directly in the table. Filled cells are included when your reasoning is checked.")

    if edited is None or len(edited) == 0:
        return ""
    clean = edited.fillna("").astype(str)
    nonempty = clean.apply(lambda row: any(v.strip() for v in row.tolist()), axis=1)
    clean = clean[nonempty]
    if clean.empty:
        return ""
    lines = ["Student working table:", "\t".join(headers)]
    for _, row in clean.iterrows():
        lines.append("\t".join(str(row[h]).strip() for h in headers))
    return "\n".join(lines)


def _looks_like_function_or_graph_question(pq: TargetedPracticeQuestion | None) -> bool:
    if pq is None:
        return False
    diagram = getattr(pq, "diagram_2d", None)
    if diagram is not None and bool(getattr(diagram, "show_axes", False)):
        return True
    text = " ".join([
        str(getattr(pq, "question", "") or ""),
        str(getattr(pq, "focus_prompt", "") or ""),
        str(getattr(pq, "target_skill", "") or ""),
    ]).lower()
    return bool(re.search(r"\b(function|graph|plot|curve|coordinate|coordinates|intercept|gradient|turning point|quadratic|linear graph)\b|f\s*\(|y\s*=", text))


def _normalise_function_expression(expr: str) -> str:
    value = str(expr or "").strip()
    value = _strip_math_transport_delimiters(value)
    value = value.replace("−", "-").replace("×", "*").replace("÷", "/").replace("^", "**")
    value = re.sub(r"^\s*(?:y|f\s*\(\s*x\s*\))\s*=\s*", "", value, flags=re.I)
    # Common implicit multiplication used by students: 2x, 3(x+1), x(x-2).
    value = re.sub(r"(?<=\d)(?=x\b)", "*", value, flags=re.I)
    value = re.sub(r"(?<=\d)(?=\()", "*", value)
    value = re.sub(r"(?<=x)(?=\()", "*", value, flags=re.I)
    value = re.sub(r"(?<=\))(?=(?:x|\d|\())", "*", value, flags=re.I)
    return value


_ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "ln": math.log, "abs": abs,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval_function(expr: str, x_value: float) -> float:
    """Evaluate a student-entered f(x) using a tiny arithmetic AST, never Python eval()."""
    tree = ast.parse(_normalise_function_expression(expr), mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id.lower() == "x":
                return float(x_value)
            if node.id.lower() in _ALLOWED_CONSTS:
                return float(_ALLOWED_CONSTS[node.id.lower()])
            raise ValueError("Only x and standard constants are allowed.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = walk(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.Pow): return a ** b
            raise ValueError("Unsupported operator.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            if name not in _ALLOWED_FUNCS or len(node.args) != 1:
                raise ValueError("Use sin, cos, tan, sqrt, exp, log/ln, or abs with one argument.")
            return float(_ALLOWED_FUNCS[name](walk(node.args[0])))
        raise ValueError("Unsupported function expression.")

    result = float(walk(tree))
    if not math.isfinite(result):
        raise ValueError("Function is not finite here.")
    return result


def _sample_function_scene(expr: str, *, x_min: float, x_max: float) -> dict[str, Any]:
    samples: list[list[float]] = []
    segments: list[list[list[float]]] = []
    n = 420
    dx = (x_max - x_min) / max(n - 1, 1)
    prior_y: float | None = None
    current: list[list[float]] = []
    finite_ys: list[float] = []
    for i in range(n):
        x = x_min + i * dx
        try:
            y = _safe_eval_function(expr, x)
            # Break the curve across vertical asymptotes / extreme jumps.
            if abs(y) > 1e5 or (prior_y is not None and abs(y - prior_y) > 80):
                raise ValueError
            current.append([round(x, 6), round(y, 6)])
            finite_ys.append(y)
            prior_y = y
        except Exception:
            if len(current) >= 2:
                segments.append(current)
            current = []
            prior_y = None
    if len(current) >= 2:
        segments.append(current)
    if not segments:
        raise ValueError("No plottable points were found in this x-range.")

    if finite_ys:
        finite_sorted = sorted(finite_ys)
        lo = finite_sorted[max(0, int(len(finite_sorted) * .03) - 1)]
        hi = finite_sorted[min(len(finite_sorted)-1, int(len(finite_sorted) * .97))]
        span = max(4.0, hi - lo)
        y_min, y_max = lo - .18 * span, hi + .18 * span
        y_min, y_max = max(-50.0, y_min), min(50.0, y_max)
        if y_max - y_min < 4:
            mid=(y_min+y_max)/2; y_min=mid-2; y_max=mid+2
    else:
        y_min, y_max = -10.0, 10.0

    polylines = [
        {"id": f"student_function_{idx}", "points": pts, "label": "", "dashed": False}
        for idx, pts in enumerate(segments)
    ]
    return {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "show_axes": True, "keep_aspect": False,
        "points": [], "segments": [], "polylines": polylines, "circles": [], "angles": [],
    }


def _render_function_graph_tool(pq: TargetedPracticeQuestion | None, *, key_base: str) -> None:
    if not _looks_like_function_or_graph_question(pq):
        return
    st.markdown("#### Function / graph tool")
    show_graph = st.toggle(
        "Show graph of a function",
        key=f"{key_base}_show_function_graph",
        help="Enter a function of x, then explore it with the same plotting and geometry tools.",
    )
    if not show_graph:
        return
    c1, c2, c3 = st.columns([2.2, .8, .8])
    with c1:
        expr = st.text_input(
            "Function",
            key=f"{key_base}_function_expr",
            placeholder="e.g. y = x^2 - 4x + 3",
            help="Use x, +, −, ×, ÷, powers (^), brackets, and sin/cos/tan/sqrt/log if needed.",
        )
    with c2:
        x_min = float(st.number_input("x min", value=-10.0, step=1.0, key=f"{key_base}_function_xmin"))
    with c3:
        x_max = float(st.number_input("x max", value=10.0, step=1.0, key=f"{key_base}_function_xmax"))
    if not expr.strip():
        st.caption("Enter the function you want to plot.")
        return
    if x_max <= x_min:
        st.warning("x max must be greater than x min.")
        return
    try:
        scene = _sample_function_scene(expr, x_min=x_min, x_max=x_max)
    except Exception as exc:
        st.warning(f"The function could not be plotted: {exc}")
        return
    if _practice_diagram_component is None:
        st.info("The interactive graph component is unavailable in this browser session.")
        return
    _practice_diagram_component(
        data={
            "scene": scene,
            "step": {"highlight_ids": [], "dim_ids": [], "animate_ids": []},
            "visible_ids": [], "animate_ids": [], "reveal_mode": False, "animation_nonce": 0,
        },
        default={},
        key=f"{key_base}_function_graph",
        width="stretch",
        height="content",
    )
    st.caption("Use Point, Line, Segment, Angle, Distance and the other graph tools to explore the function. This graph is a working aid and does not reveal the answer automatically.")


def targeted_practice_input(
    label: str,
    *,
    key_base: str,
    height: int = 150,
    practice_question: TargetedPracticeQuestion | None = None,
) -> tuple[str, str, str, list[UploadedAsset]]:
    """Collect targeted-practice working from equation editor, text, or iPad handwriting."""
    _render_function_graph_tool(practice_question, key_base=key_base)
    table_text = _student_table_tool(key_base=key_base)
    mode = st.radio(
        "Working input method",
        ["Equation editor", "Handwrite on iPad", "Text working"],
        horizontal=True,
        key=f"{key_base}_mode",
        help=(
            "Equation editor is best for typed mathematics. Handwrite on iPad lets a student write with Apple Pencil, "
            "stylus, or finger, take a camera photo, or upload an image/PDF."
        ),
    )

    if mode == "Equation editor":
        latex_lines, ascii_lines = equation_working_editor(label, key=f"{key_base}_equation")
        explanation = st.text_area(
            "Optional explanation in words",
            key=f"{key_base}_explanation",
            height=80,
            placeholder="Example: I used the gradient formula first.",
        )
        used_latex = [line for line in latex_lines if line]
        used_ascii = [line for line in ascii_lines if line]
        working_lines = [f"Step {i}: \\({line}\\)" for i, line in enumerate(used_latex, 1)]
        if explanation.strip():
            working_lines.append(f"Student explanation: {explanation.strip()}")
        main_text = "\n".join(working_lines)
        if table_text:
            main_text = (main_text + "\n\n" + table_text).strip()
        offline = "\n".join(used_ascii)
        if table_text:
            offline = (offline + "\n\n" + table_text).strip()
        return main_text, mode, offline, []

    if mode == "Text working":
        value = st.text_area(
            label,
            key=f"{key_base}_text",
            height=height,
            placeholder="Show all parts and important reasoning steps, not only the final answer.",
        )
        main_text = value.strip()
        if table_text:
            main_text = (main_text + "\n\n" + table_text).strip()
        return main_text, mode, main_text, []

    st.caption(
        "On iPad, write directly with Apple Pencil/finger. The pad no longer submits after every stroke. "
        "When the page is complete, tap **Save handwriting** once before checking the answer. "
        "You can also use the camera/upload controls. For multi-part questions, label (a), (b), (c)."
    )
    canvas_asset = handwriting_pad(key=f"{key_base}_handwriting")
    if canvas_asset is None:
        st.info("If you are using the handwriting pad, finish the page and tap **Save handwriting** before pressing the marking button.")
    camera_file = st.camera_input(
        "Take a photo of handwritten working",
        key=f"{key_base}_camera",
        help="Allow camera access in Safari/Chrome when prompted.",
    )
    upload_files = st.file_uploader(
        "Or upload handwritten page(s)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key=f"{key_base}_uploads",
        help="Useful for multiple pages or an existing photo/PDF from the iPad Files/Photos library.",
    )
    explanation = st.text_area(
        "Optional note to the tutor",
        key=f"{key_base}_hand_note",
        height=70,
        placeholder="Example: My working for (b) continues on the second page.",
    )

    browser_files: list[Any] = list(upload_files or [])
    if camera_file is not None:
        browser_files.insert(0, camera_file)
    try:
        assets = uploaded_assets(browser_files)
    except GeminiTutorError as exc:
        st.error(str(exc))
        assets = []
    if canvas_asset is not None:
        assets.insert(0, canvas_asset)
    total = sum(len(asset.data) for asset in assets)
    if total > MAX_TOTAL_BYTES:
        st.error("Handwritten working exceeds the app's 30 MB total upload limit.")
        assets = []

    text = explanation.strip()
    if table_text:
        text = (text + "\n\n" + table_text).strip()
    return text, "Handwritten working", table_text, assets


# Interactive visual explanations. The model supplies only declarative primitives;
# JavaScript rendering is owned by the app so uploaded/model content cannot execute code.
JSXGRAPH_VERSION = "2.4.0"
THREE_VERSION = "0.185.0"  # three.js r185

_VISUAL_2D_HTML = """
<div class="omt-visual2d-shell">
  <div class="omt-gg-toolbar" role="toolbar" aria-label="Interactive geometry tools">
    <button type="button" data-tool="move" class="active">Move</button>
    <button type="button" data-tool="point">Point</button>
    <button type="button" data-tool="line">Line</button>
    <button type="button" data-tool="segment">Segment</button>
    <button type="button" data-tool="ray">Ray</button>
    <button type="button" data-tool="vector">Vector</button>
    <button type="button" data-tool="circle">Circle</button>
    <button type="button" data-tool="polygon">Polygon</button>
    <button type="button" data-tool="finish" class="secondary">Finish</button>
    <button type="button" data-tool="midpoint">Midpoint</button>
    <button type="button" data-tool="perpendicular">Perpendicular</button>
    <button type="button" data-tool="parallel">Parallel</button>
    <button type="button" data-tool="angle">Measure angle</button>
    <button type="button" data-tool="distance">Distance</button>
    <button type="button" data-tool="delete">Delete</button>
    <button type="button" data-tool="undo" class="secondary">Undo</button>
    <button type="button" data-tool="clear" class="secondary">Clear</button>
    <button type="button" data-tool="snap" class="secondary active">Snap 0.5</button>
  </div>
  <div class="omt-gg-status">Use Move to pan/zoom, or select a construction tool.</div>
  <div class="omt-visual2d-board"></div>
  <div class="omt-visual-help">GeoGebra-style tools are for exploration. The tutor's construction remains separate from your added objects.</div>
</div>
"""

_VISUAL_2D_CSS = """
.omt-visual2d-shell { width: 100%; }
.omt-gg-toolbar { display:flex; gap:.38rem; overflow-x:auto; padding:.1rem 0 .48rem; scrollbar-width:thin; -webkit-overflow-scrolling:touch; }
.omt-gg-toolbar button { flex:0 0 auto; border:1px solid #cbd5e1; background:#fff; color:#334155; border-radius:.62rem; padding:.5rem .68rem; min-height:38px; font:650 .78rem/1 system-ui,sans-serif; cursor:pointer; }
.omt-gg-toolbar button.active { background:#eaf2ff; border-color:#60a5fa; color:#1d4ed8; }
.omt-gg-toolbar button.secondary { background:#f8fafc; }
.omt-gg-status { font-size:.78rem; color:#64748b; margin:0 0 .42rem; min-height:1.1rem; }
.omt-visual2d-board { width: 100%; height: min(62vw, 520px); min-height: 360px; border: 1px solid rgba(128,128,128,.28); border-radius: .75rem; overflow: hidden; background: #ffffff; touch-action:none; }
.omt-visual-help { margin-top: .35rem; font-size: .78rem; opacity: .68; }
@media (max-width: 640px) { .omt-visual2d-board { height: 420px; min-height: 340px; } .omt-gg-toolbar button { min-height:44px; padding:.62rem .76rem; } }
"""

_VISUAL_2D_JS = r"""
const JXG_URL = 'https://cdn.jsdelivr.net/npm/jsxgraph@1.12.2/distrib/jsxgraphcore.mjs';

async function loadJXG() {
  if (!globalThis.__omtJXGPromise) globalThis.__omtJXGPromise = import(JXG_URL);
  const mod = await globalThis.__omtJXGPromise;
  return mod.default || mod.JXG || mod;
}

function installGeoTools(board, toolbar, status, JXG) {
  if (!toolbar || !status) return () => {};
  let tool='move', picks=[], polygonPts=[], snap=true;
  const groups=[];
  const studentObjects=new Set();
  const pointStyle={name:'',fixed:false,size:4,strokeColor:'#dc2626',fillColor:'#fff1f2',strokeWidth:2,highlight:true};
  const lineStyle={fixed:false,strokeColor:'#dc2626',strokeWidth:2.6,highlight:true};
  const addGroup=(objs)=>{ const arr=objs.filter(Boolean); arr.forEach(o=>studentObjects.add(o)); groups.push(arr); board.update(); };
  const removeObjects=(objs)=>{ for(const o of [...objs].reverse()){ try{studentObjects.delete(o);board.removeObject(o);}catch(_){ } } board.update(); };
  const undo=()=>{ if(polygonPts.length){removeObjects([polygonPts.pop()]); return;} const g=groups.pop(); if(g) removeObjects(g); };
  const clear=()=>{ polygonPts=[]; while(groups.length) removeObjects(groups.pop()); };
  const roundSnap=(v)=>snap?Math.round(v*2)/2:v;
  const coords=(ev)=>{ const c=new JXG.Coords(JXG.COORDS_BY_SCREEN,[ev.offsetX,ev.offsetY],board); return [roundSnap(c.usrCoords[1]),roundSnap(c.usrCoords[2])]; };
  const mkPoint=(xy)=>board.create('point',xy,{...pointStyle});
  const length=(a,b)=>Math.hypot(a.X()-b.X(),a.Y()-b.Y());
  const midpointXY=(a,b)=>[(a.X()+b.X())/2,(a.Y()+b.Y())/2];
  const statusMap={
    move:'Drag to pan. Pinch or use the wheel to zoom.', point:'Tap to plot a point. Drag it to adjust.', line:'Tap two positions to draw an infinite line.', segment:'Tap two positions to draw a segment.', ray:'Tap the endpoint, then a second point for the ray direction.', vector:'Tap the start and end points of the vector.', circle:'Tap the centre, then a point on the circle.', polygon:'Tap polygon vertices, then press Finish.', midpoint:'Tap two positions to construct their midpoint.', perpendicular:'Tap two points for a reference line, then tap the point the perpendicular passes through.', parallel:'Tap two points for a reference line, then tap the point the parallel passes through.', angle:'Tap arm point 1, then the vertex, then arm point 2.', distance:'Tap two positions to measure their distance.', delete:'Tap one of your red constructions to delete it.'
  };
  const setTool=(name)=>{
    if(name!=='polygon' && polygonPts.length){ status.textContent='Finish or Undo the current polygon before switching tools.'; return; }
    tool=name; picks=[];
    toolbar.querySelectorAll('button[data-tool]').forEach(b=>{ if(!['snap','finish','undo','clear'].includes(b.dataset.tool)) b.classList.toggle('active',b.dataset.tool===name); });
    status.textContent=statusMap[name]||'Select a construction tool.';
  };
  const finishPolygon=()=>{
    if(polygonPts.length<3){status.textContent='A polygon needs at least 3 vertices.';return;}
    const poly=board.create('polygon',polygonPts,{withLines:true,fillColor:'#fecaca',fillOpacity:.12,borders:{strokeColor:'#dc2626',strokeWidth:2.4},vertices:{visible:true}});
    addGroup([poly,...polygonPts]); polygonPts=[]; status.textContent='Polygon added. Choose another tool or start another polygon.';
  };
  const deleteUnder=(ev)=>{
    let hits=[]; try{hits=board.getAllUnderMouse(ev)||[];}catch(_){ }
    const hit=hits.find(o=>studentObjects.has(o));
    if(hit){ const idx=groups.findIndex(g=>g.includes(hit)); if(idx>=0){const [g]=groups.splice(idx,1);removeObjects(g);} else removeObjects([hit]); status.textContent='Construction deleted.'; }
    else { status.textContent='Tap a red construction to delete it. If selection is difficult, use Undo.'; }
  };
  const addAngleMeasure=(a,v,b)=>{
    const ang=board.create('angle',[a,v,b],{...lineStyle,radius:.7,fillColor:'#fee2e2',fillOpacity:.18,name:'',withLabel:false});
    const txt=board.create('text',[()=>v.X()+0.55,()=>v.Y()+0.55,()=>`${(ang.Value()*180/Math.PI).toFixed(1)}°`],{fixed:true,fontSize:13,color:'#b91c1c'});
    addGroup([a,v,b,ang,txt]);
  };
  const addDistance=(a,b)=>{
    const seg=board.create('segment',[a,b],{...lineStyle,dash:2});
    const txt=board.create('text',[()=> (a.X()+b.X())/2,()=> (a.Y()+b.Y())/2,()=> length(a,b).toFixed(2)],{fixed:true,fontSize:13,color:'#b91c1c'});
    addGroup([a,b,seg,txt]);
  };
  const handlePointTool=(xy)=>{ const p=mkPoint(xy); addGroup([p]); };
  const handleMulti=(xy)=>{
    const p=mkPoint(xy); picks.push(p);
    const need=(['perpendicular','parallel','angle'].includes(tool)?3:2);
    status.textContent=`${picks.length}/${need} point${need===1?'':'s'} selected.`;
    if(picks.length<need) return;
    const [a,b,c]=picks; picks=[];
    if(tool==='line'){const o=board.create('line',[a,b],{...lineStyle,straightFirst:true,straightLast:true});addGroup([a,b,o]);}
    else if(tool==='segment'){const o=board.create('segment',[a,b],lineStyle);addGroup([a,b,o]);}
    else if(tool==='ray'){const o=board.create('line',[a,b],{...lineStyle,straightFirst:false,straightLast:true});addGroup([a,b,o]);}
    else if(tool==='vector'){const o=board.create('arrow',[a,b],lineStyle);addGroup([a,b,o]);}
    else if(tool==='circle'){const o=board.create('circle',[a,b],{...lineStyle,fillOpacity:0});addGroup([a,b,o]);}
    else if(tool==='midpoint'){const m=board.create('midpoint',[a,b],{...pointStyle,fillColor:'#fef3c7',strokeColor:'#d97706'});addGroup([a,b,m]);}
    else if(tool==='distance') addDistance(a,b);
    else if(tool==='angle') addAngleMeasure(a,b,c);
    else if(tool==='perpendicular'){
      const base=board.create('line',[a,b],{...lineStyle,strokeColor:'#94a3b8',strokeWidth:1.6,dash:2});
      const perp=board.create('perpendicular',[base,c],{...lineStyle}); addGroup([a,b,c,base,perp]);
    }
    else if(tool==='parallel'){
      const base=board.create('line',[a,b],{...lineStyle,strokeColor:'#94a3b8',strokeWidth:1.6,dash:2});
      const para=board.create('parallel',[base,c],{...lineStyle}); addGroup([a,b,c,base,para]);
    }
    status.textContent=(statusMap[tool]||'Construction added.')+' Construction added.';
  };
  const clickHandler=(ev)=>{
    const b=ev.target.closest('button[data-tool]'); if(!b)return;
    const name=b.dataset.tool;
    if(name==='clear'){clear();status.textContent='Your constructions were cleared.';return;}
    if(name==='undo'){undo();status.textContent='Last construction removed.';return;}
    if(name==='finish'){finishPolygon();return;}
    if(name==='snap'){snap=!snap;b.classList.toggle('active',snap);b.textContent=snap?'Snap 0.5':'Snap off';status.textContent=snap?'Coordinate snapping is on (0.5 units).':'Coordinate snapping is off.';return;}
    setTool(name);
  };
  toolbar.addEventListener('click',clickHandler);
  const downHandler=(ev)=>{
    if(tool==='move')return;
    if(tool==='delete'){deleteUnder(ev);return;}
    const xy=coords(ev);
    if(tool==='point'){handlePointTool(xy);return;}
    if(tool==='polygon'){const p=mkPoint(xy);polygonPts.push(p);status.textContent=`Polygon: ${polygonPts.length} vertices. Add more or press Finish.`;board.update();return;}
    handleMulti(xy);
    board.update();
  };
  board.on('down',downHandler);
  setTool('move');
  return ()=>{ try{toolbar.removeEventListener('click',clickHandler);}catch(_){ } };
}

function styleFor(id, highlight, dim, kind='line') {
  const hi = highlight.has(id);
  const low = dim.has(id);
  if (kind === 'point') {
    return { strokeColor: hi ? '#dc2626' : (low ? '#cbd5e1' : '#0f172a'), fillColor: hi ? '#dc2626' : (low ? '#e2e8f0' : '#0f172a'), opacity: low ? 0.35 : 1, size: hi ? 5 : 3.5 };
  }
  return { strokeColor: hi ? '#dc2626' : (low ? '#cbd5e1' : '#475569'), strokeWidth: hi ? 4 : 2.2, opacity: low ? 0.28 : 0.95 };
}

function pulsePoint(board, point, targetSize, targetOpacity) {
  const start = performance.now();
  const duration = 620;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const ease = 1 - Math.pow(1 - t, 3);
    const size = 0.7 + (targetSize * 1.25 - 0.7) * ease;
    point.setAttribute({ size, opacity: Math.max(0.05, targetOpacity * ease) });
    board.update();
    if (t < 1) requestAnimationFrame(tick);
    else { point.setAttribute({ size: targetSize, opacity: targetOpacity }); board.update(); }
  };
  requestAnimationFrame(tick);
}

export default async function(component) {
  const { parentElement, data } = component;
  const stage = parentElement.querySelector('.omt-visual2d-board');
  const toolbar = parentElement.querySelector('.omt-gg-toolbar');
  const status = parentElement.querySelector('.omt-gg-status');
  const scene = data?.scene || {};
  const step = data?.step || {};
  const highlight = new Set(step.highlight_ids || []);
  const dim = new Set(step.dim_ids || []);
  const animate = new Set([...(step.highlight_ids || []), ...(data?.animate_ids || step.animate_ids || [])]);
  const revealMode = Boolean(data?.reveal_mode);
  const visible = new Set(data?.visible_ids || []);
  const isVisible = (id) => !revealMode || visible.has(id) || highlight.has(id) || animate.has(id);
  // Replay/Next must always produce visible motion. If this step contains no
  // explicit construction action, replay the currently visible construction.
  if (animate.size === 0 && Number(data?.animation_nonce || 0) > 0) {
    for (const group of [scene.points || [], scene.segments || [], scene.polylines || [], scene.circles || [], scene.angles || []]) {
      for (const item of group) if (isVisible(item.id)) animate.add(item.id);
    }
  }
  let JXG;
  try { JXG = await loadJXG(); } catch (err) { console.error('JSXGraph load failed', err); stage.textContent = 'Interactive 2D visual could not load. Reload once; if this persists, the JSXGraph library may be blocked by the network.'; return; }

  try { if (parentElement.__omtBoard) JXG.JSXGraph.freeBoard(parentElement.__omtBoard); } catch (_) {}
  stage.replaceChildren();
  stage.id = `omt-jxg-${Math.random().toString(36).slice(2)}`;
  const xMin = Number(scene.x_min ?? -5), xMax = Number(scene.x_max ?? 5);
  const yMin = Number(scene.y_min ?? -5), yMax = Number(scene.y_max ?? 5);
  const board = JXG.JSXGraph.initBoard(stage.id, {
    boundingbox: [xMin, yMax, xMax, yMin],
    axis: Boolean(scene.show_axes),
    keepaspectratio: scene.keep_aspect !== false,
    showNavigation: false,
    showCopyright: false,
    pan: { enabled: true, needShift: false },
    zoom: { wheel: true, needShift: false, factorX: 1.2, factorY: 1.2 },
  });
  parentElement.__omtBoard = board;

  // If a visible construction depends on endpoints/angle arms, show those points too.
  const neededPoints = new Set();
  for (const seg of (scene.segments || [])) if (isVisible(seg.id)) { neededPoints.add(seg.start); neededPoints.add(seg.end); }
  for (const ang of (scene.angles || [])) if (isVisible(ang.id)) { neededPoints.add(ang.arm1); neededPoints.add(ang.vertex); neededPoints.add(ang.arm2); }

  const pts = new Map();
  for (const p of (scene.points || [])) {
    const st = styleFor(p.id, highlight, dim, 'point');
    const show = isVisible(p.id) || neededPoints.has(p.id);
    const obj = board.create('point', [Number(p.x), Number(p.y)], {
      name: p.label || '', fixed: true, highlight: false, withLabel: show && Boolean(p.label), visible: show,
      strokeColor: st.strokeColor, fillColor: st.fillColor, opacity: animate.has(p.id) ? 0.04 : st.opacity,
      size: animate.has(p.id) ? 0.7 : st.size,
      label: { fontSize: 14, offset: [7, 7] },
    });
    pts.set(p.id, obj);
    if (show && animate.has(p.id)) pulsePoint(board, obj, st.size, st.opacity);
  }

  for (const seg of (scene.segments || [])) {
    if (!isVisible(seg.id)) continue;
    const a = pts.get(seg.start), b = pts.get(seg.end); if (!a || !b) continue;
    const st = styleFor(seg.id, highlight, dim);
    const attrs = {
      name: seg.label || '', withLabel: Boolean(seg.label) && !animate.has(seg.id), fixed: true, highlight: false,
      strokeColor: st.strokeColor, strokeWidth: st.strokeWidth, opacity: st.opacity,
      dash: seg.dashed ? 2 : 0, label: { fontSize: 13 },
    };
    if (animate.has(seg.id)) {
      const mover = board.create('point', [a.X(), a.Y()], { visible:false, fixed:true, name:'' });
      board.create('segment', [a, mover], attrs);
      setTimeout(() => mover.moveTo([b.X(), b.Y()], 900), 90);
      if (seg.label) setTimeout(() => board.create('text', [(a.X()+b.X())/2, (a.Y()+b.Y())/2, seg.label], {fixed:true, fontSize:13, color:st.strokeColor}), 980);
    } else {
      board.create('segment', [a,b], attrs);
    }
  }

  for (const poly of (scene.polylines || [])) {
    if (!isVisible(poly.id)) continue;
    const samples = Array.isArray(poly.points) ? poly.points.filter(v => Array.isArray(v) && v.length >= 2) : [];
    if (samples.length < 2) continue;
    const st = styleFor(poly.id, highlight, dim);
    if (animate.has(poly.id)) {
      const delay = Math.max(18, Math.min(90, 850 / Math.max(1, samples.length - 1)));
      samples.slice(0, -1).forEach((a, i) => {
        const b = samples[i + 1];
        setTimeout(() => board.create('segment', [[Number(a[0]),Number(a[1])],[Number(b[0]),Number(b[1])]], {
          fixed:true, highlight:false, strokeColor:st.strokeColor, strokeWidth:st.strokeWidth, opacity:st.opacity, dash:poly.dashed ? 2 : 0,
        }), 80 + i * delay);
      });
      if (poly.label) {
        const m = samples[Math.floor(samples.length/2)];
        setTimeout(() => board.create('text', [Number(m[0]), Number(m[1]), poly.label], { fixed:true, fontSize:13, color: st.strokeColor }), 120 + samples.length * delay);
      }
    } else {
      const xs = samples.map(v => Number(v[0])), ys = samples.map(v => Number(v[1]));
      board.create('curve', [xs, ys], { fixed:true, highlight:false, strokeColor:st.strokeColor, strokeWidth:st.strokeWidth, opacity:st.opacity, dash:poly.dashed ? 2 : 0 });
      if (poly.label) {
        const m = samples[Math.floor(samples.length/2)];
        board.create('text', [Number(m[0]), Number(m[1]), poly.label], { fixed:true, fontSize:13, color: st.strokeColor });
      }
    }
  }

  for (const c of (scene.circles || [])) {
    if (!isVisible(c.id)) continue;
    const st = styleFor(c.id, highlight, dim);
    board.create('circle', [[Number(c.center_x), Number(c.center_y)], Number(c.radius)], {
      fixed:true, highlight:false, strokeColor:st.strokeColor, strokeWidth:st.strokeWidth, opacity:animate.has(c.id) ? 0.25 : st.opacity,
      fillOpacity:0, name:c.label || '', withLabel:Boolean(c.label),
    });
  }

  for (const ang of (scene.angles || [])) {
    if (!isVisible(ang.id)) continue;
    const a = pts.get(ang.arm1), v = pts.get(ang.vertex), c = pts.get(ang.arm2); if (!a || !v || !c) continue;
    const st = styleFor(ang.id, highlight, dim);
    board.create('angle', [a,v,c], {
      name:ang.label || '', withLabel:Boolean(ang.label), fixed:true, highlight:false,
      strokeColor:st.strokeColor, fillColor:st.strokeColor, fillOpacity:highlight.has(ang.id) ? 0.16 : 0.06,
      strokeWidth:st.strokeWidth, radius:0.55, label:{fontSize:13},
    });
  }
  const removeGeoTools = installGeoTools(board, toolbar, status, JXG);
  parentElement.__omtGeoToolsCleanup = removeGeoTools;
  board.update();
}
"""

_VISUAL_3D_HTML = """
<div class="omt-visual3d-shell">
  <div class="omt-visual3d-toolbar" role="toolbar" aria-label="3D view controls">
    <button type="button" data-action="rotate" class="is-active">Rotate</button>
    <button type="button" data-action="pan">Pan</button>
    <span class="omt-visual3d-divider"></span>
    <button type="button" data-action="home">Fit</button>
    <button type="button" data-action="source">Question view</button>
    <button type="button" data-action="iso">Explore iso</button>
    <button type="button" data-action="front">Front</button>
    <button type="button" data-action="top">Top</button>
    <button type="button" data-action="side">Side</button>
  </div>
  <div class="omt-visual3d-stage"></div>
  <div class="omt-visual-help">Rotate mode explores the solid; Pan moves it. For top/front/side questions, use Front, Top and Side to compare the reconstruction with the source views.</div>
</div>
"""

_VISUAL_3D_CSS = """
.omt-visual3d-shell { width: 100%; }
.omt-visual3d-toolbar { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:0 0 .45rem 0; }
.omt-visual3d-toolbar button { appearance:none; border:1px solid rgba(100,116,139,.35); background:#fff; color:#334155; border-radius:.55rem; padding:.46rem .72rem; font:600 .78rem/1 system-ui,sans-serif; cursor:pointer; min-height:36px; }
.omt-visual3d-toolbar button:hover { background:#f1f5f9; }
.omt-visual3d-toolbar button.is-active { background:#e0f2fe; border-color:#38bdf8; color:#075985; }
.omt-visual3d-divider { width:1px; height:26px; background:rgba(100,116,139,.25); margin:0 .1rem; }
.omt-visual3d-stage { width: 100%; height: min(66vw, 590px); min-height: 430px; border: 1px solid rgba(100,116,139,.32); border-radius: .75rem; overflow: hidden; background:#f8fafc; touch-action: none; position: relative; }
.omt-visual3d-stage canvas { display:block; width:100%; height:100%; }
.omt-visual-help { margin-top:.38rem; font-size:.78rem; color:#64748b; }
@media (max-width: 640px) {
  .omt-visual3d-stage { height: 520px; min-height: 430px; }
  .omt-visual3d-toolbar button { min-height:42px; padding:.55rem .78rem; font-size:.82rem; }
}
"""

_VISUAL_3D_JS = r"""
const THREE_URL = 'https://esm.sh/three@0.185.0';
const ORBIT_URL = 'https://esm.sh/three@0.185.0/examples/jsm/controls/OrbitControls.js';

async function loadThree() {
  if (!globalThis.__omtThreePromise) globalThis.__omtThreePromise = Promise.all([import(THREE_URL), import(ORBIT_URL)]);
  const [THREE, controls] = await globalThis.__omtThreePromise;
  return { THREE, OrbitControls: controls.OrbitControls };
}

function textSprite(THREE, text, color='#0f172a', scale=1.0) {
  if (!text) return null;
  const canvas=document.createElement('canvas'); canvas.width=640; canvas.height=160;
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,640,160);
  ctx.font='600 48px system-ui, sans-serif';
  const w=Math.min(600, Math.max(120, ctx.measureText(text).width+42));
  ctx.fillStyle='rgba(255,255,255,.88)'; ctx.beginPath(); ctx.roundRect((640-w)/2,34,w,92,20); ctx.fill();
  ctx.strokeStyle='rgba(100,116,139,.25)'; ctx.stroke();
  ctx.fillStyle=color; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(text,320,80);
  const texture=new THREE.CanvasTexture(canvas); texture.minFilter=THREE.LinearFilter;
  const material=new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false});
  const sprite=new THREE.Sprite(material); sprite.scale.set(2.1*scale,.52*scale,1); sprite.userData.__texture=texture; return sprite;
}

function edgeMaterial(THREE,id,highlight,dim,dashed=false){
  const hi=highlight.has(id), low=dim.has(id); const color=hi?0xdc2626:(low?0x94a3b8:0x334155); const opacity=low?0.28:1;
  if(dashed) return new THREE.LineDashedMaterial({color,transparent:true,opacity,dashSize:.18,gapSize:.1});
  return new THREE.LineBasicMaterial({color,transparent:true,opacity});
}

function hashColor(id){
  const palette=[0x60a5fa,0x34d399,0xf59e0b,0xa78bfa,0x22d3ee,0xfb7185,0x84cc16];
  let h=0; for(const ch of String(id||'')) h=(h*31+ch.charCodeAt(0))>>>0;
  return palette[h%palette.length];
}

function solidMaterial(THREE,id,highlight,dim){
  const hi=highlight.has(id), low=dim.has(id);
  return new THREE.MeshStandardMaterial({
    color: hi?0xf97316:hashColor(id),
    roughness:.72, metalness:.02,
    transparent: low,
    opacity: low?.42:1,
    side: THREE.DoubleSide,
    depthWrite: !low,
  });
}

function orientAxis(mesh,axis){
  if(axis==='x') mesh.rotation.z=-Math.PI/2;
  else if(axis==='z') mesh.rotation.x=Math.PI/2;
}

function addSolidEdges(THREE,mesh,id,highlight,dim,scene){
  const geom=new THREE.EdgesGeometry(mesh.geometry,20);
  const mat=new THREE.LineBasicMaterial({color:highlight.has(id)?0xc2410c:(dim.has(id)?0x94a3b8:0x334155),transparent:true,opacity:dim.has(id)?.42:.96});
  const lines=new THREE.LineSegments(geom,mat); lines.position.copy(mesh.position); lines.rotation.copy(mesh.rotation); lines.scale.copy(mesh.scale); scene.add(lines);
  return lines;
}

function addSolidLabel(THREE,mesh,label,id,highlight,scene){
  if(!label) return;
  const box=new THREE.Box3().setFromObject(mesh); const top=box.getCenter(new THREE.Vector3()); top.y=box.max.y+.22;
  const sp=textSprite(THREE,label,highlight.has(id)?'#c2410c':'#334155',.78); if(sp){sp.position.copy(top);scene.add(sp);}
}

export default async function(component) {
  const {parentElement,data}=component;
  const stage=parentElement.querySelector('.omt-visual3d-stage');
  const toolbar=parentElement.querySelector('.omt-visual3d-toolbar');
  const sceneData=data?.scene||{}, step=data?.step||{};
  const sourceView=sceneData.source_view||null;
  const sourceProjection=String(sourceView?.projection||'unknown').toLowerCase();
  const isOrthographicSet=sourceProjection==='orthographic_set';
  const useOrthographic=['isometric','orthographic','orthographic_set','oblique'].includes(sourceProjection);
  const sourceButton=toolbar?.querySelector('button[data-action="source"]');
  const help=parentElement.querySelector('.omt-visual-help');
  if(isOrthographicSet){
    if(sourceButton) sourceButton.textContent='Reconstruction view';
    if(help) help.textContent='This question provides top/front/side orthographic views, not a single isometric source view. Use Front, Top and Side to compare the reconstructed solid with the question; Rotate/Pan are for exploration.';
  }
  const highlight=new Set(step.highlight_ids||[]), dim=new Set(step.dim_ids||[]);
  const explicitAnimate=new Set(data?.animate_ids||step.animate_ids||[]);
  const animate=new Set([...(step.highlight_ids||[]), ...explicitAnimate]);
  const newlyRevealed=new Set(step.reveal_ids||[]);
  const revealMode=Boolean(data?.reveal_mode), visible=new Set(data?.visible_ids||[]);
  const solidGroups=[sceneData.boxes||[],sceneData.cylinders||[],sceneData.cones||[],sceneData.spheres||[],sceneData.extrusions||[]];
  const solidIds=new Set(solidGroups.flat().map(x=>x.id));
  // Physical solids remain visible throughout. revealMode is for construction geometry,
  // not for hiding the object the student is trying to understand.
  const isVisible=(id)=>solidIds.has(id)||!revealMode||visible.has(id)||highlight.has(id)||animate.has(id);
  if(animate.size===0 && Number(data?.animation_nonce||0)>0){
    for(const group of [sceneData.vertices||[],sceneData.edges||[],sceneData.faces||[],sceneData.angles||[],...solidGroups]){
      for(const item of group) if(isVisible(item.id)) animate.add(item.id);
    }
  }
  if(parentElement.__omtThreeCleanup) parentElement.__omtThreeCleanup();
  stage.replaceChildren();

  let THREE,OrbitControls;
  try{({THREE,OrbitControls}=await loadThree());}catch(err){stage.textContent='Interactive 3D visual could not load.';return;}

  const scene=new THREE.Scene();
  // Printed isometric/orthographic exam diagrams use parallel projection. Using a
  // perspective camera changes edge directions and makes a correct solid look unlike
  // the source. Source-calibrated views therefore use OrthographicCamera by default.
  const camera=useOrthographic
    ? new THREE.OrthographicCamera(-5,5,5,-5,.01,2000)
    : new THREE.PerspectiveCamera(38,1,.01,2000);
  if(Array.isArray(sourceView?.camera_up)&&sourceView.camera_up.length===3){
    camera.up.set(Number(sourceView.camera_up[0]),Number(sourceView.camera_up[1]),Number(sourceView.camera_up[2])).normalize();
  }
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setClearColor(0xf8fafc,1);
  renderer.outputColorSpace=THREE.SRGBColorSpace; stage.appendChild(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=.075; controls.enablePan=true; controls.enableRotate=true;
  controls.screenSpacePanning=true; controls.rotateSpeed=.62; controls.panSpeed=.72; controls.zoomSpeed=.82;
  controls.mouseButtons={LEFT:THREE.MOUSE.ROTATE,MIDDLE:THREE.MOUSE.DOLLY,RIGHT:THREE.MOUSE.PAN};
  controls.touches={ONE:THREE.TOUCH.ROTATE,TWO:THREE.TOUCH.DOLLY_PAN};
  scene.add(new THREE.HemisphereLight(0xffffff,0x94a3b8,2.6));
  const keyLight=new THREE.DirectionalLight(0xffffff,2.2); keyLight.position.set(7,10,8); scene.add(keyLight);
  const fillLight=new THREE.DirectionalLight(0xffffff,1.0); fillLight.position.set(-7,4,-6); scene.add(fillLight);

  const vertices=new Map(), vertexCoords=[], solidCoords=[];
  for(const v of (sceneData.vertices||[])){const pos=new THREE.Vector3(Number(v.x),Number(v.y),Number(v.z));vertexCoords.push(pos);vertices.set(v.id,pos);}
  const neededVertices=new Set();
  for(const e of (sceneData.edges||[])) if(isVisible(e.id)){neededVertices.add(e.start);neededVertices.add(e.end);}
  for(const f of (sceneData.faces||[])) if(isVisible(f.id)) for(const id of (f.vertices||[])) neededVertices.add(id);
  for(const a of (sceneData.angles||[])) if(isVisible(a.id)){neededVertices.add(a.arm1);neededVertices.add(a.vertex);neededVertices.add(a.arm2);}

  const solidTweens=[];
  const registerSolid=(mesh,id,label)=>{
    const baseScale=mesh.scale.clone(); const doAnimate=animate.has(id);
    if(doAnimate){const startScale=newlyRevealed.has(id)?.08:.86;mesh.scale.multiplyScalar(startScale); solidTweens.push({mesh,baseScale,startScale,start:performance.now()+60,duration:720});}
    scene.add(mesh); const edge=addSolidEdges(THREE,mesh,id,highlight,dim,scene); mesh.userData.__edge=edge;
    addSolidLabel(THREE,mesh,label,id,highlight,scene);
    const box=new THREE.Box3().setFromObject(mesh); if(Number.isFinite(box.min.x)){solidCoords.push(box.min.clone(),box.max.clone());}
  };

  for(const b of (sceneData.boxes||[])){
    if(!isVisible(b.id)) continue;
    const geom=new THREE.BoxGeometry(Number(b.width),Number(b.height),Number(b.depth));
    const mesh=new THREE.Mesh(geom,solidMaterial(THREE,b.id,highlight,dim));
    mesh.position.set(...(b.center||[0,0,0])); const r=b.rotation||[0,0,0]; mesh.rotation.set(Number(r[0]||0),Number(r[1]||0),Number(r[2]||0));
    registerSolid(mesh,b.id,b.label||'');
  }

  for(const c of (sceneData.cylinders||[])){
    if(!isVisible(c.id)) continue;
    const geom=new THREE.CylinderGeometry(Number(c.radius),Number(c.radius),Number(c.height),48,1,false);
    const mesh=new THREE.Mesh(geom,solidMaterial(THREE,c.id,highlight,dim)); mesh.position.set(...(c.center||[0,0,0])); orientAxis(mesh,c.axis||'y'); registerSolid(mesh,c.id,c.label||'');
  }

  for(const c of (sceneData.cones||[])){
    if(!isVisible(c.id)) continue;
    const geom=new THREE.ConeGeometry(Number(c.radius),Number(c.height),48,1,false);
    const mesh=new THREE.Mesh(geom,solidMaterial(THREE,c.id,highlight,dim)); mesh.position.set(...(c.center||[0,0,0])); orientAxis(mesh,c.axis||'y'); registerSolid(mesh,c.id,c.label||'');
  }

  for(const sp of (sceneData.spheres||[])){
    if(!isVisible(sp.id)) continue;
    const geom=new THREE.SphereGeometry(Number(sp.radius),40,24);
    const mesh=new THREE.Mesh(geom,solidMaterial(THREE,sp.id,highlight,dim)); mesh.position.set(...(sp.center||[0,0,0])); registerSolid(mesh,sp.id,sp.label||'');
  }

  for(const ex of (sceneData.extrusions||[])){
    if(!isVisible(ex.id)||!(ex.profile||[]).length) continue;
    const profile=ex.profile; const shape=new THREE.Shape(); shape.moveTo(Number(profile[0][0]),Number(profile[0][1]));
    for(let i=1;i<profile.length;i++) shape.lineTo(Number(profile[i][0]),Number(profile[i][1])); shape.closePath();
    const geom=new THREE.ExtrudeGeometry(shape,{depth:Number(ex.depth),bevelEnabled:false,steps:1}); geom.center();
    if((ex.axis||'z')==='x') geom.rotateY(Math.PI/2); else if((ex.axis||'z')==='y') geom.rotateX(-Math.PI/2);
    const mesh=new THREE.Mesh(geom,solidMaterial(THREE,ex.id,highlight,dim)); mesh.position.set(...(ex.center||[0,0,0])); registerSolid(mesh,ex.id,ex.label||'');
  }

  const focusVertices=new Set();
  for(const e of (sceneData.edges||[])) if(highlight.has(e.id)){focusVertices.add(e.start);focusVertices.add(e.end);}
  for(const a of (sceneData.angles||[])) if(highlight.has(a.id)){focusVertices.add(a.arm1);focusVertices.add(a.vertex);focusVertices.add(a.arm2);}
  for(const v of (sceneData.vertices||[])){
    if(!isVisible(v.id)&&!neededVertices.has(v.id)) continue;
    const pos=vertices.get(v.id), hi=highlight.has(v.id), low=dim.has(v.id);
    const geom=new THREE.SphereGeometry(hi?.075:.038,16,10); const mat=new THREE.MeshBasicMaterial({color:hi?0xdc2626:(low?0xcbd5e1:0x334155),transparent:true,opacity:low?.22:(hi?1:.48)});
    const mesh=new THREE.Mesh(geom,mat);mesh.position.copy(pos);scene.add(mesh);
    // Avoid the unreadable cloud of labels seen in complex composite solids.
    // Labels appear only when the current teaching step actually uses the vertex.
    if(hi||focusVertices.has(v.id)){
      const label=textSprite(THREE,v.label||v.id,hi?'#dc2626':'#334155',.64);if(label){label.position.copy(pos).add(new THREE.Vector3(.11,.15,.07));scene.add(label);}
    }
  }

  const edgeTweens=[];
  for(const e of (sceneData.edges||[])){
    if(!isVisible(e.id)) continue;
    const a=vertices.get(e.start),b=vertices.get(e.end);if(!a||!b)continue;
    const mat=edgeMaterial(THREE,e.id,highlight,dim,Boolean(e.dashed));
    if(animate.has(e.id)){
      const arr=new Float32Array([a.x,a.y,a.z,a.x,a.y,a.z]);
      const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.BufferAttribute(arr,3));
      const line=new THREE.Line(geom,mat);scene.add(line);edgeTweens.push({attr:geom.getAttribute('position'),a:a.clone(),b:b.clone(),start:performance.now()+80,duration:900,line});
    }else{
      const geom=new THREE.BufferGeometry().setFromPoints([a,b]);const line=new THREE.Line(geom,mat);if(e.dashed)line.computeLineDistances();scene.add(line);
    }
    if(e.label){const m=a.clone().add(b).multiplyScalar(.5);const sp=textSprite(THREE,e.label,highlight.has(e.id)?'#dc2626':'#334155',.66);if(sp){sp.position.copy(m).add(new THREE.Vector3(.08,.08,.08));scene.add(sp);}}
  }

  for(const f of (sceneData.faces||[])){
    if(!isVisible(f.id))continue;const vv=(f.vertices||[]).map(id=>vertices.get(id)).filter(Boolean);if(vv.length<3)continue;
    const arr=[];for(let i=1;i<vv.length-1;i++){for(const p of[vv[0],vv[i],vv[i+1]])arr.push(p.x,p.y,p.z);}const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(arr,3));geom.computeVertexNormals();
    const hi=highlight.has(f.id),low=dim.has(f.id);const mat=new THREE.MeshStandardMaterial({color:hi?0xf97316:0x94a3b8,transparent:true,opacity:low?.06:(hi?.32:.18),side:THREE.DoubleSide,depthWrite:false,roughness:.85});scene.add(new THREE.Mesh(geom,mat));
  }

  for(const aDef of (sceneData.angles||[])){
    if(!isVisible(aDef.id))continue;const pa=vertices.get(aDef.arm1),pv=vertices.get(aDef.vertex),pc=vertices.get(aDef.arm2);if(!pa||!pv||!pc)continue;
    const u=pa.clone().sub(pv).normalize(),w=pc.clone().sub(pv).normalize(),dot=THREE.MathUtils.clamp(u.dot(w),-1,1),theta=Math.acos(dot),sin=Math.sin(theta);if(theta<.02||Math.abs(sin)<1e-4)continue;
    const tangent=w.clone().sub(u.clone().multiplyScalar(dot)).normalize(),radius=.34,samples=[];for(let i=0;i<=28;i++){const t=theta*i/28;samples.push(pv.clone().add(u.clone().multiplyScalar(Math.cos(t)*radius)).add(tangent.clone().multiplyScalar(Math.sin(t)*radius)));}
    const geom=new THREE.BufferGeometry().setFromPoints(samples),mat=edgeMaterial(THREE,aDef.id,highlight,dim,false);scene.add(new THREE.Line(geom,mat));
    if(aDef.label){const mid=samples[Math.floor(samples.length/2)],sp=textSprite(THREE,aDef.label,highlight.has(aDef.id)?'#dc2626':'#334155',.66);if(sp){sp.position.copy(mid);scene.add(sp);}}
  }

  // Frame the PHYSICAL SOLID first. Scattered named vertices must never make the
  // camera zoom so far out that the object disappears.
  const fitCoords=solidCoords.length?solidCoords:vertexCoords;
  let center=new THREE.Vector3(0,0,0),radius=2,fitBox=null;
  if(fitCoords.length){fitBox=new THREE.Box3().setFromPoints(fitCoords);center=fitBox.getCenter(new THREE.Vector3());const sphere=fitBox.getBoundingSphere(new THREE.Sphere());radius=Math.max(sphere.radius,1.0);}
  const fovRad=THREE.MathUtils.degToRad(useOrthographic?38:camera.fov); const fitDistance=Math.max(radius/Math.sin(fovRad/2)*1.18,radius*2.25);
  camera.near=Math.max(.005,radius/1000); camera.far=Math.max(150,radius*80);
  let orthoHalfHeight=Math.max(radius*1.32,1.4);
  if(useOrthographic){
    controls.minZoom=.35; controls.maxZoom=5.5; camera.zoom=1;
  }else{
    controls.minDistance=Math.max(radius*.45,.25); controls.maxDistance=Math.max(radius*12,12);
  }
  camera.updateProjectionMatrix();
  const groundY=fitBox?fitBox.min.y-.05:center.y-radius*.55;
  const grid=new THREE.GridHelper(Math.max(radius*3.2,3),10,0xcbd5e1,0xe2e8f0); grid.position.set(center.x,groundY,center.z); grid.material.transparent=true; grid.material.opacity=.16; scene.add(grid);
  const sourceCp=Array.isArray(sourceView?.camera_position)&&sourceView.camera_position.length===3?sourceView.camera_position:null;
  const sourceCt=Array.isArray(sourceView?.camera_target)&&sourceView.camera_target.length===3?sourceView.camera_target:null;
  const cp=Array.isArray(step.camera_position)&&step.camera_position.length===3?step.camera_position:sourceCp;
  const ct=Array.isArray(step.camera_target)&&step.camera_target.length===3?step.camera_target:sourceCt;
  let targetLook=new THREE.Vector3(...(ct?ct:[center.x,center.y,center.z]));
  // Reject model-supplied camera targets/positions that are wildly outside the reconstructed solid.
  if(targetLook.distanceTo(center)>radius*3.5) targetLook=center.clone();
  let targetPos=cp?new THREE.Vector3(...cp):new THREE.Vector3(center.x+fitDistance*.72,center.y+fitDistance*.48,center.z+fitDistance*.72);
  const suppliedDistance=targetPos.distanceTo(targetLook);
  if(!Number.isFinite(suppliedDistance)||suppliedDistance<radius*.7||suppliedDistance>radius*7){targetPos.set(center.x+fitDistance*.72,center.y+fitDistance*.48,center.z+fitDistance*.72);}
  const pcp=Array.isArray(data?.previous_camera_position)&&data.previous_camera_position.length===3?new THREE.Vector3(...data.previous_camera_position):targetPos.clone();
  const pct=Array.isArray(data?.previous_camera_target)&&data.previous_camera_target.length===3?new THREE.Vector3(...data.previous_camera_target):targetLook.clone();
  camera.position.copy(pcp);controls.target.copy(pct);camera.lookAt(controls.target);controls.update();
  const camStart=performance.now(),camDuration=850;
  let cameraTweenActive=true;
  controls.addEventListener('start',()=>{cameraTweenActive=false;});

  const setInteractionMode=(mode)=>{
    const rotate=mode!=='pan';
    controls.mouseButtons.LEFT=rotate?THREE.MOUSE.ROTATE:THREE.MOUSE.PAN;
    controls.touches.ONE=rotate?THREE.TOUCH.ROTATE:THREE.TOUCH.PAN;
    for(const btn of toolbar?.querySelectorAll('button[data-action="rotate"],button[data-action="pan"]')||[]){
      btn.classList.toggle('is-active',btn.dataset.action===mode);
    }
  };
  const resetZoom=()=>{ if(useOrthographic){camera.zoom=1;camera.updateProjectionMatrix();} };
  const moveCamera=(position,look=center,up=null)=>{
    cameraTweenActive=false;
    if(up&&up.length===3) camera.up.set(Number(up[0]),Number(up[1]),Number(up[2])).normalize();
    camera.position.copy(position); controls.target.copy(look); camera.lookAt(look); resetZoom(); controls.update();
  };
  const sourceViewPosition=()=>{
    const p=Array.isArray(sourceView?.camera_position)&&sourceView.camera_position.length===3
      ? new THREE.Vector3(...sourceView.camera_position)
      : new THREE.Vector3(center.x+fitDistance*.72,center.y+fitDistance*.48,center.z+fitDistance*.72);
    const t=Array.isArray(sourceView?.camera_target)&&sourceView.camera_target.length===3
      ? new THREE.Vector3(...sourceView.camera_target):center.clone();
    return {p,t,up:Array.isArray(sourceView?.camera_up)?sourceView.camera_up:null};
  };
  const standardView=(name)=>{
    const d=fitDistance;
    if(name==='source'){
      if(isOrthographicSet) moveCamera(new THREE.Vector3(center.x+d*.72,center.y+d*.48,center.z+d*.72),center,[0,1,0]);
      else {const sv=sourceViewPosition();moveCamera(sv.p,sv.t,sv.up);}
    }
    else if(name==='front') moveCamera(new THREE.Vector3(center.x,center.y,center.z+d),center,[0,1,0]);
    else if(name==='top') moveCamera(new THREE.Vector3(center.x,center.y+d,center.z+.001),center,[0,0,-1]);
    else if(name==='side') moveCamera(new THREE.Vector3(center.x+d,center.y,center.z),center,[0,1,0]);
    else moveCamera(new THREE.Vector3(center.x+d*.72,center.y+d*.48,center.z+d*.72),center,[0,1,0]);
  };
  const toolbarHandler=(event)=>{
    const btn=event.target.closest('button[data-action]'); if(!btn)return;
    event.preventDefault();event.stopPropagation();const action=btn.dataset.action;
    if(action==='rotate'||action==='pan') setInteractionMode(action);
    else if(action==='home') standardView(sourceView?'source':'iso');
    else standardView(action);
  };
  toolbar?.addEventListener('click',toolbarHandler);
  setInteractionMode('rotate');

  const resize=()=>{
    const w=Math.max(stage.clientWidth,320),h=Math.max(stage.clientHeight,390);renderer.setSize(w,h,false);
    const aspect=w/h;
    if(useOrthographic){camera.left=-orthoHalfHeight*aspect;camera.right=orthoHalfHeight*aspect;camera.top=orthoHalfHeight;camera.bottom=-orthoHalfHeight;}
    else camera.aspect=aspect;
    camera.updateProjectionMatrix();
  };resize();const ro=new ResizeObserver(resize);ro.observe(stage);
  let raf=0;const animateFrame=(now)=>{raf=requestAnimationFrame(animateFrame);
    if(cameraTweenActive){
      const ctween=Math.min(1,(now-camStart)/camDuration);const ce=1-Math.pow(1-ctween,3);camera.position.lerpVectors(pcp,targetPos,ce);controls.target.lerpVectors(pct,targetLook,ce);
      if(ctween>=1) cameraTweenActive=false;
    }
    for(const tw of edgeTweens){const t=Math.max(0,Math.min(1,(now-tw.start)/tw.duration)),e=1-Math.pow(1-t,3),cur=tw.a.clone().lerp(tw.b,e);tw.attr.setXYZ(1,cur.x,cur.y,cur.z);tw.attr.needsUpdate=true;if(tw.line.material.isLineDashedMaterial)tw.line.computeLineDistances();}
    for(const tw of solidTweens){const t=Math.max(0,Math.min(1,(now-tw.start)/tw.duration)),e=1-Math.pow(1-t,3),k=tw.startScale+(1-tw.startScale)*e;tw.mesh.scale.copy(tw.baseScale).multiplyScalar(k); if(tw.mesh.userData.__edge) tw.mesh.userData.__edge.scale.copy(tw.mesh.scale);}
    controls.update();renderer.render(scene,camera);
  };raf=requestAnimationFrame(animateFrame);
  parentElement.__omtThreeCleanup=()=>{cancelAnimationFrame(raf);ro.disconnect();toolbar?.removeEventListener('click',toolbarHandler);controls.dispose();scene.traverse(o=>{o.geometry?.dispose?.();if(o.material){const mats=Array.isArray(o.material)?o.material:[o.material];for(const m of mats){m.map?.dispose?.();m.dispose?.();}}});renderer.dispose();renderer.domElement.remove();};
}

"""

try:
    _visual_2d_component = st.components.v2.component(
        "omt_visual_explanation_2d",
        html=_VISUAL_2D_HTML,
        css=_VISUAL_2D_CSS,
        js=_VISUAL_2D_JS,
        isolate_styles=False,
    )
except Exception:
    _visual_2d_component = None

# Interactive 2D workspace used inside targeted practice. Geometry questions get a
# clean schematic; graph/coordinate questions additionally get a GeoGebra-like
# student workspace with draggable points and segment construction tools. All
# interaction remains in the browser, so plotting does not rerun Streamlit.
_PRACTICE_DIAGRAM_HTML = """
<div class="omt-practice-workspace">
  <div class="omt-gg-toolbar" role="toolbar" aria-label="GeoGebra-style construction tools">
    <button type="button" data-tool="move" class="active">Move</button>
    <button type="button" data-tool="point">Point</button>
    <button type="button" data-tool="line">Line</button>
    <button type="button" data-tool="segment">Segment</button>
    <button type="button" data-tool="ray">Ray</button>
    <button type="button" data-tool="vector">Vector</button>
    <button type="button" data-tool="circle">Circle</button>
    <button type="button" data-tool="polygon">Polygon</button>
    <button type="button" data-tool="finish" class="secondary">Finish</button>
    <button type="button" data-tool="midpoint">Midpoint</button>
    <button type="button" data-tool="perpendicular">Perpendicular</button>
    <button type="button" data-tool="parallel">Parallel</button>
    <button type="button" data-tool="angle">Measure angle</button>
    <button type="button" data-tool="distance">Distance</button>
    <button type="button" data-tool="delete">Delete</button>
    <button type="button" data-tool="undo" class="secondary">Undo</button>
    <button type="button" data-tool="clear" class="secondary">Clear</button>
    <button type="button" data-tool="snap" class="secondary active">Snap 0.5</button>
  </div>
  <div class="omt-gg-status">Use Move to pan/zoom, or select a construction tool.</div>
  <div class="omt-visual2d-board"></div>
</div>
"""

_PRACTICE_DIAGRAM_CSS = """
.omt-practice-workspace { width:100%; }
.omt-gg-toolbar { display:flex; gap:.38rem; overflow-x:auto; padding:.1rem 0 .48rem; scrollbar-width:thin; -webkit-overflow-scrolling:touch; }
.omt-gg-toolbar button { flex:0 0 auto; border:1px solid #cbd5e1; background:#fff; color:#334155; border-radius:.62rem; padding:.5rem .68rem; min-height:38px; font:650 .78rem/1 system-ui,sans-serif; cursor:pointer; }
.omt-gg-toolbar button.active { background:#eaf2ff; border-color:#60a5fa; color:#1d4ed8; }
.omt-gg-toolbar button.secondary { background:#f8fafc; }
.omt-gg-status { font-size:.78rem; color:#64748b; margin:0 0 .42rem; min-height:1.1rem; }
.omt-practice-workspace .omt-visual2d-board { width:100%; height:390px; min-height:320px; border:1px solid rgba(128,128,128,.28); border-radius:.9rem; overflow:hidden; background:#fff; touch-action:none; }
@media (max-width:640px) { .omt-practice-workspace .omt-visual2d-board { height:360px; min-height:310px; } .omt-gg-toolbar button { min-height:44px; padding:.62rem .76rem; } }
"""

_PRACTICE_DIAGRAM_JS = r"""
const JXG_URL = 'https://cdn.jsdelivr.net/npm/jsxgraph@1.12.2/distrib/jsxgraphcore.mjs';
async function loadJXG(){
  if(!globalThis.__omtPracticeJXGPromise) globalThis.__omtPracticeJXGPromise=import(JXG_URL);
  const mod=await globalThis.__omtPracticeJXGPromise; return mod.default||mod.JXG||mod;
}

function installGeoTools(board, toolbar, status, JXG) {
  if (!toolbar || !status) return () => {};
  let tool='move', picks=[], polygonPts=[], snap=true;
  const groups=[];
  const studentObjects=new Set();
  const pointStyle={name:'',fixed:false,size:4,strokeColor:'#dc2626',fillColor:'#fff1f2',strokeWidth:2,highlight:true};
  const lineStyle={fixed:false,strokeColor:'#dc2626',strokeWidth:2.6,highlight:true};
  const addGroup=(objs)=>{ const arr=objs.filter(Boolean); arr.forEach(o=>studentObjects.add(o)); groups.push(arr); board.update(); };
  const removeObjects=(objs)=>{ for(const o of [...objs].reverse()){ try{studentObjects.delete(o);board.removeObject(o);}catch(_){ } } board.update(); };
  const undo=()=>{ if(polygonPts.length){removeObjects([polygonPts.pop()]); return;} const g=groups.pop(); if(g) removeObjects(g); };
  const clear=()=>{ polygonPts=[]; while(groups.length) removeObjects(groups.pop()); };
  const roundSnap=(v)=>snap?Math.round(v*2)/2:v;
  const coords=(ev)=>{ const c=new JXG.Coords(JXG.COORDS_BY_SCREEN,[ev.offsetX,ev.offsetY],board); return [roundSnap(c.usrCoords[1]),roundSnap(c.usrCoords[2])]; };
  const mkPoint=(xy)=>board.create('point',xy,{...pointStyle});
  const length=(a,b)=>Math.hypot(a.X()-b.X(),a.Y()-b.Y());
  const midpointXY=(a,b)=>[(a.X()+b.X())/2,(a.Y()+b.Y())/2];
  const statusMap={
    move:'Drag to pan. Pinch or use the wheel to zoom.', point:'Tap to plot a point. Drag it to adjust.', line:'Tap two positions to draw an infinite line.', segment:'Tap two positions to draw a segment.', ray:'Tap the endpoint, then a second point for the ray direction.', vector:'Tap the start and end points of the vector.', circle:'Tap the centre, then a point on the circle.', polygon:'Tap polygon vertices, then press Finish.', midpoint:'Tap two positions to construct their midpoint.', perpendicular:'Tap two points for a reference line, then tap the point the perpendicular passes through.', parallel:'Tap two points for a reference line, then tap the point the parallel passes through.', angle:'Tap arm point 1, then the vertex, then arm point 2.', distance:'Tap two positions to measure their distance.', delete:'Tap one of your red constructions to delete it.'
  };
  const setTool=(name)=>{
    if(name!=='polygon' && polygonPts.length){ status.textContent='Finish or Undo the current polygon before switching tools.'; return; }
    tool=name; picks=[];
    toolbar.querySelectorAll('button[data-tool]').forEach(b=>{ if(!['snap','finish','undo','clear'].includes(b.dataset.tool)) b.classList.toggle('active',b.dataset.tool===name); });
    status.textContent=statusMap[name]||'Select a construction tool.';
  };
  const finishPolygon=()=>{
    if(polygonPts.length<3){status.textContent='A polygon needs at least 3 vertices.';return;}
    const poly=board.create('polygon',polygonPts,{withLines:true,fillColor:'#fecaca',fillOpacity:.12,borders:{strokeColor:'#dc2626',strokeWidth:2.4},vertices:{visible:true}});
    addGroup([poly,...polygonPts]); polygonPts=[]; status.textContent='Polygon added. Choose another tool or start another polygon.';
  };
  const deleteUnder=(ev)=>{
    let hits=[]; try{hits=board.getAllUnderMouse(ev)||[];}catch(_){ }
    const hit=hits.find(o=>studentObjects.has(o));
    if(hit){ const idx=groups.findIndex(g=>g.includes(hit)); if(idx>=0){const [g]=groups.splice(idx,1);removeObjects(g);} else removeObjects([hit]); status.textContent='Construction deleted.'; }
    else { status.textContent='Tap a red construction to delete it. If selection is difficult, use Undo.'; }
  };
  const addAngleMeasure=(a,v,b)=>{
    const ang=board.create('angle',[a,v,b],{...lineStyle,radius:.7,fillColor:'#fee2e2',fillOpacity:.18,name:'',withLabel:false});
    const txt=board.create('text',[()=>v.X()+0.55,()=>v.Y()+0.55,()=>`${(ang.Value()*180/Math.PI).toFixed(1)}°`],{fixed:true,fontSize:13,color:'#b91c1c'});
    addGroup([a,v,b,ang,txt]);
  };
  const addDistance=(a,b)=>{
    const seg=board.create('segment',[a,b],{...lineStyle,dash:2});
    const txt=board.create('text',[()=> (a.X()+b.X())/2,()=> (a.Y()+b.Y())/2,()=> length(a,b).toFixed(2)],{fixed:true,fontSize:13,color:'#b91c1c'});
    addGroup([a,b,seg,txt]);
  };
  const handlePointTool=(xy)=>{ const p=mkPoint(xy); addGroup([p]); };
  const handleMulti=(xy)=>{
    const p=mkPoint(xy); picks.push(p);
    const need=(['perpendicular','parallel','angle'].includes(tool)?3:2);
    status.textContent=`${picks.length}/${need} point${need===1?'':'s'} selected.`;
    if(picks.length<need) return;
    const [a,b,c]=picks; picks=[];
    if(tool==='line'){const o=board.create('line',[a,b],{...lineStyle,straightFirst:true,straightLast:true});addGroup([a,b,o]);}
    else if(tool==='segment'){const o=board.create('segment',[a,b],lineStyle);addGroup([a,b,o]);}
    else if(tool==='ray'){const o=board.create('line',[a,b],{...lineStyle,straightFirst:false,straightLast:true});addGroup([a,b,o]);}
    else if(tool==='vector'){const o=board.create('arrow',[a,b],lineStyle);addGroup([a,b,o]);}
    else if(tool==='circle'){const o=board.create('circle',[a,b],{...lineStyle,fillOpacity:0});addGroup([a,b,o]);}
    else if(tool==='midpoint'){const m=board.create('midpoint',[a,b],{...pointStyle,fillColor:'#fef3c7',strokeColor:'#d97706'});addGroup([a,b,m]);}
    else if(tool==='distance') addDistance(a,b);
    else if(tool==='angle') addAngleMeasure(a,b,c);
    else if(tool==='perpendicular'){
      const base=board.create('line',[a,b],{...lineStyle,strokeColor:'#94a3b8',strokeWidth:1.6,dash:2});
      const perp=board.create('perpendicular',[base,c],{...lineStyle}); addGroup([a,b,c,base,perp]);
    }
    else if(tool==='parallel'){
      const base=board.create('line',[a,b],{...lineStyle,strokeColor:'#94a3b8',strokeWidth:1.6,dash:2});
      const para=board.create('parallel',[base,c],{...lineStyle}); addGroup([a,b,c,base,para]);
    }
    status.textContent=(statusMap[tool]||'Construction added.')+' Construction added.';
  };
  const clickHandler=(ev)=>{
    const b=ev.target.closest('button[data-tool]'); if(!b)return;
    const name=b.dataset.tool;
    if(name==='clear'){clear();status.textContent='Your constructions were cleared.';return;}
    if(name==='undo'){undo();status.textContent='Last construction removed.';return;}
    if(name==='finish'){finishPolygon();return;}
    if(name==='snap'){snap=!snap;b.classList.toggle('active',snap);b.textContent=snap?'Snap 0.5':'Snap off';status.textContent=snap?'Coordinate snapping is on (0.5 units).':'Coordinate snapping is off.';return;}
    setTool(name);
  };
  toolbar.addEventListener('click',clickHandler);
  const downHandler=(ev)=>{
    if(tool==='move')return;
    if(tool==='delete'){deleteUnder(ev);return;}
    const xy=coords(ev);
    if(tool==='point'){handlePointTool(xy);return;}
    if(tool==='polygon'){const p=mkPoint(xy);polygonPts.push(p);status.textContent=`Polygon: ${polygonPts.length} vertices. Add more or press Finish.`;board.update();return;}
    handleMulti(xy);
    board.update();
  };
  board.on('down',downHandler);
  setTool('move');
  return ()=>{ try{toolbar.removeEventListener('click',clickHandler);}catch(_){ } };
}

export default async function(component){
  const {parentElement,data}=component;
  const stage=parentElement.querySelector('.omt-visual2d-board');
  const toolbar=parentElement.querySelector('.omt-gg-toolbar');
  const status=parentElement.querySelector('.omt-gg-status');
  const scene=data?.scene||{};
  let JXG;
  try{JXG=await loadJXG();}catch(err){console.error(err);stage.textContent='Interactive maths workspace could not load.';return;}
  try{if(parentElement.__omtPracticeBoard)JXG.JSXGraph.freeBoard(parentElement.__omtPracticeBoard);}catch(_){ }
  stage.replaceChildren(); stage.id=`omt-practice-${Math.random().toString(36).slice(2)}`;
  const xMin=Number(scene.x_min??-5),xMax=Number(scene.x_max??5),yMin=Number(scene.y_min??-5),yMax=Number(scene.y_max??5);
  const graphMode=Boolean(scene.show_axes);
  const board=JXG.JSXGraph.initBoard(stage.id,{boundingbox:[xMin,yMax,xMax,yMin],axis:graphMode,keepaspectratio:scene.keep_aspect!==false,showNavigation:false,showCopyright:false,pan:{enabled:true,needShift:false},zoom:{wheel:true,needShift:false,factorX:1.18,factorY:1.18}});
  parentElement.__omtPracticeBoard=board;
  const pts=new Map();
  for(const p of(scene.points||[])){
    const obj=board.create('point',[Number(p.x),Number(p.y)],{name:p.label||'',fixed:true,highlight:false,size:3.8,strokeColor:'#0f172a',fillColor:'#0f172a',label:{fontSize:14,offset:[7,7]}});pts.set(p.id,obj);
  }
  for(const seg of(scene.segments||[])){
    const a=pts.get(seg.start),b=pts.get(seg.end);if(!a||!b)continue;
    board.create('segment',[a,b],{name:seg.label||'',withLabel:Boolean(seg.label),fixed:true,highlight:false,strokeColor:'#475569',strokeWidth:2.3,dash:seg.dashed?2:0,label:{fontSize:13}});
  }
  for(const poly of(scene.polylines||[])){
    const arr=Array.isArray(poly.points)?poly.points:[];if(arr.length<2)continue;
    const xs=arr.map(v=>Number(v[0])),ys=arr.map(v=>Number(v[1]));
    board.create('curve',[xs,ys],{fixed:true,highlight:false,strokeColor:'#475569',strokeWidth:2.3,dash:poly.dashed?2:0});
  }
  for(const c of(scene.circles||[])){
    const center=pts.get(c.center);if(center)board.create('circle',[center,Number(c.radius)],{fixed:true,highlight:false,strokeColor:'#475569',strokeWidth:2.2,fillOpacity:0});
    else if(Number.isFinite(Number(c.center_x))&&Number.isFinite(Number(c.center_y)))board.create('circle',[[Number(c.center_x),Number(c.center_y)],Number(c.radius)],{fixed:true,highlight:false,strokeColor:'#475569',strokeWidth:2.2,fillOpacity:0});
  }
  for(const a of(scene.angles||[])){
    const p1=pts.get(a.arm1),v=pts.get(a.vertex),p2=pts.get(a.arm2);if(!p1||!v||!p2)continue;
    board.create('angle',[p1,v,p2],{name:a.label||'',withLabel:Boolean(a.label),fixed:true,highlight:false,radius:Number(a.radius||.7),strokeColor:'#2563eb',fillColor:'#dbeafe',fillOpacity:.35,label:{fontSize:13}});
  }
  installGeoTools(board,toolbar,status,JXG);
  status.textContent=graphMode?'Plot points, draw lines, measure angles/distances, or pan/zoom the coordinate plane.':'Use the construction tools to explore the geometry. The given diagram remains fixed.';
  board.update();
}
"""

try:
    _practice_diagram_component = st.components.v2.component(
        "omt_targeted_practice_diagram",
        html=_PRACTICE_DIAGRAM_HTML,
        css=_PRACTICE_DIAGRAM_CSS,
        js=_PRACTICE_DIAGRAM_JS,
        isolate_styles=False,
    )
except Exception:
    _practice_diagram_component = None

try:
    _visual_3d_component = st.components.v2.component(
        "omt_visual_explanation_3d",
        html=_VISUAL_3D_HTML,
        css=_VISUAL_3D_CSS,
        js=_VISUAL_3D_JS,
        isolate_styles=False,
    )
except Exception:
    _visual_3d_component = None


def _visual_plan_is_recommended(analysis: VisualExplanationResult | GeminiAnalysis, question_text: str = "") -> bool:
    """Show simulations only when a diagram/graph/spatial view materially supports the maths."""
    if isinstance(analysis, VisualExplanationResult):
        return analysis.mode in {"geometry2d", "graph2d", "geometry3d"}

    topic = str(getattr(analysis, "likely_syllabus_topic", "") or "").lower()
    interpreted = str(getattr(analysis, "interpreted_question", "") or "").lower()
    raw_question = str(question_text or "").lower()
    haystack = f"{topic} {interpreted} {raw_question}"

    # Strong visual cues: these topics normally benefit from a diagram, graph, coordinate plane,
    # construction, or spatial model. Keep this list intentionally narrower than the old filter.
    visual_keywords = (
        "geometry", "coordinate geometry", "coordinate", "graph", "plot", "sketch",
        "straight-line graph", "straight line graph", "gradient of the line", "intercept",
        "triangle", "quadrilateral", "polygon", "circle", "angle", "bearing",
        "trigonometry", "trigonometric", "angle of elevation", "angle of depression",
        "similar triangles", "congruent", "transformation", "reflection", "rotation",
        "translation", "enlargement", "locus", "construction", "scale drawing",
        "mensuration", "perimeter", "area of",
        "cuboid", "prism", "pyramid", "cone", "cylinder", "sphere", "3d",
        "three-dimensional", "isometric", "orthographic", "top view", "front view", "side view",
        "diagram", "figure",
    )

    # Explicitly non-visual topics should not get a simulation merely because a generic word such
    # as "line" or "gradient" appears in explanatory prose.
    nonvisual_topic_keywords = (
        "standard form", "indices", "surds", "algebraic manipulation", "factorisation",
        "factorization", "equations and inequalities", "linear equation", "quadratic equation",
        "simultaneous equation", "number", "ratio", "percentage", "proportion", "sets",
        "probability", "statistics", "mean", "median", "mode", "arithmetic", "sequence",
    )

    has_visual_cue = any(token in haystack for token in visual_keywords)
    if not has_visual_cue:
        return False

    # If the syllabus topic is clearly non-visual, require an explicit visual cue in the actual
    # question itself (e.g. "plot the graph" or "in the diagram") before allowing a simulation.
    topic_is_nonvisual = any(token in topic for token in nonvisual_topic_keywords)
    question_has_explicit_visual_cue = any(token in raw_question or token in interpreted for token in (
        "diagram", "graph", "plot", "sketch", "coordinate", "triangle", "circle", "bearing",
        "elevation", "depression", "cuboid", "prism", "pyramid", "cone", "cylinder", "sphere",
        "isometric", "orthographic", "top view", "front view", "side view",
    ))
    if topic_is_nonvisual and not question_has_explicit_visual_cue:
        return False

    return True


def _render_source_3d_reference(plan: VisualExplanationResult, question_files: list[Any] | None) -> None:
    """Show the exact source isometric/orthographic drawing used to calibrate the 3D model."""
    if plan.mode != "geometry3d" or plan.scene_3d is None or not question_files:
        return
    source_view = getattr(plan.scene_3d, "source_view", None)
    if source_view is None:
        return
    source_index = int(getattr(source_view, "source_index", 1) or 1)
    page_number = int(getattr(source_view, "page_number", 1) or 1)
    if not (1 <= source_index <= len(question_files)):
        return
    image = _question_source_image(question_files[source_index - 1], page_number)
    if image is None:
        return
    box = list(getattr(source_view, "diagram_box_2d", []) or [])
    if len(box) == 4:
        px = _normalized_box_to_pixels(box, image.width, image.height)
        if px is not None:
            x1, y1, x2, y2 = px
            pad_x = max(8, int((x2 - x1) * 0.04))
            pad_y = max(8, int((y2 - y1) * 0.04))
            image = image.crop((max(0, x1-pad_x), max(0, y1-pad_y), min(image.width, x2+pad_x), min(image.height, y2+pad_y)))
    projection_raw = str(getattr(source_view, "projection", "unknown"))
    is_orthographic_set = projection_raw == "orthographic_set"
    title = "Compare reconstruction with the question's top/front/side views" if is_orthographic_set else "Compare with the question's original 3D/isometric view"
    caption = "Orthographic source views used to reconstruct the 3D object" if is_orthographic_set else "Source diagram used to calibrate the 3D model"
    with st.expander(title, expanded=True):
        st.image(image, caption=caption, use_container_width=True)
        projection = projection_raw.replace("_", " ").title()
        confidence = str(getattr(source_view, "match_confidence", "medium")).title()
        if is_orthographic_set:
            st.caption(f"Source type: {projection} · Projection-consistency confidence: {confidence}")
            st.info("This question does not provide a single isometric drawing. The 3D model is reconstructed by combining the top, front and side projections. Use the Top, Front and Side buttons in the 3D viewer to verify the reconstruction.")
        else:
            st.caption(f"Source-view projection: {projection} · Match confidence: {confidence}")
        checks = list(getattr(source_view, "view_consistency_checks", []) or [])
        if checks:
            st.markdown("**Projection checks**")
            for check in checks:
                st.markdown(f"- {check}")
        if is_orthographic_set:
            components = list(getattr(plan.scene_3d, "orthographic_components", []) or [])
            if components:
                st.markdown("**How the 3D form was inferred from the three views**")
                for item in sorted(components, key=lambda x: int(getattr(x, "vertical_order", 0))):
                    kind = str(getattr(item, "inferred_kind", "component")).replace("_", " ").title()
                    relation = str(getattr(item, "stacking_relation", "")).strip()
                    st.markdown(f"**{kind}**" + (f" — {relation}" if relation else ""))
                    cols = st.columns(3)
                    cols[0].caption("Top: " + str(getattr(item, "top_view_evidence", "")))
                    cols[1].caption("Front: " + str(getattr(item, "front_view_evidence", "")))
                    cols[2].caption("Side: " + str(getattr(item, "side_view_evidence", "")))
        note = str(getattr(source_view, "match_note", "")).strip()
        if note:
            st.caption(note)


def render_visual_explanation(plan: VisualExplanationResult, question_files: list[Any] | None = None) -> None:
    if plan.mode == "none":
        if plan.reconstruction_note:
            st.info("A reliable interactive reconstruction was not generated: " + plan.reconstruction_note)
        return
    if not plan.steps:
        return

    st.markdown("### Visual step-by-step simulation")
    st.caption(
        f"Reconstruction confidence: {plan.reconstruction_confidence.title()}. {plan.reconstruction_note}"
    )
    if plan.mode == "geometry3d" and getattr(plan, "reconstructed_parts", None):
        st.markdown("**3D form identified from the question:** " + " · ".join(plan.reconstructed_parts))
    if plan.mode == "geometry3d":
        _render_source_3d_reference(plan, question_files)
    max_index = len(plan.steps) - 1
    idx = max(0, min(int(st.session_state.get("ai_visual_step", 0)), max_index))
    st.session_state.ai_visual_step = idx

    def _go_previous() -> None:
        current = int(st.session_state.get("ai_visual_step", 0))
        st.session_state.ai_visual_step = max(0, current - 1)
        st.session_state.ai_visual_replay_nonce = int(st.session_state.get("ai_visual_replay_nonce", 0)) + 1

    def _go_next() -> None:
        current = int(st.session_state.get("ai_visual_step", 0))
        st.session_state.ai_visual_step = min(max_index, current + 1)
        st.session_state.ai_visual_replay_nonce = int(st.session_state.get("ai_visual_replay_nonce", 0)) + 1

    def _replay_current() -> None:
        # The nonce is sent through component data. Streamlit Components v2 calls
        # the frontend renderer again whenever data changes, so this reliably
        # restarts the current construction without changing the selected step.
        st.session_state.ai_visual_replay_nonce = int(st.session_state.get("ai_visual_replay_nonce", 0)) + 1

    b1, mid, b2, replay = st.columns([1, 1.7, 1, 1])
    b1.button(
        "← Previous",
        disabled=idx <= 0,
        use_container_width=True,
        key="ai_visual_prev",
        on_click=_go_previous,
    )
    mid.markdown(f"<div style='text-align:center;padding:.55rem'><strong>Step {idx + 1} of {len(plan.steps)}</strong></div>", unsafe_allow_html=True)
    b2.button(
        "Next →",
        disabled=idx >= max_index,
        use_container_width=True,
        key="ai_visual_next",
        on_click=_go_next,
    )
    replay.button(
        "↻ Replay",
        use_container_width=True,
        key="ai_visual_replay",
        on_click=_replay_current,
    )

    step = plan.steps[idx]
    # New plans progressively reveal the construction. Old saved plans without reveal/animate
    # fields continue to show the complete scene for backward compatibility.
    reveal_mode = any(bool(getattr(item, "reveal_ids", [])) or bool(getattr(item, "animate_ids", [])) for item in plan.steps)
    visible_ids: set[str] = set()
    if reveal_mode:
        for earlier in plan.steps[: idx + 1]:
            visible_ids.update(getattr(earlier, "reveal_ids", []) or [])
            visible_ids.update(getattr(earlier, "animate_ids", []) or [])
        visible_ids.update(getattr(step, "highlight_ids", []) or [])
    animate_ids = list(getattr(step, "animate_ids", []) or [])
    replay_nonce = int(st.session_state.get("ai_visual_replay_nonce", 0))

    previous_step = plan.steps[idx - 1] if idx > 0 else None
    previous_camera_position = list(getattr(previous_step, "camera_position", []) or []) if previous_step else []
    previous_camera_target = list(getattr(previous_step, "camera_target", []) or []) if previous_step else []

    scene_payload: dict[str, Any] | None = None
    component_data = {
        "step": step.model_dump(),
        "visible_ids": sorted(visible_ids),
        "animate_ids": animate_ids,
        "reveal_mode": reveal_mode,
        "animation_nonce": replay_nonce,
        "previous_camera_position": previous_camera_position,
        "previous_camera_target": previous_camera_target,
    }
    if plan.mode in {"geometry2d", "graph2d"} and plan.scene_2d is not None:
        scene_payload = plan.scene_2d.model_dump()
        if _visual_2d_component is not None:
            _visual_2d_component(
                data={"scene": scene_payload, **component_data},
                default={},
                key="ai_visual2d",
                width="stretch",
                height="content",
            )
        else:
            st.info("The interactive 2D renderer is unavailable in this browser session.")
    elif plan.mode == "geometry3d" and plan.scene_3d is not None:
        scene_payload = plan.scene_3d.model_dump()
        if _visual_3d_component is not None:
            _visual_3d_component(
                data={"scene": scene_payload, **component_data},
                default={},
                key="ai_visual3d",
                width="stretch",
                height="content",
            )
        else:
            st.info("The interactive 3D renderer is unavailable in this browser session.")

    st.markdown(f"#### {step.title}")
    st.caption(f"Matches corrected solution step {getattr(step, 'source_step_index', idx + 1)}")
    if getattr(step, "simulation_note", ""):
        st.info("Simulation: " + step.simulation_note)
    if not (getattr(step, "animate_ids", []) or getattr(step, "highlight_ids", []) or getattr(step, "reveal_ids", [])):
        st.caption("This corrected step is mainly algebraic, so no new diagram object is introduced. Replay redraws the current construction for orientation.")
    st.markdown("**Matching corrected step**")
    for formula in step.math:
        render_mathio(formula)
    st.markdown("**Why this simulation matches the step**")
    render_mathio_mixed(step.explanation)

    if plan.mode == "geometry3d":
        st.caption("iPad: drag with one finger to rotate the solid and pinch with two fingers to zoom. Use Replay to watch the current construction/camera movement again.")


def _clean_practice_display_text(text: str) -> str:
    """Keep generated practice wording compact and prevent model Markdown from taking over the UI."""
    value = str(text or "").strip()
    value = value.replace("**", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _render_practice_key_information(items: list[str]) -> None:
    for item in (items or [])[:6]:
        if not str(item).strip():
            continue
        with st.container(border=True):
            render_mathio_mixed(_clean_practice_display_text(str(item)))


def _compact_task_prompt(focus_prompt: str, full_question: str) -> str:
    """Prefer a one-line action prompt even when the model repeats the whole story."""
    candidate = _clean_practice_display_text(focus_prompt)
    if candidate and len(candidate) <= 180 and candidate.count("\n") <= 1:
        return candidate
    source = _clean_practice_display_text(full_question)
    # Choose the last sentence/clause that contains an exam-style command.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", source)
    commands = re.compile(r"\b(calculate|find|determine|solve|show|state|express|sketch|work out|give|write down)\b", re.I)
    matches = [piece.strip() for piece in pieces if commands.search(piece)]
    if matches:
        return matches[-1][:260].strip()
    return (candidate or source)[:260].strip()


def render_targeted_practice_focus(pq: TargetedPracticeQuestion, *, key: str) -> None:
    """Present the task as a compact student card, with visual information before story text."""
    full_question = _clean_practice_display_text(getattr(pq, "question", "") or "")
    focus_prompt = _compact_task_prompt(getattr(pq, "focus_prompt", "") or "", full_question)
    key_information = list(getattr(pq, "key_information", []) or [])
    diagram = getattr(pq, "diagram_2d", None)
    diagram_note = str(getattr(pq, "diagram_note", "") or "").strip()

    with st.container(border=True):
        st.markdown('<div class="omt-focus-title">Your task</div>', unsafe_allow_html=True)
        render_mathio_mixed(focus_prompt or full_question)

        if diagram is not None:
            visual_col, info_col = st.columns([1.25, .85], gap="large", vertical_alignment="top")
            with visual_col:
                st.caption("Interactive graph workspace" if bool(getattr(diagram, "show_axes", False)) else "Question diagram")
                if _practice_diagram_component is not None:
                    _practice_diagram_component(
                        data={
                            "scene": diagram.model_dump(),
                            "step": {"highlight_ids": [], "dim_ids": [], "animate_ids": []},
                            "visible_ids": [],
                            "animate_ids": [],
                            "reveal_mode": False,
                            "animation_nonce": 0,
                        },
                        default={},
                        key=f"practice_diagram_{key}",
                        width="stretch",
                        height="content",
                    )
                else:
                    st.info("The practice diagram could not load in this browser session.")
                st.caption(diagram_note or ("Plot points or draw segments to explore the graph. Your red constructions are for working only." if bool(getattr(diagram, "show_axes", False)) else "Schematic only · not drawn to scale"))
            with info_col:
                if key_information:
                    st.caption("Given")
                    _render_practice_key_information(key_information)
        elif key_information:
            st.caption("Given")
            info_cols = st.columns(2) if len(key_information) > 1 else [st.container()]
            if len(key_information) > 1:
                for idx, item in enumerate(key_information[:6]):
                    with info_cols[idx % 2]:
                        with st.container(border=True):
                            render_mathio_mixed(_clean_practice_display_text(str(item)))
            else:
                _render_practice_key_information(key_information)

        with st.expander("Full wording", expanded=False):
            render_mathio_mixed(full_question)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": secrets.token_hex(8),
        "question": None,
        "attempt_result": None,
        "history": [],
        "hint_level": 0,
        "reveal_solution": False,
        "seed_counter": 1,
        "ai_analysis": None,
        "ai_error": "",
        "ai_visual_explanation": None,
        "ai_visual_error": "",
        "ai_visual_step": 0,
        "ai_visual_replay_nonce": 0,
        "ai_fallback_result": None,
        "ai_question_detection": None,
        "ai_question_detection_error": "",
        "ai_question_file_signature": "",
        "ai_selected_question_index": 0,
        "ai_question_feasibility": None,
        "ai_question_feasibility_error": "",
        "ai_question_feasibility_signature": "",
        "ai_practice_stage": 0,
        "ai_practice_current_question": None,
        "ai_practice_evaluation": None,
        "ai_practice_last_working": "",
        "ai_practice_misses": {"Near transfer": 0, "Varied context": 0, "Stretch": 0},
        "ai_practice_consecutive_correct": {"Near transfer": 0, "Varied context": 0, "Stretch": 0},
        "ai_practice_completed": {"Near transfer": False, "Varied context": False, "Stretch": False},
        "ai_practice_ready_to_advance": False,
        "ai_practice_finished": False,
        "ai_practice_question_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

PRACTICE_STAGES = ["Near transfer", "Varied context", "Stretch"]


def clear_ai_practice_state() -> None:
    st.session_state.ai_practice_stage = 0
    st.session_state.ai_practice_current_question = None
    st.session_state.ai_practice_evaluation = None
    st.session_state.ai_practice_last_working = ""
    st.session_state.ai_practice_misses = {kind: 0 for kind in PRACTICE_STAGES}
    st.session_state.ai_practice_consecutive_correct = {kind: 0 for kind in PRACTICE_STAGES}
    st.session_state.ai_practice_completed = {kind: False for kind in PRACTICE_STAGES}
    st.session_state.ai_practice_ready_to_advance = False
    st.session_state.ai_practice_finished = False
    st.session_state.ai_practice_question_version = 0


def initialize_ai_practice(analysis: GeminiAnalysis) -> None:
    clear_ai_practice_state()
    by_kind = {q.kind: q for q in analysis.practice_questions}
    st.session_state.ai_practice_current_question = by_kind["Near transfer"]


def practice_attempt_is_secure(result: PracticeEvaluation) -> bool:
    # Older Streamlit sessions may still contain evaluation objects from the pre-multipart schema.
    # Treat those as non-secure rather than allowing an accidental category advance.
    return (
        bool(getattr(result, "is_correct", False))
        and bool(getattr(result, "all_required_parts_complete", False))
        and not list(getattr(result, "missing_or_incorrect_parts", []) or [])
        and not list(getattr(result, "presentation_errors", []) or [])
        and int(getattr(result, "answer_score", 0) or 0) >= 80
        and int(getattr(result, "reasoning_score", 0) or 0) >= 80
        and getattr(result, "mastery", "") in {"Secure", "Strong"}
    )


def initial_practice_question(analysis: GeminiAnalysis, kind: str) -> TargetedPracticeQuestion:
    for question in analysis.practice_questions:
        if question.kind == kind:
            return question
    raise ValueError(f"No initial practice question found for {kind}")

# Streamlit Community Cloud stores app secrets in st.secrets.
# Copy the Gemini key into the process environment so the service layer can read it.
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
except Exception:
    pass


def track_code(label: str) -> str:
    return TRACKS[label]


def reset_current_question() -> None:
    st.session_state.attempt_result = None
    st.session_state.hint_level = 0
    st.session_state.reveal_solution = False
    for key in ("practice_working", "practice_working_format", "practice_working_equation", "practice_working_explanation"):
        st.session_state.pop(key, None)


def make_new_question(track: str, topic: str, difficulty: str) -> None:
    seed = int(datetime.now().timestamp() * 1000) + st.session_state.seed_counter
    st.session_state.seed_counter += 1
    st.session_state.question = generate_question(track, topic, difficulty, seed=seed)
    reset_current_question()


def record_history(question: Question, result: AttemptResult) -> None:
    st.session_state.history.append(
        {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "offline generated",
            "track": question.track,
            "topic_code": official_topic_code(question.track, question.topic_code),
            "topic": question.topic_name,
            "difficulty": question.difficulty,
            "question": question.prompt,
            "correct": result.is_correct,
            "answer_score": result.answer_score,
            "reasoning_score": result.reasoning_score,
            "mastery": result.mastery,
        }
    )


def record_ai_practice_history(track: str, q: TargetedPracticeQuestion, result: PracticeEvaluation) -> None:
    st.session_state.history.append(
        {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "Gemini targeted practice",
            "track": track,
            "topic_code": "AI",
            "topic": q.target_skill,
            "difficulty": q.kind,
            "question": q.question,
            "correct": result.is_correct,
            "answer_score": result.answer_score,
            "reasoning_score": result.reasoning_score,
            "mastery": result.mastery,
        }
    )


def render_attempt(result: AttemptResult) -> None:
    st.markdown("### Feedback")
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer", f"{result.answer_score}%")
    c2.metric("Reasoning", f"{result.reasoning_score}%")
    c3.metric("Current mastery", result.mastery)
    if result.is_correct and result.first_logic_break is None:
        st.success(result.summary)
    else:
        st.info(result.summary)
    if result.first_logic_break is not None:
        st.warning(
            f"First detected logic break: line {result.first_logic_break}. "
            f"{result.first_logic_break_explanation}"
        )
    if result.step_feedback:
        st.markdown("#### Step-by-step check")
        icon = {"correct": "✅", "incorrect": "❌", "unparsed": "🔎", "checked": "•"}
        for item in result.step_feedback:
            with st.expander(f"{icon.get(item.status, '•')} Line {item.line_number}"):
                st.markdown("**Student step**")
                render_mathio(item.line)
                render_mathio_mixed(item.feedback)
    if result.strengths:
        st.markdown("**What is working**")
        for item in result.strengths:
            st.write(f"• {item}")
    if result.gaps:
        st.markdown("**What to improve**")
        for item in result.gaps:
            st.write(f"• {item}")
    st.markdown("**Next hint:**")
    render_mathio_mixed(result.next_hint)


def render_ai_analysis(a: GeminiAnalysis) -> None:
    st.markdown('<div class="omt-section-kicker">Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="omt-section-title">What the student understands — and where the reasoning breaks</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.caption(a.likely_syllabus_topic)
        st.markdown("**Question understood as**")
        render_math_text(a.interpreted_question)
        st.markdown("**Method shown in the working**")
        render_math_text(a.student_method)

    if a.first_logic_break_step > 0:
        st.markdown(
            f'<div class="omt-logic-break"><strong>First material logic break · Step {a.first_logic_break_step}</strong><br><span style="color:#7c4a10">Use the advice below for this point before continuing.</span></div>',
            unsafe_allow_html=True,
        )
        render_math_text(a.first_logic_break_explanation)
    else:
        st.markdown('<div class="omt-success-card"><strong>No material logic break found</strong><br>The visible method is mathematically coherent.</div>', unsafe_allow_html=True)
        if a.first_logic_break_explanation:
            render_math_text(a.first_logic_break_explanation)

    if a.steps:
        st.markdown("### Working, step by step")
        icons = {
            "correct": "✓",
            "partly_correct": "◐",
            "incorrect": "×",
            "unclear": "?",
            "unsupported": "•",
        }
        labels = {
            "correct": "Correct",
            "partly_correct": "Partly correct",
            "incorrect": "Needs advice",
            "unclear": "Unclear",
            "unsupported": "Needs support",
        }
        for step in a.steps:
            presentation_flag = bool(getattr(step, "presentation_error", False))
            icon = "!" if presentation_flag else icons.get(step.status, "•")
            status_label = "Presentation issue" if presentation_flag else labels.get(step.status, step.status.replace("_", " ").title())
            with st.expander(f"{icon}  Step {step.line_number} · {status_label}", expanded=(step.line_number == a.first_logic_break_step)):
                st.caption("Student wrote")
                render_mathio(step.student_step)
                if presentation_flag:
                    st.warning("This line is not written as a complete, unambiguous mathematical statement.")
                    presentation_explanation = getattr(step, "presentation_error_explanation", "")
                    if presentation_explanation:
                        render_math_text(presentation_explanation)
                detail_left, detail_right = st.columns([1.1, .9], gap="large")
                with detail_left:
                    st.markdown("**What this step is trying to do**")
                    render_mathio_mixed(step.logic_inferred)
                with detail_right:
                    st.markdown("**Tutor feedback**")
                    render_mathio_mixed(step.feedback)
                    st.caption(f"Issue type · {step.issue_type.replace('_', ' ').title()}")
                for formula in list(getattr(step, "supporting_math", []) or []):
                    render_mathio(formula)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### ✓ What is working")
            if a.strengths:
                for item in a.strengths:
                    render_math_text(f"• {item}")
            else:
                st.caption("No specific strength was identified from the visible work.")
    with c2:
        with st.container(border=True):
            st.markdown("### → Main advice focus")
            render_math_text(a.misconception_or_gap)
            st.markdown("**Check your thinking**")
            render_math_text(a.diagnostic_question)

    st.markdown("### Guided advice")
    st.caption("Reveal only as much help as the student needs.")
    hint_cols = st.columns(3)
    for i, hint in enumerate(a.hint_ladder[:3], 1):
        with hint_cols[i - 1]:
            with st.expander(f"Hint {i}"):
                render_mathio_mixed(hint)
    with st.expander("Show corrected path and final answer"):
        for i, line in enumerate(a.corrected_path, 1):
            st.caption(f"Corrected step {i}")
            render_mathio(line)
        st.markdown("**Final answer**")
        render_mathio(a.final_answer)


def render_practice_evaluation(e: PracticeEvaluation) -> None:
    st.markdown("### Attempt feedback")
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer", f"{e.answer_score}%")
    c2.metric("Reasoning", f"{e.reasoning_score}%")
    c3.metric("Mastery", e.mastery)

    with st.container(border=True):
        render_math_text(e.summary)

    if e.first_logic_break_step > 0:
        st.markdown(
            f'<div class="omt-logic-break"><strong>First reasoning break · Step {e.first_logic_break_step}</strong></div>',
            unsafe_allow_html=True,
        )
        render_math_text(e.first_logic_break_explanation)

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**✓ Strengths**")
            if e.strengths:
                for item in e.strengths:
                    render_math_text(f"• {item}")
            else:
                st.caption("No secure strength identified yet.")
    with right:
        with st.container(border=True):
            st.markdown("**→ Advice for next step**")
            if e.missing_or_incorrect_parts:
                st.warning("Complete: " + ", ".join(e.missing_or_incorrect_parts))
            presentation_errors = list(getattr(e, "presentation_errors", []) or [])
            for item in presentation_errors:
                render_mathio_mixed(item)
            if e.gaps:
                for item in e.gaps:
                    render_math_text(f"• {item}")

    with st.expander("Next hint", expanded=False):
        render_mathio_mixed(e.next_hint)
    with st.expander("Show corrected next step", expanded=False):
        render_mathio(e.corrected_next_step)


def uploaded_assets(files: list[Any] | None) -> list[UploadedAsset]:
    files = files or []
    assets: list[UploadedAsset] = []
    total = 0
    for f in files:
        data = f.getvalue()
        total += len(data)
        if len(data) > MAX_FILE_BYTES:
            raise GeminiTutorError(f"{f.name} is larger than the app's 12 MB per-file limit.", category="input")
        mime = f.type or "application/octet-stream"
        assets.append(UploadedAsset(name=f.name, mime_type=mime, data=data))
    if total > MAX_TOTAL_BYTES:
        raise GeminiTutorError("Uploads exceed the app's 30 MB total limit.", category="input")
    return assets


def question_file_signature(files: list[Any] | None) -> str:
    files = files or []
    return "|".join(f"{getattr(f, 'name', '')}:{getattr(f, 'size', 0)}:{getattr(f, 'type', '')}" for f in files)


def detected_question_context(detection: QuestionDetectionResult, index: int) -> str:
    if not detection.questions:
        return ""
    index = max(0, min(index, len(detection.questions) - 1))
    q = detection.questions[index]
    lines = [
        "[SELECTED QUESTION FROM UPLOADED WORKSHEET]",
        f"Main question number: {q.question_number}",
        f"Main question stem: {q.question_text}",
        f"Likely topic: {q.topic_hint}",
    ]
    if q.subparts:
        lines.append("Subparts:")
        for part in q.subparts:
            lines.append(f"- {part.label}: {part.question_text}")
    lines.append("IMPORTANT: Analyse this selected main question only. Ignore other unrelated questions visible in the uploaded source.")
    return "\n".join(lines)


def render_question_detection(detection: QuestionDetectionResult) -> int:
    subpart_count = sum(len(q.subparts) for q in detection.questions)
    st.success(
        f"Detected {detection.main_question_count} confirmed main question(s)"
        + (f" with {subpart_count} subpart(s)." if subpart_count else ".")
    )
    if detection.possible_additional_question_count:
        st.warning(
            f"Gemini also saw {detection.possible_additional_question_count} possible additional question(s) that were too cropped or unclear to confirm."
        )
    st.caption(f"Detection confidence: {detection.overall_confidence.title()}")
    for note in detection.notes:
        st.caption(f"• {note}")

    if not detection.questions:
        st.info("No confirmed main question could be extracted. Try a clearer image or smaller crop.")
        return 0

    options = list(range(len(detection.questions)))
    current = min(int(st.session_state.ai_selected_question_index), len(options) - 1)
    selected = st.selectbox(
        "Choose the main question to analyse",
        options,
        index=current,
        format_func=lambda i: (
            f"Question {detection.questions[i].question_number} · {detection.questions[i].topic_hint}"
            + (f" · {len(detection.questions[i].subparts)} subpart(s)" if detection.questions[i].subparts else "")
        ),
        key="ai_detected_question_selector",
    )
    st.session_state.ai_selected_question_index = int(selected)
    q = detection.questions[selected]
    with st.expander("Review detected question text", expanded=True):
        render_math_text(f"**Question {q.question_number}:** {q.question_text}")
        for part in q.subparts:
            render_math_text(f"**{part.label}** {part.question_text}")
        st.caption(f"Topic hint: {q.topic_hint} · Confidence: {q.confidence}")
    return int(selected)


def question_for_selected_analysis(
    typed_text: str,
    detection: QuestionDetectionResult | None,
    selected_index: int,
) -> str:
    parts = [typed_text.strip()]
    if detection is not None and detection.questions:
        parts.append(detected_question_context(detection, selected_index))
    return "\n\n".join(part for part in parts if part)


def question_feasibility_signature(question_text: str, files: list[Any] | None, selected_index: int) -> str:
    return f"{question_text.strip()}||{question_file_signature(files)}||selected={selected_index}"


def _question_source_image(file_obj: Any, page_number: int = 1) -> Image.Image | None:
    """Load an uploaded question image or rasterize one PDF page for visual callouts."""
    if file_obj is None:
        return None
    try:
        data = file_obj.getvalue() if hasattr(file_obj, "getvalue") else bytes(file_obj.read())
        name = str(getattr(file_obj, "name", "")).lower()
        mime = str(getattr(file_obj, "type", "")).lower()
        if name.endswith(".pdf") or mime == "application/pdf":
            import fitz

            doc = fitz.open(stream=data, filetype="pdf")
            if doc.page_count < 1:
                return None
            page_index = max(0, min(int(page_number) - 1, doc.page_count - 1))
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            doc.close()
            return image
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _normalized_box_to_pixels(box: list[int], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box]
    except (TypeError, ValueError):
        return None
    if ymax <= ymin or xmax <= xmin:
        return None
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    return x1, y1, x2, y2


def _annotate_issue_regions(image: Image.Image, callouts: list[tuple[int, Any, str]]) -> Image.Image:
    """Draw numbered issue callouts on a copy of the original question diagram/page."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    width, height = annotated.size
    line_width = max(3, round(min(width, height) * 0.006))

    for issue_number, region, severity in callouts:
        box = _normalized_box_to_pixels(list(getattr(region, "box_2d", []) or []), width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        outline = (190, 30, 45) if severity == "blocking" else (175, 115, 0)
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=line_width)

        badge = f"{issue_number}"
        left = max(0, x1)
        top = max(0, y1 - 24)
        try:
            bbox = draw.textbbox((left, top), badge, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 10, 12
        pad = 4
        draw.rectangle((left, top, left + tw + 2 * pad, top + th + 2 * pad), fill=outline)
        draw.text((left + pad, top + pad), badge, fill=(255, 255, 255), font=font)
    return annotated


def render_feasibility_visual_map(result: QuestionFeasibilityResult, question_files: list[Any] | None) -> None:
    """Show the original diagram/page with numbered highlights matching feasibility issues."""
    if not question_files:
        return

    grouped: dict[tuple[int, int], list[tuple[int, Any, str]]] = {}
    for issue_number, issue in enumerate(result.issues, 1):
        for region in list(getattr(issue, "visual_regions", []) or []):
            source_index = int(getattr(region, "source_index", 0) or 0)
            page_number = int(getattr(region, "page_number", 1) or 1)
            if source_index < 1 or source_index > len(question_files):
                continue
            grouped.setdefault((source_index, page_number), []).append((issue_number, region, issue.severity))

    if not grouped:
        return

    st.markdown("#### Visual issue map")
    st.caption("Numbered highlights show the parts of the original question diagram/page that support each issue or warning below.")
    for (source_index, page_number), callouts in sorted(grouped.items()):
        file_obj = question_files[source_index - 1]
        image = _question_source_image(file_obj, page_number)
        if image is None:
            continue
        annotated = _annotate_issue_regions(image, callouts)
        name = str(getattr(file_obj, "name", f"Question source {source_index}"))
        page_note = f" · page {page_number}" if name.lower().endswith(".pdf") else ""
        st.image(annotated, caption=f"{name}{page_note}", use_container_width=True)
        labels = []
        for issue_number, region, _severity in callouts:
            label = str(getattr(region, "label", "")).strip()
            if label:
                labels.append(f"{issue_number}: {label}")
        if labels:
            st.caption(" · ".join(labels))


def render_question_feasibility(result: QuestionFeasibilityResult, question_files: list[Any] | None = None) -> None:
    labels = {
        "feasible": "Ready to analyse",
        "feasible_with_caveats": "Ready with caveats",
        "needs_clarification": "Clarification needed",
        "infeasible": "Question issue detected",
    }
    message = labels.get(result.status, result.status.replace("_", " ").title())
    if result.status == "feasible":
        st.markdown(f'<div class="omt-success-card"><strong>✓ {message}</strong><br>The question is sufficiently clear and consistent for reasoning analysis.</div>', unsafe_allow_html=True)
    elif result.status == "feasible_with_caveats":
        st.warning(f"{message} — review the note below before marking.")
    else:
        st.error(f"{message} — student-working analysis stays locked until the question is clarified or corrected.")

    with st.container(border=True):
        st.caption("Question interpreted as")
        render_math_text(result.interpreted_question)
        c1, c2, c3 = st.columns(3)
        c1.metric("Answerability", result.answerability.replace("_", " ").title())
        c2.metric("Information", "Complete" if result.required_information_present else "Needs attention")
        c3.metric("Diagram / table", "Sufficient" if result.diagram_or_table_sufficient else "Needs attention")
        st.caption(
            f"Syllabus fit · {result.syllabus_fit.replace('_', ' ').title()}    |    "
            f"Confidence · {result.confidence.title()}"
        )

    render_feasibility_visual_map(result, question_files)

    if result.issues:
        st.markdown("### Issues to check")
        for issue_number, issue in enumerate(result.issues, 1):
            label = "Blocking issue" if issue.severity == "blocking" else "Warning"
            visual_note = f" · diagram {issue_number}" if list(getattr(issue, "visual_regions", []) or []) else ""
            with st.container(border=True):
                st.markdown(f"**{label}{visual_note}**")
                render_mathio_mixed(issue.description)
                if issue.suggested_fix:
                    st.caption("Suggested fix")
                    render_math_text(issue.suggested_fix)

    if result.suspected_corrections:
        with st.expander("Possible corrections to verify"):
            for item in result.suspected_corrections:
                render_math_text(f"• {item}")

    if result.action_needed:
        st.markdown("**Next action**")
        render_math_text(result.action_needed)


def offline_evidence_for(question_text: str, working_text: str) -> tuple[str, AttemptResult | None]:
    if not question_text.strip() or not working_text.strip():
        return "", None
    try:
        result = analyze_own_algebra_question(question_text, working_text)
    except ValueError:
        return "", None
    evidence = (
        f"Offline algebra checker says is_correct={result.is_correct}; "
        f"first_logic_break={result.first_logic_break}; "
        f"explanation={result.first_logic_break_explanation}; "
        f"answer_score={result.answer_score}; reasoning_score={result.reasoning_score}."
    )
    return evidence, result


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div class="omt-side-brand">
          <div class="title">✦ SG Math Tutor</div>
          <div class="sub">Reasoning-first support for Singapore O-Level and N-Level Mathematics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    track_label = st.selectbox("Exam track", list(TRACKS.keys()), index=0)
    tcode = track_code(track_label)
    syllabus_name = (
        "O-Level Mathematics · 4052" if tcode == "O"
        else "N(A)-Level Mathematics A · 4045" if tcode == "NA"
        else "N(T)-Level Mathematics T · 4046"
    )
    st.markdown(f'<div class="omt-status-pill neutral">📘 <span>{syllabus_name}</span></div>', unsafe_allow_html=True)

    with st.expander("⚙️ Gemini connection", expanded=False):
        explicit_key = st.text_input(
            "Gemini API key (optional here)",
            type="password",
            help="Prefer Streamlit Community Cloud Secrets with the name GEMINI_API_KEY.",
        )
        has_key = bool(get_api_key(explicit_key))
        model = st.selectbox(
            "Gemini model",
            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            index=0,
            help="Free-tier availability and quotas depend on the Google account/project.",
        )
    if has_key:
        st.markdown('<div class="omt-status-pill good">● <span>Gemini online</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="omt-status-pill neutral">○ <span>Offline tools available</span></div>', unsafe_allow_html=True)

    with st.expander("Privacy & data", expanded=False):
        st.caption(
            "Online analysis sends only the selected question/work to Gemini. Remove names, NRICs and unnecessary identifiers. "
            "Offline practice and typed-algebra checking do not call Gemini."
        )

    if st.button("↻ Reset learning session", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in {"session_id"}:
                del st.session_state[key]
        st.rerun()


st.markdown(
    """
    <section class="omt-hero">
      <div class="omt-eyebrow">Singapore secondary mathematics</div>
      <h1>Reasoning Tutor</h1>
      <p>Understand the student's method, find the first reasoning break, advise the student clearly, then build mastery through adaptive practice.</p>
      <div class="omt-chip-row">
        <span class="omt-chip">✍️ Handwriting & iPad</span>
        <span class="omt-chip">∑ MathIO equation view</span>
        <span class="omt-chip">◫ Visual geometry</span>
        <span class="omt-chip">↗ Adaptive mastery</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

ai_tab, practice_tab, own_tab, syllabus_tab, progress_tab = st.tabs(
    [
        "✨ Analyse",
        "🧠 Offline practice",
        "✎ Algebra check",
        "📚 Syllabus",
        "📈 Progress",
    ]
)

# ---------- Gemini online analysis ----------
with ai_tab:
    st.markdown('<div class="omt-section-kicker">Step 1 · Submit</div>', unsafe_allow_html=True)
    st.markdown('<div class="omt-section-title">Question + student working</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='omt-section-copy'>Upload a photo/PDF or type the question, then add the student's working. The tutor keeps the question and solution separate during diagnosis.</div>",
        unsafe_allow_html=True,
    )

    input_left, input_right = st.columns([.95, 1.05], gap="large")
    with input_left:
        with st.container(border=True):
            st.markdown("#### 📄 Question")
            q_text = st.text_area(
                "Question text",
                key="ai_question_text",
                height=132,
                placeholder="Type the question here, or leave blank if it is visible in the upload.",
                label_visibility="collapsed",
            )
            q_files = st.file_uploader(
                "Upload question image/PDF",
                type=["png", "jpg", "jpeg", "webp", "pdf"],
                accept_multiple_files=True,
                key="ai_question_files",
                help="Photos, screenshots and PDFs are supported.",
            )

    with input_right:
        with st.container(border=True):
            st.markdown("#### ✍️ Student working")
            w_text, w_input_mode, w_offline_text = working_input(
                "Student working",
                text_key="ai_working_text",
                format_key="ai_working_format",
                height=160,
                plain_placeholder="Type the steps, use the equation editor, or leave blank when the working is uploaded.",
            )
            w_files = st.file_uploader(
                "Upload student working image/PDF",
                type=["png", "jpg", "jpeg", "webp", "pdf"],
                accept_multiple_files=True,
                key="ai_working_files",
            )

    # Clear stale detection results when the uploaded source changes.
    current_signature = question_file_signature(q_files)
    if current_signature != st.session_state.ai_question_file_signature:
        st.session_state.ai_question_file_signature = current_signature
        st.session_state.ai_question_detection = None
        st.session_state.ai_question_detection_error = ""
        st.session_state.ai_selected_question_index = 0
        st.session_state.ai_question_feasibility = None
        st.session_state.ai_question_feasibility_error = ""
        st.session_state.ai_question_feasibility_signature = ""
        st.session_state.ai_analysis = None
        st.session_state.ai_visual_explanation = None
        st.session_state.ai_visual_error = ""
        st.session_state.ai_visual_step = 0
        clear_ai_practice_state()
        st.session_state.pop("ai_detected_question_selector", None)

    consent = st.checkbox(
        "Allow Gemini to analyse the selected question and working",
        key="gemini_consent",
        help="Remove names, NRICs and other unnecessary personal identifiers before sending student work.",
    )

    selected_detection_index = 0
    if q_files:
        st.markdown("### Detect questions in the upload")
        st.write(
            "Gemini can count the **main questions** in the uploaded image/PDF, keep subparts grouped under their main question, "
            "and let the student choose which question to analyse."
        )
        if st.button("Detect questions in uploaded file(s)", use_container_width=True):
            st.session_state.ai_question_detection = None
            st.session_state.ai_question_detection_error = ""
            st.session_state.ai_question_feasibility = None
            st.session_state.ai_question_feasibility_error = ""
            st.session_state.ai_question_feasibility_signature = ""
            st.session_state.pop("ai_detected_question_selector", None)
            if not consent:
                st.session_state.ai_question_detection_error = (
                    "Confirm the Gemini data-sharing acknowledgement before detecting questions."
                )
            else:
                try:
                    assets_q = uploaded_assets(q_files)
                    with st.spinner("Detecting main questions and subparts in the upload..."):
                        detection = detect_questions_in_assets(
                            track_label=track_label,
                            question_assets=assets_q,
                            api_key=explicit_key,
                            model=model,
                        )
                    st.session_state.ai_question_detection = detection
                    st.rerun()
                except GeminiTutorError as exc:
                    st.session_state.ai_question_detection_error = str(exc)
                    st.rerun()

        if st.session_state.ai_question_detection_error:
            st.error(st.session_state.ai_question_detection_error)

        detection: QuestionDetectionResult | None = st.session_state.ai_question_detection
        if detection is not None:
            selected_detection_index = render_question_detection(detection)

    detection = st.session_state.ai_question_detection
    question_for_analysis = question_for_selected_analysis(q_text, detection, selected_detection_index)
    current_feasibility_signature = question_feasibility_signature(
        question_for_analysis, q_files, selected_detection_index
    )
    if (
        st.session_state.ai_question_feasibility_signature
        and st.session_state.ai_question_feasibility_signature != current_feasibility_signature
    ):
        st.session_state.ai_question_feasibility = None
        st.session_state.ai_question_feasibility_error = ""
        st.session_state.ai_question_feasibility_signature = ""
        st.session_state.ai_analysis = None
        st.session_state.ai_visual_explanation = None
        st.session_state.ai_visual_error = ""
        st.session_state.ai_visual_step = 0
        clear_ai_practice_state()

    st.markdown("### Check the question before marking")
    st.write(
        "Before looking at the student's solution, Gemini checks whether the selected question is complete, internally consistent, "
        "mathematically meaningful, and sufficiently clear to mark reliably."
    )
    if st.button("Check question feasibility", use_container_width=True):
        st.session_state.ai_question_feasibility = None
        st.session_state.ai_question_feasibility_error = ""
        st.session_state.ai_analysis = None
        st.session_state.ai_visual_explanation = None
        st.session_state.ai_visual_error = ""
        st.session_state.ai_visual_step = 0
        clear_ai_practice_state()
        if not consent:
            st.session_state.ai_question_feasibility_error = (
                "Confirm the Gemini data-sharing acknowledgement before checking the question."
            )
        elif not question_for_analysis.strip() and not q_files:
            st.session_state.ai_question_feasibility_error = "Provide the question as text or an upload first."
        else:
            try:
                assets_q = uploaded_assets(q_files)
                with st.spinner("Checking the question for missing information, contradictions, ambiguity, and mathematical feasibility..."):
                    feasibility = assess_question_feasibility(
                        track_label=track_label,
                        question_text=question_for_analysis,
                        question_assets=assets_q,
                        api_key=explicit_key,
                        model=model,
                    )
                st.session_state.ai_question_feasibility = feasibility
                st.session_state.ai_question_feasibility_signature = current_feasibility_signature
                st.rerun()
            except GeminiTutorError as exc:
                st.session_state.ai_question_feasibility_error = str(exc)
                st.rerun()

    if st.session_state.ai_question_feasibility_error:
        st.error(st.session_state.ai_question_feasibility_error)

    feasibility: QuestionFeasibilityResult | None = st.session_state.ai_question_feasibility
    if feasibility is not None:
        render_question_feasibility(feasibility, q_files)

    feasibility_ready = bool(
        feasibility is not None
        and feasibility.can_analyse_student_work
        and st.session_state.ai_question_feasibility_signature == current_feasibility_signature
    )
    if not feasibility_ready:
        st.info("Student-working analysis remains locked until the current question passes the feasibility check.")

    if st.button(
        "Analyse student working with Gemini",
        type="primary",
        use_container_width=True,
        disabled=not feasibility_ready,
    ):
        st.session_state.ai_analysis = None
        st.session_state.ai_error = ""
        st.session_state.ai_fallback_result = None
        st.session_state.ai_visual_explanation = None
        st.session_state.ai_visual_error = ""
        st.session_state.ai_visual_step = 0
        clear_ai_practice_state()
        if not consent:
            st.error("Confirm the Gemini data-sharing acknowledgement before sending the submission.")
        elif not feasibility_ready:
            st.error("Run the question feasibility check and resolve any blocking question issue before analysing the student's work.")
        else:
            # The visual equation editor also returns ASCIIMath for the deterministic algebra fallback.
            evidence, offline_result = offline_evidence_for(question_for_analysis, w_offline_text)
            working_for_gemini = (
                f"[Student working input method: {w_input_mode}]\n{w_text}" if w_text.strip() else w_text
            )
            try:
                assets_q = uploaded_assets(q_files)
                assets_w = uploaded_assets(w_files)
                with st.spinner("Gemini is checking the mathematics and the student's reasoning..."):
                    analysis = analyze_submission(
                        track_label=track_label,
                        question_text=question_for_analysis,
                        working_text=working_for_gemini,
                        question_assets=assets_q,
                        working_assets=assets_w,
                        offline_evidence=evidence,
                        api_key=explicit_key,
                        model=model,
                    )
                st.session_state.ai_analysis = analysis
                if _visual_plan_is_recommended(analysis, question_for_analysis):
                    try:
                        with st.spinner("Building an interactive visual explanation for this geometry/graph question..."):
                            visual_plan = generate_visual_explanation(
                                track_label=track_label,
                                question_text=question_for_analysis,
                                analysis=analysis,
                                question_assets=assets_q,
                                api_key=explicit_key,
                                model=model,
                            )
                        st.session_state.ai_visual_explanation = visual_plan
                    except GeminiTutorError as visual_exc:
                        # Visuals are an enhancement; never lose the verified reasoning analysis if this second call fails.
                        st.session_state.ai_visual_error = str(visual_exc)
                initialize_ai_practice(analysis)
                st.rerun()
            except GeminiTutorError as exc:
                st.session_state.ai_error = str(exc)
                if offline_result is not None:
                    st.session_state.ai_fallback_result = offline_result
                st.rerun()

    if st.session_state.ai_error:
        st.error(st.session_state.ai_error)
        if st.session_state.ai_fallback_result is not None:
            st.info("Gemini was unavailable, so the tutor automatically used its deterministic offline algebra fallback for this typed submission.")
            render_attempt(st.session_state.ai_fallback_result)
        else:
            st.info("Use **Offline practice** or **Offline algebra check** while Gemini is unavailable.")

    analysis: GeminiAnalysis | None = st.session_state.ai_analysis
    if analysis is not None:
        render_ai_analysis(analysis)
        visual_plan: VisualExplanationResult | None = st.session_state.ai_visual_explanation
        if visual_plan is not None:
            st.markdown("---")
            render_visual_explanation(visual_plan, q_files)
        elif st.session_state.ai_visual_error:
            st.caption("Interactive visual explanation unavailable for this attempt: " + st.session_state.ai_visual_error)
        st.markdown("---")
        st.markdown('<div class="omt-section-kicker">Adaptive practice</div>', unsafe_allow_html=True)
        st.markdown('<div class="omt-section-title">Build mastery one transfer level at a time</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="omt-section-copy">Near transfer → Varied context → Stretch. A mistake keeps the student on the same skill until the reasoning becomes secure.</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.ai_practice_current_question is None and not st.session_state.ai_practice_finished:
            initialize_ai_practice(analysis)

        stage_index = int(st.session_state.ai_practice_stage)
        completed = st.session_state.ai_practice_completed
        stage_html = []
        for i, kind in enumerate(PRACTICE_STAGES):
            if completed.get(kind):
                css = "done"; icon = "✓"; detail = "Mastered"
            elif not st.session_state.ai_practice_finished and i == stage_index:
                css = "current"; icon = "●"; detail = "Current focus"
            else:
                css = "locked"; icon = "◌"; detail = "Locked"
            stage_html.append(
                f'<div class="omt-stage {css}"><div class="name">{icon} {kind.title()}</div><div class="detail">{detail}</div></div>'
            )
        st.markdown('<div class="omt-stage-row">' + ''.join(stage_html) + '</div>', unsafe_allow_html=True)

        if st.session_state.ai_practice_finished:
            st.success(
                "Adaptive practice complete: the student demonstrated secure reasoning through Near transfer, "
                "Varied context, and Stretch."
            )
        else:
            kind = PRACTICE_STAGES[stage_index]
            pq: TargetedPracticeQuestion = st.session_state.ai_practice_current_question
            misses = st.session_state.ai_practice_misses[kind]
            streak = st.session_state.ai_practice_consecutive_correct[kind]

            st.markdown(f'<div class="omt-section-kicker">Current focus</div><div class="omt-section-title">{kind.title()}</div>', unsafe_allow_html=True)
            if misses:
                st.warning(
                    f"This category remains active because the student has had {misses} non-secure attempt(s). "
                    f"Current recovery streak: {streak}/2 secure attempts."
                )
            render_targeted_practice_focus(
                pq,
                key=f"{stage_index}_{st.session_state.ai_practice_question_version}",
            )
            st.caption("Skill being checked")
            render_mathio_mixed(_clean_practice_display_text(pq.target_skill))
            required_parts = required_parts_for_question(pq)
            if required_parts != ["whole question"]:
                st.caption("All parts required for mastery: " + ", ".join(required_parts))
            with st.expander("Why this question"):
                render_mathio_mixed(_clean_practice_display_text(pq.why_this_tests_understanding))
            with st.expander("Practice hints"):
                for i, hint in enumerate(pq.hints, 1):
                    st.markdown(f"**Hint {i}:**")
                    render_mathio_mixed(hint)

            working_key = f"ai_practice_working_{stage_index}_{st.session_state.ai_practice_question_version}"
            attempt, practice_input_mode, _practice_offline_text, practice_assets = targeted_practice_input(
                f"Student working for {kind}",
                key_base=working_key,
                height=150,
                practice_question=pq,
            )

            if st.button(f"Check {kind} reasoning", key=f"ai_practice_check_{stage_index}_{st.session_state.ai_practice_question_version}", type="primary"):
                if practice_input_mode == "Handwritten working" and not attempt.strip() and not practice_assets:
                    st.warning("No saved handwriting was received. Return to the handwriting pad, tap **Save handwriting**, then check the reasoning again.")
                    st.stop()
                try:
                    with st.spinner("Checking the practice reasoning..."):
                        evaluation = evaluate_practice_attempt(
                            track_label=track_label,
                            practice_question=pq,
                            student_working=(
                                f"[Student working input method: {practice_input_mode}]\n{attempt}"
                                if attempt.strip() else f"[Student working input method: {practice_input_mode}]"
                            ),
                            working_assets=practice_assets,
                            original_gap=analysis.misconception_or_gap,
                            api_key=explicit_key,
                            model=model,
                        )
                    secure = practice_attempt_is_secure(evaluation)
                    if secure:
                        st.session_state.ai_practice_consecutive_correct[kind] += 1
                    else:
                        st.session_state.ai_practice_misses[kind] += 1
                        st.session_state.ai_practice_consecutive_correct[kind] = 0

                    current_misses = st.session_state.ai_practice_misses[kind]
                    current_streak = st.session_state.ai_practice_consecutive_correct[kind]
                    st.session_state.ai_practice_ready_to_advance = bool(
                        secure and (current_misses == 0 or current_streak >= 2)
                    )
                    st.session_state.ai_practice_evaluation = evaluation
                    st.session_state.ai_practice_last_working = (
                        attempt if attempt.strip() else f"[{practice_input_mode} submitted; use the marking feedback as the diagnostic summary.]"
                    )
                    record_ai_practice_history(tcode, pq, evaluation)
                    st.rerun()
                except GeminiTutorError as exc:
                    st.error(str(exc))

            evaluation: PracticeEvaluation | None = st.session_state.ai_practice_evaluation
            if evaluation is not None:
                render_practice_evaluation(evaluation)
                secure = practice_attempt_is_secure(evaluation)
                ready = bool(st.session_state.ai_practice_ready_to_advance)

                if ready:
                    st.success(f"{kind} is secure. The next transfer level can now be unlocked.")
                    if stage_index < len(PRACTICE_STAGES) - 1:
                        next_kind = PRACTICE_STAGES[stage_index + 1]
                        if st.button(f"Continue to {next_kind}", use_container_width=True):
                            st.session_state.ai_practice_completed[kind] = True
                            st.session_state.ai_practice_stage = stage_index + 1
                            st.session_state.ai_practice_current_question = initial_practice_question(analysis, next_kind)
                            st.session_state.ai_practice_evaluation = None
                            st.session_state.ai_practice_last_working = ""
                            st.session_state.ai_practice_ready_to_advance = False
                            st.session_state.ai_practice_question_version += 1
                            st.rerun()
                    else:
                        if st.button("Complete adaptive practice", use_container_width=True):
                            st.session_state.ai_practice_completed[kind] = True
                            st.session_state.ai_practice_finished = True
                            st.session_state.ai_practice_evaluation = None
                            st.rerun()
                else:
                    if secure:
                        remaining = max(0, 2 - st.session_state.ai_practice_consecutive_correct[kind])
                        st.info(
                            f"Good recovery. Because there was an earlier miss in {kind}, "
                            f"complete {remaining} more secure attempt(s) in this same category before advancing."
                        )
                    else:
                        st.warning(
                            f"Stay on {kind}. The next category remains locked until the student can apply the advice securely."
                        )

                    if st.button(f"Generate another {kind} question", use_container_width=True):
                        try:
                            with st.spinner(f"Creating another {kind} question focused on the same gap..."):
                                followup = generate_followup_practice_question(
                                    track_label=track_label,
                                    kind=kind,
                                    previous_question=pq,
                                    previous_working=st.session_state.ai_practice_last_working,
                                    evaluation=evaluation,
                                    original_gap=analysis.misconception_or_gap,
                                    api_key=explicit_key,
                                    model=model,
                                )
                            st.session_state.ai_practice_current_question = followup
                            st.session_state.ai_practice_evaluation = None
                            st.session_state.ai_practice_last_working = ""
                            st.session_state.ai_practice_ready_to_advance = False
                            st.session_state.ai_practice_question_version += 1
                            st.rerun()
                        except GeminiTutorError as exc:
                            st.error(str(exc))

            with st.expander("Reveal reference answer and worked solution"):
                st.markdown("**Answer**")
                render_mathio(pq.answer)
                st.markdown("**Worked solution**")
                for i, line in enumerate(pq.worked_solution, 1):
                    st.caption(f"Step {i}")
                    render_mathio(line)

# ---------- Offline generated practice ----------
with practice_tab:
    st.subheader("No-credit syllabus-generated practice")
    st.caption("This tab never calls Gemini. It keeps working even if the API key is missing or a free-tier quota is reached.")
    available = topics_for_track(tcode)
    topic_labels = {f"{official_topic_code(tcode, t.code)} · {t.name}": t.code for t in available}
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        topic_label = st.selectbox("Topic", list(topic_labels.keys()), key="topic_choice")
    with c2:
        difficulty = st.selectbox("Difficulty", ["Foundation", "Similar", "Stretch"], index=1)
    with c3:
        st.write("")
        st.write("")
        if st.button("Generate question", type="primary", use_container_width=True):
            make_new_question(tcode, topic_labels[topic_label], difficulty)
            st.rerun()

    question: Question | None = st.session_state.question
    if question is None or question.track != tcode:
        st.info("Choose a topic and click **Generate question**.")
    else:
        st.markdown(f"### {official_topic_code(question.track, question.topic_code)} · {question.topic_name}")
        st.caption(f"{question.strand} · {question.difficulty}")
        st.markdown(f'<div class="soft-card"><strong>{question.prompt}</strong></div>', unsafe_allow_html=True)
        st.markdown(f"**Target skill:** {question.target_skill}")

        if st.button("Show next hint", key="show_hint"):
            st.session_state.hint_level = min(len(question.hints), st.session_state.hint_level + 1)
        for i in range(st.session_state.hint_level):
            st.markdown(f"**Hint {i+1}:**")
            render_mathio_mixed(question.hints[i])
        if st.session_state.hint_level == 0:
            st.caption("Try the question before revealing a hint.")

        working, working_mode, working_offline = working_input(
            "Your working and answer",
            text_key="practice_working",
            format_key="practice_working_format",
            height=190,
            plain_placeholder="Show the important steps, one line at a time where possible.",
        )
        working_to_check = working_offline if working_mode == "Equation editor" else working
        if st.button("Check my reasoning offline", type="primary", use_container_width=True):
            if not working_to_check.strip():
                st.error("Enter your working and answer first.")
            else:
                result = evaluate_attempt(question, working_to_check)
                st.session_state.attempt_result = result
                record_history(question, result)
                st.rerun()

        result: AttemptResult | None = st.session_state.attempt_result
        if result is not None:
            render_attempt(result)

        st.markdown("---")
        reveal = st.checkbox("Reveal verified answer and worked solution", key="reveal_solution")
        if reveal:
            st.markdown(f"**Answer:** {question.answer_display}")
            for i, line in enumerate(question.worked_solution, 1):
                st.write(f"{i}. {line}")

        cnext1, cnext2 = st.columns(2)
        with cnext1:
            if st.button("Generate a similar question", use_container_width=True):
                seed = int(datetime.now().timestamp() * 1000) + st.session_state.seed_counter
                st.session_state.seed_counter += 1
                st.session_state.question = generate_similar(question, seed=seed, difficulty="Similar")
                reset_current_question()
                st.rerun()
        with cnext2:
            if st.button("Generate a stretch question", use_container_width=True):
                seed = int(datetime.now().timestamp() * 1000) + st.session_state.seed_counter
                st.session_state.seed_counter += 1
                st.session_state.question = generate_similar(question, seed=seed, difficulty="Stretch")
                reset_current_question()
                st.rerun()

# ---------- Own typed algebra ----------
with own_tab:
    st.subheader("Check a student's own typed algebra question — offline")
    st.write(
        "This deterministic checker supports **one-variable equations** typed as text. "
        "It verifies equation equivalence line by line and identifies the earliest parseable logic break. No API key is used."
    )
    st.info("Example: `Solve 3(x + 2) = 18.` Enter each working line separately, such as `3x + 6 = 18`.")

    own_q = st.text_area("Question", key="own_question", height=95, placeholder="Solve 3(x + 2) = 18.")
    own_w, own_mode, own_w_offline = working_input(
        "Student working",
        text_key="own_working",
        format_key="own_working_format",
        height=190,
        plain_placeholder="3(x + 2) = 18\n3x + 6 = 18\n3x = 12\nx = 4",
    )
    own_working_to_check = own_w_offline if own_mode == "Equation editor" else own_w

    if st.button("Check typed algebra", type="primary"):
        if not own_q.strip() or not own_working_to_check.strip():
            st.error("Enter both the question and the student's working.")
        else:
            try:
                res = analyze_own_algebra_question(own_q, own_working_to_check)
                render_attempt(res)
            except ValueError as exc:
                st.warning(str(exc))

# ---------- Coverage ----------
with syllabus_tab:
    st.subheader("2026 Singapore Mathematics syllabus coverage")
    st.write(
        "Offline generated practice spans the three syllabus strands: Number and Algebra, Geometry and Measurement, "
        "and Statistics and Probability. Gemini online mode broadens interpretation to uploaded handwriting, diagrams, PDFs, "
        "word problems and alternative methods, but AI feedback can still be wrong and should be reviewed for high-stakes use."
    )

    selected = topics_for_track(tcode)
    strong = sum(1 for t in selected if t.offline_support == "Strong")
    partial = sum(1 for t in selected if t.offline_support == "Partial")
    c1, c2, c3 = st.columns(3)
    c1.metric("Topics mapped", len(selected))
    c2.metric("Strong offline generated support", strong)
    c3.metric("Partial offline generated support", partial)

    for strand in ("Number and Algebra", "Geometry and Measurement", "Statistics and Probability"):
        st.markdown(f"### {strand}")
        for t in [x for x in selected if x.strand == strand]:
            badge = "✅ Strong" if t.offline_support == "Strong" else "🟡 Partial"
            with st.expander(f"{official_topic_code(tcode, t.code)} · {t.name} — {badge}"):
                st.write(t.notes)

    st.warning(
        "Coverage means the tutor has practice/checking support for these areas; it does not guarantee perfect interpretation of every past-paper question. "
        "For school assessment decisions, verify AI feedback against a teacher or official marking scheme."
    )

# ---------- Progress ----------
with progress_tab:
    st.subheader("Session progress")
    hist = st.session_state.history
    if not hist:
        st.info("Complete offline or Gemini-generated practice questions to build a progress record for this browser session.")
    else:
        total = len(hist)
        correct = sum(1 for x in hist if x["correct"])
        avg_reason = round(sum(x["reasoning_score"] for x in hist) / total)
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts", total)
        c2.metric("Correct", f"{correct}/{total}")
        c3.metric("Average reasoning", f"{avg_reason}%")

        topic_counts = Counter(x["topic"] for x in hist if x["correct"])
        if topic_counts:
            st.markdown("**Most successful topics this session**")
            for name, count in topic_counts.most_common(5):
                st.write(f"• {name}: {count} correct")

        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button(
            "Download session history (JSON)",
            data=json.dumps(hist, indent=2),
            file_name="singapore_math_tutor_session.json",
            mime="application/json",
        )

st.markdown("---")
st.caption(
    f"Educational tool, not an official SEAB/MOE product. Gemini default model: {DEFAULT_MODEL}. "
    "Generated questions are original and are not past-year examination questions."
)
