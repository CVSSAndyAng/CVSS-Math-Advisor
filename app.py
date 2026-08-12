from __future__ import annotations

import base64
import json
import os
import secrets
from io import BytesIO
from collections import Counter
from datetime import datetime, timezone
from typing import Any

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
    UploadedAsset,
    analyze_submission,
    assess_question_feasibility,
    detect_questions_in_assets,
    evaluate_practice_attempt,
    generate_followup_practice_question,
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
.block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1220px;}
.soft-card {border: 1px solid rgba(128,128,128,.28); border-radius: .8rem; padding: 1rem 1.1rem; margin: .4rem 0 1rem 0;}
.small {font-size:.88rem; opacity:.82;}
.ok {background: rgba(0,160,90,.08); border-radius:.6rem; padding:.65rem .8rem;}
.warn {background: rgba(255,170,0,.08); border-radius:.6rem; padding:.65rem .8rem;}
@media (max-width: 1100px) {
  .block-container {max-width: 100%; padding-left: 1rem; padding-right: 1rem;}
}
@media (pointer: coarse) {
  button, [role="button"] {min-height: 44px;}
  input, textarea, select {font-size: 16px !important;}
}
</style>
""",
    unsafe_allow_html=True,
)

MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024


def math_markdown(text: str) -> str:
    """Convert model-safe LaTeX delimiters into Streamlit Markdown math delimiters.

    Gemini is prompted to return \\( ... \\) / \\[ ... \\] so raw dollar signs
    never appear in stored model output. Streamlit Markdown renders the converted
    delimiters with KaTeX in the browser.
    """
    if not text:
        return ""
    return (
        str(text)
        .replace(r"\[", "$$")
        .replace(r"\]", "$$")
        .replace(r"\(", "$")
        .replace(r"\)", "$")
    )


def render_math_text(text: str) -> None:
    st.markdown(math_markdown(text))


def _mathio_latex(text: str) -> str:
    """Normalize model LaTeX for equation-view rendering without showing delimiters."""
    if not text:
        return r"\text{No reference answer available}"
    value = str(text).strip()
    for token in (r"\(", r"\)", r"\[", r"\]", "$$", "$"):
        value = value.replace(token, "")
    return value.strip()


def render_mathio(text: str) -> None:
    """Render reference mathematics in MathIO/equation view."""
    st.latex(_mathio_latex(text))


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


def equation_working_editor(label: str, *, key: str) -> tuple[list[str], list[str]]:
    """Render a visual multi-step MathLive editor and return LaTeX + ASCIIMath lines."""
    if _equation_editor_component is None:
        fallback = st.text_area(
            label,
            key=f"{key}_fallback",
            height=150,
            placeholder=r"Fallback: type one LaTeX step per line, e.g. m=\frac{4-1}{-2-7}",
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
        st.caption("The editor stores LaTeX internally for accurate maths rendering; students do not need to type LaTeX commands.")
        return working_for_gemini, mode, offline_text

    value = st.text_area(label, key=text_key, height=height, placeholder=plain_placeholder)
    return value, mode, value



_HANDWRITING_HTML = """
<div class="omt-handwriting-pad">
  <div class="omt-handwriting-help">Write directly with Apple Pencil, stylus, or finger. Use one line per step and label parts such as (a), (b), (c).</div>
  <div class="omt-handwriting-toolbar">
    <button type="button" class="omt-clear-pad">Clear</button>
  </div>
  <canvas class="omt-handwriting-canvas" aria-label="Handwritten mathematics working area"></canvas>
  <div class="omt-handwriting-status" aria-live="polite"></div>
</div>
"""

_HANDWRITING_CSS = """
.omt-handwriting-pad { width: 100%; font-family: var(--st-font, sans-serif); }
.omt-handwriting-help { opacity: .74; font-size: .88rem; margin: 0 0 .55rem 0; }
.omt-handwriting-toolbar { display:flex; justify-content:flex-end; margin-bottom:.45rem; }
.omt-clear-pad { min-height:44px; padding:.45rem .85rem; border:1px solid rgba(128,128,128,.42); border-radius:.5rem; background:transparent; color:var(--st-text-color,#222); }
.omt-handwriting-canvas { width:100%; height:420px; display:block; background:white; border:1px solid rgba(128,128,128,.45); border-radius:.7rem; touch-action:none; box-sizing:border-box; }
.omt-handwriting-status { min-height:1rem; margin-top:.35rem; font-size:.78rem; opacity:.7; }
@media (max-width: 900px) { .omt-handwriting-canvas { height:360px; } }
@media (pointer: coarse) { .omt-handwriting-canvas { height:46vh; min-height:320px; max-height:520px; } }
"""

_HANDWRITING_JS = r"""
function dataUrlIsPresent(value) {
  return typeof value === 'string' && value.startsWith('data:image/png;base64,') && value.length > 100;
}

export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const canvas = parentElement.querySelector('.omt-handwriting-canvas');
  const clearButton = parentElement.querySelector('.omt-clear-pad');
  const status = parentElement.querySelector('.omt-handwriting-status');
  const ctx = canvas.getContext('2d', { alpha: false });
  let drawing = false;
  let hasInk = false;
  let lastX = 0;
  let lastY = 0;
  let restoreData = data?.image_data_url || '';

  const setCanvasSize = () => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const old = hasInk ? canvas.toDataURL('image/png') : restoreData;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 2.4;
    if (dataUrlIsPresent(old)) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, rect.width, rect.height);
        hasInk = true;
      };
      img.src = old;
    }
  };

  const point = (ev) => {
    const rect = canvas.getBoundingClientRect();
    return [ev.clientX - rect.left, ev.clientY - rect.top];
  };

  const emit = () => {
    if (!hasInk) {
      setStateValue('image_data_url', '');
      status.textContent = 'Canvas is blank';
      return;
    }
    setStateValue('image_data_url', canvas.toDataURL('image/png'));
    status.textContent = 'Handwriting saved';
  };

  const start = (ev) => {
    ev.preventDefault();
    drawing = true;
    hasInk = true;
    canvas.setPointerCapture?.(ev.pointerId);
    [lastX, lastY] = point(ev);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
  };

  const move = (ev) => {
    if (!drawing) return;
    ev.preventDefault();
    const [x, y] = point(ev);
    const pressure = ev.pressure && ev.pressure > 0 ? ev.pressure : 0.5;
    ctx.lineWidth = 1.8 + pressure * 2.4;
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x; lastY = y;
  };

  const end = (ev) => {
    if (!drawing) return;
    ev.preventDefault();
    drawing = false;
    ctx.closePath();
    emit();
  };

  canvas.onpointerdown = start;
  canvas.onpointermove = move;
  canvas.onpointerup = end;
  canvas.onpointercancel = end;
  canvas.onpointerleave = (ev) => { if (drawing && ev.buttons === 0) end(ev); };

  clearButton.onclick = () => {
    const rect = canvas.getBoundingClientRect();
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, rect.width, rect.height);
    hasInk = false;
    restoreData = '';
    emit();
  };

  setCanvasSize();
  const observer = new ResizeObserver(() => setCanvasSize());
  observer.observe(canvas);
  parentElement.__omtHandwritingObserver?.disconnect?.();
  parentElement.__omtHandwritingObserver = observer;
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



def targeted_practice_input(
    label: str,
    *,
    key_base: str,
    height: int = 150,
) -> tuple[str, str, str, list[UploadedAsset]]:
    """Collect targeted-practice working from equation editor, text, or iPad handwriting."""
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
        return "\n".join(working_lines), mode, "\n".join(used_ascii), []

    if mode == "Text working":
        value = st.text_area(
            label,
            key=f"{key_base}_text",
            height=height,
            placeholder="Show all parts and important reasoning steps, not only the final answer.",
        )
        return value, mode, value, []

    st.caption(
        "On iPad, write directly in the pad with Apple Pencil/finger, or use the camera/upload controls. "
        "For multi-part questions, label the working (a), (b), (c) where possible."
    )
    canvas_asset = handwriting_pad(key=f"{key_base}_handwriting")
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
    return text, "Handwritten working", "", assets

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
            with st.expander(f"{icon.get(item.status, '•')} Line {item.line_number}: {item.line}"):
                st.write(item.feedback)
    if result.strengths:
        st.markdown("**What is working**")
        for item in result.strengths:
            st.write(f"• {item}")
    if result.gaps:
        st.markdown("**What to repair**")
        for item in result.gaps:
            st.write(f"• {item}")
    st.markdown(f"**Next hint:** {result.next_hint}")


def render_ai_analysis(a: GeminiAnalysis) -> None:
    render_math_text(f"**Interpreted question:** {a.interpreted_question}")
    st.markdown(f"**Likely syllabus topic:** {a.likely_syllabus_topic}")
    render_math_text(f"**Method evidenced by the working:** {a.student_method}")

    if a.first_logic_break_step > 0:
        st.warning(f"First material logic break: step {a.first_logic_break_step}.")
        render_math_text(a.first_logic_break_explanation)
    else:
        st.success("No material logic break was identified.")
        if a.first_logic_break_explanation:
            render_math_text(a.first_logic_break_explanation)

    if a.steps:
        st.markdown("### Step-by-step reasoning")
        icons = {
            "correct": "✅",
            "partly_correct": "🟡",
            "incorrect": "❌",
            "unclear": "🔎",
            "unsupported": "⚪",
        }
        for step in a.steps:
            presentation_flag = bool(getattr(step, "presentation_error", False))
            title_icon = "⚠️" if presentation_flag else icons.get(step.status, "•")
            with st.expander(f"{title_icon} Step {step.line_number}"):
                st.markdown("**Student step**")
                render_mathio(step.student_step)
                if presentation_flag:
                    st.error("Presentation error — this written line does not form a clear mathematical statement.")
                    presentation_explanation = getattr(step, "presentation_error_explanation", "")
                    if presentation_explanation:
                        render_math_text(presentation_explanation)
                st.markdown(f"**What the step appears to be doing:** {step.logic_inferred}")
                st.write(f"**Issue type:** {step.issue_type}")
                st.write(step.feedback)
                supporting_math = list(getattr(step, "supporting_math", []) or [])
                for formula in supporting_math:
                    render_mathio(formula)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Strengths")
        for item in a.strengths:
            render_math_text(f"• {item}")
    with c2:
        st.markdown("### Main gap to repair")
        render_math_text(a.misconception_or_gap)
        render_math_text(f"**Diagnostic question:** {a.diagnostic_question}")

    st.markdown("### Guided correction")
    for i, hint in enumerate(a.hint_ladder, 1):
        with st.expander(f"Hint {i}"):
            render_math_text(hint)
    with st.expander("Reveal corrected path and answer"):
        for i, line in enumerate(a.corrected_path, 1):
            st.caption(f"Step {i}")
            render_mathio(line)
        st.markdown("**Final answer**")
        render_mathio(a.final_answer)


def render_practice_evaluation(e: PracticeEvaluation) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer", f"{e.answer_score}%")
    c2.metric("Reasoning", f"{e.reasoning_score}%")
    c3.metric("Mastery", e.mastery)
    render_math_text(e.summary)
    if e.first_logic_break_step > 0:
        st.warning(f"First logic break: step {e.first_logic_break_step}.")
        render_math_text(e.first_logic_break_explanation)
    if e.strengths:
        st.markdown("**Strengths**")
        for item in e.strengths:
            render_math_text(f"• {item}")
    if e.missing_or_incorrect_parts:
        st.warning("Parts still to complete correctly: " + ", ".join(e.missing_or_incorrect_parts))
    presentation_errors = list(getattr(e, "presentation_errors", []) or [])
    if presentation_errors:
        st.markdown("**Presentation errors**")
        for item in presentation_errors:
            st.warning(item)
    if e.gaps:
        st.markdown("**Gaps**")
        for item in e.gaps:
            render_math_text(f"• {item}")
    render_math_text(f"**Next hint:** {e.next_hint}")
    st.markdown("**Corrected next step**")
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
        st.success(f"Question feasibility: {message}.")
    elif result.status == "feasible_with_caveats":
        st.warning(f"Question feasibility: {message}.")
    else:
        st.error(f"Question feasibility: {message}. Student-working analysis is blocked until the question is clarified or corrected.")

    render_math_text(f"**Interpreted question:** {result.interpreted_question}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Answerability", result.answerability.replace("_", " ").title())
    c2.metric("Information", "Complete" if result.required_information_present else "Missing / unclear")
    c3.metric("Diagram / table", "Sufficient" if result.diagram_or_table_sufficient else "Needs attention")
    st.caption(
        f"Syllabus fit: {result.syllabus_fit.replace('_', ' ').title()} · "
        f"Feasibility confidence: {result.confidence.title()}"
    )

    render_feasibility_visual_map(result, question_files)

    if result.issues:
        st.markdown("**Question issues / warnings**")
        for issue_number, issue in enumerate(result.issues, 1):
            label = "Blocking" if issue.severity == "blocking" else "Warning"
            visual_note = ""
            if list(getattr(issue, "visual_regions", []) or []):
                visual_note = f" · see diagram callout {issue_number}"
            if issue.severity == "blocking":
                st.error(f"{label}{visual_note} — {issue.description}")
            else:
                st.warning(f"{label}{visual_note} — {issue.description}")
            if issue.suggested_fix:
                render_math_text(f"**Suggested clarification/fix:** {issue.suggested_fix}")

    if result.suspected_corrections:
        st.markdown("**Possible corrections to verify**")
        for item in result.suspected_corrections:
            render_math_text(f"• {item}")

    if result.action_needed:
        render_math_text(f"**Next action:** {result.action_needed}")


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
    st.title("🇸🇬 Math Reasoning Tutor")
    st.caption("Singapore O/N-Level tutor with Gemini online analysis plus a no-credit offline fallback.")

    track_label = st.selectbox("Exam track", list(TRACKS.keys()), index=0)
    tcode = track_code(track_label)

    st.markdown("---")
    st.markdown("**2026 syllabus mode**")
    if tcode == "O":
        st.caption("GCE O-Level Mathematics, syllabus 4052")
    elif tcode == "NA":
        st.caption("GCE N(A)-Level Mathematics Syllabus A, syllabus 4045")
    else:
        st.caption("GCE N(T)-Level Mathematics Syllabus T, syllabus 4046")

    st.markdown("---")
    st.markdown("**Gemini online mode**")
    explicit_key = st.text_input(
        "Gemini API key (optional here)",
        type="password",
        help="Prefer Streamlit Community Cloud Secrets with the name GEMINI_API_KEY. Leave this blank when the secret is configured.",
    )
    has_key = bool(get_api_key(explicit_key))
    if has_key:
        st.success("Gemini key detected")
    else:
        st.info("No Gemini key detected — offline modes still work")
    model = st.selectbox(
        "Gemini model",
        ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        index=0,
        help="Flash-Lite is intended for lower-cost/high-volume multimodal work. Free-tier availability and quotas depend on the Google account/project.",
    )

    st.markdown("---")
    st.caption(
        "Online mode sends the submitted question/work to Google Gemini. Offline practice and typed-algebra checking do not call Gemini."
    )
    if st.button("Reset learning session", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in {"session_id"}:
                del st.session_state[key]
        st.rerun()


st.title("Singapore O-Level & N-Level Mathematics Tutor")
st.write(
    "Use **Gemini online analysis** for uploaded/handwritten work and complex questions, with automatic question detection and a visual equation editor. "
    "If Gemini is unavailable or its free quota is reached, the **offline practice and algebra checker remain usable**."
)

ai_tab, practice_tab, own_tab, syllabus_tab, progress_tab = st.tabs(
    [
        "1 · Gemini analyse work",
        "2 · Offline practice",
        "3 · Offline algebra check",
        "4 · Syllabus coverage",
        "5 · Progress",
    ]
)

# ---------- Gemini online analysis ----------
with ai_tab:
    st.subheader("Analyse a student's question and working")
    st.write(
        "Type the question/working, upload images or PDFs, or combine both. Gemini is used for flexible interpretation; "
        "for supported typed algebra, the deterministic offline checker is also passed in as verification evidence."
    )
    st.warning(
        "Privacy: Gemini Free Tier terms can differ from paid API terms, including how submitted content may be used. "
        "Do not upload student names, NRICs, school identifiers, or other unnecessary personal data. Use this only where your school/guardian policy permits it."
    )

    q_text = st.text_area(
        "Question text",
        key="ai_question_text",
        height=120,
        placeholder="Type the question here, or leave blank if it is fully visible in the uploaded file.",
    )
    q_files = st.file_uploader(
        "Question image/PDF (optional)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key="ai_question_files",
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
        clear_ai_practice_state()
        st.session_state.pop("ai_detected_question_selector", None)

    w_text, w_input_mode, w_offline_text = working_input(
        "Student working",
        text_key="ai_working_text",
        format_key="ai_working_format",
        height=190,
        plain_placeholder="Type the student's steps here, use the equation editor, or leave blank if the working is in the uploaded file.",
    )
    w_files = st.file_uploader(
        "Student working image/PDF (optional)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key="ai_working_files",
    )

    consent = st.checkbox(
        "I understand that Gemini features send the selected inputs to Google's Gemini API.",
        key="gemini_consent",
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
        st.markdown("---")
        st.markdown("## Adaptive targeted practice")
        st.write(
            "Practice is now mastery-gated: the student works through **Near transfer → Varied context → Stretch** in order. "
            "If a category is not secure, the tutor stays there and generates more questions targeting the same gap."
        )
        st.caption(
            "Mastery rule: a secure first attempt unlocks the next category. After any miss in a category, "
            "the student must produce two consecutive secure attempts before moving on."
        )

        if st.session_state.ai_practice_current_question is None and not st.session_state.ai_practice_finished:
            initialize_ai_practice(analysis)

        stage_index = int(st.session_state.ai_practice_stage)
        completed = st.session_state.ai_practice_completed
        status_cols = st.columns(3)
        for i, kind in enumerate(PRACTICE_STAGES):
            if completed.get(kind):
                label = f"✅ {kind}"
                detail = "Mastered"
            elif not st.session_state.ai_practice_finished and i == stage_index:
                label = f"🟠 {kind}"
                detail = "Current focus"
            else:
                label = f"🔒 {kind}"
                detail = "Locked"
            status_cols[i].markdown(f"**{label}**")
            status_cols[i].caption(detail)

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

            st.markdown(f"### Current focus: {kind}")
            if misses:
                st.warning(
                    f"This category remains active because the student has had {misses} non-secure attempt(s). "
                    f"Current recovery streak: {streak}/2 secure attempts."
                )
            with st.container(border=True):
                render_math_text(f"**{pq.question}**")
            render_math_text(f"**Target skill:** {pq.target_skill}")
            required_parts = required_parts_for_question(pq)
            if required_parts != ["whole question"]:
                st.caption("All parts required for mastery: " + ", ".join(required_parts))
            render_math_text(pq.why_this_tests_understanding)
            with st.expander("Practice hints"):
                for i, hint in enumerate(pq.hints, 1):
                    render_math_text(f"**Hint {i}:** {hint}")

            working_key = f"ai_practice_working_{stage_index}_{st.session_state.ai_practice_question_version}"
            attempt, practice_input_mode, _practice_offline_text, practice_assets = targeted_practice_input(
                f"Student working for {kind}",
                key_base=working_key,
                height=150,
            )

            if st.button(f"Check {kind} reasoning", key=f"ai_practice_check_{stage_index}_{st.session_state.ai_practice_question_version}", type="primary"):
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
                            f"Stay on {kind}. The next category remains locked until this reasoning is repaired."
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
            st.markdown(f"**Hint {i+1}:** {question.hints[i]}")
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
