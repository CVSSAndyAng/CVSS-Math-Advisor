from __future__ import annotations

import json
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from gemini_service import (
    DEFAULT_MODEL,
    GeminiAnalysis,
    GeminiTutorError,
    PracticeEvaluation,
    TargetedPracticeQuestion,
    UploadedAsset,
    analyze_submission,
    evaluate_practice_attempt,
    get_api_key,
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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1220px;}
.soft-card {border: 1px solid rgba(128,128,128,.28); border-radius: .8rem; padding: 1rem 1.1rem; margin: .4rem 0 1rem 0;}
.small {font-size:.88rem; opacity:.82;}
.ok {background: rgba(0,160,90,.08); border-radius:.6rem; padding:.65rem .8rem;}
.warn {background: rgba(255,170,0,.08); border-radius:.6rem; padding:.65rem .8rem;}
</style>
""",
    unsafe_allow_html=True,
)

MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

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
    st.session_state.pop("practice_working", None)


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
    st.markdown("## Gemini reasoning analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Judgement", a.overall_judgement.replace("_", " ").title())
    c2.metric("Confidence", a.confidence.title())
    c3.metric("Human review", "Recommended" if a.needs_human_review else "Not flagged")

    st.markdown(f"**Interpreted question:** {a.interpreted_question}")
    st.markdown(f"**Likely syllabus topic:** {a.likely_syllabus_topic}")
    st.markdown(f"**Method evidenced by the working:** {a.student_method}")

    if a.first_logic_break_step > 0:
        st.warning(
            f"First material logic break: step {a.first_logic_break_step}. "
            f"{a.first_logic_break_explanation}"
        )
    else:
        st.success(a.first_logic_break_explanation or "No material logic break was identified.")

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
            title = f"{icons.get(step.status, '•')} Step {step.line_number}: {step.student_step}"
            with st.expander(title):
                st.write(f"**What the step appears to be doing:** {step.logic_inferred}")
                st.write(f"**Issue type:** {step.issue_type}")
                st.write(step.feedback)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Strengths")
        for item in a.strengths:
            st.write(f"• {item}")
    with c2:
        st.markdown("### Main gap to repair")
        st.write(a.misconception_or_gap)
        st.markdown(f"**Diagnostic question:** {a.diagnostic_question}")

    st.markdown("### Guided correction")
    for i, hint in enumerate(a.hint_ladder, 1):
        with st.expander(f"Hint {i}"):
            st.write(hint)
    with st.expander("Reveal corrected path and answer"):
        for i, line in enumerate(a.corrected_path, 1):
            st.write(f"{i}. {line}")
        st.markdown(f"**Final answer:** {a.final_answer}")


def render_practice_evaluation(e: PracticeEvaluation) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer", f"{e.answer_score}%")
    c2.metric("Reasoning", f"{e.reasoning_score}%")
    c3.metric("Mastery", e.mastery)
    st.info(e.summary)
    if e.first_logic_break_step > 0:
        st.warning(
            f"First logic break: step {e.first_logic_break_step}. "
            f"{e.first_logic_break_explanation}"
        )
    if e.strengths:
        st.markdown("**Strengths:** " + "; ".join(e.strengths))
    if e.gaps:
        st.markdown("**Gaps:** " + "; ".join(e.gaps))
    st.markdown(f"**Next hint:** {e.next_hint}")
    st.markdown(f"**Corrected next step:** {e.corrected_next_step}")
    st.caption(f"Gemini confidence: {e.confidence}")


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
    "Use **Gemini online analysis** for uploaded/handwritten work and complex questions. "
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
    w_text = st.text_area(
        "Student working",
        key="ai_working_text",
        height=190,
        placeholder="Type the student's steps here, or leave blank if the working is in the uploaded file.",
    )
    w_files = st.file_uploader(
        "Student working image/PDF (optional)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key="ai_working_files",
    )

    consent = st.checkbox(
        "I understand that clicking Analyse with Gemini sends these inputs to Google's Gemini API.",
        key="gemini_consent",
    )

    if st.button("Analyse with Gemini", type="primary", use_container_width=True):
        st.session_state.ai_analysis = None
        st.session_state.ai_error = ""
        st.session_state.ai_fallback_result = None
        for key in list(st.session_state.keys()):
            if str(key).startswith("ai_practice_eval_"):
                del st.session_state[key]
        if not consent:
            st.error("Confirm the Gemini data-sharing acknowledgement before sending the submission.")
        else:
            evidence, offline_result = offline_evidence_for(q_text, w_text)
            try:
                assets_q = uploaded_assets(q_files)
                assets_w = uploaded_assets(w_files)
                with st.spinner("Gemini is checking the mathematics and the student's reasoning..."):
                    analysis = analyze_submission(
                        track_label=track_label,
                        question_text=q_text,
                        working_text=w_text,
                        question_assets=assets_q,
                        working_assets=assets_w,
                        offline_evidence=evidence,
                        api_key=explicit_key,
                        model=model,
                    )
                st.session_state.ai_analysis = analysis
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
        st.markdown("## Targeted practice to test understanding")
        st.write("These three questions are generated from the diagnosed gap, not just from the final answer.")

        for idx, pq in enumerate(analysis.practice_questions):
            st.markdown(f"### {idx + 1}. {pq.kind}")
            st.markdown(f'<div class="soft-card"><strong>{pq.question}</strong></div>', unsafe_allow_html=True)
            st.caption(f"Target skill: {pq.target_skill}")
            st.write(pq.why_this_tests_understanding)
            with st.expander("Practice hints"):
                for i, hint in enumerate(pq.hints, 1):
                    st.write(f"**Hint {i}:** {hint}")
            attempt = st.text_area(
                f"Student working for {pq.kind}",
                key=f"ai_practice_working_{idx}",
                height=140,
            )
            if st.button(f"Check {pq.kind} reasoning", key=f"ai_practice_button_{idx}"):
                try:
                    with st.spinner("Checking the practice reasoning..."):
                        evaluation = evaluate_practice_attempt(
                            track_label=track_label,
                            practice_question=pq,
                            student_working=attempt,
                            original_gap=analysis.misconception_or_gap,
                            api_key=explicit_key,
                            model=model,
                        )
                    st.session_state[f"ai_practice_eval_{idx}"] = evaluation
                    record_ai_practice_history(tcode, pq, evaluation)
                    st.rerun()
                except GeminiTutorError as exc:
                    st.error(str(exc))
            evaluation = st.session_state.get(f"ai_practice_eval_{idx}")
            if evaluation is not None:
                render_practice_evaluation(evaluation)
            with st.expander("Reveal AI reference answer and worked solution"):
                st.markdown(f"**Answer:** {pq.answer}")
                for i, line in enumerate(pq.worked_solution, 1):
                    st.write(f"{i}. {line}")

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

        working = st.text_area(
            "Your working and answer",
            key="practice_working",
            height=190,
            placeholder="Show the important steps, one line at a time where possible.",
        )
        if st.button("Check my reasoning offline", type="primary", use_container_width=True):
            if not working.strip():
                st.error("Enter your working and answer first.")
            else:
                result = evaluate_attempt(question, working)
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
    own_w = st.text_area("Student working", key="own_working", height=190, placeholder="3(x + 2) = 18\n3x + 6 = 18\n3x = 12\nx = 4")

    if st.button("Check typed algebra", type="primary"):
        if not own_q.strip() or not own_w.strip():
            st.error("Enter both the question and the student's working.")
        else:
            try:
                res = analyze_own_algebra_question(own_q, own_w)
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
