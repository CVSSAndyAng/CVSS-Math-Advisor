# Singapore O/N-Level Math Reasoning Tutor — Gemini + Offline

A Streamlit tutor built for Singapore secondary mathematics. It combines:

1. **Gemini online analysis** for typed questions, images, handwritten workings and PDFs.
2. **Deterministic offline checking** with Python/SymPy for supported algebra.
3. **Offline syllabus-generated practice** across O-Level / N(A)-Level / N(T)-Level topic areas.
4. **Targeted transfer questions** generated from the student's diagnosed gap: Near transfer, Varied context and Stretch.
5. **Automatic fallback**: if Gemini is unavailable, its quota is reached, or no key is configured, the offline tabs continue to work.

The tutor focuses on the reasoning visible in the student's work. It does not claim to read hidden thoughts or make judgments about intelligence, motivation, personality or learning/medical diagnoses.

## Curriculum tracks

The interface keeps the 2026 examination labels used by the offline curriculum map:

- GCE O-Level Mathematics — syllabus 4052
- GCE N(A)-Level Mathematics Syllabus A — syllabus 4045
- GCE N(T)-Level Mathematics Syllabus T — syllabus 4046

This is an educational tool, not an official SEAB or MOE product.

## Streamlit Community Cloud: fastest setup

### 1. Import the ZIP

Create a new Streamlit Community Cloud app and import this ZIP. The included `.replit` file starts Streamlit on port 3000.

### 2. Run without a key first

Press **Run**. The offline practice and offline algebra checker work with no API key.

### 3. Create a Gemini API key

Use Google AI Studio:

https://aistudio.google.com/apikey

Google's Gemini API documentation:

https://ai.google.dev/gemini-api/docs/api-key

### 4. Store the key in Streamlit Community Cloud Secrets

In Streamlit Community Cloud, open **Tools → Secrets** and add:

```text
Key:   GEMINI_API_KEY
Value: paste_your_Gemini_key_here
```

Do not put the key in `app.py`, screenshots, chat messages or a public repository.

Restart the Streamlit Community Cloud app after adding the secret.

### 5. Use Gemini online analysis

Open **1 · Gemini analyse work**. A student can:

- type the question;
- upload a PNG/JPEG/WebP question image;
- upload a PDF;
- type their working;
- upload handwritten working as an image/PDF; or
- combine typed text and uploads.

After analysis, the tutor displays:

- interpreted question;
- likely syllabus topic;
- method evidenced by the work;
- step-by-step reasoning check;
- earliest material logic break;
- misconception/gap;
- diagnostic question;
- progressive hints;
- corrected path and final answer behind a reveal control; and
- three targeted practice questions.

Each targeted practice attempt can itself be checked by Gemini for both answer accuracy and reasoning quality.

## Gemini model

Default:

```text
gemini-3.5-flash-lite
```

The app also exposes `gemini-3.1-flash-lite` as a fallback model choice.

Google currently describes Gemini 3.5 Flash-Lite as a low-latency, cost-effective multimodal model supporting text, image and PDF input. Free-tier eligibility and active quotas depend on the Google account/project and can change. Check Google AI Studio for the active limits on your project.

Model docs:

https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite

Rate limits:

https://ai.google.dev/gemini-api/docs/rate-limits

Pricing/free tier:

https://ai.google.dev/gemini-api/docs/pricing

## Privacy note for student work

Online mode sends submitted content to Google's Gemini API. The app displays a consent acknowledgement before calling Gemini. Do not include unnecessary personal identifiers such as a student's full name, NRIC, school identifier, address or contact details.

Google's free-tier data terms may differ from paid-tier data terms. Review the current Gemini API terms and your school's policies before using online mode with real student data.

Offline practice and offline algebra checking do **not** call Gemini.

## Hybrid verification design

For a supported typed one-variable equation, the app first runs the deterministic SymPy checker. It passes the result to Gemini as supporting evidence while instructing Gemini to independently verify the mathematics. This helps catch obvious algebraic equivalence breaks while retaining Gemini's flexibility for explanations and alternative methods.

If Gemini returns a quota/rate-limit/network/authentication error:

- the offline tabs remain fully available;
- for typed one-variable algebra that the deterministic engine understands, the app automatically displays the offline result as a fallback.

## Run locally

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

If using Gemini locally, set `GEMINI_API_KEY` in your environment or `.env` loader of choice. The app itself reads environment variables and Streamlit Community Cloud Secrets directly.

## Project structure

```text
singapore_math_gemini_hybrid/
├── app.py
├── ai_tutor/
│   ├── offline_engine.py
│   └── gemini_service.py
├── tests/
│   ├── test_offline_engine.py
│   └── test_gemini_service.py
├── .replit
├── .streamlit/config.toml
├── requirements.txt
├── README.md
├── REPLIT_START_HERE.md
├── ARCHITECTURE.md
└── VALIDATION.md
```

## Important limitations

- AI analysis can be wrong, especially with poor handwriting, ambiguous diagrams or incomplete working.
- The online model is not a substitute for an official marking scheme or teacher in high-stakes decisions.
- Offline free-form checking is intentionally narrower than Gemini. Its strongest free-form capability is typed one-variable equation equivalence checking.
- Offline generated practice has broader topic coverage because the engine knows the exact mathematical structure of the questions it generates.
- Generated questions are original templates, not past-year SEAB questions.


## GitHub browser-friendly flat layout

This deployment copy intentionally keeps all runtime Python files at the repository root so it can be uploaded through GitHub's **Choose your files** dialog without uploading folders.

Required files:
- `app.py`
- `gemini_service.py`
- `offline_engine.py`
- `requirements.txt`

Optional documentation:
- `README.md`

For Streamlit Community Cloud, set `GEMINI_API_KEY` in **App settings → Secrets**.
