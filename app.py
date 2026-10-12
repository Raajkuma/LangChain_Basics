"""
breakfast.py - 5 healthy breakfast ideas via a LangChain chain + Gemini, streamed in a Streamlit UI.

Setup:
    pip install streamlit langchain-google-genai
    export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

Run:
    streamlit run breakfast.py
"""

import os

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.1-flash-lite"

# Don't let the model decide what "healthy" means -- tell it. This is the default
# for the {criteria} variable; the user can edit it in the sidebar.
DEFAULT_CRITERIA = (
    "under 400 calories, at least 10 g protein or 5 g fiber, "
    "no added sugar, built mostly from whole foods"
)

# {criteria} is a template variable, filled in at invoke/stream time.
PROMPT = ChatPromptTemplate.from_template(
    "Give me exactly 5 healthy breakfast ideas.\n"
    "'Healthy' here means: {criteria}.\n"
    "Format rules: a numbered list from 1 to 5, one line per idea, "
    "no preamble, no closing remarks, nothing else."
)


def chunk_text(chunk) -> str:
    """Text of one streamed AIMessageChunk (content may be a str or a list of parts)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)


def ideas_as_lines(text: str) -> list[str]:
    return [line.strip().lower() for line in text.splitlines() if line.strip()]


# ----------------------------- UI -----------------------------

st.set_page_config(page_title="Healthy Breakfasts", page_icon="🥣", layout="centered")
st.title("🥣 5 Healthy Breakfast Ideas")
st.caption("LangChain chain + Gemini, streamed. Same ask, run it twice, and see what changes.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Set the `GEMINI_API_KEY` environment variable, then restart the app.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.0, 0.1,
        help="0 = most repeatable output. Higher = more varied ideas each run.",
    )
    criteria = st.text_area("What does 'healthy' mean?", DEFAULT_CRITERIA, height=120)
    show_raw = st.checkbox(
        "Lesson mode: show the raw message object",
        value=False,
        help="Streams `prompt | llm` (no parser) and shows the AIMessage it adds up to.",
    )

if "runs" not in st.session_state:
    st.session_state.runs = []

with st.expander("Prompt being sent"):
    st.code(PROMPT.format(criteria=criteria), language="text")

if st.button("Get breakfast ideas", type="primary"):
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=temperature, google_api_key=api_key)

    raw_chain = PROMPT | llm                       # returns AIMessage(Chunk)s
    chain = raw_chain | StrOutputParser()          # returns plain str chunks
    inputs = {"criteria": criteria}
    raw_repr = None

    try:
        if show_raw:
            # Stream the un-parsed chain, collect chunks into one message, show text live.
            collected = {"msg": None}

            def stream_raw():
                for chunk in raw_chain.stream(inputs):
                    collected["msg"] = chunk if collected["msg"] is None else collected["msg"] + chunk
                    yield chunk_text(chunk)

            text = st.write_stream(stream_raw())
            raw_repr = repr(collected["msg"])
        else:
            text = st.write_stream(chain.stream(inputs))   # str chunks, shown as they arrive
    except Exception as e:
        st.error(f"Model call failed: {e}")
        st.stop()

    st.session_state.runs.append({"temp": temperature, "text": text.strip(), "raw": raw_repr})

runs = st.session_state.runs
if runs:
    latest = runs[-1]

    if latest["raw"]:
        st.subheader("What the stream adds up to (an AIMessage, not a string)")
        st.code(latest["raw"], language="text")
        st.caption("The part we want is `.content`. In normal mode, StrOutputParser does this for you.")

    if len(runs) >= 2:
        same = ideas_as_lines(runs[-1]["text"]) == ideas_as_lines(runs[-2]["text"])
        st.success("Identical to the previous run.") if same else st.warning("Different from the previous run.")

    with st.expander(f"Run history ({len(runs)})"):
        for i, r in enumerate(reversed(runs)):
            st.markdown(f"**Run {len(runs) - i}** (temperature {r['temp']})")
            st.text(r["text"])

    if st.button("Clear history"):
        st.session_state.runs = []
        st.rerun()