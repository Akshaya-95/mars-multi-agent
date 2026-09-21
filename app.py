"""
MARS — Multi-Agent Research System (Streamlit)
Six AI agents research any topic and produce a cited report.
"""
import os, json, time, warnings
import streamlit as st
from google import genai
from ddgs import DDGS

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is not configured. Add it in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ---------- THROTTLE ----------
_last = [0.0]
def _throttle():
    gap = time.time() - _last[0]
    if gap < 13.0:
        time.sleep(13.0 - gap)
    _last[0] = time.time()

TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")

def llm(prompt, system="", json_mode=False):
    cfg = {}
    if system: cfg["system_instruction"] = system
    if json_mode: cfg["response_mime_type"] = "application/json"
    for attempt in range(6):
        _throttle()
        try:
            r = client.models.generate_content(
                model=MODEL_NAME, contents=prompt, config=cfg or None
            )
            return (r.text or "").strip()
        except Exception as e:
            if any(k in str(e) for k in TRANSIENT):
                time.sleep(20 + attempt * 15)
                continue
            return ""
    return ""

def search_web(q, n=3):
    try:
        with DDGS() as d:
            hits = list(d.text(q, max_results=n))
        return [{"title": h.get("title",""), "url": h.get("href") or h.get("url",""),
                 "snippet": h.get("body","")} for h in hits]
    except Exception:
        return []

def plan(query):
    raw = llm(
        f"Break this into AT MOST 3 research sub-questions. "
        f'Return JSON: {{"q": ["...", "...", "..."]}}\n\nQuery: {query}',
        system="You are a research planner. Return strict JSON only.",
        json_mode=True,
    )
    try:
        return json.loads(raw).get("q", [])[:3] or [query]
    except Exception:
        return [query]

def research(query):
    sub_qs = plan(query)
    sections = []
    for i, sq in enumerate(sub_qs, 1):
        srcs = search_web(sq, n=3)
        if not srcs:
            continue
        ctx = "\n\n".join(f"### {s['title']}\nURL: {s['url']}\n{s['snippet']}" for s in srcs)
        summary = llm(
            f"Summarize these sources for: {sq}\n\n5 bullets + 1 key fact. Cite URLs.\n\n{ctx}",
            system="You are a research summarizer."
        )
        if summary:
            sections.append(f"## {sq}\n\n{summary}")

    if not sections:
        return {"report": "❌ No data collected. Please try again.", "critique": {}}

    report = llm(
        f"Write a markdown research report answering: {query}\n\n"
        f"Use these sections. Add # Title, ## Executive Summary, ## Findings, ## References.\n\n"
        + "\n\n---\n\n".join(sections),
        system="You are a report synthesizer. Use clean markdown."
    )

    critique_raw = llm(
        f"Evaluate this report. Query: {query}\n\nReport:\n{report}\n\n"
        f'Return JSON: {{"score": int, "critique": str, "recommendations": [str]}}',
        system="You are a research critic. Return strict JSON.",
        json_mode=True,
    )
    try:
        critique = json.loads(critique_raw)
    except Exception:
        critique = {"score": 0, "critique": "parse failed", "recommendations": []}

    return {"report": report, "critique": critique}


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="MARS — Multi-Agent Research System", page_icon="🚀", layout="centered")

st.title("🚀 MARS — Multi-Agent Research System")
st.caption("Six AI agents research any topic and produce a cited report. ⏱️ ~3 minutes per query.")

query = st.text_area(
    "Research Topic",
    placeholder="e.g., What is quantum computing?",
    height=100,
)

if st.button("🚀 Run Research", type="primary"):
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.status("Running MARS pipeline...", expanded=True) as status:
            st.write("Planning research...")
            sub_qs = plan(query)
            st.write(f"Split into {len(sub_qs)} sub-questions")

            sections = []
            for i, sq in enumerate(sub_qs, 1):
                st.write(f"Step {i}/{len(sub_qs)}: {sq[:60]}...")
                srcs = search_web(sq, n=3)
                if not srcs:
                    st.write("   → no sources, skipping")
                    continue
                ctx = "\n\n".join(f"### {s['title']}\nURL: {s['url']}\n{s['snippet']}" for s in srcs)
                summary = llm(
                    f"Summarize these sources for: {sq}\n\n5 bullets + 1 key fact. Cite URLs.\n\n{ctx}",
                    system="You are a research summarizer."
                )
                if summary:
                    sections.append(f"## {sq}\n\n{summary}")
                    st.write(f"   → {len(srcs)} sources, summary {len(summary)} chars")

            if not sections:
                status.update(label="No data collected", state="error")
                st.error("No data collected. The model may be temporarily unavailable. Please try again.")
            else:
                st.write("Writing final report...")
                report = llm(
                    f"Write a markdown research report answering: {query}\n\n"
                    f"Use these sections. Add # Title, ## Executive Summary, ## Findings, ## References.\n\n"
                    + "\n\n---\n\n".join(sections),
                    system="You are a report synthesizer. Use clean markdown."
                )

                st.write("Evaluating quality...")
                critique_raw = llm(
                    f"Evaluate this report. Query: {query}\n\nReport:\n{report}\n\n"
                    f'Return JSON: {{"score": int, "critique": str, "recommendations": [str]}}',
                    system="You are a research critic. Return strict JSON.",
                    json_mode=True,
                )
                try:
                    critique = json.loads(critique_raw)
                except Exception:
                    critique = {"score": 0, "critique": "parse failed", "recommendations": []}

                status.update(label="Done!", state="complete")

                st.markdown(report)

                st.divider()
                score = critique.get("score", "N/A")
                st.subheader(f"⚖️ Critic Score: {score}/100")
                if critique.get("critique"):
                    st.markdown(f"**Feedback:** {critique['critique']}")
                if critique.get("recommendations"):
                    st.markdown("**Suggestions:**")
                    for r in critique["recommendations"]:
                        st.markdown(f"- {r}")
