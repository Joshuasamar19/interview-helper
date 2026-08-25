import queue

import numpy as np
import pydub
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Interview Helper", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


# ---------- Session state ----------
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

# ---------- Header ----------
st.title("🎙️ Interview Helper")
st.caption("Practice answering interview questions out loud. Record only with everyone's permission.")

col1, col2 = st.columns([2, 1])
with col1:
    role = st.text_input("Target role", "MEP Designer")
with col2:
    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=1,
        help="Bigger = more accurate but slower to run.",
    )

st.divider()

# ---------- Question input ----------
st.subheader("1. Get a question")

typed_question = st.text_area(
    "Type an interview question",
    value=st.session_state.question,
    placeholder="e.g. Tell me about a time you resolved a conflict with a contractor.",
)
if typed_question != st.session_state.question:
    st.session_state.question = typed_question

st.write("**Or record it with your microphone:**")

webrtc_ctx = webrtc_streamer(
    key="interview-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"video": False, "audio": True},
)

status_indicator = st.empty()

if "sound_chunk" not in st.session_state:
    st.session_state.sound_chunk = pydub.AudioSegment.empty()

if webrtc_ctx.state.playing:
    status_indicator.info("🔴 Recording — speak your question, then click the widget's stop button above.")

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                break

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                st.session_state.sound_chunk += sound
        else:
            break

elif len(st.session_state.sound_chunk) > 0:
    status_indicator.empty()
    with st.spinner("Transcribing your recording..."):
        segment = st.session_state.sound_chunk.set_frame_rate(16000).set_channels(1)
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(segment.array_type).max  # normalize to [-1, 1]

        model = load_model(model_size)
        segments, info = model.transcribe(samples, language="en", vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()

    st.session_state.sound_chunk = pydub.AudioSegment.empty()

    if not text:
        st.error("I couldn't make out any speech. Try recording again, or type the question instead.")
    else:
        st.session_state.question = text
        st.session_state.last_transcript = text
        st.success("Heard you loud and clear:")
        st.rerun()

if st.session_state.last_transcript:
    st.caption(f"Last transcript: \u201c{st.session_state.last_transcript}\u201d")

st.divider()

# ---------- Talking points ----------
st.subheader("2. Build your answer")

STRUCTURE_HINTS = {
    "conflict": "Situation → who disagreed and why → Action you took to resolve it → Result/relationship outcome.",
    "mistake": "Situation → what went wrong and why → Action to fix it and prevent recurrence → Result/lesson learned.",
    "challenge": "Situation → what made it hard → Action you took → Result and what you'd do differently.",
    "team": "Situation → your role on the team → Action that helped the team succeed → Result for the project/team.",
    "deadline": "Situation → the time pressure → Action you took to prioritize → Result: did you hit it, and how.",
}


def pick_structure_hint(q: str) -> str:
    q_lower = q.lower()
    for keyword, hint in STRUCTURE_HINTS.items():
        if keyword in q_lower:
            return hint
    return "Situation → the specific context → Action you personally took → Result, ideally with a number or outcome."


if st.button("✨ Generate talking points", type="primary", use_container_width=True):
    q = st.session_state.question.strip()
    if not q:
        st.warning("Type or record an interview question first.")
    else:
        st.markdown(f"**Question:** {q}")
        st.markdown("##### Suggested structure (STAR)")
        st.write(pick_structure_hint(q))

        st.markdown("##### Talking points")
        st.write("1. Answer the question directly in your first sentence — don't bury the lead.")
        st.write(f"2. Give one concrete, real example from MEP design work relevant to a {role} role.")
        st.write("3. Quantify the result if you can (time saved, cost avoided, clashes resolved, coordination improved).")
        st.write(f"4. Close by explicitly tying the example back to what a {role} needs to deliver day-to-day.")

        with st.expander("💡 Extra tip"):
            st.write(
                "Keep your spoken answer to about 60–90 seconds. Practice it out loud using the recorder above, "
                "then play back your own transcript to check pacing and clarity."
            )
