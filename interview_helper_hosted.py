import os
import queue
import threading
import time

import av
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import WebRtcMode, webrtc_streamer

try:
    import anthropic
except Exception:
    anthropic = None

st.set_page_config(page_title="Live Transcriber", page_icon="🎙️", layout="centered")

WHISPER_SR = 16000
STEP_SECONDS = 0.8           # re-transcribe the growing utterance this often
SILENCE_RMS = 0.0015         # below this = "silence"
SILENCE_HANG = 0.8           # seconds of silence that ends an utterance
MAX_UTTERANCE_SEC = 14.0     # force-finalize very long utterances

POLISH_SYSTEM = (
    "You are a real-time transcription editor. The user sends you a raw "
    "speech-to-text fragment that may have grammar mistakes, missing "
    "punctuation, wrong capitalization, or misheard words. Rewrite it into "
    "clean, correct, natural English. Fix obvious transcription errors from "
    "context. Do not add new information, do not answer questions in the text, "
    "do not add commentary. Output ONLY the corrected text — no quotes, no "
    "preamble, no explanation."
)


@st.cache_resource(show_spinner=False)
def load_model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def require_password():
    """Optional password gate. Active only when APP_PASSWORD is set (env var or
    Streamlit secret). When unset, the app is unlocked."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        try:
            if "APP_PASSWORD" in st.secrets:
                expected = str(st.secrets["APP_PASSWORD"])
        except Exception:
            pass
    if not expected:
        return
    if st.session_state.get("auth_ok"):
        return
    st.title("🔒 Private Transcriber")
    st.caption("This app is password protected.")
    pw = st.text_input("Enter password", type="password")
    if pw == expected:
        st.session_state.auth_ok = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_shared():
    """Shared state that survives reruns and is visible to the WebRTC audio
    callback and the background worker threads. Created exactly once."""
    return {
        "audio_q": queue.Queue(),     # float32 mono 16k samples from the browser mic
        "level_q": queue.Queue(),     # floats: RMS volume for the meter
        "result_q": queue.Queue(),    # {"kind": "partial"|"final"|"polished", "id": int, "text": str}
        "stop_event": threading.Event(),
        "workers_started": False,
        "resampler": None,
    }


def resolve_api_key(entered: str) -> str:
    if entered and entered.strip():
        return entered.strip()
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def polish_text(client, model: str, raw: str) -> str:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=POLISH_SYSTEM,
            messages=[{"role": "user", "content": raw}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or raw
    except Exception:
        return raw


def _stream_thread(whisper_model, shared, polish_q, use_polish):
    """Live-caption engine: accumulate mic audio, re-transcribe every
    STEP_SECONDS to show growing PARTIAL text, finalize on a pause."""
    stop_event = shared["stop_event"]
    audio_q = shared["audio_q"]
    result_q = shared["result_q"]

    utterance = np.zeros((0,), dtype=np.float32)
    silent_time = 0.0
    last_tx = time.time()
    line_id = 0
    last_partial = ""

    def transcribe(buf):
        segs, _ = whisper_model.transcribe(
            buf, language="en", beam_size=1, best_of=1, temperature=0,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    while not stop_event.is_set():
        collected = []
        try:
            while True:
                collected.append(audio_q.get_nowait())
        except queue.Empty:
            pass

        if collected:
            new = np.concatenate(collected)
            utterance = np.concatenate([utterance, new])
            new_rms = float(np.sqrt(np.mean(new ** 2))) if len(new) else 0.0
            if new_rms < SILENCE_RMS:
                silent_time += len(new) / WHISPER_SR
            else:
                silent_time = 0.0

        has_speech = len(utterance) > 0 and float(np.sqrt(np.mean(utterance ** 2))) > SILENCE_RMS
        now = time.time()

        if not has_speech:
            keep = int(WHISPER_SR * 0.3)
            if len(utterance) > keep:
                utterance = utterance[-keep:]
            time.sleep(0.03)
            continue

        dur = len(utterance) / WHISPER_SR

        if now - last_tx >= STEP_SECONDS:
            text = transcribe(utterance)
            if text and text != last_partial:
                last_partial = text
                result_q.put({"kind": "partial", "id": line_id, "text": text})
            last_tx = now

        if silent_time >= SILENCE_HANG or dur >= MAX_UTTERANCE_SEC:
            text = transcribe(utterance)
            if text:
                result_q.put({"kind": "final", "id": line_id, "text": text})
                if use_polish:
                    polish_q.put((line_id, text))
                line_id += 1
            utterance = np.zeros((0,), dtype=np.float32)
            silent_time = 0.0
            last_partial = ""
            last_tx = now

        time.sleep(0.03)


def _polish_thread(client, model, shared, polish_q):
    stop_event = shared["stop_event"]
    result_q = shared["result_q"]
    while not stop_event.is_set():
        try:
            line_id, raw = polish_q.get(timeout=0.2)
        except queue.Empty:
            continue
        polished = polish_text(client, model, raw)
        result_q.put({"kind": "polished", "id": line_id, "text": polished})


# ---------- shared + session state ----------
shared = get_shared()

for key, val in [("lines", []), ("line_ids", []), ("partial", ""), ("current", ""), ("level", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- password gate (only if APP_PASSWORD is set) ----------
require_password()

# ---------- UI ----------
st.title("🎙️ Live Transcriber")
st.caption("Live captions from your microphone, polished by Claude. Works in any browser.")

with st.sidebar:
    st.header("Settings")
    ai_polish = st.toggle("✨ Claude AI polish", value=True,
                          help="Live text shows instantly; finished sentences get "
                               "cleaned up by Claude a moment later.")
    ai_model = st.selectbox("Claude model", ["claude-haiku-4-5", "claude-opus-4-8"], index=0,
                            help="Haiku = fastest / cheapest. Opus = best quality.")
    api_key_input = st.text_input(
        "Anthropic API key", type="password",
        help="On Streamlit Cloud, set ANTHROPIC_API_KEY in Secrets instead of typing it here.",
    )
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=0,
                                help="tiny = fastest (best for the free cloud tier).")

api_key = resolve_api_key(api_key_input)

if ai_polish and anthropic is None:
    st.warning("The `anthropic` package isn't installed, so polishing is off. Add it to requirements.txt.")
    ai_polish = False
if ai_polish and not api_key:
    st.info("Enter an Anthropic API key (or set it in Secrets) to enable Claude polish. "
            "Transcription still works without it.")

# ---------- WebRTC audio capture ----------


def audio_frame_callback(frame: av.AudioFrame):
    """Runs in WebRTC's thread. Resample each mic frame to mono 16k float32
    and hand it to the transcription pipeline."""
    if shared["resampler"] is None:
        shared["resampler"] = av.AudioResampler(format="s16", layout="mono", rate=WHISPER_SR)
    try:
        for rf in shared["resampler"].resample(frame):
            arr = rf.to_ndarray().flatten().astype(np.float32) / 32768.0
            shared["audio_q"].put(arr)
            if len(arr):
                shared["level_q"].put(float(np.sqrt(np.mean(arr ** 2))))
    except Exception:
        pass
    return frame


# STUN finds a direct path; TURN relays audio when firewalls block a direct
# connection (common on the cloud). The free openrelay servers cover most cases.
RTC_CONFIG = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:openrelay.metered.ca:80"],
         "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443"],
         "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
         "username": "openrelayproject", "credential": "openrelayproject"},
    ]
}

webrtc_ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration=RTC_CONFIG,
)

st.divider()

if st.button("🗑 Clear transcript", use_container_width=True):
    st.session_state.lines = []
    st.session_state.line_ids = []
    st.session_state.partial = ""
    st.session_state.current = ""

status_box = st.empty()
level_box = st.empty()
caption_box = st.empty()
st.markdown("#### Full transcript")
transcript_box = st.empty()

# ---------- start / stop workers with the stream ----------
if webrtc_ctx.state.playing and not shared["workers_started"]:
    shared["stop_event"].clear()
    while not shared["audio_q"].empty():
        try:
            shared["audio_q"].get_nowait()
        except queue.Empty:
            break

    model = load_model(whisper_size)
    polish_q = queue.Queue()

    threads = [threading.Thread(
        target=_stream_thread, args=(model, shared, polish_q, ai_polish), daemon=True)]
    if ai_polish and api_key and anthropic is not None:
        client = anthropic.Anthropic(api_key=api_key)
        threads.append(threading.Thread(
            target=_polish_thread, args=(client, ai_model, shared, polish_q), daemon=True))

    for t in threads:
        t.start()
    shared["workers_started"] = True

if not webrtc_ctx.state.playing and shared["workers_started"]:
    shared["stop_event"].set()
    shared["workers_started"] = False

# ---------- drain results ----------
while not shared["result_q"].empty():
    msg = shared["result_q"].get_nowait()
    kind = msg["kind"]
    if kind == "partial":
        st.session_state.partial = msg["text"]
    elif kind == "final":
        st.session_state.partial = ""
        st.session_state.lines.append(msg["text"])
        st.session_state.line_ids.append(msg["id"])
        st.session_state.current = msg["text"]
    else:  # polished
        if msg["id"] in st.session_state.line_ids:
            idx = st.session_state.line_ids.index(msg["id"])
            st.session_state.lines[idx] = msg["text"]
            if idx == len(st.session_state.lines) - 1:
                st.session_state.current = msg["text"]

while not shared["level_q"].empty():
    st.session_state.level = shared["level_q"].get_nowait()

# ---------- render ----------
if webrtc_ctx.state.playing:
    status_box.success("🔴 Live — listening to your microphone")
    rms = st.session_state.level
    bars = int(min(rms * 500, 20))
    level_box.markdown(f"`Vol: [{'█' * bars}{'░' * (20 - bars)}]  {rms:.4f}`")
else:
    status_box.info("Click **START** above and allow microphone access to begin.")

live_text = st.session_state.partial or st.session_state.current
if live_text:
    cursor = " ▌" if st.session_state.partial else ""
    caption_box.markdown(
        f"""<div style="
            background:#1e1e2e;color:#cdd6f4;font-size:1.5rem;
            line-height:1.6;padding:1.2rem 1.5rem;border-radius:12px;
            border-left:4px solid #89b4fa;margin-bottom:0.5rem;
        ">{live_text}{cursor}</div>""",
        unsafe_allow_html=True,
    )

if st.session_state.lines:
    transcript_box.text_area(
        label="", value="\n".join(st.session_state.lines),
        height=250, label_visibility="collapsed",
    )
else:
    transcript_box.caption("Transcribed text will appear here.")

# Keep refreshing while the stream is live so captions update.
if webrtc_ctx.state.playing:
    time.sleep(0.4)
    st.rerun()
