import os
import queue
import threading
import time

import anthropic
import numpy as np
# noinspection PyPackageRequirements
import pyaudiowpatch as pyaudio  # Windows-only; listed in requirements with a platform marker
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Live Transcriber", page_icon="🎙️", layout="centered")

WHISPER_SR = 16000
REFRESH_RATE = 0.3           # how often the UI redraws
CHUNK = 512
MAX_LINES = 400              # cap transcript length to avoid unbounded memory growth

STEP_SECONDS = 0.5           # re-transcribe the growing utterance this often
SILENCE_RMS = 0.0015         # below this = "silence"
SILENCE_HANG = 0.7           # seconds of silence that ends an utterance
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

ANSWER_SYSTEM = (
    "You are an expert interview coach assisting the user DURING a live "
    "interview. You are given the running transcript of the conversation. "
    "Identify the interviewer's most recent question or point, and write a "
    "strong, natural answer the user can say out loud, in the first person "
    "('I'). Keep it concise — about 20–45 seconds of speaking. Be specific and "
    "confident. If there is no clear question, suggest a good thing for the "
    "user to say next. Output ONLY the words the user should say — no preamble, "
    "no labels, no explanation."
)

ANALYZE_SYSTEM = (
    "You are an interview analyst. You are given the transcript of a "
    "conversation. Provide a short, skimmable analysis in markdown with these "
    "sections: **Summary** (2–3 sentences), **Key points** (bullets), "
    "**Questions asked** (bullets), and **Suggestions** (bullets). Keep it brief."
)

ASSIST_MODEL = "claude-opus-4-8"


def ai_assist(client, system, transcript, max_tokens=1024):
    """Send the transcript to Claude Opus for an answer suggestion or analysis."""
    resp = client.messages.create(
        model=ASSIST_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": transcript}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# noinspection PyBroadException
def require_password():
    """Optional password gate. Active only when APP_PASSWORD is set (env var or
    Streamlit secret). When no password is configured, the app is unlocked so
    local use is unaffected."""
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
    """Single shared state object that survives Streamlit reruns AND is
    visible to the background worker threads. Created exactly once."""
    return {
        "result_q": queue.Queue(),   # dicts: {"kind": "partial"|"final"|"polished", "id": int, "text": str}
        "level_q": queue.Queue(),     # floats: RMS volume
        "error_q": queue.Queue(),     # strings: error messages
        "stop_event": threading.Event(),
        "native_sr": None,
    }


@st.cache_resource(show_spinner=False)
def load_model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def make_client(api_key, workspace_id="", timeout=None):
    """Build an Anthropic client. Identity-linked API keys also require a
    workspace id, sent as the anthropic-workspace-id header."""
    kwargs = {"api_key": api_key.strip()}
    if workspace_id and workspace_id.strip():
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id.strip()}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return anthropic.Anthropic(**kwargs)


# noinspection PyBroadException
def polish_text(client: anthropic.Anthropic, model: str, raw: str) -> str:
    """Ask Claude to clean up a raw transcription fragment. Returns the raw
    text unchanged if the call fails."""
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


def _resample_to_16k(segment, native_sr):
    if native_sr == WHISPER_SR:
        return segment
    new_len = int(len(segment) * WHISPER_SR / native_sr)
    indices = np.linspace(0, len(segment) - 1, new_len)
    return np.interp(indices, np.arange(len(segment)), segment).astype(np.float32)


# noinspection PyBroadException
def reduce_noise(x):
    """Light spectral-subtraction noise reduction: estimate a steady noise
    floor from the quietest frequency bins and subtract it. Reduces background
    hiss/hum so Whisper hears speech more clearly. Returns x unchanged on any
    problem so it can never break the pipeline."""
    try:
        if len(x) < 1024:
            return x
        spectrum = np.fft.rfft(x)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        noise_floor = np.percentile(mag, 25)          # steady background level
        mag = np.maximum(mag - 1.5 * noise_floor, 0.0)  # subtract it
        cleaned = np.fft.irfft(mag * np.exp(1j * phase), n=len(x))
        return cleaned.astype(np.float32)
    except Exception:
        return x


# noinspection PyBroadException
def get_loopback_devices():
    p = pyaudio.PyAudio()
    devices = []
    default_name = None
    try:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            default_out_name = default_out["name"]
        except Exception:
            default_out_name = None

        for loopback in p.get_loopback_device_info_generator():
            devices.append(loopback)
            if default_out_name and default_out_name in loopback["name"]:
                default_name = loopback["name"]
    finally:
        p.terminate()
    return devices, default_name


# noinspection PyBroadException
def _capture_thread(device_info, shared, audio_q):
    """Reads system audio continuously and pushes small mono chunks to audio_q.
    Does nothing heavy, so no audio is ever dropped."""
    stop_event = shared["stop_event"]
    level_q = shared["level_q"]
    error_q = shared["error_q"]

    p = None
    stream = None
    try:
        p = pyaudio.PyAudio()
        native_sr = int(device_info["defaultSampleRate"])
        channels = min(int(device_info["maxInputChannels"]), 2)
        shared["native_sr"] = native_sr

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=native_sr,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=CHUNK,
        )

        while not stop_event.is_set():
            raw = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(raw, dtype=np.float32)
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            level_q.put(float(np.sqrt(np.mean(audio ** 2))))
            audio_q.put(audio.copy())
    except Exception as e:
        error_q.put(f"{type(e).__name__}: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()


def _stream_thread(whisper_model, shared, audio_q, polish_q, use_polish, denoise):
    """The live-caption engine. Accumulates audio into the current utterance,
    re-transcribes it every STEP_SECONDS to show growing PARTIAL text, and on a
    pause finalizes the line (and hands it to Claude for polishing)."""
    stop_event = shared["stop_event"]
    result_q = shared["result_q"]

    native_sr = None
    while native_sr is None and not stop_event.is_set():
        native_sr = shared.get("native_sr")
        if native_sr is None:
            time.sleep(0.05)
    if native_sr is None:
        return

    utterance = np.zeros((0,), dtype=np.float32)
    silent_time = 0.0
    last_tx = time.time()
    line_id = 0
    last_partial = ""

    # noinspection PyBroadException
    def transcribe(buf):
        try:
            audio16k = _resample_to_16k(buf, native_sr)
            if denoise:
                audio16k = reduce_noise(audio16k)
            segs, _ = whisper_model.transcribe(
                audio16k,
                language="en", beam_size=1, best_of=1, temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,          # skip non-speech noise
            )
            return " ".join(s.text.strip() for s in segs).strip()
        except Exception:
            return ""

    while not stop_event.is_set():
        # 1) Drain whatever audio has arrived.
        collected = []
        try:
            while True:
                collected.append(audio_q.get_nowait())
        except queue.Empty:
            pass

        if collected:
            new = np.concatenate(collected)
            utterance = np.concatenate([utterance, new])
            new_rms = float(np.sqrt(np.mean(new ** 2)))
            if new_rms < SILENCE_RMS:
                silent_time += len(new) / native_sr
            else:
                silent_time = 0.0

        has_speech = len(utterance) > 0 and float(np.sqrt(np.mean(utterance ** 2))) > SILENCE_RMS
        now = time.time()

        # 2) No speech yet: keep only a short lead-in tail so silence never piles up.
        if not has_speech:
            keep = int(native_sr * 0.3)
            if len(utterance) > keep:
                utterance = utterance[-keep:]
            time.sleep(0.03)
            continue

        dur = len(utterance) / native_sr

        # 3) Live partial update.
        if now - last_tx >= STEP_SECONDS:
            text = transcribe(utterance)
            if text and text != last_partial:
                last_partial = text
                result_q.put({"kind": "partial", "id": line_id, "text": text})
            last_tx = now

        # 4) Finalize on a pause (or if the utterance got too long).
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


def _polish_thread(ai_client, ai_model, shared, polish_q):
    """Pulls finalized lines, asks Claude to clean them up, emits the polished
    text which the UI swaps in place of the raw line."""
    stop_event = shared["stop_event"]
    result_q = shared["result_q"]
    while not stop_event.is_set():
        try:
            line_id, raw = polish_q.get(timeout=0.2)
        except queue.Empty:
            continue
        polished = polish_text(ai_client, ai_model, raw)
        result_q.put({"kind": "polished", "id": line_id, "text": polished})


# ---------- shared state ----------
shared = get_shared()

# ---------- session state ----------
for key, val in [
    ("lines", []), ("line_ids", []), ("running", False),
    ("partial", ""), ("current", ""), ("level", 0.0), ("threads", []),
    ("ai_output", ""), ("ai_title", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- password gate (only if APP_PASSWORD is set) ----------
require_password()

# ---------- UI ----------
st.title("🎙️ Live Transcriber")
st.caption("Live captions from your system audio, polished by Claude.")

with st.sidebar:
    st.header("Settings")
    ai_polish = st.toggle("✨ Claude AI polish", value=True,
                          help="Live text shows instantly; finished sentences get "
                               "cleaned up by Claude a moment later.")
    ai_model = st.selectbox(
        "Claude model",
        ["claude-haiku-4-5", "claude-opus-4-8"],
        index=0,
        help="Haiku = fastest / lowest latency (recommended). Opus = best quality.",
    )
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com. Or set ANTHROPIC_API_KEY env var.",
    )
    workspace_id = st.text_input(
        "Workspace ID (only if your key needs it)",
        value=os.environ.get("ANTHROPIC_WORKSPACE_ID", ""),
        help="Only needed for identity-linked keys that ask for "
             "'anthropic-workspace-id'. Leave blank for a normal key.",
    )
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=0,
                                help="tiny = fastest live updates, small = most accurate.")
    denoise = st.toggle("🔇 Noise reduction", value=True,
                        help="Reduces background hiss/hum so noisy interviews "
                             "transcribe more cleanly.")

devices, default_name = get_loopback_devices()

if not devices:
    st.error("No WASAPI loopback devices found.")
    st.stop()

device_names = [d["name"] for d in devices]
default_index = device_names.index(default_name) if default_name in device_names else 0

if default_name:
    st.caption(f"Your current default output is: **{default_name}**")

selected_name = st.selectbox("Speaker output to capture", device_names, index=default_index)
selected_device = devices[device_names.index(selected_name)]

c1, c2, c3 = st.columns(3)
start_btn = c1.button("▶ Start", type="primary", use_container_width=True)
stop_btn = c2.button("⏹ Stop", use_container_width=True)
clear_btn = c3.button("🗑 Clear", use_container_width=True)

st.divider()

status_box = st.empty()
level_box = st.empty()
caption_box = st.empty()
st.markdown("#### Full transcript")
transcript_box = st.empty()

# ---------- button logic ----------
if start_btn and not st.session_state.running:
    if ai_polish and not api_key.strip():
        st.error("Enter your Anthropic API key in the sidebar, or turn off Claude AI polish.")
        st.stop()

    shared["stop_event"].clear()
    shared["native_sr"] = None
    for q in ("result_q", "level_q", "error_q"):
        while not shared[q].empty():
            try:
                shared[q].get_nowait()
            except queue.Empty:
                break

    with st.spinner("Loading transcription model…"):
        _model = load_model(whisper_size)

    _client = make_client(api_key, workspace_id) if ai_polish else None

    audio_q = queue.Queue()
    polish_q = queue.Queue()

    threads = [
        threading.Thread(target=_capture_thread,
                         args=(selected_device, shared, audio_q), daemon=True),
        threading.Thread(target=_stream_thread,
                         args=(_model, shared, audio_q, polish_q, ai_polish, denoise), daemon=True),
    ]
    if ai_polish:
        threads.append(threading.Thread(target=_polish_thread,
                                        args=(_client, ai_model, shared, polish_q), daemon=True))

    for t in threads:
        t.start()
    st.session_state.threads = threads
    st.session_state.running = True
    st.session_state.partial = ""
    st.session_state.current = ""

if stop_btn:
    shared["stop_event"].set()
    st.session_state.running = False
    st.session_state.partial = ""

if clear_btn:
    st.session_state.lines = []
    st.session_state.line_ids = []
    st.session_state.partial = ""
    st.session_state.current = ""

# ---------- drain queues ----------
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
        if len(st.session_state.lines) > MAX_LINES:
            st.session_state.lines = st.session_state.lines[-MAX_LINES:]
            st.session_state.line_ids = st.session_state.line_ids[-MAX_LINES:]
    else:  # polished — swap in place of the finalized line
        if msg["id"] in st.session_state.line_ids:
            idx = st.session_state.line_ids.index(msg["id"])
            st.session_state.lines[idx] = msg["text"]
            if idx == len(st.session_state.lines) - 1:
                st.session_state.current = msg["text"]

while not shared["level_q"].empty():
    st.session_state.level = shared["level_q"].get_nowait()

if not shared["error_q"].empty():
    err = shared["error_q"].get_nowait()
    st.session_state.running = False
    shared["stop_event"].set()
    st.error(f"Capture failed: {err}")

# ---------- render ----------
if st.session_state.running:
    status_box.success("🔴 Live — listening")
    rms = st.session_state.level
    bars = int(min(rms * 500, 20))
    level_box.markdown(f"`Vol: [{'█' * bars}{'░' * (20 - bars)}]  {rms:.4f}`")
else:
    status_box.info("Press ▶ Start to begin")

# Big live box: show the growing partial if speaking, else the last finished line.
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

# ---------- AI Assistant (Claude Opus) ----------
st.divider()
st.markdown("#### 🤖 AI Assistant")
st.caption("Uses Claude Opus on the conversation above.")
_a, _b = st.columns(2)
suggest_btn = _a.button("💡 Suggest an answer", use_container_width=True)
analyze_btn = _b.button("🔍 Analyze conversation", use_container_width=True)

if suggest_btn or analyze_btn:
    transcript = "\n".join(st.session_state.lines).strip()
    if not transcript:
        st.warning("No conversation yet — transcribe something first.")
    elif not api_key.strip():
        st.warning("Enter your Anthropic API key in the sidebar to use the AI Assistant.")
    else:
        # 30s timeout so a slow/stuck request can never freeze the app.
        _client = make_client(api_key, workspace_id, timeout=30.0)
        # noinspection PyBroadException
        try:
            if suggest_btn:
                with st.spinner("Claude is drafting an answer…"):
                    st.session_state.ai_output = ai_assist(
                        _client, ANSWER_SYSTEM, transcript, max_tokens=800)
                st.session_state.ai_title = "💡 Suggested answer"
            else:
                with st.spinner("Claude is analyzing the conversation…"):
                    st.session_state.ai_output = ai_assist(
                        _client, ANALYZE_SYSTEM, transcript, max_tokens=1500)
                st.session_state.ai_title = "🔍 Conversation analysis"
        except Exception as e:
            st.error(f"AI request failed: {e}")

if st.session_state.ai_output:
    with st.container(border=True):
        st.markdown(f"**{st.session_state.ai_title}**")
        st.markdown(st.session_state.ai_output)

if st.session_state.running:
    time.sleep(REFRESH_RATE)
    st.rerun()
