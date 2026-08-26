"""Diagnostic: list loopback devices and measure volume for 10 seconds.
Run this WHILE playing a YouTube video, then paste the full output.
"""
import time

import numpy as np
import pyaudiowpatch as pyaudio

CHUNK = 512

p = pyaudio.PyAudio()

print("=" * 60)
print("DEFAULT OUTPUT DEVICE")
print("=" * 60)
try:
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    print(f"  Default output: {default_out['name']}")
except Exception as e:
    print(f"  Could not detect default output: {e}")

print()
print("=" * 60)
print("ALL LOOPBACK DEVICES")
print("=" * 60)
loopbacks = list(p.get_loopback_device_info_generator())
for d in loopbacks:
    print(f"  index={d['index']:>3}  ch={int(d['maxInputChannels'])}  "
          f"sr={int(d['defaultSampleRate'])}  name={d['name']}")

if not loopbacks:
    print("  NONE FOUND")
    p.terminate()
    raise SystemExit

# Test each loopback for 3 seconds
print()
print("=" * 60)
print("VOLUME TEST — play audio now! (3s per device)")
print("=" * 60)

for d in loopbacks:
    idx = d["index"]
    sr = int(d["defaultSampleRate"])
    ch = min(int(d["maxInputChannels"]), 2)
    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=sr,
            input=True,
            input_device_index=idx,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        print(f"  [{idx}] {d['name'][:40]:40}  OPEN FAILED: {e}")
        continue

    peak = 0.0
    start = time.time()
    while time.time() - start < 3.0:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = max(peak, rms)

    stream.stop_stream()
    stream.close()

    flag = "  <-- AUDIO DETECTED!" if peak > 0.001 else ""
    print(f"  [{idx}] {d['name'][:40]:40}  peak_rms={peak:.5f}{flag}")

p.terminate()
print()
print("Done. Paste this entire output back.")
