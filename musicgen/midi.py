"""Small Standard MIDI File reader for inspecting the bundled format-0/1 samples.

No synthesizer, network, or ML dependency. Supports PPQN timing and tempo changes;
rejects format 2 and SMPTE timing rather than silently assigning wrong durations.
"""
from __future__ import annotations

import bisect
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Note:
    track: int
    channel: int
    pitch: int
    velocity: int
    start: float
    end: float


def _vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise ValueError("Truncated MIDI variable-length integer")
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 127)
        if not byte & 128:
            return value, pos
    raise ValueError("MIDI variable-length integer exceeds four bytes")


def read_midi(path: str | Path) -> list[Note]:
    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File")
    hlen = struct.unpack_from(">I", data, 4)[0]
    if hlen < 6 or 8 + hlen > len(data):
        raise ValueError("Invalid MIDI header length")
    fmt, count, ppqn = struct.unpack_from(">HHH", data, 8)
    if fmt not in (0, 1) or not count or (fmt == 0 and count != 1):
        raise ValueError("Only MIDI format 0/1 is supported")
    if ppqn & 0x8000 or ppqn == 0:
        raise ValueError("SMPTE or zero time division is unsupported")
    pos = 8 + hlen
    events, tempos = [], [(0, 500000)]
    for track in range(count):
        if data[pos:pos+4] != b"MTrk" or pos+8 > len(data):
            raise ValueError("Missing MIDI track")
        size = struct.unpack_from(">I", data, pos+4)[0]
        pos += 8
        chunk = data[pos:pos+size]; pos += size
        if len(chunk) != size:
            raise ValueError("Truncated MIDI track")
        i, tick, running = 0, 0, None
        while i < len(chunk):
            delta, i = _vlq(chunk, i); tick += delta
            if i >= len(chunk): raise ValueError("Missing event status")
            status = chunk[i]
            if status >= 128:
                i += 1
                running = status if status < 0xF0 else None
            elif running is not None:
                status = running
            else:
                raise ValueError("Running status without a channel event")
            if status == 0xFF:
                if i >= len(chunk): raise ValueError("Missing meta type")
                kind = chunk[i]; i += 1
                length, i = _vlq(chunk, i)
                payload = chunk[i:i+length]; i += length
                if len(payload) != length: raise ValueError("Truncated meta event")
                if kind == 0x51:
                    if length != 3: raise ValueError("Invalid tempo event")
                    tempo = int.from_bytes(payload, "big")
                    if tempo <= 0: raise ValueError("Nonpositive tempo")
                    tempos.append((tick, tempo))
                if kind == 0x2F: break
            elif status in (0xF0, 0xF7):
                length, i = _vlq(chunk, i); i += length
                if i > len(chunk): raise ValueError("Truncated SysEx")
            elif status < 0xF0:
                kind, channel = status >> 4, status & 15
                n = 1 if kind in (12, 13) else 2
                payload = chunk[i:i+n]; i += n
                if len(payload) != n or any(x >= 128 for x in payload):
                    raise ValueError("Invalid channel data")
                if kind in (8, 9):
                    pitch, velocity = payload
                    on = kind == 9 and velocity > 0
                    events.append((tick, track, channel, pitch, velocity, on))
            else:
                raise ValueError(f"Unsupported system status {status:#x}")
    # MIDI tempo is global for synchronous format-1 tracks.
    tempo_map = {}
    for tick, tempo in tempos: tempo_map[tick] = tempo
    ticks = sorted(tempo_map)
    seconds = [0.0]
    for k in range(1, len(ticks)):
        seconds.append(seconds[-1] + (ticks[k]-ticks[k-1]) * tempo_map[ticks[k-1]] / ppqn / 1e6)
    def at(tick):
        k = bisect.bisect_right(ticks, tick)-1
        return seconds[k] + (tick-ticks[k]) * tempo_map[ticks[k]] / ppqn / 1e6
    active = defaultdict(deque)
    notes = []
    for tick, track, channel, pitch, velocity, on in sorted(events, key=lambda e:e[0]):
        key = (track, channel, pitch)
        if on:
            active[key].append((tick, velocity))
        elif active[key]:
            start, vel = active[key].popleft()
            if tick > start: notes.append(Note(track, channel, pitch, vel, at(start), at(tick)))
    if any(active.values()): raise ValueError("Unterminated MIDI note")
    return sorted(notes, key=lambda n:(n.start, n.track, n.pitch))


def summarize(notes: list[Note]) -> dict:
    return {"note_count": len(notes), "tracks_with_notes": len({n.track for n in notes}),
            "duration_seconds": round(max((n.end for n in notes), default=0), 4),
            "pitch_min": min((n.pitch for n in notes), default=None),
            "pitch_max": max((n.pitch for n in notes), default=None)}


def render_wav(notes: list[Note], target: Path, sample_rate=22050) -> None:
    """Render an explicitly synthetic listening preview, not an instrument model."""
    import numpy as np
    import wave
    if not notes: raise ValueError("Cannot render an empty MIDI")
    duration = max(n.end for n in notes) + 0.2
    out = np.zeros(int(duration*sample_rate)+1, dtype=np.float64)
    for n in notes:
        start = int(n.start*sample_rate)
        length = max(1, int((n.end-n.start)*sample_rate))
        t = np.arange(length)/sample_rate
        frequency = 440 * 2**((n.pitch-69)/12)
        tone = np.sin(2*np.pi*frequency*t) + 0.20*np.sin(4*np.pi*frequency*t) + 0.08*np.sin(6*np.pi*frequency*t)
        envelope = np.minimum(t/0.015, 1) * np.minimum((length/sample_rate-t)/0.05, 1)
        out[start:start+length] += tone*envelope*(n.velocity/127)
    peak = np.max(np.abs(out))
    if peak: out *= 0.75/peak
    with wave.open(str(target), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes((out*32767).astype("<i2").tobytes())
