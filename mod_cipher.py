#!/usr/bin/env python3
"""
Chiptune 2-tone chord modulation → Generates ProTracker .MOD (4-channel)
Supports text or file payloads, optional zstd compression + Argon2id/XChaCha20-Poly1305.

Note: Embedded payloads are capped at MAX_PAYLOAD bytes (65535) because the format
stores the payload length in a 2-byte field. Attempting to embed larger files will
produce a clear warning and exit.
"""

import argparse
import math
import struct
import sys
import os
import getpass
import random

import numpy as np

# try crypto
HAS_CRYPTO = True
try:
    from argon2.low_level import hash_secret_raw, Type as ArgonType
    from nacl.bindings import (
        crypto_aead_xchacha20poly1305_ietf_encrypt,
        crypto_aead_xchacha20poly1305_ietf_decrypt,
    )
except ImportError:
    HAS_CRYPTO = False


# try zstd
HAS_ZSTD = True
try:
    import zstandard as zstd
except ImportError:
    zstd = None
    HAS_ZSTD = False

# zstd frame magic (always defined for safe checks)
ZSTD_MAGIC = b"\x28\xB5\x2F\xFD"

# ----------------------------------------
# Audio/mod constants & framing parameters
# ----------------------------------------

SR             = 44100
BPM            = 140

PREAMBLE_FRAMES = 8
HEADER_FRAMES   = 4   # 2 bytes → 4 nibbles

ENC_MAGIC      = b"ENC1"
SALT_LEN       = 16
NONCE_LEN      = 24
KEY_LEN        = 32

# Maximum payload length that can be represented in the 2-byte header (bytes)
MAX_PAYLOAD = 0xFFFF  # 65535 bytes (~64 KiB)

# Musical space: 16 dyads → 4-bit symbols
DYADS = [
    ("A3","C4"),("A3","E4"),("A3","G4"),("A3","A4"),
    ("C4","E4"),("C4","G4"),("C4","A4"),("C4","C5"),
    ("D4","G4"),("D4","A4"),("D4","C5"),("D4","E5"),
    ("E4","A4"),("E4","C5"),("E4","E5"),("G4","C5"),
]
SYMBOL_TO_DYAD = {i:DYADS[i] for i in range(16)}
DYAD_TO_SYMBOL = {DYADS[i]:i for i in range(16)}

ARPEGGIO_SETS = [
    [0,2,4,7,4,2,0,5],[1,3,6,3,1,4,7,4],
    [2,4,6,4,2,5,2,4],[0,7,0,5,0,7,0,4],
    [3,4,5,4,3,6,4,3],[7,6,5,4,3,2,1,0],
    [0,2,2,4,4,7,5,5],[1,4,7,1,4,7,4,1],
    [0,2,4,6,4,2,0,5],[1,3,5,6,5,3,1,4],
    [0,7,5,0,7,4,0,7],[2,3,4,5,4,3,2,6],
    [0,3,5,0,3,5,7,5],[2,5,7,2,5,7,4,2],
    [1,6,3,7,4,1,6,3],[0,4,7,0,4,7,5,4],
]
LEAD_NOTES = ["A3","C4","D4","E4","G4","A4","C5","E5"]

# ----------------------------------------
# CRC16 & nibble/frame utilities
# ----------------------------------------

def crc16(data: bytes) -> int:
    poly, crc = 0xA001, 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc>>1) ^ (poly if (crc & 1) else 0)
    return crc & 0xFFFF

def bytes_to_nibbles(data: bytes) -> list:
    out = []
    for b in data:
        out.append((b>>4)&0xF)
        out.append(b &0xF)
    return out

def nibbles_to_bytes(nibs: list) -> bytes:
    out = bytearray()
    for i in range(0, len(nibs), 2):
        hi, lo = nibs[i]&0xF, nibs[i+1]&0xF
        out.append((hi<<4) | lo)
    return bytes(out)

# ----------------------------------------
# Simple waveform generators
# ----------------------------------------

def square_wave(freq, length, amp, duty):
    t = np.arange(length) / SR
    ph = (t*freq) % 1.0
    w  = np.where(ph < duty, 1.0, -1.0)
    return (amp*w).astype(np.float32)

def triangle_wave(freq, length, amp):
    t  = np.arange(length) / SR
    ph = (t*freq) % 1.0
    tri= 2.0*np.abs(2.0*(ph - np.floor(ph+0.5))) - 1.0
    return (amp*tri).astype(np.float32)

def apply_fade(buf):
    edge = int(0.003 * SR)
    if len(buf) < 2*edge:
        return buf
    fi = np.linspace(0,1,edge, dtype=np.float32)
    fo = fi[::-1]
    buf[:edge]  *= fi
    buf[-edge:] *= fo
    return buf

# ----------------------------------------
# Procedural instrument builders (randomized naming)
# ----------------------------------------

def _rand_name(prefixes=("DRM","SYN","WAV","NOI","ORG"),
               suffixes=("01","02","A","B","X","Z")):
    return (random.choice(prefixes) + random.choice(suffixes)).ljust(4)[:4]

def kick_drum(length=512, amp=1.0):
    t = np.arange(length) / length
    # pitch drops over time; exponential decay envelope
    freq = 60 + 120 * (1 - t)
    phase = np.cumsum(freq) / length
    wave = np.sin(2*np.pi*phase)
    env  = np.exp(-6*t)
    return (amp * wave * env).astype(np.float32)

def snare_drum(length=512, amp=0.8):
    t = np.arange(length) / length
    noise = np.random.randn(length)
    # add a weak tone component for body
    tone  = np.sin(2*np.pi*200*np.arange(length)/SR) * 0.15
    env   = np.exp(-12*t)
    return (amp * (noise + tone) * env).astype(np.float32)

def hi_hat(length=256, amp=0.6):
    noise = np.sign(np.random.randn(length))
    env   = np.linspace(1,0,length)
    # simple high-pass feel by alternating sign
    hiss  = noise * np.sign(np.sin(2*np.pi*4000*np.arange(length)/SR))
    return (amp * hiss * env).astype(np.float32)

def saw_wave(freq, length, amp):
    t = np.arange(length) / SR
    ph = (t*freq) % 1.0
    return (amp*(2*ph - 1)).astype(np.float32)

def pulse_wave(freq, length, amp):
    duty = random.choice([0.125,0.25,0.33,0.5,0.67,0.75])
    return square_wave(freq, length, amp, duty)

def organ_wave(freq, length, amp):
    t = np.arange(length) / SR
    wave = (np.sin(2*np.pi*freq*t) +
            0.5*np.sin(2*np.pi*2*freq*t) +
            0.25*np.sin(2*np.pi*3*freq*t))
    return (amp*wave/1.75).astype(np.float32)

def make_instrument_bank(seed=None):
    """
    Build a minimal band with roles: chord1, chord2, lead, drums (kick/snare/hat).
    Names are randomized; instruments are procedurally varied.
    Returns (samples, sample_map) where samples is a list of dicts for write_mod,
    and sample_map maps logical voices to sample indices (1-based for MOD).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

    # Melodic voices
    chord1_name = _rand_name(("SYN","PUL","TRI","WAV"), ("01","A","X"))
    chord2_name = _rand_name(("SYN","PUL","TRI","WAV"), ("02","B","Z"))
    lead_name   = _rand_name(("LEAD","SYN","ORG"), ("01","A","X"))

    # Slight parameter jitter
    def rf(base): return base * (1.0 + random.uniform(-0.03, 0.03))
    def ra(base): return max(0.3, min(0.9, base + random.uniform(-0.1, 0.1)))

    # Choose wave types for variety
    chord1_gen = lambda l, f=rf(440), a=ra(0.8): pulse_wave(f, l, a)
    chord2_gen = lambda l, f=rf(330), a=ra(0.8): triangle_wave(f, l, a)
    lead_gen   = lambda l, f=rf(550), a=ra(0.75): organ_wave(f, l, a)

    # Drums (always include; rhythm channel prefers these)
    kick_name  = _rand_name(("DRM","KIK"), ("01","A"))
    snare_name = _rand_name(("DRM","SNR"), ("01","B"))
    hat_name   = _rand_name(("DRM","HAT"), ("01","C"))

    samp_len = 1024
    samples = []
    def add_sample(name, genfunc, play_len_mult=2):
        buf = apply_fade(genfunc(samp_len*play_len_mult))
        samples.append({
            "name":         name,
            "length":       samp_len,
            "finetune":     0,
            "volume":       64,
            "repeat_offset":0,
            "repeat_length":1,
            "wave":         buf
        })

    add_sample(chord1_name, chord1_gen)
    add_sample(chord2_name, chord2_gen)
    add_sample(lead_name,   lead_gen)
    add_sample(kick_name,   lambda l: kick_drum(l, amp=1.0), play_len_mult=1)
    add_sample(snare_name,  lambda l: snare_drum(l, amp=0.9), play_len_mult=1)
    add_sample(hat_name,    lambda l: hi_hat(l, amp=0.7),     play_len_mult=1)

    # 1-based indices for MOD samples in pack_note
    sample_map = {
        "chord1": 1,
        "chord2": 2,
        "lead":   3,
        "kick":   4,
        "snare":  5,
        "hat":    6
    }
    return samples, sample_map

def make_groove_plan(seed=None):
    """
    Build an 8-step drum plan favoring kick/snare/hat with occasional rests.
    Returns a list like ['kick','hat','snare','rest',...]
    """
    if seed is not None:
        random.seed(seed + 0x9E3779B9)

    choices = ['kick','snare','hat','rest']
    # bias toward kick on downbeats, snare on backbeats
    plan = []
    for i in range(8):
        if i % 4 == 0:
            plan.append('kick')
        elif i % 4 == 2:
            plan.append('snare')
        else:
            plan.append(random.choice(['hat','rest','hat']))
    # small random mutation to avoid fixed pattern
    for i in range(8):
        if random.random() < 0.2:
            plan[i] = random.choice(choices)
    return plan

# ----------------------------------------
# MOD note periods & name mappings
# ----------------------------------------

PERIODS = {
    "A-2":508,"A-3":254,"A-4":127,
    "C-4":214,"D-4":190,"E-4":170,"G-4":143,"C-5":107,"E-5":85,
    "D-3":381,"E-3":339,"G-2":570,"C-3":428
}

NOTE_TO_MOD = {
    "A3":"A-3","C4":"C-4","D4":"D-4","E4":"E-4",
    "G4":"G-4","A4":"A-4","C5":"C-5","E5":"E-5",
    "A2":"A-2","E3":"E-3","G2":"G-2","D3":"D-3"
}

PERIODS_REVERSE = {v:k for k,v in PERIODS.items()}

def pack_note(period, sample_idx, effect=0, param=0) -> bytes:
    b0 = ((sample_idx & 0x10)) | ((period >> 8) & 0x0F)
    b1 = period & 0xFF
    b2 = ((sample_idx & 0x0F) << 4) | (effect & 0x0F)
    b3 = param & 0xFF
    return bytes((b0,b1,b2,b3))

def write_mod(filename: str, title: str, samples: list, order: list, patterns: list):
    with open(filename, "wb") as f:
        f.write(title[:20].ljust(20, "\0").encode("ascii"))
        for smp in samples + [None]*(31-len(samples)):
            if smp is None:
                f.write(b"\0"*30); continue
            name  = smp["name"][:22].ljust(22, "\0").encode("ascii")
            length= smp["length"]
            f.write(name)
            f.write(struct.pack(">H", length))
            f.write(bytes((smp["finetune"]&0x0F, smp["volume"]&0x7F)))
            f.write(struct.pack(">HH",
                                smp["repeat_offset"],
                                smp["repeat_length"]))
        f.write(bytes((len(order), 0x7F)))
        tbl = order + [0]*(128-len(order))
        f.write(bytes(tbl))
        f.write(b"M.K.")
        for pat in patterns:
            for row in pat:
                for chan in row:
                    f.write(chan)
        for smp in samples:
            wav = np.clip(smp["wave"], -1, 1)
            pcm = (wav * 127).astype(np.int8).tobytes()
            f.write(pcm)

def encode_to_mod(data: bytes,
                  encrypt: bool,
                  password: str,
                  out: str,
                  title: str,
                  seed: int = None):
    # Final payload length check before framing
    if len(data) > MAX_PAYLOAD:
        sys.exit(f"Payload too large for embedding: {len(data)} bytes > {MAX_PAYLOAD} bytes maximum")

    if encrypt:
        if not HAS_CRYPTO:
            sys.exit("Install argon2-cffi + pynacl for encryption")
        if not password:
            password = getpass.getpass("Password: ")
        salt  = os.urandom(SALT_LEN)
        key   = hash_secret_raw(password.encode(), salt,
                                time_cost=2, memory_cost=64*1024,
                                parallelism=1, hash_len=KEY_LEN,
                                type=ArgonType.ID)
        nonce = os.urandom(NONCE_LEN)
        ct    = crypto_aead_xchacha20poly1305_ietf_encrypt(data, None, nonce, key)
        data  = ENC_MAGIC + salt + nonce + ct

    length_b = struct.pack(">H", len(data))
    crc_b    = struct.pack(">H", crc16(data))
    nibs     = bytes_to_nibbles(length_b) \
             + bytes_to_nibbles(data)    \
             + bytes_to_nibbles(crc_b)

    # Band setup: procedural instruments and groove plan
    samples, sample_map = make_instrument_bank(seed)
    drum_plan = make_groove_plan(seed)
    # lead arpeggio offsets (per symbol) to vary melodies
    arp_offsets = [random.randint(0,7) for _ in range(len(ARPEGGIO_SETS))]

    all_syms     = [0]*PREAMBLE_FRAMES + nibs
    total_frames = len(all_syms)
    num_patterns = math.ceil(total_frames/64)

    patterns = []
    for p in range(num_patterns):
        pat = []
        for r in range(64):
            idx = p*64 + r
            sym = all_syms[idx] if idx < total_frames else 0

            # Channels 0 & 1: chord dyad (critical for decode)
            d0, d1 = SYMBOL_TO_DYAD[sym]
            p0 = PERIODS[NOTE_TO_MOD[d0]]
            p1 = PERIODS[NOTE_TO_MOD[d1]]

            # Channel 2: lead melody with per-render arp offset
            arp = ARPEGGIO_SETS[sym % len(ARPEGGIO_SETS)]
            lead_note = LEAD_NOTES[arp[(idx + arp_offsets[sym % len(ARPEGGIO_SETS)]) % len(arp)]]
            p2 = PERIODS[NOTE_TO_MOD[lead_note]]

            # Channel 3: drums preferred (kick/snare/hat) based on drum_plan
            step = drum_plan[idx % len(drum_plan)]
            if step == 'kick':
                samp4 = sample_map["kick"]
                p4    = PERIODS["C-3"]
            elif step == 'snare':
                samp4 = sample_map["snare"]
                p4    = PERIODS["C-3"]
            elif step == 'hat':
                samp4 = sample_map["hat"]
                p4    = PERIODS["C-3"]
            else:
                # Optional: rest – use a silent note by employing an unused period with volume 0,
                # but since we don't push effects here, we can place a low bass tone very quietly.
                # Simpler: pick hat sample but it will be very short; or use a safe bass touch:
                # We'll just place a hat with minimal impact (same period).
                samp4 = sample_map["hat"]
                p4    = PERIODS["C-3"]

            notes = (
                pack_note(p0, sample_map["chord1"]),
                pack_note(p1, sample_map["chord2"]),
                pack_note(p2, sample_map["lead"]),
                pack_note(p4, samp4)
            )
            pat.append((notes[0], notes[1], notes[2], notes[3]))
        patterns.append(pat)

    order = list(range(num_patterns))
    song_title = title if title else os.path.basename(out)
    write_mod(out, song_title, samples, order, patterns)
    print(f"Wrote {out}: {total_frames} frames → {num_patterns} patterns")
    if seed is not None:
        print(f"Render seed: {seed}")

def decode_from_mod(path: str, password: str = None) -> bytes:
    with open(path, "rb") as f:
        f.seek(20 + 31*30)
        song_len,_ = struct.unpack("BB", f.read(2))
        order      = list(f.read(128))[:song_len]
        f.read(4)  # signature
        num_pats   = max(order)+1 if order else 0

        frames = []
        for _ in range(num_pats*64):
            row = f.read(16)
            b0,b1,b2,_ = row[0:4]
            per0 = ((b0&0x0F)<<8) | b1
            b0,b1,b2,_ = row[4:8]
            per1 = ((b0&0x0F)<<8) | b1

            note0 = PERIODS_REVERSE.get(per0)
            note1 = PERIODS_REVERSE.get(per1)
            if note0 is None:
                note0 = min(PERIODS, key=lambda k:abs(PERIODS[k]-per0))
            if note1 is None:
                note1 = min(PERIODS, key=lambda k:abs(PERIODS[k]-per1))

            inv  = {v:k for k,v in NOTE_TO_MOD.items()}
            d0,d1 = inv[note0], inv[note1]
            frames.append(DYAD_TO_SYMBOL.get((d0,d1), 0))

    payload_syms = frames[PREAMBLE_FRAMES:]
    hdr_syms     = payload_syms[:HEADER_FRAMES]
    payload_len  = struct.unpack(">H", nibbles_to_bytes(hdr_syms))[0]
    start        = HEADER_FRAMES
    data_syms    = payload_syms[start:start+payload_len*2]
    crc_syms     = payload_syms[start+payload_len*2:start+payload_len*2+4]

    payload_b = nibbles_to_bytes(data_syms)
    crc_recv  = struct.unpack(">H", nibbles_to_bytes(crc_syms))[0]
    if crc_recv != crc16(payload_b):
        print("CRC mismatch"); return None

    if payload_b.startswith(ENC_MAGIC):
        if not HAS_CRYPTO:
            print("Encrypted data; install crypto libs"); return None
        # Use provided password if available; otherwise prompt.
        pw    = password if password is not None else getpass.getpass("Password: ")
        salt  = payload_b[4:4+SALT_LEN]
        nonce = payload_b[4+SALT_LEN:4+SALT_LEN+NONCE_LEN]
        ct    = payload_b[4+SALT_LEN+NONCE_LEN:]
        key   = hash_secret_raw(pw.encode(), salt,
                                time_cost=2, memory_cost=64*1024,
                                parallelism=1, hash_len=KEY_LEN,
                                type=ArgonType.ID)
        try:
            pt = crypto_aead_xchacha20poly1305_ietf_decrypt(ct, None, nonce, key)
        except:
            print("Decryption failed"); return None
    else:
        pt = payload_b

    if pt.startswith(ZSTD_MAGIC):
        if not HAS_ZSTD:
            print("Compressed payload; install zstandard"); return None
        dctx = zstd.ZstdDecompressor()
        try:
            pt = dctx.decompress(pt)
        except zstd.ZstdError:
            print("ZSTD decompression failed"); return None

    return pt

def main():
    ap = argparse.ArgumentParser(
        description="Chiptune 2-tone → Generates ProTracker .MOD (text/file payloads,"
                    " optional zstd + encryption)"
    )
    ap.add_argument("--encode",   help="String to embed into .MOD")
    ap.add_argument("--infile",   help="Path of file to embed")
    ap.add_argument("--out",      default="out.mod", help="Output .mod filename")
    ap.add_argument("--title",    default=None,    help="Song title (max 20 chars)")
    ap.add_argument("--compress", action="store_true",
                    help="Compress payload with zstd before encryption")
    ap.add_argument("--encrypt",  action="store_true", help="Use encryption")
    ap.add_argument("--password", help="Password (optional)")
    ap.add_argument("--seed",     type=lambda s: int(s, 0), help="Optional render seed (int, supports hex like 0xdeadbeef)")
    ap.add_argument("--decode",   help="Path to .MOD to decode")
    ap.add_argument("--outfile",  help="Path to write extracted file")
    args = ap.parse_args()

    if args.encode or args.infile:
        if args.infile:
            try:
                fsize = os.path.getsize(args.infile)
            except OSError:
                sys.exit(f"Cannot stat file: {args.infile}")
            # Quick pre-check: if raw file already exceeds the maximum, warn and exit.
            if fsize > MAX_PAYLOAD:
                sys.exit(f"Input file is too large to embed ({fsize} bytes). Maximum allowed is {MAX_PAYLOAD} bytes.")
            with open(args.infile, "rb") as f:
                payload = f.read()
        else:
            payload = args.encode.encode("utf-8")

        # If compression requested, we compress now and then validate final size
        if args.compress:
            if not HAS_ZSTD:
                sys.exit("Install zstandard for compression")
            cctx    = zstd.ZstdCompressor()
            payload = cctx.compress(payload)

        # Final size check before encoding/framing
        if len(payload) > MAX_PAYLOAD:
            sys.exit(f"Payload too large after processing: {len(payload)} bytes > {MAX_PAYLOAD} bytes maximum")

        encode_to_mod(payload,
                      args.encrypt,
                      args.password,
                      args.out,
                      args.title,
                      args.seed)

    elif args.decode:
        pt = decode_from_mod(args.decode, args.password)
        if pt is None:
            sys.exit(1)
        if args.outfile:
            with open(args.outfile, "wb") as f:
                f.write(pt)
        else:
            try:
                print("Decoded:", pt.decode("utf-8"))
            except UnicodeDecodeError:
                print("Decoded binary payload")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
