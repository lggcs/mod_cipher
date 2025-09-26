# 🎶 mod_cipher — Chiptune Payload Encoder/Decoder

Ever wanted to hide data inside a **ProTracker .MOD chiptune**?
**mod_cipher** is a playful yet powerful tool that transforms text or files into 4-channel tracker music, embedding your payload into musical dyads and arpeggios. Think of it as **crypto-meets-chiptune steganography**.

---

## ✨ Features

- 🎼 **Chiptune Encoding**
  Converts text or binary data into a valid `.MOD` file playable in classic trackers.

- 🔒 **Optional Encryption** 
  Uses **Argon2id** for key derivation and **XChaCha20-Poly1305** for authenticated encryption.

- 📦 **Optional Compression** 
  Shrinks payloads with **Zstandard (zstd)** before embedding.

- 🧩 **Robust Framing**
  - CRC16 integrity checks
  - 2-byte payload length header (max ~64 KiB)
  - Musical dyads mapped to 4-bit symbols

- 🎹 **Authentic Sound**
  Generates square, triangle, pulse, noise, and organ-like waveforms for that retro tracker feel.

- 🔄 **Bidirectional**
  Encode → `.MOD` file with hidden data
  Decode → Extract original payload from `.MOD`

---

## ⚙️ Installation

Dependencies:
- Python 3.8+
- `numpy`
- Optional:
  - `argon2-cffi` + `pynacl` (for encryption)
  - `zstandard` (for compression)

Install requirements:

```
pip install numpy argon2-cffi pynacl zstandard
```

## 🚀 Usage

### Encode a String into a MOD
```
python mod_cipher.py --encode "Hello, world!" --out secret.mod --title "Chip Secrets"
```

### Encode a binary < 64kb with compression and encryption
```
python mod_cipher.py --infile data.bin --out secure.mod --compress --encrypt --password "myreallysecurepassword"
```

### Decode from a mod
```
python mod_cipher.py --decode secure.mod --outfile recovered.bin
```
If no `--outfile` is given, the tool will attempt to print the decoded payload as UTF-8 text.


## 🎛️ Options

| Flag          | Description |
|---------------|-------------|
| `--encode`    | Embed a string into a MOD |
| `--infile`    | Embed a file into a MOD |
| `--out`       | Output `.mod` filename (default: `out.mod`) |
| `--title`     | Song title (max 20 chars) |
| `--compress`  | Compress payload with zstd |
| `--encrypt`   | Encrypt payload with Argon2id + XChaCha20-Poly1305 |
| `--password`  | Password for encryption (optional, otherwise prompted) |
| `--decode`    | Decode a `.MOD` back into data |
| `--outfile`   | File to write extracted payload |


## ⚠️ Limitations
- Maximum payload size: 65535 bytes (~64 KiB)
- Requires optional libraries for encryption/compression
- Not intended for serious cryptographic security — this is more art-meets-tech than hardened stego
