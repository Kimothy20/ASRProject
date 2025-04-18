"""Utility script to convert CommonVoice clips into WAV and create newline-delimited train/test JSON files for speech recognition."""

import os
import argparse
import json
import random
import csv
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def main(args):
    # Ensure output directory exists
    os.makedirs(args.save_json_path, exist_ok=True)

    # Determine clips directory
    clips_dir = args.clips_dir or os.path.join(os.path.dirname(args.file_path), "clips")
    if not os.path.isdir(clips_dir):
        print(f"[Error] Clips directory not found: {clips_dir}")
        return
    print(f"Using clips directory: {clips_dir}")

    # Count total lines in TSV (including header)
    with open(args.file_path, 'r', encoding='utf-8', errors='replace') as f:
        total_lines = sum(1 for _ in f)
    print(f"TSV line count (including header): {total_lines}")

    data = []

    # Read TSV rows
    with open(args.file_path, 'r', newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader, start=1):
            raw_path = row.get('path', '').strip()
            text     = row.get('sentence', '').strip()

            # Build candidate paths:
            candidates = []
            # 1. absolute from TSV
            if os.path.isabs(raw_path):
                candidates.append(raw_path)
            # 2. as-is under clips_dir
            candidates.append(os.path.join(clips_dir, raw_path))
            # 3. with .mp3 extension
            candidates.append(os.path.join(clips_dir, raw_path + '.mp3'))
            # 4. with .wav extension
            candidates.append(os.path.join(clips_dir, raw_path + '.wav'))

            # Pick the first that exists
            src_path = next((c for c in candidates if os.path.isfile(c)), None)
            if not src_path:
                print(f"[Warning] Audio file not found for row {idx}: tried {candidates}")
                continue

            key_path = src_path
            # Convert if requested
            if args.convert:
                base_name = os.path.splitext(os.path.basename(src_path))[0]
                wav_name  = base_name + ".wav"
                dst_path  = os.path.join(clips_dir, wav_name)
                try:
                    ext = os.path.splitext(src_path)[1].lower().lstrip('.') or 'mp3'
                    sound = AudioSegment.from_file(src_path, format=ext)
                    sound.export(dst_path, format='wav')
                    key_path = dst_path
                    print(f"\r[{idx}/{total_lines}] Converted {os.path.basename(src_path)} → {wav_name}", end='', flush=True)
                except CouldntDecodeError as e:
                    print(f"\n[Warning] Decode failed for {raw_path}: {e} (using original)")
                except Exception as e:
                    print(f"\n[Warning] Error converting {raw_path}: {e} (using original)")

            data.append({"key": key_path, "text": text})

    # Shuffle and split data
    print("\nShuffling and splitting data…")
    random.shuffle(data)

    n = len(data)
    if n == 0:
        print("[Error] No valid entries; verify TSV paths and clips directory.")
        return
    split = int(n * (1 - args.percent / 100.0))
    train = data[:split]
    test  = data[split:]

    # Write JSONL outputs
    for name, subset in [("train.json", train), ("test.json", test)]:
        out_file = os.path.join(args.save_json_path, name)
        with open(out_file, 'w', encoding='utf-8') as outf:
            for entry in subset:
                outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} entries to {out_file}")

    print(f"Done! Total processed entries: {n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CommonVoice clips to WAV and produce newline-delimited train/test JSON files."
    )
    parser.add_argument('--file_path',      required=True, help='Path to the CommonVoice TSV file')
    parser.add_argument('--clips_dir',      default=None, help='Directory where audio clips live (default: <tsv_parent>/clips)')
    parser.add_argument('--save_json_path', required=True, help='Directory to save train.json and test.json')
    parser.add_argument('--percent', type=int, default=10, help='Test split percentage (1-99)')
    parser.add_argument('--convert', action='store_true', default=False, help='Convert audio to WAV before JSON')
    args = parser.parse_args()

    if not (0 < args.percent < 100):
        parser.error("--percent must be between 1 and 99")
    main(args)
