import os
import argparse
import json
import random
import csv
from pydub import AudioSegment
from sox import SoxiError

def main(args):
    # ensure output dir exists
    os.makedirs(args.save_json_path, exist_ok=True)

    total_lines = 0
    with open(args.file_path, 'r', newline='') as f:
        total_lines = sum(1 for _ in f)

    clip_dir = os.path.join(os.path.dirname(args.file_path), "clips")
    data = []
    for idx, row in enumerate(csv.DictReader(open(args.file_path, newline=''), delimiter='\t'), start=1):
        src_name = row['path']
        text     = row['sentence']
        src_path = os.path.join(clip_dir, src_name)

        if args.convert:
            # build new .wav name
            base, _ = os.path.splitext(src_name)
            dst_name = base + ".wav"
            dst_path = os.path.join(clip_dir, dst_name)

            try:
                ext = os.path.splitext(src_path)[1].lower().lstrip('.')
                sound = AudioSegment.from_file(src_path, format=ext)
                sound.export(dst_path, format="wav")
            except SoxiError as e:
                print(f"\n[Warning] Skipping {src_name}: {e}")
                continue

            data.append({"key": dst_path, "text": text})
            print(f"\rConverting {idx}/{total_lines} → {dst_name}", end="", flush=True)
        else:
            data.append({"key": src_path, "text": text})

    print("\nShuffling and splitting data…")
    random.shuffle(data)

    n = len(data)
    split = int(n * (1 - args.percent/100.0))

    train = data[:split]
    test  = data[split:]

    for name, subset in [("train.json", train), ("test.json", test)]:
        out_path = os.path.join(args.save_json_path, name)
        with open(out_path, 'w', encoding='utf8') as j:
            for entry in subset:
                j.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CommonVoice clips to WAV and produce train/test JSONs."
    )
    parser.add_argument('--file_path',       required=True, help='path to the .tsv file')
    parser.add_argument('--save_json_path',  required=True, help='directory to save JSONs')
    parser.add_argument('--percent',  type=int, default=10, help='percentage for test set (0–100)')
    parser.add_argument('--convert', action='store_true', default=False,
                        help='whether to convert to WAV; if omitted, just splits JSON')
    args = parser.parse_args()

    if not (0 < args.percent < 100):
        parser.error("--percent must be between 1 and 99")

    main(args)
