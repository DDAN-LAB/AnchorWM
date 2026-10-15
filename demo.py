import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from attacks import DEFAULTS, SAMPLE_RATE, perturb, read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embed = commands.add_parser("embed", help="Generate watermarked audio and its private key record")
    embed.add_argument("--prompt", required=True)
    embed.add_argument("--model", default="cvssp/audioldm")
    embed.add_argument("--output", type=Path, default=Path("outputs/example"))
    embed.add_argument("--steps", type=int, default=50)
    embed.add_argument("--guidance-scale", type=float, default=7.5)
    embed.add_argument("--seed", type=int, default=0)
    embed.add_argument("--message", help="Exactly 256 binary digits; omitted: random payload")
    embed.add_argument("--device", default="cuda")

    attack = commands.add_parser("attack", help="Apply a waveform perturbation")
    attack.add_argument("--input", type=Path, required=True)
    attack.add_argument("--output", type=Path, required=True)
    attack.add_argument("--kind", choices=DEFAULTS, default="noise")
    attack.add_argument("--strength", type=float, help="SNR dB / crop fraction / speed rate / sample rate Hz")
    attack.add_argument("--seed", type=int, default=0)

    extract = commands.add_parser("extract", help="Invert audio and extract bits using a private key record")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--key", type=Path, required=True)
    extract.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    extract.add_argument("--model", help="Override the checkpoint location stored in the key record")
    extract.add_argument("--inner-steps", type=int, default=8, help="Adam updates per timestep; 0: plain DDIM")
    extract.add_argument("--lr", type=float, default=0.01)
    extract.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.command == "attack":
        wave = perturb(read_audio(args.input), args.kind, args.strength, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Float WAV avoids clipping the noise attack and changing its requested SNR.
        sf.write(args.output, wave, SAMPLE_RATE, subtype="FLOAT")
        print(args.output)
        return

    import torch
    from audioldm import AudioLDM
    from watermark import Watermark

    if args.command == "embed":
        bits = None
        if args.message is not None:
            if len(args.message) != 256 or set(args.message) - {"0", "1"}:
                parser.error("--message must contain exactly 256 binary digits")
            bits = np.array([int(b) for b in args.message], dtype=np.uint8)
        # Avoid mixing a newly generated waveform with an existing private record.
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("--output must be a new or empty directory")
        wm, noise = Watermark.create(args.seed, bits)
        model = AudioLDM(args.model, args.device, args.steps)
        wave = model.generate(noise, args.prompt, args.guidance_scale)
        anchors = model.anchor_trajectory(noise)
        args.output.mkdir(parents=True, exist_ok=True)
        sf.write(args.output / "watermarked.wav", wave, SAMPLE_RATE, subtype="FLOAT")
        key_path = args.output / "key.npz"
        np.savez_compressed(key_path, bits=wm.bits, key=np.frombuffer(wm.key, dtype=np.uint8),
                            nonce=np.frombuffer(wm.nonce, dtype=np.uint8),
                            anchors=anchors.numpy(), steps=args.steps, model=args.model)
        key_path.chmod(0o600)
        print(f"Audio: {args.output / 'watermarked.wav'}\nPrivate record: {key_path}")
    else:
        with np.load(args.key, allow_pickle=False) as record:
            wm = Watermark(record["bits"], record["key"].tobytes(), record["nonce"].tobytes())
            anchors = torch.from_numpy(record["anchors"].copy())
            model_id = args.model or str(record["model"].item())
            steps = int(record["steps"].item())
        model = AudioLDM(model_id, args.device, steps)
        latent = model.encode_audio(args.input)
        noise, losses = model.invert(latent, anchors, args.inner_steps, args.lr)
        result = wm.score(wm.extract(noise))
        result.update(input=str(args.input), model=model_id, steps=steps,
                      inner_steps=args.inner_steps, lr=args.lr, losses=losses)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Recovered {result['correct_bits']}/256 bits ({result['bit_accuracy']:.2%})")
        print(args.output)


if __name__ == "__main__":
    main()
