# AnchorWM

## Install

Python 3.10/3.11, PyTorch with CUDA, and a CUDA toolkit with NVVM/libdevice for Numba.

```bash
pip install -r requirements.txt
```

## Run

```bash
python demo.py embed --prompt "A dog barks." --output outputs/example

python demo.py attack --input outputs/example/watermarked.wav \
  --output outputs/example/noisy.wav --kind noise --strength 20

python demo.py extract --input outputs/example/noisy.wav \
  --key outputs/example/key.npz --output outputs/example/result.json
```

Keep `key.npz` private and retain it for extraction. Use `--help` for options;
`embed --model /path/to/audioldm` loads a local checkpoint.

## Acknowledgements

Code references: [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading),
[pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda)
[AudioLDM](https://github.com/haoheliu/AudioLDM),
[Timbre Watermarking](https://github.com/TimbreWatermarking/TimbreWatermarking).
