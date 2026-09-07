# How fast will this run on my hardware?

Estimated time to process **1,000 trail-camera photos**.

## Read this first

Only two numbers here are measured. Everything else is extrapolated, and extrapolation
across GPU architectures is rough — treat the estimates as ±40%, useful for "is this
minutes or hours" decisions and not much finer than that.

**Measured on the development machine** (Ryzen AI 5 340, Radeon 840M iGPU, MegaDetector
v5a at 1280px):

| What | Throughput | 1,000 photos |
| --- | --- | --- |
| Detector on Radeon 840M (DirectML) | 0.86 img/s | **19 min** |
| Detector on CPU, 11 cores | 0.59 img/s | 28 min |
| Detector on CPU, 1 core | 0.22 img/s | 76 min |
| Species classifier (CPU, always) | 3.2 img/s | **5 min** |

Everything below scales that 840M number by peak FP32 throughput, assuming ~25% of peak
is achievable through CUDA and ~12% through DirectML (12% is what the 840M actually
achieved, so the DirectML column is anchored to reality; the CUDA figure is an
assumption).

## The classifier floor matters more than your GPU

The pipeline has two heavy stages. Only the detector uses the GPU — the species
classifier runs on CPU and takes **about 5 minutes per 1,000 photos no matter what
graphics card you own**.

So total time is roughly *detector time + 5 minutes*:

| GPU | Detector | Classifier | Total for 1,000 |
| --- | --- | --- | --- |
| Radeon 840M (this laptop) | 19 min | 5 min | **24 min** |
| GTX 1070 | ~2 min | 5 min | **7 min** |
| RTX 4070 | ~30 s | 5 min | **5.5 min** |
| RTX 4090 | ~10 s | 5 min | **5 min** |

Past roughly a GTX 1070, **the GPU stops being the bottleneck.** Going from a 1070 to a
4090 is a ~12x faster detector but only a ~1.4x faster job. That is the single most
useful thing on this page if you are deciding what to buy.

## NVIDIA (CUDA)

| GPU | ~FP32 TFLOPS | Est. img/s | 1,000 photos |
| --- | --- | --- | --- |
| GTX 1060 6GB | 4.4 | ~5 | ~3 min |
| GTX 1070 | 6.5 | ~8 | ~2 min |
| GTX 1080 Ti | 11.3 | ~13 | ~1.2 min |
| RTX 2060 | 6.5 | ~8 | ~2 min |
| RTX 2070 Super | 9.1 | ~11 | ~1.5 min |
| RTX 2080 Ti | 13.4 | ~16 | ~1 min |
| RTX 3050 | 9.1 | ~11 | ~1.5 min |
| RTX 3060 12GB | 12.7 | ~15 | ~1.1 min |
| RTX 3070 | 20.3 | ~24 | ~40 s |
| RTX 3080 | 29.8 | ~36 | ~30 s |
| RTX 3090 | 35.6 | ~43 | ~25 s |
| RTX 4060 | 15.1 | ~18 | ~55 s |
| RTX 4070 | 29.1 | ~35 | ~30 s |
| RTX 4080 | 48.7 | ~58 | ~17 s |
| RTX 4090 | 82.6 | ~99 | ~10 s |
| RTX 5070 | ~31 | ~37 | ~27 s |
| RTX 5080 | ~56 | ~67 | ~15 s |
| RTX 5090 | ~105 | ~126 | ~8 s |

Anything faster than ~20 img/s will likely be limited by JPEG decoding rather than the
GPU, so the bottom rows are optimistic in practice — see Caveats.

## AMD and Intel (DirectML on Windows)

DirectML is a translation layer and gives up perhaps half the efficiency of CUDA, which
is why these land lower than raw specs suggest. On Linux, ROCm would do better.

| GPU | ~FP32 TFLOPS | Est. img/s | 1,000 photos |
| --- | --- | --- | --- |
| Radeon 840M (measured) | 1.5 | 0.86 | **19 min** |
| Radeon 780M iGPU | 8.9 | ~5 | ~3 min |
| RX 6600 | 8.9 | ~5 | ~3 min |
| RX 6700 XT | 13.2 | ~8 | ~2 min |
| RX 6800 XT | 20.7 | ~12 | ~1.4 min |
| RX 7600 | 21.5 | ~12 | ~1.3 min |
| RX 7800 XT | 37.3 | ~21 | ~47 s |
| RX 7900 XTX | 61 | ~35 | ~29 s |
| Intel Arc A750 | 17.2 | ~10 | ~1.7 min |
| Intel Arc A770 | 19.7 | ~11 | ~1.5 min |

## Caveats

- **FP32 TFLOPS is a mediocre predictor.** Real throughput also depends on memory
  bandwidth, driver quality, and how well the model maps to the hardware. Two cards with
  the same TFLOPS can differ by 2x.
- **JPEG decode becomes the ceiling.** Above roughly 20-30 img/s the CPU can't decode
  photos fast enough to keep the GPU fed, so the fastest cards will not hit the numbers
  above. Reading from a slow SD card makes this worse — copy photos to an SSD first.
- **Tensor cores are not in play.** These figures are FP32. Running the model in FP16
  could roughly double NVIDIA throughput, but that is not currently enabled here.
- **eGPU enclosures** (like the Alienware Graphics Amplifier) run at reduced PCIe
  bandwidth. That hurts gaming far more than it hurts this workload, which is
  compute-bound rather than bandwidth-bound, so expect close to the figures above.
- **Laptop GPUs** run 20-40% slower than their desktop namesakes at the same model
  number, due to power limits.

## Reproducing this on your own hardware

```
python -m megadetector.detection.run_detector_batch <model.pt> <folder> out.json
```

It prints an images-per-second figure at the end. Divide 1,000 by that for your own
number, and please replace the estimate above with the measurement.
