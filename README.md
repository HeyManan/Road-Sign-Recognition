# Road Sign Recognition — CPU vs GPU CNN Benchmark (GTSRB)

A custom convolutional neural network that classifies German traffic signs — a core **perception** task for ADAS and self-driving systems — built from scratch and benchmarked across **CPU vs. GPU** to study hardware-acceleration tradeoffs.

> Built for *CIS 4930/6930 — Hardware Accelerators for Machine Learning*, University of South Florida (Spring 2025).

---

## Results at a glance

| Model | Hardware | Test Accuracy | Throughput | Params |
|-------|----------|--------------:|-----------:|-------:|
| Deep CNN | **NVIDIA Tesla T4 (GPU)** | **97.5%** | **22.5 GOPS** | 18.4M |
| Lightweight CNN | Apple M4 (CPU) | 91.9% | 1.0 GOPS | 321K |

The GPU-accelerated model delivered **~22× higher inference throughput** than the CPU model while running a network ~56× more computationally complex (473M vs. 8.5M MACs).

---

## Why this project

Traffic-sign recognition has to run reliably and in real time inside driver-assistance and autonomous-driving stacks. This project asks a practical hardware question on top of the ML one: *given the same task, how does the choice of hardware change what model you can realistically deploy?* The answer drove a **model–hardware co-design** approach — a lean network tuned for the CPU and a deeper, higher-capacity network for the GPU.

## Dataset

[GTSRB — German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/): 50,000+ real-world images across **43 sign classes**, with natural variation in lighting, occlusion, angle, and resolution.

- Images resized to 32×32×3, pixel values normalized to `[0,1]` with per-channel mean subtraction.
- Split: 80% train (20% of which is held out for validation) / 20% test.
- **Augmentation (training only):** random rotation (±5°), translation (±10%), and zoom (±10%) to mimic real sign appearance and improve generalization.

## Approach

Two CNN architectures were designed and trained from scratch in PyTorch:

| | CPU model (lightweight) | GPU model (deep) |
|---|---|---|
| Conv layers | 3 | 4 (two blocks) |
| Regularization | Dropout | Batch norm + dropout (0.3) |
| Dense layers | 256 → 43 | 512 → 43 |
| Parameters | ~321K | ~18.4M |
| Epochs | 50 | 30 |

Both models use the **Adam** optimizer (lr `0.0008`), **StepLR** scheduling (halved every 10 epochs), and cross-entropy loss.

## Hardware benchmarking

Inference was profiled over 5 runs per device, measuring runtime, throughput (GOPS), power, and energy efficiency:

| Metric | GPU (Tesla T4) | CPU (Apple M4) |
|--------|---------------:|---------------:|
| Throughput | 22.49 GOPS | 1.03 GOPS |
| Power | 33.9 W | — |
| Energy efficiency | 0.108 GOPS/W | — |
| MACs | 473M | 8.5M |

GPU power metrics were collected with NVIDIA's **`pynvml`** library; CPU usage was approximated with `psutil`.

**Takeaway:** hardware sets the ceiling on feasible model complexity. The GPU's parallelism made it practical to train a deeper network that generalizes better (97.5%), while the CPU favored a lighter model to keep inference tractable (91.9%).

## Run it

The fastest way to explore the project is the Colab badge at the top — it opens the notebook with no local setup. To run locally:

```bash
git clone https://github.com/HeyManan/road-sign-recognition.git
cd road-sign-recognition
pip install -r requirements.txt
```

Then open the notebook and run all cells. (A GPU runtime is recommended for the deep model.)

## Authors

- Manan Hareshbhai Patel
- Manjal Amrishkumar Patel
- Pearl Viralkumar Patel

Instructor: Dr. Dayane A. Reis · Teaching support: Parsa Khorrami

## References

Key works that informed the design: Krizhevsky et al. (2012, AlexNet); Simonyan & Zisserman (2014, VGG); Stallkamp et al. (2011, GTSRB); Khan et al. (2023, lightweight TSR CNN).
