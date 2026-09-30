| Agent | Success (%) [95% CI] | Invalid (%) | WM cost / ep. |
|---|---|---|---|
| Base agent, no WM (C) | 17.0 [7.3, 29.2] | 87.8 | – |
| + failure memory, no WM (C_fm) | 79.5 [70.5, 87.5] | 35.0 | – |
| JEV, typed, frozen (A) | 93.4 [85.1, 98.6] | 15.4 | 129 req., $0.010 |
| Qwen, typed, zero-shot (B0) | 83.0 [73.3, 91.3] | 48.6 | 49 GPU-s |
| Qwen, generative, zero-shot (D0) | 59.0 [48.6, 69.1] | 36.3 | 115 GPU-s |
| Qwen+LoRA, typed (B) | 92.7 [85.8, 97.9] | 13.2 | 42 GPU-s |
| Qwen+LoRA, generative (D) | 92.7 [85.8, 97.9] | 13.2 | 73 GPU-s |
| Validity oracle | 85.8 [77.8, 92.7] | 10.8 | – |
| Full oracle | 92.7 [85.8, 97.9] | 12.8 | – |

N = 6,000 training transitions for B and D (B 13.9, D 4.5 GPU-h). Parse failures per one-step prediction: D0 26.0%, D 0.1%. B and D matched the full oracle's outcome in every episode.
