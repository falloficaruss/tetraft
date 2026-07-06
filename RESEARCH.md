Project TetraFT (Quantization-Aware Fine-Tuning for 2-Bit Quaternary LLMs)
I am starting a new, independent research track focused on Quantization-Aware Fine-Tuning (QAFT). The goal of this project, codenamed TetraFT, is to take an existing, pre-trained full-precision (FP16/BF16) Transformer model (e.g., GPT-2 or a tiny Llama variant) and surgically convert its linear layers into a multiplier-free, 2-bit quaternary format.
By utilizing a 4-state quaternary grid instead of BitNet's 3-state ternary grid, we want to prove that the extra representation capacity dramatically minimizes "representation shock" during the initial phase of fine-tuning, allowing the model to retain its pre-trained knowledge while compressing its disk footprint by up to 87.5%.

1. Mathematical Formulation
We represent our quaternary weights using a symmetric 4-state grid mapped to the set $\{-1, -c, c, 1\}$, where $c$ is a fixed, hardware-native power-of-two scaling parameter ($c = 0.25$ or $c = 0.5$).
Let $W$ be the continuous latent weight matrix initialized from a pre-trained layer. The forward pass quantization function $Q(W)$ is defined as:
$$Q(W) = \gamma \cdot \text{sign\_segment}\left( \frac{W}{\gamma} \right)$$
Where $\gamma$ is the scale factor (Mean Absolute Value of the weights):
$$\gamma = \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \sum_{i,j} |W_{i,j}|$$
And the segmentation function maps the normalized weight $x = \frac{W}{\gamma}$ to the discrete grid:
$$\text{sign\_segment}(x) = \begin{cases} 
-1.0 & \text{if } x < -\frac{1+c}{2} \\
-c & \text{if } -\frac{1+c}{2} \le x < 0 \\
c & \text{if } 0 \le x < \frac{1+c}{2} \\
1.0 & \text{if } x \ge \frac{1+c}{2}
\end{cases}$$
The Backward Pass (Straight-Through Estimator):
During backpropagation, we bypass the zero-derivative of the step function. The gradient of the loss $\mathcal{L}$ passes directly through to the continuous latent weights $W$, but we clamp gradient flow for saturated weights where $|x| > 1.0$ to ensure stability:
$$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial Q(W)} \cdot \mathbb{I}(|x| \le 1.0)$$
