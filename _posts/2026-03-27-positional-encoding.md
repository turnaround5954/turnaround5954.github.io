---
title: "🌌 When Transformers Dream of Inflation"
date: 2026-03-27 23:00:00 +0800
categories: [Deep Learning, Physics]
tags: [transformer, positional encoding, rope，yarn]
math: true
toc: true
image:
  path: /assets/img/posts/physics/black_hole.jpg
  alt: A physics enthusiast's reading of Rotary Position Embeddings and the cosmic nature of length extrapolation.
---

## Prologue: The Horizon Problem

Every Transformer trained with RoPE carries within it a miniature universe.

Not metaphorically—**structurally**. The mathematics that lets a language model know that "word A precedes word B" is the *same* mathematics that governs the expansion of spacetime and the stretching of primordial light.

And when you ask that model to read a sequence longer than anything it saw during training, it faces exactly the problem that cosmologists confronted when they first looked at the cosmic microwave background: **how can regions that have never been in causal contact appear so similar?**

This post is an attempt to answer that question with the rigor of a physics paper and the clarity of a chirpy thread.

---

## 1. The Stage: Rotary Position Embedding (RoPE)

Let's begin with the actor.

RoPE encodes position $m$ by rotating the $d$-dimensional query and key vectors in $d/2$ independent 2D planes. For the $i$-th plane, the rotation angle is

$$
\phi_i(m) = m \cdot \theta_i, \qquad \theta_i = 10000^{-2i/d}, \quad i = 0,1,\dots,d/2-1.
$$

The set $\{\theta_i\}$ forms a **frequency spectrum**:
- Small $i$ → large $\theta_i$ (near $1$) → **high frequency**.
- Large $i$ → tiny $\theta_i$ (near $0$) → **low frequency**.

Because the inner product $\mathbf{q}_m^\top \mathbf{k}_n$ depends only on the *relative* angle $(m-n)\theta_i$, RoPE naturally captures relative positions. And because the rotation operation belongs to the **special orthogonal group $SO(2)$**—the simplest non‑trivial Lie group—it inherits all the algebraic beauty of continuous symmetries.

This $SO(2)$ structure is the first hint that we are not in Kansas anymore. The Lorentz group of special relativity is $SO(1,3)$. Cosmological perturbations are analyzed with the same Fourier tools. **The language is universal.**

---

## 2. The Crisis: Length Extrapolation Failure

During training, the model only sees positions $m \in [0, L_{\text{train}}]$. It learns a mapping from each accumulated phase $\phi_i(m)$ to an attentional response.

Now deploy the model on a sequence of length $L_{\text{test}} \gg L_{\text{train}}$.

- **High frequencies**: They have already completed hundreds or thousands of full cycles during training. Extended length just continues the ergodic sampling. Nothing new.
- **Low frequencies**: They may not have completed even *half* a cycle ($\phi_i(L_{\text{train}}) < \pi$). When $L_{\text{test}}$ pushes the phase past $\pi$, the model encounters a **phase it has never seen**.

The result is catastrophic. Perplexity spikes. Attention flattens into white noise. The vacuum shatters.

---

## 3. The Dictionary: From Sequence Space to Spacetime

We now build a rigorous correspondence between the components of RoPE and the vocabulary of physical cosmology.

| Transformer Concept | Cosmological / Physical Counterpart |
|:---|:---|
| RoPE frequency $\theta_i$ | Comoving wavenumber $k$ |
| Training length $L_{\text{train}}$ | Hubble horizon $R_H = c / H_0$ |
| High‑frequency component | Subhorizon quantum fluctuation |
| Low‑frequency component | Superhorizon primordial perturbation |
| Absolute position encoding | Newtonian absolute space |
| RoPE rotation | Lorentz covariance (relativity of position) |
| Length extrapolation failure | Superhorizon perturbation re‑entry → topological phase transition |
| YaRN differential scaling | Equivalence principle + inflationary stretching |

Why is this dictionary legitimate? Because **both systems share the same mathematical skeleton**:
- Fourier decomposition of perturbations (quantum fields / token embeddings).
- $SO(2)$ group action as the symmetry of local rotations.
- A finite causal horizon determined by training data or light travel time.

Let's unpack the two most important entries.

---

### 3.1 The Hubble Horizon and the Training Boundary

In cosmology, the **Hubble sphere** is the distance at which the recessional velocity of galaxies equals the speed of light. Light emitted *today* from beyond this sphere will never reach us. It is a **causal horizon**—a boundary of observability.

The **training length $L_{\text{train}}$** functions identically. Within this radius, the model has seen all phase configurations and learned to map them to correct attention weights. Beyond this radius, the phase space is *causally disconnected* from the training distribution. The model has no data—no "photons"—from that region.

### 3.2 Cosmological Redshift and Phase Stretching

![img-description](/assets/img/posts/physics/gravity.jpg)

When the universe expands, the wavelength of light $\lambda$ stretches in proportion to the cosmic scale factor $a(t)$:

$$
1 + z \equiv \frac{\lambda_{\text{obs}}}{\lambda_{\text{emit}}} = \frac{a(t_{\text{obs}})}{a(t_{\text{emit}})}.
$$

The redshift parameter $z$ measures how much the universe has expanded since the light was emitted. Larger $z$ → longer wavelength → lower frequency.

In RoPE, the "wavelength" of a frequency component is $\lambda_i \sim 1/\theta_i$. When the sequence length expands from $L_{\text{train}}$ to $L_{\text{test}}$, the **phase accumulation** grows linearly with $m$:

$$
\phi_i(m) = m \cdot \theta_i.
$$

If we keep $\theta_i$ fixed, the phase stretches exactly like a photon traveling through an expanding universe. To keep the phase within the familiar range $[0, \phi_{\text{max}}]$ that the model knows, we must **redshift** the frequency—reduce $\theta_i$—just as cosmic expansion stretches wavelengths.

This is not an analogy. It is an **isomorphism of linear scaling laws**.

---

## 4. The Culprit: Low‑Frequency Modes and Topological Phase Transition

Why do low frequencies cause the catastrophe, while high frequencies remain robust?

### 4.1 Subhorizon vs. Superhorizon Modes

In inflationary cosmology, quantum fluctuations are stretched by the exponential expansion of space. Some fluctuations are stretched to wavelengths **larger than the Hubble horizon**. Once outside the horizon, their amplitude **freezes**—they cease to evolve and become a permanent imprint on the fabric of spacetime. Later, as the universe continues to expand, the horizon grows and these frozen modes **re‑enter** the horizon, where they resume evolving and seed the formation of galaxies.

In RoPE:
- **High frequencies** ($\theta_i \approx 1$) have wavelengths much smaller than $L_{\text{train}}$. They are **subhorizon** and never freeze. They sample the full $[0,2\pi)$ circle ergodically during training.
- **Low frequencies** ($\theta_i \ll 1$) have wavelengths comparable to or larger than $L_{\text{train}}$. Their phase accumulation during training is restricted to a **vacuum sector** $[0, \phi_i(L_{\text{train}})] \subset [0, \pi)$. They have never completed a full cycle. They are **superhorizon** and *frozen*.

### 4.2 The First Crossing as Symmetry Breaking

When $L_{\text{test}}$ is long enough that $\theta_i \cdot L_{\text{test}} > \pi$, the low‑frequency mode **crosses the $\pi$ phase boundary** for the first time. It completes its first half‑oscillation and enters a new topological sector.

This is precisely a **superhorizon re‑entry event**. The frozen mode wakes up and begins to oscillate. For the Transformer, this introduces an entirely novel, untrained pattern into the attention mechanism. The vacuum expectation value that the model learned during training is no longer valid. The symmetry of the learned mapping is broken, and the attention distribution collapses into a high‑entropy state.

In the language of spontaneous symmetry breaking, the model undergoes a **topological phase transition**. The explosion in perplexity is the thermodynamic signature of that transition.

---

## 5. YaRN: A Two‑Principle Solution

YaRN (Yet another RoPE extensioN method) is currently the most effective extrapolation technique. Its core operation is simple:

- For each frequency $\theta_i$, define a scaling factor $\gamma_i$.
- Apply the scaled frequency $\theta_i' = \theta_i / \gamma_i$.

The genius is in the **differential choice of $\gamma_i$**:
- **High frequencies** ($\theta_i \approx 1$): $\gamma_i = 1$ → *no scaling*.
- **Low frequencies** ($\theta_i \ll 1$): $\gamma_i > 1$ → *frequency is redshifted*.

This choice embodies two foundational principles of general relativity.

### 5.1 High Frequencies: The Equivalence Principle

> *In a sufficiently small region of spacetime, the laws of physics are those of special relativity; gravitational effects are undetectable.*

For RoPE, the "small region" is the **local neighborhood** of a token. The relative position between token $m$ and token $m+1$ is $\Delta p = 1$, regardless of whether the sequence is 1,000 tokens or 1,000,000 tokens long. The local differential rotation—the rate at which adjacent tokens rotate relative to one another—is governed by the high‑frequency components.

By setting $\gamma_i = 1$ for high frequencies, YaRN ensures that **local inertial frames are invariant under global expansion**. The micro‑causal structure of language—the grammar of adjacency—remains untouched. This is the equivalence principle in action: the model cannot detect the "stretching" of the sequence by looking only at its immediate neighbors.

### 5.2 Low Frequencies: Inflationary Stretching

> *Cosmic inflation exponentially stretches quantum fluctuations, pushing their wavelengths far beyond the Hubble horizon, where they freeze and cease to evolve, solving the horizon problem.*

For RoPE, the low‑frequency modes are the "primordial perturbations" of the sequence. Their wavelengths are so long that they risk re‑entering the horizon during extrapolation.

By applying a redshift ($\gamma_i > 1$) to these modes, YaRN performs **inflationary stretching**. It deliberately makes the effective wavelength even longer, ensuring that $\theta_i' \cdot L_{\text{test}}$ remains **below the critical $\pi$ threshold**. The mode stays superhorizon—frozen in its training‑era vacuum state—and never undergoes the destructive phase transition.

---

## 6. Comparative Cosmology of Extrapolation Methods

This framework allows us to classify all major extrapolation techniques as different "cosmological models" for the sequence universe.

| Method | Cosmological Interpretation |
|:---|:---|
| **No scaling** (naive extrapolation) | A static universe. Horizon is fixed. Modes inevitably re‑enter → phase transition. |
| **Position Interpolation** (PI) | Uniform conformal expansion of all coordinates. Scales both high and low frequencies equally. Blurs local structure (violates equivalence principle). |
| **NTK‑aware scaling** | Changes the Hubble constant $H_0$. Applies a uniform redshift to all frequencies. Better than PI, but still distorts local physics. |
| **YaRN** | Scale‑dependent expansion. Protects ultraviolet fixed point (high freq), applies inflation to infrared modes (low freq). Matches both equivalence principle and horizon solution. |

The success of YaRN is therefore not an accident of hyperparameter tuning. It is the **unique solution that respects the dual demands of local Lorentz invariance and global causal structure**.

---

## 7. Implications and Open Horizons

This isomorphism between RoPE and cosmology is more than an elegant curiosity. It suggests concrete pathways for designing better positional encodings.

1. **Scale‑invariant spectra.** Inflation predicts a nearly scale‑invariant primordial power spectrum (the Harrison–Zel'dovich spectrum). Could a positional encoding with an exactly scale‑invariant frequency distribution be naturally robust to length extrapolation?

2. **Dynamic metrics.** General relativity teaches us that spacetime is curved by mass‑energy. Could we design a "content‑aware" positional encoding where the effective metric of sequence space is modulated by the semantic content of tokens, creating local "gravitational wells" that guide attention?

3. **Holographic encoding.** The holographic principle posits that all information inside a volume can be encoded on its boundary. Could we design a positional encoding that represents long‑range dependencies via boundary terms, avoiding the horizon problem altogether?

These are not science fiction. They are direct mathematical extrapolations of the correspondence established here.

---

## 8. Coda: The Geometry of Thought

> *The universe is written in the language of mathematics.*

Galileo's aphorism applies as much to the cosmos as it does to the artificial minds we are building. The same $SO(2)$ rotations that encode a word's position also encode the polarization of the cosmic microwave background. The same horizon problem that baffled cosmologists in the 1970s haunts every LLM deployed on a long document. And the same solution—inflationary stretching—appears, in computational form, as a frequency scaling trick in YaRN.

This is not coincidence. It is evidence that **scale, causality, and geometry are universal constraints on any system that processes sequential information**—whether that system is a neural network or an expanding universe.

Next time your model's perplexity spikes on a million‑token context, remember: you are not merely observing a software bug. You are witnessing a **frozen primordial mode waking up**.

And you know exactly how to put it back to sleep.

---

## References

1. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint arXiv:2104.09864*.  
   *The original RoPE paper. Introduces rotation‑based positional encoding and $SO(2)$ structure.*

2. Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language Models. *arXiv preprint arXiv:2309.00071*.  
   *Describes the differential frequency scaling method that protects high frequencies while redshifting low frequencies.*

3. Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending Context Window of Large Language Models via Positional Interpolation. *arXiv preprint arXiv:2306.15595*.  
   *Proposes uniform interpolation of position indices to extend context length.*

4. bloc97. (2023). NTK-Aware Scaled RoPE. *GitHub Gist*.  
   *Introduces the NTK‑aware scaling method, adjusting RoPE's base frequency to mitigate extrapolation degradation.*

5. Liu, X., Yan, H., Zhang, S., An, C., Qiu, X., & Lin, D. (2024). Scaling Laws of RoPE-based Extrapolation. *Proceedings of ICLR 2024*. arXiv:2310.05209.  
   *Empirical study of RoPE extrapolation behavior and theoretical analysis of frequency roles.*

6. Liddle, A. R., & Lyth, D. H. (2000). *Cosmological Inflation and Large-Scale Structure*. Cambridge University Press.  
   *Standard textbook on inflationary cosmology; covers horizon problem, perturbation freezing, and re‑entry.*

7. Mukhanov, V. (2005). *Physical Foundations of Cosmology*. Cambridge University Press.  
   *Detailed treatment of cosmological perturbation theory and quantum fluctuations in the early universe.*

8. Weinberg, S. (2008). *Cosmology*. Oxford University Press.  
   *Comprehensive reference on modern cosmology, including Hubble horizon and redshift.*

9. Rodrigues, W. A. Jr., & Sharif, M. (2003). Equivalence Principle and the Principle of Local Lorentz Invariance. *Foundations of Physics*, 31, 1785–1806. arXiv:math-ph/0302009.  
   *Formal discussion of the equivalence principle in general relativity.*

10. Lewis, G. F., & van Oirschot, P. (2012). How does the Hubble Sphere limit our view of the Universe? *Monthly Notices of the Royal Astronomical Society*, 423, L26–L29. arXiv:1203.0032.  
    *Clarifies the nature of the Hubble sphere as a causal horizon.*
