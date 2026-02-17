# Readme.md

<aside>
🌃

from London weather example, we'll stop at building state space model from scratch. then we jump into what we have in mamba. the continuous case of state space model.

-

then introduce what is hippo matrix-then introduce the whole chapter of discretisation.

-

then explain why the state space model could be viewed as ODE/RNN/CNN(the inference between each view).

-

then we'll introduce linear time invariant is not enough, why?

-

selective state space model (mamba) is the solution, we have a quick example of how to make $\Delta t$/B/C time variant via the code.

- 

here remains the question, what about A? then we'll have a whole chapter explaining the evolution of state matrix A throughout the line of work. 

- 

now we have all ingredient of state-space model, let's talk about duality. why state space model could be viewed as a causal transformer. 

-

then we talk about the engineering side of mamba, list some application papers and view mamba as a block. In which scenarios we could use mamba?    

</aside>

---

### Chapter 1 — The London Weather Problem: what memory must do

**Storyline**

- Weather stream + “alien invasion” shock.
- Define the three memory behaviours: **retain**, **write**, **reset**.
- Conclude: we need a *machine* that updates an internal memory state over time.

**Leads to:** “Let’s build such a memory machine from first principles.”

---

### Chapter 2 — Build a State Space Model from scratch (continuous-time)

**Storyline**

- Introduce the idea of a **state** (hidden memory) and **observations** (what you see).
- Write down the continuous-time SSM form (no heavy derivations yet):
    - state evolves + input drives it + readout produces outputs
- Interpret each matrix’s job at a high level:
    
    **A = memory dynamics**, **B = how input writes**, **C = how memory is read**.
    

**Leads to:** “To make this memory useful, we need a principled choice of the memory dynamics (A).”

---

### Chapter 3 — HiPPO: where the state matrix comes from (and why it matters)

**Storyline**

- Problem: arbitrary A gives arbitrary (often bad) memory.
- Introduce **HiPPO** as “a principled way to compress the past.”
- Explain the intuition (projection/compression) and why it produces structured A.
- Key takeaway: **A isn’t random — it encodes a memory scheme.**

**Leads to:** “But our data is discrete tokens. We must convert continuous dynamics into discrete updates.”

*Remark: HiPPO强调reconstruction, which has it resonates with how brain functions*

<aside>
🌃

In cognitive neuroscience, memory is fundamentally reconstructive. The hippocampal-medial temporal lobe acts as an index, using partial cues to reactivate distributed cortical traces and rebuild past episodes. This mechanism enables flexible recall and future simulation without requiring passive, exhaustive storage.

HiPPO computationally formalises this principle. It compresses extensive data histories into finite-dimensional states while maintaining the ability to reconstruct the original signal. The parallel is not biological mimicry, but a shared abstraction: both systems prioritise compact, reconstructive representations over exhaustive archival.

</aside>

[The cognitive neuroscience of constructive memory: remembering the past and imagining the future](https://pmc.ncbi.nlm.nih.gov/articles/PMC2429996/)

Schacter DL, Addis DR. The cognitive neuroscience of constructive memory: remembering the past and imagining the future. Philos Trans R Soc Lond B Biol Sci. 2007 May 29;362(1481):773-86. doi: 10.1098/rstb.2007.2087. PMID: 17395575; PMCID: PMC2429996.

---

### Chapter 4 — Discretisation: turning continuous SSMs into token-time machines

**Storyline**

- Why discretisation exists: tokens arrive at steps; you need a stable update rule.
- Introduce the main discretisation choices (Forward/Backward Euler, Bilinear) as *design decisions*.
- Show what discretisation produces:
    - ( \bar A ) controls retention/forgetting per step
    - ( \bar B ) controls writing per step
- (Optional) show your reconstruction/empirical figure as a payoff.

**Leads to:** “Now that we have discrete SSMs, we can look at them from different computational viewpoints.”

---

### Chapter 5 — One model, three faces: ODE / RNN / CNN views

**Storyline**

- Present the three equivalent views and what each is *good for*:
    - **ODE view**: conceptual, stable dynamics story
    - **RNN view**: streaming inference (step-by-step)
    - **CNN/convolution view**: parallel training (kernel perspective)
- Explain the inference mapping between views (high-level, not a proof).
- Key takeaway: SSMs give you both **streaming** and **parallel training**.

**Leads to:** “So we’re done, right? Not yet—classic SSMs are still missing something crucial.”

*Remark: 在量子力学中，上帝掷骰子吗？Thinking hat metaphor.*

<aside>
🌃

Edward de Bono’s "Six Thinking Hats" and the Bohr-Einstein debates on quantum complementarity illustrate a shared epistemological principle: seemingly competing perspectives can be simultaneously valid, yielding different insights depending on the analytical objective.

This framework directly applies to State Space Models (SSMs). While an SSM is governed by a single underlying mathematical structure, it is operationalised through three distinct, complementary representations: continuous dynamics (Ordinary Differential Equations, ODE), sequential state updates (Recurrent Neural Networks, RNN), and parallelisable kernels (Convolutional Neural Networks, CNN). A rigorous understanding of applied SSMs requires integrating all three paradigms.

</aside>

[Completeness of Quantum Theory](https://sites.pitt.edu/~jdnorton/teaching/HPS_0410/chapters/quantum_theory_completeness/)

[Six Thinking Hats](https://www.debonogroup.com/services/core-programs/six-thinking-hats/)

---

### Chapter 6 — Why Linear Time-Invariant (LTI) is not enough

**Storyline**

- Tie back to London:
    - LTI means *fixed* response/forgetting curve.
    - It can’t treat “routine drizzle” and “alien invasion” differently.
- Make the limitation concrete: “importance is context-dependent.”
- Conclusion: we need **selectivity** (time-varying behaviour).

**Leads to:** “Mamba’s core idea is to make the SSM selective—input-dependent at each step.”

*Remark: 在神经科学中，强写入和弱写入，强重置，PTSD以及失忆。*

<aside>
🌃

In cognitive neuroscience, memory formation is not “written” at a constant strength. Salient or stressful events can trigger neuromodulatory and hormonal signals (for example noradrenaline and glucocorticoids) that prioritise certain experiences for stronger consolidation, while deemphasising irrelevant details; this is one reason emotionally charged memories can remain unusually vivid.

The same selectivity can become maladaptive. In **PTSD**, trauma-related memories may become over-strengthened and intrusive, and neurobiological reviews describe how stress-related systems and memory circuits contribute to these symptoms; conversely, disruptions to consolidation or reconsolidation processes can impair or even weaken stored memories—an intuition-level analogue of “resetting” or “erasing” what was written.

Abstractly, this looks like an efficiency trade-off: a limited-capacity system must decide *where* to spend its representational budget. 

That is why **linear time-invariant (LTI)** state space models feel insufficient for the London drizzle-versus-alien-invasion story. If biological memory benefits from context-dependent strong/weak write and occasional reset, then the modelling analogue is **selectivity**—time-varying behaviour that adapts its update to what is currently happening.

</aside>

[Interacting Brain Systems Modulate Memory Consolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3315607/)

[Post‐traumatic stress disorder: a state‐of‐the‐art review of evidence and challenges](https://onlinelibrary.wiley.com/doi/full/10.1002/wps.20656)

*Remark: in information theory, the tradeoff in Entropy.*

<aside>
🌃

In the London weather story, imagine each day falls into one of $K$ discrete “weather types”:

$$
X \in {1,2,\dots,K}
$$

where the categories might include *drizzle*, *overcast*, *sunny*, and — extremely rarely — *alien invasion*. Let the probability of each type be $p_X(x)$. In information theory, the information gained from observing a particular outcome $x$ is measured by its **surprisal**:

$$
⁍
$$

This captures a simple intuition: **rare events carry more information when they occur**. For everyday London drizzle, $p_X(\text{drizzle})$ might be relatively large, so $-\log p_X(\text{drizzle})$ is modest, it tells you little you didn’t already expect. But for alien invasion, $p_X(\text{alien})$ is undoubtedly tiny, so

$$
\log p_X(\text{alien}) \ \text{is huge},
$$

meaning that observing it delivers a very large “information shock”.

The **average** information in the stream is the Shannon entropy:

$$
H(X) = -\sum_{x=1}^{K} p_X(x)\log p_X(x)
$$

Which you can read as: *on average, how surprising is the world I’m observing?* 

In the London example, most days contribute small-to-moderate surprisal, but the rare categories contribute massive surprisal when they happen.

This is the metaphorical bridge to selectivity: a sensible memory system should not “write” each day into memory with equal strength. Routine drizzle (low surprisal) should lead to a gentle update, whereas alien invasion (high surprisal) should trigger **strong write** and possibly **reset**, because it fundamentally changes what the future is likely to look like. An LTI model, with a fixed update/forgetting profile, cannot naturally reallocate its memory budget in response to these sharp changes in information content; a selective model can.

</aside>

![image.png](Readme%20md/image.png)

[GitHub - felipe-tobar/Probabilistic-Generative-Models: Repository for the Probabilistic Generative Models module (MSc in Applied Maths, Imperial)](https://github.com/felipe-tobar/Probabilistic-Generative-Models/tree/main)

---

### Chapter 7 — Selective State Space: how Mamba makes B, C, and Δt time-varying

**Storyline**

- Explain selectivity with one knob: **update rate** (Δt).
- Show how Mamba generates **Δt, B, C** from the current input/context (your small code example).
- Interpret the effect:
    - small Δt → retain/slow update
    - large Δt → overwrite/reset quickly
- End with your exact question: “Great—but what about **A**?”

**Leads to:** “To answer ‘what about A,’ we need the historical evolution of how A is structured and computed.”

---

### Chapter 8 — The evolution of A: from structured SSMs to modern Mamba-style designs

**Storyline**

- A “guided history,” but *purpose-driven*:
    - Why structure A at all (speed + stability + long memory)
    - How different lines of work shaped A (and how that enables fast computation)
- Your main payoff:
    - A is constrained/parameterised so the model is efficient **and** expressive.
- Tie back: how this interacts with selectivity (what stays fixed vs what becomes input-dependent).

**Leads to:** “Now that we understand all ingredients (A/B/C/Δt), we can discuss a deeper equivalence: duality.”

*Remark: HiPPO-DPLR-Diagonal-scalar*identity*

---

### Chapter 9 — Duality: why an SSM can look like a causal Transformer

**Storyline**

- Start with the question: “How can a memory machine resemble attention?”
- Build the bridge via the convolution/kernel view:
    - causal weighting over the past = a learned history-to-now operator
- Explain the *conceptual* equivalence: both compute a causal mixing of past information, but with different parameterisations and compute profiles.
- End with: “So how do we actually use Mamba as a block in systems?”

**Leads to:** “Let’s talk about engineering: architecture block, practical usage, and where it wins.”

*Remark: 波粒二象性+时频二象性引出SSD*

<aside>
🌃

In perception and cognition research, it is common to meet a single underlying phenomenon through complementary descriptions that are each valid, yet emphasise different aspects of the same object. A famous metaphor is wave–particle duality: quantum entities are not “really” waves or “really” particles in a naïve sense, but our best understanding often requires both lenses, depending on what we are measuring and how we set up the experiment.

A second, more engineering-flavoured duality is time–frequency duality: the very same signal can be understood as an evolving waveform in time or as a spectrum over frequencies; and crucially, convolution in time becomes multiplication in frequency, making some computations dramatically simpler in one view than the other.

In this light, a state space model (and in particular the Mamba-style SSM) is the object that benefits from a similar dual description. In the “state update” view it is a memory machine—it rolls forward a compact state as new tokens arrive. In the “kernel/convolution” view it becomes a causal history-to-now operator—a learned weighting over the past that mixes previous inputs into the present. These are not two different models; they are complementary ways of looking at the same mechanism, and each view makes different properties obvious (streaming in the recurrent view; global mixing and parallelism in the convolutional view).

**This theoretical bridge is formally unified as State Space Duality (SSD). SSD mathematically proves that these linear state space models and structured masked attention mechanisms are actually computing the same underlying semi-separable matrices.**  That is why an SSM can start to look like a causal Transformer: both ultimately compute a causal mixture of past information to produce the present representation, but they arrive there with different parameterisations and computational trade-offs. The duality is the point: once you accept that one object can wear two “hats” **through the lens of SSD**, the comparison stops being superficial (“SSM vs attention”) and becomes structural (“two ways to implement a causal mixing operator”).

</aside>

[Quantum mechanics | Definition, Development, & Equations | Britannica](https://www.britannica.com/science/quantum-mechanics-physics)

[Convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem)

---

### Chapter 10 — Engineering Mamba: the block, the recipes, and when to use it

**Storyline**

- Present Mamba as a reusable **sequence block** (like you would present an attention block).
- Practical considerations:
    - training/inference behaviour (streaming, memory footprint)
    - integration patterns (hybrids, stacking, normalisation, etc.)
- Survey of applications (list papers + short bullets per domain).
- Clear guidance: scenarios where Mamba is a strong choice vs when alternatives are better.