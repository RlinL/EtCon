<div align="center">

# EtCon: Edit-then-Consolidate for Reliable Knowledge Editing

</div>

## 📖 Abstract

Knowledge editing aims to update specific facts in large language models (LLMs) without full retraining. Prior efforts sought to tune the knowledge layers of LLMs, proving effective for making selective edits. However, a significant gap exists between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning scenarios, which greatly limits their practical applicability.
This work's empirical analysis reveals two recurring issues associated with this gap:
(1) Most traditional methods lead the edited model to **overfit** to the new fact, thereby degrading pre-trained capabilities.
(2) There is a critical absence of a **knowledge consolidation stage**, leaving new facts insufficiently integrated into LLMs' inference-time behavior.
To this end, we propose **Edit-then-Consolidate (EtCon)**, a novel knowledge editing paradigm that aims to bridge the gap between theoretical knowledge editing methods and their real-world applicability. Specifically:
**(1) Edit Stage:** Our framework mitigates overfitting via **Targeted Proximal Supervised Fine-Tuning (TPSFT)** that localizes the edit via a trust-region objective to limit policy drift.
**(2) Consolidate Stage:** A consolidation stage using **Group Relative Policy Optimization (GRPO)** aligns the edited knowledge with CoT-based inference policy by optimizing trajectory-level behavior under comprehensive reward signals.
Extensive experiments demonstrate our framework consistently improves editing reliability and generalization under real-world evaluations, while better preserving locality and pre-trained capabilities.

---

## 💡 Motivation

### The Problem: Missing Knowledge Consolidation
Most traditional methods appear reliable under teacher-forced evaluation, but their performance drops in autoregressive generation (as in **Figure 1**), where edited knowledge is not consistently applied.

<div align="center">
  <img src="assets/figure1.png" alt="Figure 1: Illustration of the knowledge editing problem" width="85%">
  <br>
  <em>Figure 1: Illustration of the knowledge editing problem and our Edit-then-Consolidate solution.</em>
</div>

<br>

## 🧭 Our Method

### The Solution: EtCon Framework
**Figure 3** provides an overview of our two-stage approach:
1.  **Edit Stage (TPSFT):** Localized edits within selected FFN layers.
2.  **Consolidate Stage (GRPO):** Aligning parametric knowledge with CoT-based inference policy.

<div align="center">
  <img src="assets/figure3.png" alt="Figure 3: Overview of the Edit-then-Consolidate Framework" width="100%">
  <br>
  <em>Figure 3: Overview of the Edit-then-Consolidate (EtCon) Framework.</em>
</div>

## 🧑‍💻 Running EtCon

> 💪 **Work in Progress:** We are currently consolidating code from two codebases into a unified framework to simplyfy the usage of EtCon. 
**Coming soon!**

### 🛠️ Edit Stage (TPSFT)

```bash

```

### 🚀 Consolidation Stage (GRPO)

After finishing the edit stage, run the consolidation stage to align the edited knowledge with CoT-based inference:

1. Open `KE-Con/examples/train.sh` and fill in the paths for `MODEL_PATH`, `TRAIN_DATA`, `VAL_DATA`, `TENSORBOARD_DIR`, `EXPERIMENT_NAME`
2. Launch the GRPO consolidation run:

```bash
bash KE-Con/examples/train.sh
```

3. Merge the consolidated checkpoint to Hugging Face format (update the path to your checkpoint):

```bash
python3 KE-Con/scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```
