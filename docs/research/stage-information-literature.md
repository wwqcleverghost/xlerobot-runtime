# Stage information in visuomotor policies: literature check

中文：[stage-information-literature.zh.md](stage-information-literature.zh.md)

Search date: 2026-11-06. Purpose: for each of the six research points of the XLeRobot handover project (9 stages from
[`examples/xlerobot/stage_labels.py`](../../examples/xlerobot/stage_labels.py); v6 variants v6stage, v6stagepred, v6progpred,
v6rabc, v6stagehead vs baseline v6_flow_h16_s8), find the closest prior work, say what is the same and what is different, and
judge whether the point is **already done / partly done / possibly new**.

How it was checked: WebSearch, then the arXiv abstract page (and the HTML full text where marked) was opened for the key papers.
**✓** = read on arXiv (abstract, or the quoted section of the HTML). **(unverified)** = only seen in search snippets or
second-hand summaries; numbers marked unverified must not be cited without reading the paper. "Not found" means not found in this
search, not proof that nothing exists.

## Summary

| # | Point | Verdict |
|---|---|---|
| 1 | Causal, event-rule frame-wise stage labels, no human / language labels | **Partly done**. Automatic stage labels from gripper events exist (StaKe, MimicGen-style termination signals); a causal labeler that is *the same code* in training and live deployment, with multi-signal events for a bimanual handover, was not found |
| 2 | Stage as policy **input** in the closed loop, vs auxiliary target, vs RA-BC weighting, vs learned memory | Each use alone is **done** (PHASER, ABC, MS-Bot / StaKe, SADP / SARM / CAMP, TF-DP). A **same-data, same-robot comparison of all four** was **not found → possibly new** |
| 3 | Closed-loop stage-misjudgement failures and fixes (debounce, allow regression, HMM/HSMM forward filter) | Fix techniques are **done** in other fields (surgical phase, speech, online action segmentation, RL switching). Analysis of the *policy ↔ stage-estimator feedback loop* in a visuomotor policy, and an HSMM filter feeding a diffusion policy: **not found → possibly new** |
| 4 | Stage-gated inputs (cameras / joints scaled by a stage gate, FiLM, adapters) | **Done** (MS-Bot, CoRL 2024 oral; PHASER FiLM; Selective Perception camera router; MoE-DP) |
| 5 | Low-human stage labeling (rules + classifier, few labels + DTW/TCC, VLM, active learning) | **Done** as separate pieces; the combination for robot stage labels is engineering, not a research claim |
| 6 | Predicted stage fed back as input → copying / error accumulation | General mechanism **done** (copycat problem, causal confusion, surgical error cascades). No current variant feeds the prediction back (v6stagehead's head is logged only); a controlled test is **planned**, and showing it for a policy's *own* predicted stage was **not found → possibly new (as an analysis)** |

**On "am I just copying SARM?"** SARM ([2509.25358](https://arxiv.org/abs/2509.25358) ✓) is a *reward model*: a stage
classifier + within-stage progress regressor trained from human subtask annotations (protocol-based; "Qwen" does not appear in the
paper's HTML v2; LeRobot's SARM implementation defaults to Qwen3-VL-30B for annotation, see point 5), used to *weight* BC samples
(RA-BC). It never feeds the stage to the policy. Of our variants only **v6rabc** follows SARM (with rule labels instead of a learned
reward model) and is an application of it. v6stage / v6stagehead (rule stage as input; v6stagehead adds an auxiliary
stage head that does not see the input and is not fed back) are closer to PHASER and ABC; v6stagepred /
v6progpred (stage as an extra denoised target) are closer to StaKe / SADP. None of these papers compares the uses against each other.

## 1. Causal, rule-based frame-wise stage labels

**Closest work**

- **StaKe** — *Improving VLA Fine-Tuning with Structured Stage and Keyframe Supervision*, [2606.26801](https://arxiv.org/abs/2606.26801) ✓ (HTML read).
  "A fully automatic pipeline that extracts stage labels and keyframe targets from demonstration gripper states … without any manual
  annotation." Stages come from gripper open/close transitions (both arms merged for bimanual). Labels are used **only as auxiliary
  heads during training**; inference unchanged. Reported relative gains 14 % (bimanual sim) and 56 % (Franka real) ✓ (abstract).
- **MimicGen subtask termination signals** — [docs](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) ✓.
  Binary per-subtask signals from the environment (privileged sim state), or manual annotation; used to split source demos for data
  generation, not as policy input.
- Other automatic segmentation (unverified, snippets): SparTa (gripper state / low velocity), DynaMimicGen ([2511.16223](https://arxiv.org/abs/2511.16223), object state events),
  *The Art of Imitation* ([2407.13432](https://arxiv.org/abs/2407.13432), velocity thresholds), UVD ([2310.08581](https://arxiv.org/abs/2310.08581), embedding-distance plateaus, no labels, NeurIPS 2023).
- **PHASER** ([2605.29407](https://arxiv.org/abs/2605.29407) ✓ HTML): phases labelled **by the operator pressing a controller button** during teleoperation.
- **MS-Bot** ([2408.01366](https://arxiv.org/abs/2408.01366) ✓ HTML): stage labels added to every sample; stage division by humans, with automatic (LLM) labeling named as future work (the latter from search summary, unverified).

**Difference from ours**: StaKe and MimicGen-style labels use one signal (gripper) or privileged state, offline over the whole
trajectory. Ours: (a) strictly **causal** — each event uses only current and past frames and the *previous* command, so the
**same function** produces training labels and the live stage at eval; (b) multi-signal events (FK tip speed along the gripper
axis, tape diameter in the wrist camera, the other gripper entering A's wrist-camera cone, return-to-start distance);
(c) bimanual handover roles A/B decided online; (d) explicit regressions on drops.

**Verdict: partly done.** "Automatic stage labels from gripper events, no human" is done (StaKe). The new part is narrow:
train/deploy-identical causal labeling that can be used as a live policy input. On its own it is an engineering contribution;
it becomes a research point only together with point 2 or 3 (e.g. showing label/live mismatch hurts, or that causality matters).

## 2. Stage as policy input vs other uses

**Stage as input, closed loop**

- **PHASER** — [2605.29407](https://arxiv.org/abs/2605.29407) ✓ (HTML, IEEE/ASME T-Mech). FiLM-conditioned ACT on the phase; a separate multimodal phase
  predictor (vision + force + pose, frozen policy backbone + MLP) estimates the phase in real time and its output is fed back to the
  policy. Ablation: FiLM beats token-level conditioning and unconditioned ✓. Hanging success 56 % → 87 % with recovery ✓. Argues that
  without phase the policy learns a mixture whose "competing modes produce averaged or oscillatory behavior at aliased states" ✓.
- **ABC** — *Scalable Behavior Cloning with Open Data, Training, and Evaluation*, [2606.27375](https://arxiv.org/abs/2606.27375) ✓ (HTML App. H.3). A small subtask
  classifier "inspired by the stage-aware reward model of SARM" sets the subtask prompt of a DiT policy at inference. T-shirt,
  half-folded start: without prompting re-flattening in 5/10 trials, with prompting none (1 trial timed out after a missed grasp) ✓.
  Subtask conditioning is applied with probability 0.2 in training ✓. Same group as SARM (Wu, Shentu, Yu, Abbeel).
- **MS-Bot** — [2408.01366](https://arxiv.org/abs/2408.01366) ✓ (HTML, CoRL 2024 oral). Stage predicted *inside* the policy (gate MLP on a state token,
  soft stage tokens, a loss that tolerates errors near stage boundaries) and used to drive fusion.
- **SADP** — [2605.16871](https://arxiv.org/abs/2605.16871) ✓ (RA-L 2026). Diffusion policy conditioned on natural-language subgoals (labels from foundation
  models) plus an auxiliary continuation-score head that drives online subgoal switching.
- **BEACON** — [2609.22730](https://arxiv.org/abs/2609.22730) ✓. Diffusion policy conditioned on a Bayesian belief over hidden state (estimate + uncertainty).

**Stage as auxiliary target**

- **StaKe** (above) ✓: stage classifier + keyframe head as auxiliary heads, separate from the action head.
- Multi-granularity language-guided IL, [2609.37135](https://arxiv.org/abs/2609.37135) ✓ (abstract); the "subtask-aware loss" detail is from snippets (unverified).
- PALM, [2601.07060](https://arxiv.org/abs/2601.07060) (unverified): stage + within-stage progress prediction.

**Stage / progress as training weight**: **SARM + RA-BC** [2509.25358](https://arxiv.org/abs/2509.25358) ✓ (HTML). Progress labels from the average
temporal proportion of each annotated subtask, linear inside a subtask ✓; weights from the progress delta over one chunk ✓.
Related: π\*0.6 / RECAP [2511.14759](https://arxiv.org/abs/2511.14759) conditions the policy on a binarized advantage instead of weighting (unverified, secondary sources).

**Learned memory / history**: **CAMP** [2606.21188](https://arxiv.org/abs/2606.21188) ✓ (compressed action memory, "implicitly track generalized task progress …
without any additional supervision"); **TF-DP** [2602.07388](https://arxiv.org/abs/2602.07388) ✓ (execution trace projected into the image, "stage-aware context");
DSSP [2605.14598](https://arxiv.org/abs/2605.14598), SeedPolicy [2603.05117](https://arxiv.org/abs/2603.05117) (unverified).

**Difference from ours**: every paper above tests **one** way to use stage information against an unconditioned baseline. No paper
found compares *input (v6stage)*, *joint-denoised auxiliary target (v6stagepred / v6progpred)*, *RA-BC weighting (v6rabc)*,
*rule input + auxiliary head that does not see the input (v6stagehead)* and *learned memory* on the same data, robot and scoring. Putting the stage one-hot **into the
diffusion/flow target vector** (denoised together with the action) rather than a separate head was not seen either (StaKe, SADP,
MS-Bot use separate heads).

**Verdict**: "stage as input helps in closed loop" is **already shown** (PHASER, ABC) — do not claim it as new. The **systematic
comparison** is **possibly new** and is the strongest candidate contribution, provided it is run with matched seeds/steps and the
eval-scoring CIs.

## 3. Closed-loop stage-misjudgement failures and fixes

**Closest work**

- **Online surgical phase recognition** — *Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition*,
  [2605.16387](https://arxiv.org/abs/2605.16387) ✓. Early misclassifications "propagate forward to form error cascades"; memoryless frame-wise decisions are
  unstable; an *Evidence-Gated Transition Predictor* allows a phase change only when accumulated evidence exceeds a threshold
  (≈ hysteresis/debounce). Earlier: left-to-right HMMs for online phase recognition (Blum et al., MICCAI 2008,
  [pdf](https://www.campar.in.tum.de/pub/blum2008miccai/blum2008miccai.pdf), unverified); TeCNO causal TCN ([2003.10751](https://arxiv.org/abs/2003.10751), unverified details).
- **Online temporal action segmentation**: OnlineTAS ([2411.01122](https://arxiv.org/abs/2411.01122), NeurIPS 2024) and O-TALC ([2404.06894](https://arxiv.org/abs/2404.06894)) — post-processing against
  over-segmentation in streaming prediction (unverified details).
- **Hybrid NN/HMM speech recognition** (Bourlard & Morgan, *Connectionist Speech Recognition*, 1994): classifier posteriors divided
  by class priors = scaled likelihoods used as HMM emissions ([Edinburgh ASR notes](https://www.inf.ed.ac.uk/teaching/courses/asr/2024-25/asr10-hybrid_hmm_nn.pdf)). This is exactly the "classifier output as
  emission, HMM forward filter" recipe.
- **HSMM for robot skills**: GP-HSMM segmentation (Nakamura et al., 2017, [PMC5742615](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5742615/), batch, unverified); task-parameterized HSMM
  (Rozo et al., [pdf](https://lcas.lincoln.ac.uk/wp/wp-content/uploads/2019/07/Rozoetal_RSS_TIG.pdf), unverified); online phase estimation for interaction ProMPs (Maeda et al., IJRR, [pdf](https://www.ias.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf), unverified).
- **Hysteresis switching for RL policies** to stop chattering (RLJ 2024, [par.nsf.gov](https://par.nsf.gov//servlets/purl/10541296), unverified title/details).
- **Gait phase estimation** (exoskeletons): FSMs vs neural nets, FSM switching discontinuities ([2107.03746](https://arxiv.org/abs/2107.03746), unverified).
- **Robot IL**: PHASER (oscillation at aliased states without phase ✓; its predictor switches into recovery phases and back ✓);
  SADP continuation score for switching ✓; StreamVLA [2602.01100](https://arxiv.org/abs/2602.01100) ✓ "Lock-and-Gated" — the subtask is locked until a transition is
  detected, motivated by "goal instability"; ABC re-execution of earlier subtasks ✓.

**Difference from ours**: the papers treat stage estimation as an open-loop recognition problem (surgery, speech, video) or show that
a phase predictor helps. Our failure is a **feedback loop**: the rule says "holding" too early → the stage-conditioned policy
executes the next stage's motion → the observation no longer matches → the stage flips back → oscillation; and the pre-grasp hover.
No paper found analyses this policy ↔ estimator interaction, or supplies an HSMM-filtered stage (with classifier outputs as
emissions) to a diffusion/flow policy.

**Verdict**: debounce, hysteresis, allowing regressions and HMM/HSMM forward filtering are **all standard** — cite them, do not
claim them. The **closed-loop failure analysis** and **measuring which fix works for a stage-conditioned visuomotor policy on a real
robot** are **possibly new**.

## 4. Stage-gated inputs

**Closest work**

- **MS-Bot** — [2408.01366](https://arxiv.org/abs/2408.01366) ✓ (HTML). "Stage-guided dynamic multi-sensory fusion": a stage-injected state token is the
  query of a cross-attention over vision / audio / touch features, so modality weights change with the predicted stage. This is
  essentially point 4.
- **Selective Perception for Robot** — [2602.15543](https://arxiv.org/abs/2602.15543) ✓. A lightweight router reads the wrist camera + prompt and predicts the
  task relevance of each camera view, attenuating low-utility views; router labels from a VLM pipeline.
- **PHASER** ✓: FiLM modulation of the encoder by the phase.
- **MoE-DP** — [2511.05007](https://arxiv.org/abs/2511.05007) ✓. MoE layer between visual encoder and diffusion head; experts "dynamically activated to handle
  different phases", interpretable as task primitives.
- TacFiLM ([2603.14604](https://arxiv.org/abs/2603.14604), unverified), FiLM in diffusion policies (survey [2608.07558](https://arxiv.org/abs/2608.07558), unverified).

**Difference from ours**: our version would gate cameras/joint features by a **rule** stage (not a learned stage) for a handover
(wrist vs head camera). That is a variant of MS-Bot / PHASER.

**Verdict: done.** Only a new empirical finding (e.g. which camera matters in which handover stage, and whether gating fixes the
start-of-episode 复位 confusion) could be reported; the mechanism is not new.

## 5. Low-human stage labeling

- **Weak supervision**: labeling functions → probabilistic labels → end model (Snorkel, Ratner et al., [1711.10160](https://arxiv.org/abs/1711.10160)); heuristics
  directly on time series (ECG, [PMC8861672](https://pmc.ncbi.nlm.nih.gov/articles/PMC8861672), unverified); segmentation from imprecise timestamps (Adams & Marlin, UAI 2018, unverified).
- **Few labels + alignment**: TCC ([1904.07846](https://arxiv.org/abs/1904.07846), CVPR 2019) — phase classification "with as few as a single labeled video"
  ([Google blog](https://research.google/blog/video-understanding-using-temporal-cycle-consistency-learning/)); DTW alignment of demonstrations (event-informed DTW, PoliMi thesis, unverified); ProMP phase estimation (Maeda, above).
- **VLM auto-labeling**: SARM's LeRobot implementation — subtask annotation step whose `--model` default is
  `Qwen/Qwen3-VL-30B-A3B-Instruct` ([LeRobot SARM docs](https://huggingface.co/docs/lerobot/sarm), from search summary; the SARM paper itself describes human annotation protocols ✓);
  StreamVLA ✓ (Qwen3-VL-plus proposes sub-stages + timestamps, humans verify boundaries — this detail from snippets, unverified);
  NILS ([2410.17772](https://arxiv.org/abs/2410.17772), unverified); *Actions as Language* ([2509.22195](https://arxiv.org/abs/2509.22195), Gemini labels, unverified); disassembly VLM comparison
  ([2609.17714](https://arxiv.org/abs/2609.17714), unverified). A recurring finding: VLM boundaries are imprecise.
- **Active learning only on disagreement/boundaries**: Su & Elhamifar, active learning for temporal action segmentation, ECCV 2024
  (unverified); B-ACT boundary-centric active learning ([2604.15173](https://arxiv.org/abs/2604.15173), unverified).

**Verdict: done** (each piece). A pipeline "causal rules as labeling functions → classifier → humans review only rule/classifier
disagreements near boundaries" is a sensible tool, not a paper contribution, unless it is shown to fix the closed-loop failures of
point 3.

## 6. Predicted stage fed back as input

- **Copycat problem** — Wen et al., NeurIPS 2020, [2010.14876](https://arxiv.org/abs/2010.14876) ✓. With history inputs and temporally correlated expert actions
  "the imitator learns to cheat by predicting the expert's previous action"; fix: adversarially remove previous-action information.
- **Causal confusion** — de Haan et al., NeurIPS 2019, [1905.11979](https://arxiv.org/abs/1905.11979). "More information can yield worse performance", especially
  with history inputs (via search results; not re-read).
- **Surgical error cascades** — [2605.16387](https://arxiv.org/abs/2605.16387) ✓ (an early wrong phase propagates).
- **π0.5** — [2504.16054](https://arxiv.org/abs/2504.16054) ✓ (HTML). At runtime "the model first predicts the semantic subtask … and then" the actions; trained
  on annotated subtask labels. The sections read do not discuss copying of the predicted subtask.
- **ABC** ✓: external classifier at inference; subtask conditioning used only 20 % of the time in training.
- **VISTA** [2602.10983](https://arxiv.org/abs/2602.10983) ✓ (abstract): hierarchical world model + VLA; snippets say random goal offsets around subtask boundaries
  for robustness to misalignment (unverified). **τ0-VLA** [2608.16885](https://arxiv.org/abs/2608.16885) ✓ (abstract): high-level subtask from execution memory;
  training the low level with predicted vs. ground-truth subtasks is from snippets (unverified).

**Our current variants do not feed a predicted stage back.** v6stagehead (commit d7fb1d86,
`src/lerobot/policies/diffusion/modeling_diffusion.py`) = rule stage input + an auxiliary head: the head predicts the stage from the
joint state and image features only (it does not see the stage input), its cross-entropy is weighted 0.1, the stage input is always
the rule-computed fine9, and the head's prediction (`stage_head_pred`) is only logged during eval. So there is no feedback and no
copycat risk in the current runs. Feeding the predicted stage back as input (and its copycat risk) is a **planned experiment**.

**Difference from ours (for that planned experiment)**: the general risk is the copycat problem (the stage is almost constant over
time, so "copy the input" gets low loss). No paper found shows this specifically for a stage/subtask channel of a visuomotor policy,
or compares fixes (external causal labeler, dropout of the stage input, noise / scheduled sampling, HSMM filter between head and input).

**Verdict**: mechanism **known** (cite copycat + causal confusion); a **controlled demonstration in this setting**, once run, would be
**possibly new** but small — best presented as part of point 2/3, not as a standalone claim.

## What can be claimed

1. **Strongest**: a controlled, same-data real-robot comparison of the ways to use stage information (input / joint-denoised target /
   rule input + auxiliary head / RA-BC weighting / learned memory) — none found. Feeding a predicted stage back is a planned
   addition, not part of the current runs.
2. **Second**: closed-loop failure analysis of stage-conditioned policies (premature "holding" → oscillation, hover) and an evaluation of
   fixes (debounce/hysteresis, regressions, HSMM filtering of a classifier) — techniques cited from surgery/speech, the loop analysis is new.
3. **Supporting**: causal, train/deploy-identical rule labeler for a bimanual handover (engineering; cite StaKe as the closest).
4. **Do not claim as new**: stage as input helps (PHASER, ABC); stage auxiliary heads (StaKe, SADP); RA-BC (SARM); stage-gated fusion
   (MS-Bot); VLM / TCC / active-learning labeling.

## Suggested citations

| Paper | Use | Checked |
|---|---|---|
| SARM, Chen et al., ICLR 2026, [2509.25358](https://arxiv.org/abs/2509.25358) | stage + progress reward, RA-BC | ✓ HTML |
| PHASER, Chen et al., T-Mech, [2605.29407](https://arxiv.org/abs/2605.29407) | phase as FiLM input, closed-loop predictor | ✓ HTML |
| ABC, Allshire et al., [2606.27375](https://arxiv.org/abs/2606.27375) | SARM-style subtask classifier → subtask prompt | ✓ HTML App. H |
| StaKe, [2606.26801](https://arxiv.org/abs/2606.26801) | automatic gripper-based stage labels, auxiliary heads | ✓ HTML |
| MS-Bot, Feng et al., CoRL 2024, [2408.01366](https://arxiv.org/abs/2408.01366) | stage-guided modality fusion | ✓ HTML |
| SADP, Hu & Horii, RA-L 2026, [2605.16871](https://arxiv.org/abs/2605.16871) | subgoal-conditioned DP + continuation head | ✓ abstract |
| CAMP, Wang et al., [2606.21188](https://arxiv.org/abs/2606.21188) | learned action memory | ✓ abstract |
| TF-DP, [2602.07388](https://arxiv.org/abs/2602.07388) | execution-trace conditioning, stage ambiguity | ✓ abstract |
| BEACON, [2609.22730](https://arxiv.org/abs/2609.22730) | belief-conditioned diffusion policy | ✓ abstract |
| MoE-DP, [2511.05007](https://arxiv.org/abs/2511.05007) | phase experts in DP | ✓ abstract |
| Selective Perception, [2602.15543](https://arxiv.org/abs/2602.15543) | camera-view router | ✓ abstract |
| StreamVLA, [2602.01100](https://arxiv.org/abs/2602.01100) | locked subtask, goal instability | ✓ abstract |
| π0.5, [2504.16054](https://arxiv.org/abs/2504.16054) | predicted subtask → action | ✓ HTML |
| Copycat, Wen et al., NeurIPS 2020, [2010.14876](https://arxiv.org/abs/2010.14876) | copying of fed-back inputs | ✓ abstract |
| Causal confusion, de Haan et al., NeurIPS 2019, [1905.11979](https://arxiv.org/abs/1905.11979) | history inputs hurt | search only |
| Online SPR stabilization, [2605.16387](https://arxiv.org/abs/2605.16387) | error cascades, evidence-gated transitions | ✓ abstract |
| Bourlard & Morgan 1994, *Connectionist Speech Recognition* | hybrid NN/HMM emissions | secondary |
| TCC, Dwibedi et al., CVPR 2019, [1904.07846](https://arxiv.org/abs/1904.07846) | few-label phase alignment | secondary |
| UVD, Zhang et al., NeurIPS 2023, [2310.08581](https://arxiv.org/abs/2310.08581) | label-free subgoal discovery | search only |
| OnlineTAS, NeurIPS 2024, [2411.01122](https://arxiv.org/abs/2411.01122) | streaming segmentation flicker | search only |
| VISTA [2602.10983](https://arxiv.org/abs/2602.10983), τ0-VLA [2608.16885](https://arxiv.org/abs/2608.16885) | hierarchical subtask robustness | abstract only |
