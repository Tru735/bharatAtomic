# Phase 2 — Open-Set SEM Segmentation & Quantitative Microstructure Analysis: Research Foundation & Strategy

> Companion planning document to `README.md` (Phase 1) and `semPhase1/README.md`. This covers the literature review, methodology, model selection, and dataset strategy for Phase 2: open-set instance segmentation across arbitrary sample materials, with morphometric analysis (PSD, aspect ratio, circularity, porosity) planned as a downstream consumer of the segmentation output.

---

## Table of Contents

- [Phase 2 — Open-Set SEM Segmentation \& Quantitative Microstructure Analysis: Research Foundation \& Strategy](#phase-2--open-set-sem-segmentation--quantitative-microstructure-analysis-research-foundation--strategy)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Constraints, Restated](#1-scope-and-constraints-restated)
  - [2. Literature Review](#2-literature-review)
    - [2.1 Deep learning for EM/SEM segmentation — surveys and reviews](#21-deep-learning-for-emsem-segmentation--surveys-and-reviews)
    - [2.2 Materials-SEM-specific segmentation (closest analogs to your task)](#22-materials-sem-specific-segmentation-closest-analogs-to-your-task)
    - [2.3 Foundation models (SAM/SAM2 family) for microscopy](#23-foundation-models-samsam2-family-for-microscopy)
    - [2.4 Why this matters for your design](#24-why-this-matters-for-your-design)
  - [3. The Open-Set Constraint — Why It Drives Everything Else](#3-the-open-set-constraint--why-it-drives-everything-else)
    - [3.1 Closed-set vs. open-set: what each path actually buys and costs](#31-closed-set-vs-open-set-what-each-path-actually-buys-and-costs)
    - [3.2 Decision](#32-decision)
  - [4. Methodology and Model Selection](#4-methodology-and-model-selection)
    - [4.1 Instance segmentation, not pure semantic segmentation](#41-instance-segmentation-not-pure-semantic-segmentation)
    - [4.2 Architecture shortlist](#42-architecture-shortlist)
    - [4.3 Recommendation](#43-recommendation)
  - [5. Self-Supervised Pretraining — What's Actually Worth Doing](#5-self-supervised-pretraining--whats-actually-worth-doing)
    - [5.1 What SSL can and can't help with here](#51-what-ssl-can-and-cant-help-with-here)
    - [5.2 Continued pretraining of SAM's encoder — the careful version](#52-continued-pretraining-of-sams-encoder--the-careful-version)
    - [5.3 What we are explicitly NOT doing](#53-what-we-are-explicitly-not-doing)
  - [6. Hitting the Latency Ceiling — What the Numbers Actually Say](#6-hitting-the-latency-ceiling--what-the-numbers-actually-say)
  - [7. Dataset Strategy — Where the General Model Ends and Fine-Tuning Begins](#7-dataset-strategy--where-the-general-model-ends-and-fine-tuning-begins)
    - [7.1 What NFFA-Europe actually gives you (and what it doesn't)](#71-what-nffa-europe-actually-gives-you-and-what-it-doesnt)
    - [7.2 Our actual starting point — be honest about this](#72-our-actual-starting-point--be-honest-about-this)
    - [7.3 Building the fine-tuning dataset](#73-building-the-fine-tuning-dataset)
    - [7.4 Annotation scale targets, summarized](#74-annotation-scale-targets-summarized)
  - [8. Phase 2.1 Preview — Where Morphometric Analysis Plugs In](#8-phase-21-preview--where-morphometric-analysis-plugs-in)
  - [9. Checklist](#9-checklist)
  - [10. Summary of Recommendations](#10-summary-of-recommendations)
  - [11. References](#11-references)

---

## 1. Scope and Constraints, Restated

Before picking models, it's worth pinning down what "open-set," "real-time," and "segmentation" actually mean for this project, because each covers a wide range of very different engineering problems:

- **Product constraint, and the one that overrides every other decision in this document: Bharat Atomic is shipping an open SEM.** Any sample a customer puts under the beam — powders, particles, fibres, MEMS structures, biological tissue, porous ceramics, battery materials, or something none of us have seen yet — must be segmentable. This is an **open-set** requirement, not a fixed-catalog one, and it rules out architectures that only know a predefined list of classes unless we explicitly scope a narrower fast-path for known categories (Section 3).
- **Latency target:** under 40 ms per frame, hard ceiling, no exceptions. This is a per-frame inference budget, not an average — a model that's 25 ms most of the time but spikes to 80 ms on dense particle fields fails the requirement.
- **Task type:** instance segmentation (need per-object masks to compute per-particle area, aspect ratio, circularity — not just semantic class maps), though semantic segmentation is the right tool for porosity/defect-area tasks where individual object identity doesn't matter.
- **Deployment hardware:** unspecified beyond the M3 Max dev machine and the "incoming NVIDIA GPU" mentioned for Phase 1. The 40 ms target effectively assumes a CUDA + TensorRT path — this is addressed in Section 6.
- **Downstream consumer:** the segmentation masks feed a measurement pipeline (PSD, surface area, volume estimates, aspect ratio, circularity, porosity), which depends on (a) accurate masks and (b) a pixel-to-physical-unit calibration step using the SEM's scale bar/metadata. This second part has no real precedent for real-time operation in the literature — it's traditionally a batch, offline analysis step (see Section 8).
- **Annotation reality:** we currently have **zero labeled segmentation masks**. The realistic near-term annotation budget is on the order of ~200 manually reviewed images, not the thousands-of-masks scale several precedent papers started from. This is the single constraint most likely to be underestimated, and it shapes Sections 5 and 7 directly.

---

## 2. Literature Review

### 2.1 Deep learning for EM/SEM segmentation — surveys and reviews

These give the lay of the land for how DL segmentation has been applied to electron microscopy broadly, mostly in connectomics/cellular EM rather than materials SEM, but the architectural lessons transfer:

- **Aswath, A. et al., "Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey"** (arXiv:2206.07171, 2022). Surveys how semantic and instance segmentation architectures were adapted for cellular/sub-cellular EM structures, examining the special challenges posed by EM images and the network architectures that addressed them, alongside a review of the major datasets that enabled progress. Good starting point for understanding why generic Cityscapes/COCO-tuned architectures don't transfer cleanly to microscopy.
- **Khadangi, A., Boudier, T., & Rajagopal, V., "EM-stellar: benchmarking deep learning for electron microscopy image segmentation"** (*Bioinformatics*, 37(1), 2021). Notes that the inherent low contrast of EM datasets is a major challenge for rapid segmentation of cellular ultrastructures, especially with high-resolution datasets from electron tomography and serial block-face imaging — and that no rigorous benchmark of DL methods existed prior to this work. Directly relevant to our noise/contrast situation, since we already characterized SEM-specific noise sources in Phase 1.
- **"Deep learning for brain electron microscopy segmentation"** (*Computers & Graphics*, Sept 2025). A 2025 meta-analysis reviewing 60 deep learning approaches for brain EM segmentation, covering self-supervised learning to reduce manual annotation needs, topology-aware loss functions for better continuity in neuron segmentation, and transformer architectures for capturing long-range context. Useful for loss-function ideas (topology-aware losses are relevant to grain/pore continuity).
- **M-SegNet with Global Attention** (*Computers in Biology and Medicine*, 136, 2021). Not an SEM paper — this is brain MRI tissue segmentation — but architecturally relevant as a worked example of combining multiscale side inputs, deep supervision, a global-attention module, and combined encoder-decoder connections to recover spatial detail lost during downsampling, achieving a mean Dice improvement of 1.5–10% over SegNet, U-Net, M-Net, U-Net++, CE-Net, and CFPNet-M baselines. The ablation methodology here (isolating each architectural addition one at a time: dilated kernels, patch-wise input, combined-connections, global attention) is a useful template for how we should structure our own GSEFE/MDFF ablations in Section 5.

### 2.2 Materials-SEM-specific segmentation (closest analogs to your task)

This is the more directly relevant cluster — segmentation of particles, grains, and defects in materials-science SEM imagery rather than biological EM:

- **PerovSegNet** (arXiv:2509.26548, 2025) — segments lead iodide, perovskite, and defect domains in solar cell SEM images. Built on an improved YOLOv8x architecture with an Adaptive Shuffle Dilated Convolution Block for multi-scale feature extraction and a Separable Adaptive Downsampling module for boundary recognition, trained on an augmented dataset of 10,994 SEM images, achieving 87.25% mAP with 265.4 GFLOPs while reducing model size and computational load by roughly a quarter relative to baseline YOLOv8x-seg. **Important scope note:** this is a closed-set, fixed-three-class detector trained on one material system (perovskite thin films). It is not a candidate for our primary open-set pipeline — see Section 3 for why — but its architectural ideas around multi-scale feature extraction and boundary-preserving downsampling are worth borrowing for our own edge/texture modules, and it remains a credible blueprint for a future vertical-specific fast-path if Bharat Atomic ever has a dedicated perovskite/solar customer segment with budget for full annotation.
- **LBMS-SAM** (*Neural Networks*, 196, 2026) — SAM-guided segmentation for lithium battery material SEM images. Built a 244-image dataset of lithium battery SEM images at 960×600 resolution, with 10,609 annotated masks for training and 2,274 for validation/testing, specifically targeting the high-density, adhesion-heavy particle clusters that cause CNN-based methods to suffer from edge blurring or missed segmentation. Freezes SAM's encoder and decoder entirely and adds two small trainable modules — a Sobel/Gabor edge feature extractor (GSEFE) and a wavelet-denoised multi-layer attention fusion module (MDFF) — adding only ~1.3M learnable parameters against SAM's 312M frozen ones, while outperforming SAM, HQ-SAM, and µSAM on every reported metric. **This is the architectural template our primary pipeline is built on** — see Sections 3 and 4.
- **Uncertainty-aware particle segmentation for EM at varied length scales** (*npj Computational Materials*, 2024). Enhances Mask R-CNN to segment particles in SEM images of powder samples, explicitly addressing image blur and particle agglomeration — useful baseline if we want an accuracy-first reference point against which to measure the speed/accuracy tradeoff of our deployed model.
- **Workflow for agglomerated, non-spherical particle segmentation** (*Scientific Reports*, 2021). Flags the core dataset problem directly: there is a severe lack of high-quality annotated image data for training and validating algorithms for nanoparticle segmentation in SEM images, despite how ubiquitous electron microscopy is in materials science — and proposes generating ground truth automatically from paired STEM-in-SEM acquisitions of the same sample area. This is a useful trick if our hardware can acquire STEM-mode images alongside standard SEM, and it produces the TiO2 dataset used as a zero-shot generalization benchmark in the LBMS-SAM paper.
- **Machine vision-driven particle size/morphology recognition** (*Nanoscale*, RSC, 2020). The closest published analog to our eventual Phase 2.1 deliverable — it doesn't stop at segmentation, it goes all the way to physical measurement. It performs scale-bar and embedded-text recognition to extract calibration information from the SEM image itself, converting pixel-based size estimates into physical units like µm or nm, and explicitly notes that prior high-throughput methods often failed to measure diameters of overlapping nanoparticles. Read this one closely — it's effectively a worked example of the full pipeline we're describing (segmentation → calibration → PSD).
- **MatSegNet** (arXiv:2312.17251) — carbide precipitate analysis in steels. Demonstrates the standard calibration approach we'll need: carbide sizes are derived from segmentation masks through a calibrated pixel-to-area conversion, where spatial resolution is determined by measuring pixel density along the SEM's own scale bar, yielding the physical area represented by a single pixel.
- **Leveraging unlabeled SEM datasets with self-supervised learning for enhanced particle segmentation** (*npj Computational Materials*, 11:289, 2025). The key precedent for Section 5. Curates a 24,751-image unlabeled SEM dataset and benchmarks ConvNeXtV2 (masked-autoencoder SSL) against DenseCL (contrastive SSL), ImageNet pretraining, and a no-pretraining baseline, on a downstream Mask R-CNN particle-segmentation task with only 91 labeled images. ConvNeXtV2 wins decisively at every backbone scale and both magnifications, beating even ImageNet's 14-million-image supervised pretraining despite training on roughly 25,000 images. Critically, the ablation shows ConvNeXtV2 surpasses ImageNet-pretrained performance using as few as ~1,000 unlabeled pretraining samples for low magnification, with gains saturating around 4,000–8,000 samples for high magnification — and that a medium backbone (Pico, 8.6M params) consistently beat much larger variants, with larger models showing diminishing or negative returns. This is direct evidence that masked-autoencoder-style SSL is markedly more data-efficient than contrastive or ViT-heavy SSL approaches in the SEM domain specifically, which is the basis for the caution in Section 5 about how we pretrain (or choose not to pretrain) any ViT-based component.

### 2.3 Foundation models (SAM/SAM2 family) for microscopy

This matters because SAM/SAM2 fine-tuning is our primary architecture path (Section 4), and there's a meaningful and fast-growing body of work specifically on adapting it to EM:

- **SAM-EM** (arXiv:2501.03153, 2025) — a domain-adapted foundation model built on SAM2 that unifies segmentation, tracking, and statistical analysis for liquid-phase TEM data, derived through full-model fine-tuning on 46,600 curated synthetic video frames, and integrates particle tracking with statistical tools including mean-squared displacement and particle displacement distribution analysis as part of an end-to-end framework. **Scope note:** despite the name, this targets a different imaging modality (liquid-phase TEM video, with temporal particle tracking) than our static SEM instance segmentation problem, and is not directly transplantable — but the pairing of "fine-tuned SAM-family model for segmentation" + "statistical particle analysis bolted on afterward" is structurally identical to what we're planning across Phase 2 and Phase 2.1.
- **µSAM ("Segment Anything for Microscopy")** (*Nature Methods*, 22, 2025) — integrates SAM into the napari platform for a unified interactive/automatic microscopy segmentation workflow, with domain-finetuned checkpoints for electron and light microscopy (including organelle/mitochondria-focused variants). Directly benchmarked against LBMS-SAM and underperforms it on every metric on the lithium-battery dataset (IoU 92.4 vs. 96.2 on the large subset), a gap the LBMS-SAM authors attribute to µSAM's watershed-based decoder relying on global spatial context that doesn't transfer well to single-particle, high-density-adhesion segmentation. µSAM's zero-shot generalization is competitive only on datasets that resemble its own biological-microscopy pretraining distribution (e.g., TiO2, where its full-image mode edges out SAM single-particle mode) — a caution against assuming "more EM-specific" automatically means "better for our task," documented further in Section 5.
- **Lightweight SAM2 fine-tuning for microscopy** (bioRxiv, Nov 2025). Introduces a Colab-based pipeline for fine-tuning SAM2's mask decoder on small, curated microscopy datasets without adding architectural complexity, motivated by the fact that prior adaptations like μSAM and CellSAM require additional transformer or convolutional layers that make them computationally demanding and harder to scale. If our annotated dataset stays small (likely — see Section 7), decoder-only fine-tuning is the cheaper, faster-converging path, consistent with our Phase 1 finding that decoder-only fine-tuning of RDUNet/NAFNet outperformed from-scratch training under compute constraints.
- **Foundation models for zero-shot segmentation of scientific images** (arXiv:2506.24039, 2025). Surveys the lightweight SAM variants relevant to a 40ms target: FastSAM (a YOLOv8-based architecture built for efficiency), MobileSAM (optimized for mobile/edge deployment), MedSAM (medical imaging adaptation), and µSAM (microscopy-specific adaptation). FastSAM and MobileSAM are worth benchmarking for latency, but neither has materials-SEM precedent the way LBMS-SAM does, and both would need our own validation before trusting their boundaries on adherent particles.
- **SAM-I-Am** (arXiv:2404.06638, 2024) — a non-learned, rule-based post-processing booster for zero-shot SAM on atomic-scale electron micrographs. Extracts geometric and textural features of intermediate masks to perform mask removal and mask merging, reporting absolute mean IoU gains of +21.35%, +12.6%, and +5.27% across three difficulty tiers over vanilla SAM (ViT-L), with no additional training required at all. Worth keeping as a zero-cost baseline to compare against GSEFE/MDFF before committing engineering time to trained adapter modules — if a free post-processing heuristic closes most of the gap, that changes the priority order in Section 5.

### 2.4 Why this matters for your design

A recurring pattern across nearly every materials-SEM segmentation paper above is the same one we already learned the hard way in Phase 1 with noise modeling: **generic, natural-image-trained segmentation models underperform on microscopy until either (a) fine-tuned on domain data, or (b) architecturally adapted for the specific failure mode (overlapping particles, low contrast, fine boundaries)**. None of the papers above achieved strong results from an out-of-the-box COCO/SA-1B-pretrained model with zero adaptation. This sets expectations for Section 7 (dataset strategy) — we should plan for some fine-tuning from the start, not as a contingency, even where the bulk of the architecture stays frozen.

---

## 3. The Open-Set Constraint — Why It Drives Everything Else

This section did not exist in earlier planning and needs to be explicit, because it overturns what would otherwise be the "obvious" model choice.

### 3.1 Closed-set vs. open-set: what each path actually buys and costs

| | Closed-set detector (YOLO-seg, RF-DETR, PerovSegNet-style) | Open-set / promptable segmenter (SAM-family) |
|---|---|---|
| What it needs to function at all | Substantial labeled instance data per class — it has no prior concept of "object" beyond what it's trained on | Nothing — SAM's 1-billion-mask pretraining already gives it a general "find object boundaries" prior, usable zero-shot |
| Behavior on an unseen material/morphology | Misclassifies, fails to detect, or confidently produces wrong boxes with no signal it's out of distribution | Degrades gracefully — still attempts boundary segmentation based on general shape/edge cues, without needing to "recognize" the material |
| Annotation burden to reach usable quality | High and roughly per-class — PerovSegNet needed ~11,000 images for 3 classes on one material system | Lower — fine-tuning only needs to correct a known, narrow failure mode (e.g., merged adherent particles), not teach the concept of "object" from scratch |
| Inference speed | Generally faster, single forward pass, mature TensorRT export tooling | Heavier — full ViT encoder pass, though this is mitigated by lightweight variants (MobileSAM, FastSAM) and by GSEFE/MDFF adding negligible overhead (~1.3M params) over a frozen backbone |
| Fits "any sample" requirement | **No, by construction** | **Yes, by construction** |

### 3.2 Decision

Given Bharat Atomic is shipping an **open SEM**, a closed-set detector cannot be the primary segmentation path — it solves a narrower problem than the one stated in the product brief, regardless of how good its in-distribution accuracy looks on a benchmark. A YOLO- or DETR-family detector remains a legitimate **secondary, opt-in fast-path** for specific, high-volume material categories once (a) Bharat Atomic has a defined customer vertical with predictable sample types, and (b) the annotation investment for that category is justified by volume. Until then, it is out of scope for the primary deliverable, and PerovSegNet's strong numbers, however good, are evidence for a future module — not a reason to revisit this decision now.

The primary architecture for Phase 2 is therefore the **LBMS-SAM pattern**: frozen SAM backbone + small trainable adapter modules (GSEFE edge/texture extraction, MDFF denoised multi-layer feature fusion), fine-tuned on a small, growing, in-house-annotated dataset. This is detailed in Section 4.

---

## 4. Methodology and Model Selection

### 4.1 Instance segmentation, not pure semantic segmentation

Our stated downstream goals — particle size distribution, aspect ratio, circularity per particle — require **per-instance masks**, since these are all per-object measurements. Pure semantic segmentation (DeepLabV3+, SegFormer, BiSeNet-class models) only gives a class map, not separated object identities, so it would need an additional instance-separation step (e.g., watershed on the semantic mask, which reintroduces the over-segmentation/under-segmentation problems classical methods have struggled with for decades).

Porosity analysis is the one sub-task where pure semantic segmentation (pore vs. matrix) is actually sufficient and standard — see the thresholding-based porosity workflows in Section 8 — so the pipeline should ideally support both an instance path (individual particle morphology) and a semantic path (porosity/defect-area), or run two lightweight passes within the latency budget.

### 4.2 Architecture shortlist

| Candidate | Type | Open-set? | Why it's in the running | Why it might not be the final answer |
|---|---|---|---|---|
| **Frozen SAM + GSEFE/MDFF (LBMS-SAM pattern)** | Promptable foundation model + lightweight adapters | **Yes** | Direct precedent on dense, adherent-particle SEM data; needs only ~1.3M trainable params; works zero-shot before any fine-tuning even starts, so there's always a usable fallback; matches our annotation-scarce reality | Heavier per-frame than a one-stage detector (full ViT-L encoder pass); needs either a prompt source or an automatic-everything mode, both addressed below |
| **MobileSAM / FastSAM** | Distilled/lightweight SAM variants | Yes | Designed explicitly for the speed problem vanilla SAM has; FastSAM is itself YOLOv8-based, inheriting mature deployment tooling | Less literature precedent specifically on SEM/materials imagery — would need our own validation before trusting it on grain/particle boundaries |
| **SAM-I-Am-style post-processing booster** | Rule-based mask correction, no training | Yes | Zero training cost, zero additional labeled data needed, documented IoU gains over vanilla SAM | Likely a smaller ceiling than a trained adapter approach; best used as a baseline/complement, not a replacement, for GSEFE/MDFF |
| **YOLO11-seg / YOLOv8-seg / RF-DETR** | One-stage closed-set instance segmentation | **No** | Real-world materials-SEM precedent (PerovSegNet); mature TensorRT export path; fastest realistic route to the 40ms target | Ruled out as primary by Section 3 — closed-set by construction; remains a valid future fast-path for specific known material verticals |
| **Mask R-CNN (ResNet/ConvNeXt backbone)** | Two-stage closed-set instance segmentation | No | Best accuracy ceiling among classic architectures; used in the most accuracy-focused SEM particle papers | Too slow for 40ms on any benchmark surveyed, and still closed-set — fails both the latency and the open-set requirement |
| **DeepLabV3+ / U-Net family (semantic only)** | Semantic segmentation | Depends on training data | We already have U-Net-family infrastructure and experience from Phase 1 (RDUNet, Attention U-Net); good fit specifically for the porosity/defect-area task | Doesn't give per-instance masks natively; not a complete answer to the PSD/aspect-ratio/circularity goals on its own; would itself need open-set-scale training data to avoid the same closed-set trap |

### 4.3 Recommendation

A two-track approach, with the open-set constraint setting which track is primary:

1. **Primary track — frozen SAM + GSEFE/MDFF, fine-tuned on our annotated SEM data.** This is the LBMS-SAM architecture, adopted as-is rather than reinvented, because (a) it's the only candidate that satisfies the open-set requirement, (b) it needs the least labeled data of any path under consideration, given we are starting from zero labels, and (c) frozen SAM alone already provides a working zero-shot fallback the moment we have any pipeline at all, before a single epoch of GSEFE/MDFF training. **SAM's encoder and decoder are not fine-tuned at any point in this track** — see Section 5 for why, and what we tested instead.

2. **Secondary/parallel track — lightweight SAM variants (MobileSAM/FastSAM) and SAM-I-Am-style post-processing**, benchmarked against the primary track on both latency and mask quality, since neither has materials-SEM precedent and both need our own validation before being trusted in production.

3. **Deferred, opt-in track — a closed-set detector (YOLO-seg or RF-DETR) for specific known material verticals**, only once Bharat Atomic has a customer segment with predictable sample types and the annotation budget to support per-class training at the scale PerovSegNet-style results require (~10,000+ images for a handful of classes). This is explicitly not part of the Phase 2 primary deliverable.

A reasonable evaluation matrix going into experiments:

| Model | Expected role | Target metric |
|---|---|---|
| Frozen SAM (ViT-L), zero-shot | Baseline / always-available fallback | IoU, BIoU, Fa on hand-corrected eval set |
| Frozen SAM + GSEFE only | Ablation — boundary/edge correction in isolation | IoU, BIoU vs. frozen baseline |
| Frozen SAM + MDFF only | Ablation — global context/denoising in isolation | IoU, BIoU vs. frozen baseline |
| Frozen SAM + GSEFE + MDFF (full LBMS-SAM) | Primary candidate | IoU, BIoU, Dice, Fa |
| SAM-I-Am-style rule-based booster | Zero-training-cost baseline | IoU vs. frozen SAM, no training data spent |
| MobileSAM / FastSAM | Lightweight alternative | Latency vs. primary track on identical hardware |

---

## 5. Self-Supervised Pretraining — What's Actually Worth Doing

This section exists because it's tempting to assume "we have a large unlabeled SEM pool (NFFA-Europe), so let's pretrain everything self-supervised first." That instinct is partially right and partially wrong, and getting the split correct matters enough to spell out explicitly.

### 5.1 What SSL can and can't help with here

**GSEFE and MDFF cannot be meaningfully pretrained with SSL, and we are not attempting it.** Both modules are thin signal-processing adapters, not representation-learning backbones:

- GSEFE applies fixed Sobel and Gabor operators (mathematical edge/texture transforms, not learned from data) followed by a small enhancement network (pooling → 1×1 conv → batch norm → ReLU).
- MDFF applies a fixed discrete wavelet transform, soft-thresholding with one learnable scalar, inverse wavelet transform, and a small fusion convolution.

Together these add roughly 1.3M learnable parameters on top of SAM's 312M frozen ones. There is no representation-learning problem here large enough for a self-supervised pretext task to do real work — these modules go straight to supervised training on our (small, growing) labeled set, exactly as the original LBMS-SAM paper does it.

**Full supervised fine-tuning of SAM's own encoder/decoder is actively harmful at our data scale, and is explicitly out of scope.** The LBMS-SAM paper's own ablation is the evidence: fine-tuning all of SAM on their (much larger, ~10,000-mask) dataset dropped IoU from 95.6 to 80.5 and BIoU from 86.5 to 52.6 — catastrophic forgetting of SAM's general segmentation prior. With our smaller dataset, the same failure mode would be worse, not better. SAM stays frozen throughout the primary track.

### 5.2 Continued pretraining of SAM's encoder — the careful version

There is one legitimate, evidence-adjacent way to use our large unlabeled NFFA-Europe pool to improve SAM's encoder, distinct from supervised fine-tuning:

- Resume SAM's encoder training with a **self-supervised objective** (masked-patch reconstruction, in the spirit of ConvNeXtV2's FCMAE approach documented in Section 2.2) on unlabeled SEM images, using no labels at all — the "ground truth" is simply the original image's own pixels at the masked locations.
- This must run at a low learning rate, likely with only the later encoder blocks unfrozen, to avoid drifting away from SAM's already-strong general prior faster than our smaller SEM corpus can teach it something better.
- **Mandatory validation gate before this touches the labeled pipeline at all:** compare zero-shot mask quality (automatic mode, no GSEFE/MDFF) between vanilla frozen SAM and the continued-pretrained version, on a held-out hand-checked sample. If the adapted encoder is not clearly better, the experiment is discarded and we proceed with vanilla SAM. This is a real research experiment with a genuine chance of net-negative results, not a guaranteed upgrade, and it is the single most compute-expensive item in this document (training a 300M+ parameter ViT, even partially unfrozen) — it should be scheduled accordingly, after the primary GSEFE/MDFF track has a working baseline, not before.
- The npj ConvNeXtV2 paper's own finding (Section 2.2) is a relevant caution, not just a green light: their result favors CNN/MAE-style SSL over ViT/contrastive SSL specifically because ViT-style SSL needed far more data to be competitive in their setting. SAM's encoder is a ViT. Continued pretraining at NFFA-Europe's scale (tens of thousands of images, not the 100M+ scale typical ViT-SSL recipes assume) is an open question, not a validated result, for us to test rather than assume.

### 5.3 What we are explicitly NOT doing

For clarity and to avoid relitigating these later:

- **Not** pretraining GSEFE/MDFF with a fabricated SSL pretext task (Section 5.1 — no capacity for it to do useful work).
- **Not** fully fine-tuning SAM's encoder/decoder on our small labeled set (Section 5.1 — destroys the pretrained prior).
- **Not** swapping SAM's backbone for a separately-pretrained ConvNeXtV2 or DINOv2 encoder mid-architecture — this was evaluated and rejected as a cross-architecture transplant with real integration risk and no validated precedent at our data scale; if we want EM-domain-adapted ViT features inside SAM specifically, that is the continued-pretraining path in Section 5.2, not a backbone swap.
- **Not** treating µSAM ("SAM-EM"/"SAM-M" checkpoints) as a drop-in replacement backbone for SAM in the LBMS-SAM architecture — its decoder design (global-context watershed) is documented to underperform specifically on single-particle, high-adhesion segmentation, which is our exact failure mode.

---

## 6. Hitting the Latency Ceiling — What the Numbers Actually Say

Our Phase 1 README is explicit that the M3 Max/MPS setup is a real bottleneck and that CUDA + TensorRT is the planned target hardware. The 40ms requirement should be evaluated **only against that target hardware**, because the evidence is unambiguous that it's out of reach on MPS:

- On an RTX 5070 Ti with TensorRT optimization, YOLOv8 segmentation reached up to 374 FPS (roughly 2.7ms/frame) in a real industrial deployment, with TensorRT FP16 providing the dominant share of the speedup over plain ONNX Runtime — for a single-class, fairly simple task, so treat this as an optimistic ceiling for one-stage closed-set detectors rather than a number applicable to our SAM-based primary track.
- For SAM-family models specifically, the relevant comparison is MobileSAM/FastSAM-class latency rather than vanilla ViT-L SAM: full SAM2 (even fine-tuned) is unlikely to hit 40ms without aggressive distillation, which is precisely why the secondary track in Section 4.3 exists — to find out empirically whether a lightweight SAM variant can meet the ceiling, or whether GSEFE/MDFF's modest overhead on a smaller SAM backbone (ViT-B rather than ViT-L) is the more realistic path.
- On embedded/edge-class hardware rather than a desktop RTX card, one-stage detector latency drops further with INT8 quantization — relevant background for the deferred closed-set fast-path (Section 3), and a useful reference point for what "fast" looks like in this domain even though our primary track is architecturally different.
- The honest caveat, directly relevant to our own Phase 1 NAFNet-on-MPS-to-RTX projection: real-world TensorRT latency can diverge meaningfully from vendor documentation depending on GPU variant, driver stack, and export settings — so treat all of the above as directional, not as a guarantee, and budget for our own benchmarking pass before committing to a final model size.

**Practical implication for our 40ms requirement:** the primary risk to the latency budget isn't just the segmentation model itself — it's everything wrapped around it: image preprocessing, postprocessing (mask upsampling, NMS-equivalent merge/split logic for SAM's automatic mode), and especially the downstream morphometric analysis if it's naively run synchronously in the same loop. Keep the measurement/PSD pipeline (Section 8) decoupled from the real-time segmentation loop — segment in real time, batch the morphometric analysis asynchronously or on a slower cadence, since there is no published precedent for running full PSD/porosity/circularity computation inside a 40ms budget, and forcing it there is an unnecessary self-imposed constraint that doesn't serve the actual goal.

---

## 7. Dataset Strategy — Where the General Model Ends and Fine-Tuning Begins

This is the part of the plan most likely to be underestimated, because Phase 1 had a genuine advantage that Phase 2 does not: a large, clean, purpose-built dataset (NFFA-Europe) already existed for denoising-style training. Segmentation does not have an equivalent off-the-shelf resource for SEM imagery, and **we are starting this phase with zero labeled masks.**

### 7.1 What NFFA-Europe actually gives you (and what it doesn't)

We already used the NFFA-Europe Majority dataset in Phase 1, so it's worth being precise about its actual annotation type, since the framing matters for what's reusable in Phase 2:

- The Majority dataset consists of roughly 21,000–25,500 SEM images (reported variously as 21,272 / 25,430 / 25,537 across NFFA-Europe's dataset releases) at 1,024×728 pixels, classified into 10 categories (tips, particles, patterned surfaces, MEMS devices and electrodes, nanowires, porous sponge, biological, powder, films and coated surfaces, and fibres) based on majority agreement among a panel of nanoscientists. No scientific metadata beyond the classification label is attached to the images — meaning **NFFA-Europe has no segmentation masks, no instance annotations, and no bounding boxes**. It is purely an image-classification dataset, and the 10 category labels are image-level, not pixel-level — they cannot substitute for instance masks even if we wanted to train a closed-set detector on them.
- This means NFFA-Europe is useful for Phase 2 in exactly two ways, neither of which is direct supervised segmentation training: (1) as a large pool of **unlabeled SEM images for the continued-pretraining experiment described in Section 5.2**, and (2) as a source of category-diverse images for **building our own annotation set** via SAM-assisted bootstrapping (Section 7.3).
- One practically useful detail for our eventual calibration step: NFFA-Europe SEM images include a white information bar at the bottom of the frame showing scale and acquisition settings, typically cropped out before use in classification/denoising pipelines. For our Phase 2.1 scale-calibration work, this metadata bar (or the equivalent one on our own SEM's image output) is precisely the kind of structured region a scale-bar-reading model (Section 8) would need to parse, rather than discard.

### 7.2 Our actual starting point — be honest about this

- **Raw, unlabeled SEM images:** the full NFFA-Europe pool (tens of thousands), usable for the Section 5.2 continued-pretraining experiment only — not for direct segmentation supervision.
- **Labeled segmentation masks:** zero, currently. The realistic near-term plan is ~200 manually reviewed source images. Standard 2–3x geometric augmentation (flips, rotations, crops) is necessary but not sufficient at this scale — it doesn't change the underlying noise/texture statistics. **Our existing Phase 1 SEM noise simulator (`NoiseImage`/`new_augment_sem`) is a meaningfully better augmentation source than generic geometric transforms here**, since it produces realistic, varied noise realizations of the same annotated particle boundaries, directly targeting the noise modes our hardware will actually produce.
- **The actual unit that matters is masks, not images.** 200 source images could yield anywhere from a few hundred to several thousand mask instances depending on particle density per image — this should be counted empirically on a sample of our own images before assuming we are "data-poor" in the way the number 200 implies.
- **SAM needs no labels to begin generating candidate masks.** Frozen SAM (zero-shot) can run in either promptable mode (point/box click → mask for that object, no annotation required) or automatic mode (grid-sampled, segment-everything, no prompt required). Both are usable immediately, before any training. The realistic annotation workflow is: run frozen SAM automatic mode on raw images, accept/correct/reject its proposals (correcting is dramatically faster than drawing boundaries from scratch), and treat the corrected output as the labeled dataset. Wherever particles are densely adhered, frozen SAM is expected to merge them — exactly the LBMS-SAM paper's documented failure mode — so these regions will need more manual correction or point-prompting per particle.

### 7.3 Building the fine-tuning dataset

A realistic dataset-building path, given the starting point above:

1. **First concrete action, before any training:** run frozen SAM (automatic mode, zero-shot, no fine-tuning) on a handful of our actual raw SEM images and visually inspect the output. This single test tells us (a) how good a bootstrapped-labeling starting point we have, and (b) how badly the adherent-particle merging problem shows up on our specific samples — which tells us how urgently GSEFE/MDFF are needed versus how far frozen SAM alone gets us.
2. **Seed set:** ~200 manually reviewed SEM images, bootstrapped via SAM-assisted correction rather than hand-drawn from scratch, from material categories matching our actual hardware's likely use cases.
3. **Annotation tooling:** an open-source annotation tool (e.g., Label-Studio) paired with a SAM-based ML backend for AI-assisted annotation, so annotators correct SAM-generated proposals rather than drawing from scratch.
4. **Domain-specific synthetic augmentation:** apply Phase 1's `NoiseImage`/`new_augment_sem` engine to the annotated seed set's clean reference crops to synthetically expand the effective training set with realistic SEM degradation — the same logic that made the Phase 1 denoiser generalize, and a genuine advantage Phase 2 inherits from Phase 1 that most precedent papers didn't have.
5. **Module priority under data scarcity:** rather than training GSEFE and MDFF jointly from the start, validate each in isolation first (Section 4.3's evaluation matrix). The original LBMS-SAM ablation shows GSEFE-alone and MDFF-alone each already capture most of the gain over frozen SAM, with the combination adding a comparatively small further improvement — at our data scale, training only the module that addresses our most visible failure mode (likely GSEFE, for boundary precision on adherent particles) may generalize better than fitting both sets of parameters to a small dataset at once.
6. **Pseudo-labeling at scale, once a working seed model exists:** use the fine-tuned model to auto-generate candidate masks across a larger unlabeled pool (NFFA particle/powder categories, plus our own hardware's raw output), with human spot-checking rather than full re-annotation.

### 7.4 Annotation scale targets, summarized

| Stage | Approx. scale | Source / rationale |
|---|---|---|
| Unlabeled pool (continued pretraining experiment only, Section 5.2) | Full NFFA-Europe pool, ~21,000–25,500 images | Existing resource from Phase 1; not a source of segmentation supervision |
| Seed annotated set (SAM-assisted, current realistic target) | ~200 source images, mask count TBD per-image density | Our actual near-term capacity — count instances per image before assuming scarcity |
| Reference precedent for what "more" looks like | 244 images / ~13,000 masks (LBMS-SAM) | Aspirational scale, not our Phase 2 starting point |
| Synthetic augmentation | As large as needed | Reuse Phase 1's `NoiseImage` engine on annotated reference crops |
| Pseudo-labeled expansion | Spot-checked, not fully re-annotated | Only once a working seed model (GSEFE and/or MDFF fine-tuned) exists |

---

## 8. Phase 2.1 Preview — Where Morphometric Analysis Plugs In

Not the focus of the current phase, but worth scoping now so the segmentation output format doesn't have to be redesigned later. The standard pipeline observed across the porosity/PSD literature is consistent:

1. **Scale calibration:** read the SEM's embedded scale bar or metadata (length in physical units corresponding to a pixel span) — measuring pixel density along the scale bar gives a calibration factor representing the physical length (and by extension, area) of a single pixel. This is the "read a scale reference from the image" step, and it should also attempt automatic extraction from the image's metadata bar when available, falling back to a user-provided value otherwise.
2. **Per-instance measurement from masks:** once masks exist, standard connected-component-style analysis is sufficient — no novel research needed here. Segmented masks are fed into a connected component analysis to compute particle size, shape, and orientation distributions, typically expressed as equivalent spherical diameter. Aspect ratio and circularity are computed directly from each mask's geometric moments/contour, not a separate model.
3. **Porosity specifically** is conventionally a thresholding/segmentation-area ratio problem, not a per-instance one: total porosity is calculated as the area covered by pores above a minimum size threshold (filtering out single-pixel noise), with bin-wise size distributions computed from each individual pore's area, including extrapolation to an equivalent-sphere radius. The semantic segmentation path's pore class output maps directly onto this without modification.
4. **3D/volume estimates** are the one place where genuine caution is warranted: true volume from a single 2D SEM image is an inherently approximate exercise (inferring 3D structure from a 2D projection), and the literature handles it either via stereological approximations (equivalent spherical diameter assumptions, as above) or by requiring actual 3D acquisition (FIB-SEM tomography, serial sectioning) when true volumetry is required. Decide early whether "volume" in our requirements means a stereological estimate from 2D masks (fast, approximate, no new acquisition hardware needed) or genuine 3D reconstruction (accurate, but a materially larger scope than segmentation alone).

---

## 9. Checklist

**Foundational (do first, before any training)**
- [ ] Run frozen SAM (automatic mode, zero-shot) on a handful of real Bharat Atomic SEM images; visually assess merged-particle failure rate
- [ ] Count actual mask instances per image on a representative sample of our raw images — establish the real annotation unit, not just image count
- [ ] Set up SAM-assisted annotation tooling (Label-Studio + SAM ML backend, or equivalent)
- [ ] Build a held-out evaluation split (small, hand-verified) before any training begins — needed for every gate below

**Primary track — frozen SAM + GSEFE/MDFF**
- [ ] Annotate seed set (~200 source images, SAM-assisted correction)
- [ ] Apply Phase 1 `NoiseImage`/`new_augment_sem` to seed set as augmentation
- [ ] Train GSEFE alone; evaluate against frozen-SAM baseline
- [ ] Train MDFF alone; evaluate against frozen-SAM baseline
- [ ] Train GSEFE + MDFF jointly; compare against both isolated results and baseline
- [ ] Benchmark SAM-I-Am-style rule-based post-processing as a zero-training-cost comparison point

**Self-supervised pretraining experiment (deferred, after primary track has a working baseline)**
- [ ] Implement masked-patch continued pretraining for SAM's encoder on unlabeled NFFA-Europe pool
- [ ] Validate: compare zero-shot mask quality, adapted encoder vs. vanilla frozen SAM, on held-out eval set
- [ ] Gate: only proceed to combine with GSEFE/MDFF if validation shows clear improvement; otherwise discard and retain vanilla SAM

**Latency and deployment**
- [ ] Benchmark primary track (SAM ViT-L/ViT-B + GSEFE/MDFF) latency on target CUDA hardware once available
- [ ] Benchmark MobileSAM/FastSAM as lightweight alternatives on identical hardware
- [ ] Decouple morphometric analysis (Section 8) from the real-time segmentation loop — confirm this architecturally, not just as an intention

**Deferred / future scope**
- [ ] Revisit closed-set fast-path (YOLO-seg/RF-DETR/PerovSegNet-style) only once a specific customer vertical and matching annotation budget exist
- [ ] Phase 2.1 scale calibration and morphometric pipeline, once segmentation output format is stable

---

## 10. Summary of Recommendations

- **Architecture:** Frozen SAM + GSEFE/MDFF (the LBMS-SAM pattern) as the primary deliverable, because it is the only candidate under consideration that satisfies the open-set "any sample" product requirement, and because it needs the least labeled data of any path given our current zero-label starting point. SAM's own encoder and decoder are **not** fine-tuned in this track.
- **Closed-set detectors (YOLO-seg, RF-DETR, PerovSegNet-style):** explicitly deferred, not rejected outright — they remain a legitimate future fast-path for specific, high-volume material verticals once Bharat Atomic has a defined customer segment and the annotation budget such models require (precedent: PerovSegNet needed ~11,000 images for 3 classes on one material system).
- **Self-supervised pretraining:** useful only where there's real capacity to absorb it. GSEFE/MDFF are too small to benefit from SSL pretexts and go straight to supervised training. Continued self-supervised pretraining of SAM's own ViT encoder on our unlabeled NFFA-Europe pool is a legitimate but genuinely uncertain experiment, gated by a mandatory zero-shot validation check before it's allowed anywhere near the labeled pipeline.
- **Latency:** 40ms is achievable for a SAM-family pipeline on CUDA + TensorRT hardware if a lightweight variant (MobileSAM/FastSAM-class backbone) is used; vanilla SAM ViT-L fine-tuned is the higher-quality, higher-latency reference point. The actual risk to the latency budget is pipeline overhead (preprocessing, mask postprocessing, and any temptation to run morphometric analysis synchronously), not the segmentation model itself.
- **Dataset:** NFFA-Europe is classification-only and has no segmentation masks — usable for continued pretraining experiments and as a source pool for SAM-assisted annotation, not for direct supervised segmentation training. We are starting from zero labeled masks; the realistic near-term target is a ~200-image SAM-assisted seed set, expanded via Phase 1's noise simulator and later pseudo-labeling, not the thousands-of-masks scale of published precedents.
- **General-vs-fine-tuned boundary:** SAM's general SA-1B pretraining is the right starting point and is left untouched; the only thing we fine-tune is the small GSEFE/MDFF adapter pair, mirroring the same "freeze the broadly-useful part, fine-tune only the domain-specific part" logic already validated in Phase 1's RDUNet/NAFNet decoder-only fine-tuning.
- **Phase 2.1 scoping:** scale calibration from the SEM's own metadata/scale bar, per-instance geometric measurement from masks (standard, not novel), and an early decision on whether "volume" means a 2D stereological estimate or requires true 3D acquisition.

---

## 11. References

1. Aswath, A., et al. (2022). Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey. arXiv:2206.07171.
2. Khadangi, A., Boudier, T., & Rajagopal, V. (2021). EM-stellar: benchmarking deep learning for electron microscopy image segmentation. *Bioinformatics*, 37(1), 97–106.
3. "Deep learning for brain electron microscopy segmentation: Advances, challenges, and future directions in connectomics and ultrastructure analysis." *Computers & Graphics*, 2025.
4. Yamanakkanavar, N., & Lee, B. (2021). A novel M-SegNet with global attention CNN architecture for automatic segmentation of brain MRI. *Computers in Biology and Medicine*, 136, 104761.
5. Pan, J. G., Wang, L., & Cai, X. (2025). Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework (PerovSegNet). arXiv:2509.26548.
6. Qi, Y., Zhang, J., Kuang, J., Ren, T., Wang, D., Wu, Z., Zheng, H., & Zhang, Q. (2026). LBMS-SAM: Segment anything model guided SEM image segmentation for lithium battery materials. *Neural Networks*, 196, 108325.
7. Rettenberger, L., et al. (2024). Uncertainty-aware particle segmentation for electron microscopy at varied length scales. *npj Computational Materials*, 10, 124.
8. Rühle, B., Krumrey, J. F., & Hodoroaba, V.-D. (2021). Workflow towards automated segmentation of agglomerated, non-spherical particles from electron microscopy images using artificial neural networks. *Scientific Reports*, 11, 4942.
9. Machine vision-driven automatic recognition of particle size and morphology in SEM images. *Nanoscale* (RSC Publishing), 2020. DOI:10.1039/D0NR04140H.
10. MatSegNet: a New Boundary-aware Deep Learning Model for Accurate Carbide Precipitate Analysis in High-Strength Steels. arXiv:2312.17251.
11. Rettenberger, L., Szymanski, N. J., Giunto, A., Dartsi, O., Jain, A., Ceder, G., Hagenmeyer, V., & Reischl, M. (2025). Leveraging unlabeled SEM datasets with self-supervised learning for enhanced particle segmentation. *npj Computational Materials*, 11, 289.
12. SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy. arXiv:2501.03153, 2025.
13. Archit, A., et al. (2025). Segment Anything for Microscopy. *Nature Methods*, 22, 579–591.
14. Lightweight open-source fine-tuning of SAM2 enables domain-specific microscopy segmentation. bioRxiv, 2025.
15. Foundation Models for Zero-Shot Segmentation of Scientific Images without AI-Ready Data. arXiv:2506.24039, 2025.
16. SAM-I-Am: Semantic boosting for zero-shot atomic-scale electron micrograph segmentation. arXiv:2404.06638, 2024.
17. Aversa, R., et al. (2018). The first annotated set of scanning electron microscopy images for nanoscience (NFFA-Europe). *Scientific Data*, 5, 180172.
18. Semi-supervised spatiotemporal segmentation of in situ TEM for nanoparticle dynamics (SwinTCN-Seg). *ScienceDirect*, 2026.
19. Comparing YOLOv8 and Mask R-CNN for instance segmentation in complex orchard environments. arXiv:2312.07935, 2023.
20. Achieving 374 FPS with YOLOv8 Segmentation on NVIDIA RTX 5070 Ti GPU. Medium / cvRealtime, 2026.
21. Evolution of Porosity in Suspension Thermal Sprayed YSZ Thermal Barrier Coatings through Neutron Scattering and Image Analysis Techniques. arXiv:2010.07599.
22. Microstructural investigation of hybrid CAD/CAM restorative dental materials by micro-CT and SEM. arXiv:2308.07341.
23. Kirillov, A., et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4015–4026.
24. Liu, Z., et al. (2022). A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11976–11986.
25. Woo, S., et al. (2023). ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16133–16142.

---

*This document reflects the current state of architecture and dataset planning for Phase 2 as of this writing. The open-set constraint (Section 3) and the zero-label starting point (Section 7.2) are the two facts most likely to be forgotten under time pressure — re-read those two sections before any decision that would point the project back toward a closed-set, large-labeled-dataset architecture by default.*
