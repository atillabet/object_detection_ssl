# Experimental design

## Primary question

Does shape-aware self-supervised pretraining improve low-label object detection
and shape-preserving robustness relative to supervised initialization and a
compute-matched conventional SSL baseline?

## Required controls

- supervised training from random initialization;
- global VICReg with the same unlabeled images and optimizer updates;
- object-centric or mixed-crop VICReg;
- uniform aligned contour consistency;
- high-contour region-aligned consistency;
- mismatched-contour negative control;
- optional supervised object-centered and stylized-data controls.

## Evaluation protocol

1. Generate nested labeled splits for every seed with `ssl-splits`.
2. Generate nested unlabeled scaling manifests with `ssl-manifests`.
3. Keep architecture, optimizer updates, labeled split, fine-tuning schedule, and
   evaluation sample fixed within each comparison.
4. Run independent clean validation and the V2 shape-reliance benchmark.
5. Aggregate across seeds with Student-t confidence intervals and paired deltas.
6. Use image-paired bootstrap intervals for thresholded shape-benchmark metrics.
7. Inspect representative successes and failures with `ssl-explain`.

## Interpretation

Better clean mAP alone does not establish a shape-aware representation. Evidence
requires a clean-performance result together with improved appearance/context
retention or a larger shape-versus-appearance reliance gap. The aligned method
must also outperform the mismatched-contour control to attribute an improvement
to meaningful spatial correspondence rather than additional loss or compute.

Object-centered sampling based on ground-truth boxes is label-guided SSL. Report
it separately from fully label-free pretraining, even though class identities are
not used as learning targets. Negative images remain in every condition and use a
global-view fallback so sample counts do not become a hidden confound.
