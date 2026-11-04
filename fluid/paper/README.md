# Scale-aware GPU liquid paper

The paper is authored in `fluid-paper.tex` and uses verified citations in
`references.bib`. It follows a two-column academic format, without authors,
and deliberately does not claim results that have not been measured.

The compiled preview is `fluid-paper.pdf` (generated automatically by the
`fluid-paper` GitHub Actions workflow; a workflow artifact is also uploaded).

### Build

In this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error fluid-paper.tex
```

### Inserting future experimental figures

Place the finished image in `figures/` using one of these filenames:
- `teaser.png` -- overview of small drops, coalescence, filament and shading
- `scale-regimes.png` -- dimensionless-number and scale diagnostic plots
- `pipeline.png` -- actual buffer and shader stage diagram
- `laplace-validation.png` -- measured pressure-jump convergence plots
- `microdrop-sequence.png` -- measured physical validation sequences
- `rendering-ablation.png` -- measured optical comparison
- `performance.png` -- measured GPU-stage timings on the M2
- `hardware-comparison.png` -- matched-device reference

The LaTeX paper automatically prefers these images over placeholders
and also accepts matching `.pdf` and `.jpg` filenames. Caption text and
cross references survive future replacement. Captions must be changed
to describe actual experimental figures, not just the planned contents.

### Scope and evidence

The verified material physics functions currently on `main` cover
capillary length, Young--Laplace pressure, and dimensionless numbers.
GPU compute SPH, visual reconstruction, and 60 FPS M2 measurements are
research/implementation targets. Until validated, they are not working
claims. The paper contains an explicit evidence ledger and acceptance
gates to keep the manuscript synchronized with the code.
