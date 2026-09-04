## mask-interpolation

`mask-interpolation` completes sparse sequences of 2D masks without depending
on image files, axis metadata, a GUI, or FITS orchestration.

It is designed for masks drawn on a few representative time points or Z
planes. Shapes are interpolated between known planes, while optional start/end
extrapolation copies the nearest known mask into outer gaps. Non-spatial axis
combinations are handled independently, so channels or other batches never
bleed into each other.

```python
from mask_interpolation import fill_missing_masks

completed = fill_missing_masks(
    sparse_masks,
    axes="TZYX",
    interpolation_axis="Z",
)
```

The input may have any shape with at least three dimensions and must include
`Y` and `X` axes. Entirely zero planes along the selected interpolation axis are
missing. Every combination of the remaining axes is processed independently.
Leading and trailing gaps are copied from the nearest known mask by default.
The input is never modified.

## Role in FITS

The reference- and ROI-mask viewers preserve the user's raw drawing planes and
call this package to produce a completed preview or save-ready stack. FITS owns
the drawing interaction and artifact encoding; `mask-interpolation` only
performs the array geometry.
