## mask-interpolation

`mask-interpolation` completes sparse sequences of 2D masks without depending
on image files, axis metadata, a GUI, or FITS orchestration.

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
