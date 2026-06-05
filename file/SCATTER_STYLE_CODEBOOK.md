# Scatter Style Codebook

This file merges two existing scatter-plot style references:

- Green, smaller-point style from `GetForYeast/train_yeast_single_peak_scatter_style.md`
- Blue, larger-point benchmarking style from `Benchmarking/SCATTER_PLOT_SPEC.md`

Use the style codes below in future requests so a plot can be specified concisely.

## Style Codes

| Code | Name | Main Use | Visual Intent |
|---|---|---|---|
| `GSM` | Green Small Marker | Main result figures, cleaner reports, presentation-style plots | Better looking, less visually heavy, dense points blend smoothly |
| `BLD` | Blue Large Diagnostic | Benchmarking, debugging, checking where predictions fail | Easier to see individual points and inaccurate regions |

Mnemonic:

- `GSM` = Green / Small / Manuscript-style
- `BLD` = Blue / Large / Diagnostic

## Quick Selection Rule

Use `GSM` when the figure should look polished and visually calm.

Use `BLD` when the priority is inspection: seeing bad regions, local scatter structure, outliers, and model failure patterns.

## `GSM`: Green Small Marker

Recommended code phrase:

```text
scatter style = GSM
```

Core marker settings:

```python
ax.scatter(
    x_plot,
    y_plot,
    s=1.5,
    alpha=0.15,
    c="#1b7c3d",
    edgecolors="none",
    zorder=1,
)
```

Recommended figure settings:

```python
fig, ax = plt.subplots(figsize=(10, 10))
fig.savefig(out_path, dpi=300, bbox_inches="tight")
```

Recommended axis and annotation style:

```python
lo, hi = 0.0, 20.0
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x", zorder=10)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="both", labelsize=12)
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
```

Optional regression line:

```python
x_line = np.linspace(lo, hi, 200)
y_line = slope * x_line + intercept
ax.plot(
    x_line,
    y_line,
    color="green",
    linewidth=1.2,
    zorder=3,
    label=f"fit: y={slope:.3f}x+{intercept:.3f}",
)
```

Optional metric box:

```python
ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#f0e6c8",
        alpha=0.9,
        edgecolor="#a98f5a",
    ),
)
```

Notes:

- This style is based on the GetForYeast final test scatter.
- It is best when there are many points and the plot should not look too noisy.
- The fixed `0-20` axis range is appropriate for expression-like log-scale values in the existing GetForYeast script. If another task uses a different value space, keep the marker style but adapt the limits.

## `BLD`: Blue Large Diagnostic

Recommended code phrase:

```text
scatter style = BLD
```

Core marker settings:

```python
ax.scatter(
    x,
    y,
    s=4,
    alpha=0.2,
    c="#2563eb",
    edgecolors="none",
    rasterized=True,
)
```

Recommended figure settings:

```python
fig, ax = plt.subplots(constrained_layout=True)
fig.savefig(out_path, dpi=150)
```

Recommended diagonal:

```python
ax.plot(
    [diag_lo, diag_hi],
    [diag_lo, diag_hi],
    "k--",
    lw=0.8,
    alpha=0.6,
)
```

Recommended axis limits:

```python
# Default idea:
# Use 0.5-99.5 percentiles per axis, plus 3% padding.
# If x and y share units, use shared limits from true and pred together.
# If x and y have different units, use separate limits and clip the diagonal.
```

Recommended title/subtitle metrics:

```text
Title: Pearson r, Spearman rho, MAE computed on the full merged n
Subtitle: panel label + (n=..., in frame ...)
```

Notes:

- This style is based on the cross-project benchmarking spec.
- Larger points make it easier to notice inaccurate regions and local failure structure.
- `rasterized=True` keeps vector outputs or complex figure rendering lighter when points are numerous.
- Percentile-based limits are better for diagnostic comparison across models because outliers do not crush the main cloud.

## Parameter Comparison

| Parameter | `GSM` Green Small Marker | `BLD` Blue Large Diagnostic |
|---|---|---|
| Point size | `s=1.5` | `s=4` |
| Alpha | `alpha=0.15` | `alpha=0.2` |
| Color | `#1b7c3d` | `#2563eb` |
| Edge | `edgecolors="none"` | `edgecolors="none"` |
| Rasterized | optional / not used in original | `rasterized=True` |
| Main diagonal | red dashed, `linewidth=1` | black dashed, `lw=0.8`, `alpha=0.6` |
| Axis limits | usually fixed `0-20` for GetForYeast expression plots | percentile limits, usually `0.5-99.5%` plus padding |
| DPI | `300` | `150` |
| Best for | polished output | diagnostic inspection |

## Canonical Helper Snippets

### Apply `GSM`

```python
def scatter_gsm(ax, x, y, *, label=None):
    return ax.scatter(
        x,
        y,
        s=1.5,
        alpha=0.15,
        c="#1b7c3d",
        edgecolors="none",
        zorder=1,
        label=label,
    )
```

### Apply `BLD`

```python
def scatter_bld(ax, x, y, *, label=None):
    return ax.scatter(
        x,
        y,
        s=4,
        alpha=0.2,
        c="#2563eb",
        edgecolors="none",
        rasterized=True,
        label=label,
    )
```

## How To Specify In Future Requests

Examples:

```text
用 GSM 画这个预测散点图。
```

```text
这次用 BLD，我想看哪些区域预测不准。
```

```text
同一个结果输出两张图：GSM 作为正式图，BLD 作为诊断图。
```

If no style is specified:

- Use `GSM` for final result figures.
- Use `BLD` for benchmarking, debugging, or comparison figures.
