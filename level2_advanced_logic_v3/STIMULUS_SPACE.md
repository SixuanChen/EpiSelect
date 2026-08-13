# Stimulus space

Every Level-2 object has **all four attributes** below. Exactly **two feature dimensions are logically relevant in any one base rule**; the other two are nuisance dimensions and never determine category membership.

| Dimension | Values |
|---|---|
| color | `red`, `blue`, `green`, `purple` |
| shape | `circle`, `square`, `triangle`, `star` |
| texture | `solid`, `horizontal_stripes`, `dots` |
| size | `small`, `large` |

The complete single-object universe is therefore `4 × 4 × 3 × 2 = 96` objects.

## Relevant feature pairs

All six unordered dimension pairs are represented equally:

1. color + shape
2. color + texture
3. color + size
4. shape + texture
5. shape + size
6. texture + size

Within a base rule, the two relevant target predicates are called **A** and **B**. A/B orientation is counterbalanced across rule families so that direction-sensitive rules are not always tied to the same feature modality.

Example:

- A = `is RED`
- B = `is a CIRCLE`
- rule = `A -> B`

The nuisance texture and size still vary across all history and action objects.

## Target-value balance

Across the 60 generated base category rules, each feature dimension is active in 30 bases. Target-value counts differ by at most one where exact equality is mathematically impossible:

- color: `8, 8, 7, 7`
- shape: `8, 8, 7, 7`
- texture: `10, 10, 10`
- size: `15, 15`

The exact assignment for every base rule is in `RULE_DISTRIBUTION.csv`.

## Text rendering

Objects always use the same attribute order:

`SIZE COLOR TEXTURE SHAPE`

Examples:

- `small red solid circle`
- `large purple horizontally striped star`
- `small blue dotted triangle`

## Optional visual rendering

`render_stimuli.py` deterministically maps the same symbolic objects to 256 × 256 PNGs.

- white canvas, center `(128, 128)`
- small maximum shape bounding box: **72 px**
- large maximum shape bounding box: **128 px**
- every shape template is scaled to the same maximum bounding-box dimension within a size level
- outline: 3 px dark gray
- colors:
  - red `#D62728`
  - blue `#1F77B4`
  - green `#2CA02C`
  - purple `#9467BD`
- horizontal stripes: fixed pixel-density mask clipped to the shape
- dots: fixed pixel-density grid clipped to the shape

Thus a 72-px square and a 72-px star have the same maximum bounding-box extent; size is not secretly encoded differently by shape.

The shipped benchmark is text-ready. The renderer exists so a later text-vs-vision comparison can use the **identical latent object JSON and gold labels**.
