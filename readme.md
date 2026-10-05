# MVGPUtensils

MVGPUtensils (namespace `mvgpu`) is MVUtensils on the GPU, through VapourSynth's Vulkan GPU frames
(API 4.3): every clip it takes and returns is GPU resident. The goal is a drop-in replacement for
`mvu`, the same functions with the same arguments; there are `Super`, `Analyse`, `AnalyseMany`,
`Recalculate`, `Degrain` (with `Degrain1` … `Degrain25`), `Compensate`, `Flow`, `FlowBlur`,
`FlowInter`, `FlowFPS`, `SCDetection`, `VectorLengthMask`, `SADMask` and `OcclusionMask` (not the
`Depan` family), and `ToMVU` and `FromMVU`, which convert vector clips to and from mvu's form. The
rest of this file documents MVUtensils, which the GPU filters follow.

```python
clip = core.bs.VideoSource('video.mkv', gpu=True)
sup = core.mvgpu.Super(clip, blksize=16, overlap=8)
vectors = core.mvgpu.AnalyseMany(sup, radius=2)
denoised = core.mvgpu.Degrain(clip, sup, vectors)
```

## Status

* `Super` takes everything mvu.Super does: GRAY and YUV 4:2:0, 4:2:2, 4:4:0 and 4:4:4 at 8 to 16
  bits or float, `pel` 1, 2 and 4, every `sharp` and `rfilter`, `pelclip`, `onelevel`, and every
  block size, overlap and padding, horizontal and vertical apart; but a `pelclip` only at `pel=2`
  (below). As mvu.Super does, it refuses a padding the chroma subsampling doesn't divide (an odd
  horizontal one at 4:2:0 and 4:2:2, an odd vertical one at 4:2:0 and 4:4:0). Its level 0 is
  mvu.Super's bit for bit (`test/check_super.py`). Its coarse levels are the GPU search's own,
  reduced with `rfilter`'s filter.
* The other filters so far take Gray and YUV supers of every subsampling (4:2:0, 4:2:2, 4:4:0 and
  4:4:4) and 8 to 16-bit or float samples at any `pel` (1, 2 or 4) with every block size mvu takes
  (4×4 to 128×128), any overlap and padding, horizontal and vertical apart, MVUtensils' extended
  grids included; but `Analyse` and `Recalculate` refuse the blocks whose SADs can pass 2^31:
  128×128 blocks with 4:4:4 chroma at 16 bits or float, and with `satd`, whose SATD can reach twice
  the SAD, 128×128 blocks with chroma of any subsampling at 16 bits or float.
  `Analyse` and `Recalculate` take a grid other than the super's, as in mvu, where it fits the
  super's block-aligned frame; the other filters take vectors of any such grid with the super's
  planes. As in mvu, vectors analysed
  on an 8-bit copy of a clip serve the filters that compensate motion on the 9 to 16-bit or float
  clip, their SAD thresholds scaled to the depth the vectors were analysed at, and a Gray clip's SADs
  are luma's.
* The filters check the clips they're given as mvu's do, with mvu's messages: a super, centersuper
  or vector clip shorter than the clip it goes with (for `Recalculate`, vectors shorter than the
  super) is refused, since the frames past its end would be its last one again; every super frame
  a filter loads must have been made with the same `Super` arguments as the clip's first frame, and
  every vector frame with vectors with the same `Analyse` or `Recalculate` arguments as the first
  frame of its clip (a spliced clip can mix them); and the vector clips one filter takes must all
  come from one analysis but for their delta, bit depth included. `Recalculate` gives a frame whose
  reference frame lies past the end of the super no vectors, also where a longer vector clip has
  some for it. `satd` on 16×2 blocks is refused as mvu refuses it.
  `test/check_robustness.py` runs each of these cases on both plugins.
* `Analyse` searches float supers as the 16-bit samples they stand for: luma's 0 to 1 and
  chroma's −0.5 to 0.5 scaled to 0 to 65535, rounded and clamped, each sample as it is read, which
  puts the SADs on the 16-bit scale mvu gives float SADs. `Degrain`, `FlowInter` and `FlowFPS` take
  mvu's float arithmetic operation for operation. mvu's float `FlowInter` and `FlowFPS` round
  differently with each CPU's kernels (AVX-512, AVX2, SSE2); mvgpu's are bit for bit those of
  mvu's AVX-512 kernels.
* `Analyse` searches its own way: a coarse search on a pyramid of the frame, a seed list per block,
  checkerboard passes under mvu's cost (`mvlambda`, `lsad`; `plevel` scales the coarse levels'
  lambda as mvu scales it per level), a wide search for the blocks still
  above `badsad` (radius `badrange`, every `badstep` pixels, an argument mvu doesn't have) and a
  half-pel step (none at `pel=1`). At `pel=4` the seeds and the passes stay on the half-pel grid
  (chained and inverted seeds are rounded to it, toward zero) and a quarter-pel step follows the
  half-pel one. A super without the coarse levels (`onelevel=True`, or a frame narrower than 192
  pixels, which has none) leaves out the coarse search: the seeds are zero and, with `AnalyseMany`,
  the chained and inverted vectors.
  With `chroma=False` every SAD is luma's alone, at every level. Above 8 bits `mvlambda`, `lsad` and
  `badsad` are scaled to the depth as mvu scales them, and `lsad` relaxes lambda by the worst
  neighbour's SAD in steps of 2^(bits − 8), as the search tabulates it.
  With `satd=True` luma's SAD is mvu's SATD (the absolute values of each 4×4 tile's Hadamard
  transform added up, halved) wherever the full-size grid's vectors are measured, chroma's staying
  SADs, so the vectors and their SADs are chosen and scored as mvu's are; the coarse search, mvgpu's
  own, keeps the SAD. As in mvu, `satd` refuses 16×2 blocks.
  `search`, `searchparam`, `pelsearch`, `levels`, `pnew`, `pzero`, `pglobal`, `globalmv`,
  `meander` and `trymany` tune mvu's search; they are checked and otherwise ignored.
  The wide search defaults to the tested `badsad=1000`, `badrange=40` and `badstep=2` rather than
  mvu's `badsad=10000` (which it would almost never reach) and `badrange=24`.
* `AnalyseMany` also seeds each field from the fields refined before it (chained and inverted
  vectors), which separate `Analyse` calls can't.
* The super and the vectors are GPU frames laid out for the GPU, described by frame properties
  under the prefix `MVGPUtensils` by default. A super frame is one Gray frame of the clip's sample
  type holding the whole super: the padded luma planes in its rows, chroma and the coarse levels
  after them at offsets its properties give. A vector clip's frames are the vectors themselves, a
  32-bit record (x, y, SAD, 0) per block and a row of records per row of blocks (Gray32,
  4·blocks wide), vectors in units of the super's `pel`; a frame whose reference frame lies outside
  the clip has `MVGPUtensilsAnalysisHasVectors` 0. Neither clip is the source's size or format, so
  view them through the filters that take them.
* `ToMVU` and `FromMVU` convert vector clips between mvgpu's form and mvu's, which keeps the vectors
  in frame properties (`MVUtensilsAnalysisVectors` and `MVUtensilsAnalysisSAD`, none on a frame
  without vectors). `mvgpu.ToMVU(vectors)` gives the vectors of mvgpu's `Analyse`, `AnalyseMany` or
  `Recalculate` to mvu's filters; `mvgpu.FromMVU(vectors, super)` gives mvu's to mvgpu's, `super`
  being mvgpu's super of the clip they were analysed on (or of the clip at another bit depth) with
  their `blksize`, `overlap`, `pad` and `pel`. Both take a list of vector clips and return one for
  each, and `prefix` and `mvuprefix` name the two sides' property prefixes. The vectors and SADs pass
  unchanged, so either plugin's filters make the same frames from them: the checks run each plugin's
  filters on the other's vectors this way, and `test/check_convert.py` checks the conversions both
  ways. `ToMVU`'s frames are the records, downloaded, with mvu's properties added, and remain mvgpu
  vector frames; `FromMVU` checks every vector as mvu checks the vectors it loads (none may read
  outside the padded frame) and copies the records into GPU memory. mvu's filters also need mvu's
  super of the clip, made with the same arguments (`onelevel=True` will do: they read only its
  full-size level). mvgpu's filters refuse mvu's vectors with an error that names `FromMVU`; mvu's
  filters given mvgpu's prefix would take mvgpu's unconverted vector frames for frames without
  vectors, so convert them. The masks and `SCDetection` read only the vectors, so they take
  `FromMVU`'s vectors of any grid and format mvu makes, not just those listed above: they matched
  mvu's exactly on 4:2:2 16×16, 4:2:0 16×8, 32×16, 64×64 and 4×4, and 4:4:4 8×4 grids.

  ```python
  sup = core.mvgpu.Super(core.std.GPUUpload(clip), blksize=16, overlap=8)
  vectors = core.mvgpu.ToMVU(core.mvgpu.AnalyseMany(sup, radius=2))
  denoised = core.mvu.Degrain(clip, core.mvu.Super(clip, blksize=16, overlap=8, onelevel=True), vectors)
  ```
* A `pel=4` super stores luma's four half-pel planes only, as `pel=2` does: mvu's quarter-pel
  samples are rounded averages of the half-pel ones around them, whatever `sharp`, which the
  quarter-pel step computes as it goes; so does 4:4:4 chroma, which takes the luma vector itself.
  Subsampled chroma, which the half-pel passes already read at quarter-pel positions, keeps all
  sixteen planes' samples, as one image of the quarter-pel grid per plane rather than sixteen
  planes: neighbouring blocks' vectors mostly differ in sub-pel phase, and in the image their reads
  stay in one region. That made a `pel=4` chain of `Super`, `AnalyseMany` and `Degrain` 7-19%
  faster on the GPU on moving content and 1-4% on static (RX 6900 XT, 16×16 blocks, radius 3). A
  4:2:0 super is twice the size of a `pel=2` one instead of four times (4:2:2 and 4:4:0 two and a
  half times), a 4:4:4 one the same size.
* Divergence from MVUtensils: `Super` refuses a `pelclip` at `pel=4`. A pelclip's quarter-pel
  samples needn't be averages of its half-pel ones, so its luma would need all sixteen planes, twice
  the memory of a `pel=4` super that computes them. A `pelclip` at `pel=2` works as in mvu.
* `Degrain` takes all of mvu.Degrain's arguments, on mvgpu's supers and vectors. Given the same
  vectors its output is mvu.Degrain's bit for bit, at `pel=2` and `pel=4` (`test/check_degrain.py`
  runs both on mvgpu.AnalyseMany's vectors, or on mvu.Analyse's, which mvgpu's search wouldn't find);
  its weights, which mvu computes in double precision, are reproduced exactly in integers.
* `FlowInter` and `FlowFPS` take all of mvu's arguments, on mvgpu's supers and vectors, and given
  the same vectors their output is mvu's bit for bit (`test/check_flow.py`, on mvgpu.Analyse's
  vectors, mvu.Analyse's, or constant fields with scene changes in between). mvu resizes the blocks'
  vectors and occlusion masks to the pixels with zimg's bilinear resize, in 64×64 tiles; the GPU
  does the same arithmetic, the taps computed for each tile as zimg computes them.
* `Compensate`, `Flow` and `FlowBlur` take all of mvu's arguments, and given the same super and
  vectors their output is mvu's bit for bit (`test/check_flow.py --filter compensate`, `flow`,
  `blur`), float `FlowBlur` included, which sums in double precision as mvu does (on a device
  without 64-bit floats in float, which can round apart). As in mvu, `Compensate` takes the blocks
  whose SAD isn't under `thsad` from the frame's own super, which matters where the super isn't the
  clip's.
* `VectorLengthMask`, `SADMask` and `OcclusionMask` take all of mvu's arguments, and given the same
  vectors their masks are mvu's bit for bit (`test/check_masks.py`), zimg's resize of the whole plane
  included, but where a `gamma` other than 1 (other than 1 or 2 for `VectorLengthMask`) takes a
  power: mvgpu computes it correctly rounded, in double precision, while the C library's `powf` mvu
  calls misrounds about one value in ten thousand by a unit in the last place, which can move a
  mask's sample a step (on a device without 64-bit floats mvgpu uses the GPU's `pow`, a step off
  more often). `SCDetection` sets mvu's properties on the clip's frames, GPU resident or not; it
  counts the badly matched blocks on the GPU and waits for the count, the one filter whose frames
  wait for their GPU work.
* `Recalculate` takes all of mvu's arguments, for the super's grid or another that fits it (the old
  vectors can be of any grid and `pel`, from mvgpu.Analyse or Recalculate of the same bit depth; as
  in mvu, they're rescaled to the super's `pel`, which the new vectors take), and its vectors are
  mvu's bit for bit (`test/check_recalculate.py`), every search type
  replicated candidate for candidate: unlike `Analyse`'s, its blocks don't depend on one another.
  `satd` makes luma's SAD mvu's SATD, as in `Analyse`.
  As in mvu, a frame whose old vectors are missing (their reference frame outside the clip) stays
  without vectors, with the new grid's description. Float supers
  are searched as `Analyse` searches them, as the 16-bit samples they stand for (mvu sums float
  differences in the order its CPU kernels take), so at `pel=1`, where the samples are the clip's,
  the vectors are mvu's of the clip so quantized.
* `test/matrix.py` runs the checks over their case matrices (`super`, `analyse`, `degrain`, `flow`,
  `masks`, `motion`, `recalculate`, `convert`, `robustness`) on a directory of raw test clips, and
  `smoke` a cross section of all of them in a few minutes.
* With the environment variable `MVGPU_PROFILE=1`, the filters time their stages on the GPU and
  print the averages to stderr when the filter is freed. They wait for every frame, so they run
  slower.
* The search's results are bit-identical to its CPU reference implementation,
  `test/reference/reference.cpp` (built with `meson compile mvgpu_reference`, or on its own with any
  C++20 compiler); `test/check_reference.py --reference` runs it on the same frames with the same
  arguments and compares every block.
* Needs a Vulkan device with 32- or 64-lane subgroups, 64-bit integers and 64-bit buffer atomics,
  and for `Degrain` 158 storage buffers per kernel (2 · 25 references with three buffers each);
  64-bit floats where the device has them (the masks' powers, float `FlowBlur`'s sums).
  Building needs the Vulkan headers (`-Dvulkan_include=` for meson, the Vulkan SDK for
  `msvc/MVGPUtensils.slnx`); nothing is linked, the core hands out every entry point.

# MVUtensils

MVUtensils (namespace `mvu`) is a large refactoring and cleanup of the original
VapourSynth [mvtools](https://github.com/dubhater/vapoursynth-mvtools) port. It keeps the same
overall workflow (build a *super* clip, estimate motion vectors with *Analyse*, then feed those
vectors to filters like *Degrain* or *Flow*) but fixes a number of long-standing bugs, is faster,
and adds full high-bit-depth and float support.

## Goals

* Fix long-standing correctness bugs — most notably the right/bottom border not being processed.
* Be faster through more cache-friendly algorithms and wider SIMD (SSE2/AVX2/AVX-512 where it helps).
* Greatly reduce memory usage in `Super`, `Analyse`, `Degrain` and the `Flow*` filters.
* Full float (32-bit) support in every filter except the `Depan*` family.
* A smaller, more consistent API: arrays instead of cryptic per-direction arguments, and consistent
  frame-property propagation.

## Notes

* Supported pixel formats are GRAY and YUV at 8–16 bit integer or 32 bit float. The bit depth and
  subsampling are taken from the input/super clip; masks derive their format from the vector clip.
* A vector clip stores block geometry and per-block SAD, never pixels, so it does **not** have to be
  analysed at the same bit depth as the clip it is later applied to. Analysing an 8-bit copy and
  using the result on the 16-bit or float original is a supported way to trade motion precision for
  analysis speed. Resolution, subsampling, `pel` and the super's padding must still match. Vectors analysed
  with `chroma=False` are the exception to the subsampling rule (e.g. analysed on a GRAY copy and applied to
  4:2:0), as long as their overlap is divisible by the clip's subsampling, even if only luma is processed.
  SAD-derived arguments (`thsad`, `thscd1`) are interpreted against the depth the vectors were
  analysed at, so they keep their documented 8-bit 8×8 meaning either way.
  [Recalculate](#recalculate) is the one exception and still requires equal bit depths.
* Every frame of a super or vector clip must be made with the same arguments, and the vector clips passed
  to one filter together must match each other, including the `chroma` setting they were analysed with.
  Splicing clips made with different settings is not supported: filters stop with an error instead.
* The attached frame properties use the prefix `MVUtensils` by default. Every function accepts a
  `prefix` argument to change it, which lets two independent MVUtensils graphs coexist on one clip.
* Motion vectors are stored as frame properties: `<prefix>AnalysisVectors` (an int array where the
  low 32 bits hold *x* and the high 32 bits hold *y*) and `<prefix>AnalysisSAD` (the per-block SAD).
* On x86-64 the plugin is built for several instruction-set levels (baseline, AVX2 and znver4) and the
  matching build is loaded for the running CPU. Most of the SIMD gain comes from that choice — the
  `Degrain` kernels in particular are plain auto-vectorized C, so their width is whatever the loaded
  build was compiled for. A few hand-written kernels additionally pick a wider variant at runtime.

## Porting from MVTools

MVUtensils is API-compatible in spirit but not verbatim. The main differences:

| MVTools | MVUtensils |
| --- | --- |
| namespace `mv` | namespace `mvu` |
| `Super(clip, pel=2)` (blksize implicit) | `Super(clip, blksize=8, overlap=4, pel=2)` — **blksize and overlap are mandatory** |
| `hpad` / `vpad` | `pad=[h, v]` |
| `levels` | `onelevel=True/False` |
| `blksize`/`blksizev`, `overlap`/`overlapv` | `blksize=[h, v]`, `overlap=[h, v]` (a single value applies to both axes) |
| `Analyse(isb=False, delta=1)` (forward) | `Analyse(delta=-1)` — **negative delta = forward, positive = backward** |
| `dct=0` / `dct=5` | `satd=False` / `satd=True` |
| `search` modes 0–7 | modes 0–5 (old modes 0 and 1 dropped, everything shifted by −2). `pelsearch` is a radius, not a mode: keep its value |
| `rfilter` 2 / 4 | `rfilter` 1 / 2 (modes 1 and 3 dropped) |
| `lambda`, `global` | `mvlambda`, `globalmv` (avoid Python keywords) |
| `Degrain1(clip, super, mvbw, mvfw, ...)` | `Degrain(clip, super, [mvbw, mvfw], ...)` — vectors in a list |
| `thsad` + `thsadc` | `thsad=[luma, chroma]` |
| `limit` + `limitc` | `limit=[luma, chroma]` (float; non-finite or > max = no limit) |
| `Degrain` centre frame taken from `clip` | taken from `super`; if the render super is built from a different clip (MCDegrainSharp, SMDegrain `mfilter`, MLDegrain `soft`), also pass `centersuper=Super(clip, …, onelevel=True)` — see [the centre frame](#the-centre-frame) |
| `Flow(mode=...)`, `BlockFPS`, `Finest`, `search_coarse`, `divide`, `scbehavior`, `truemotion` | removed |
| `fields`, `tff` (`Analyse`, `Recalculate`, `Compensate`, `Flow`) | removed: process `SeparateFields` output as is; vectors between fields of opposite parity already contain the half-line offset |
| `FlowFPS(mask=1/2)` | `FlowFPS(extramask=False/True)` |
| `Mask(kind=0/1/2)` | `VectorLengthMask` / `SADMask` / `OcclusionMask` |
| `thscd2` (0–256 int, default 130) | `thscd2` (0–100 float percentage, default 51) |

A typical denoise, before and after:

```py
# MVTools
super  = core.mv.Super(clip, pel=2)
mvbw = core.mv.Analyse(super, isb=True,  delta=1)
mvfw = core.mv.Analyse(super, isb=False, delta=1)
out  = core.mv.Degrain1(clip, super, mvbw, mvfw)

# MVUtensils
super  = core.mvu.Super(clip, blksize=8, overlap=4, pel=2)  # blksize/overlap mandatory
mvbw = core.mvu.Analyse(super, delta=1)    # positive delta = backward
mvfw = core.mvu.Analyse(super, delta=-1)   # negative delta = forward (was isb=False)
out  = core.mvu.Degrain(clip, super, [mvbw, mvfw])  # vectors passed as a list
```

## Table of Contents

* [Quick start](#quick-start)
* [Common parameters](#common-parameters)
* [Super](#super)
* [Analyse](#analyse)
* [AnalyseMany](#analysemany)
* [Recalculate](#recalculate)
* [Degrain](#degrain)
* [Compensate](#compensate)
* [Flow](#flow)
* [FlowInter](#flowinter)
* [FlowFPS](#flowfps)
* [FlowBlur](#flowblur)
* [VectorLengthMask / SADMask / OcclusionMask](#vectorlengthmask--sadmask--occlusionmask)
* [SCDetection](#scdetection)
* [Depan functions](#depan-functions)
  * [DepanEstimate](#depanestimate)
  * [DepanAnalyse](#depananalyse)
  * [DepanStabilise](#depanstabilise)
  * [DepanCompensate](#depancompensate)

## Quick start

The motion vector estimators (`Analyse`, `AnalyseMany`, `Recalculate`) consume a *super* clip and emit a
*vector* clip; the consumers (`Degrain`, `Compensate`, `Flow*`, masks) take the super and the
vector clip(s). `AnalyseMany` is the easy way to produce the list of forward/backward vectors that
`Degrain`, `FlowInter`, `FlowFPS` and `FlowBlur` expect.

```py
# Temporal denoise (radius 3 = 6 vector clips)
super = core.mvu.Super(clip, blksize=8, overlap=4)
out = core.mvu.Degrain(clip, super, core.mvu.AnalyseMany(super, radius=3))

# Motion-interpolated frame doubling
super = core.mvu.Super(clip, blksize=8, overlap=4)
out = core.mvu.FlowFPS(clip, super, core.mvu.AnalyseMany(super), num=clip.fps_num*2, den=clip.fps_den)
```

Only `Analyse` uses the full hierarchical pyramid (`Super(..., onelevel=False)`, the default). For
`Recalculate` and every consumer filter a single level is enough, so build their super with
`onelevel=True` to save time and memory if it's not also used with `Analyse`.

## Common parameters

`prefix` is accepted by every function; `thscd1`/`thscd2` are accepted by the consumer filters
(`Compensate`, `Degrain`, `Flow`, `FlowInter`, `FlowFPS`, `FlowBlur`, the mask functions and
`SCDetection`). They behave identically everywhere and, to avoid repetition, are listed here once
and omitted from the per-function tables below.

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| thscd1 | int | (400) | Scene-change SAD threshold. A block whose SAD exceeds this is considered "changed". The value is defined for an 8×8 luma block at 8-bit and is scaled internally to the actual block size, chroma usage and the bit depth the vectors were analysed at. |
| thscd2 | float | 0–100 (51) | Percentage of blocks that must be "changed" (SAD above `thscd1`) for the whole frame to be treated as a scene change. 0 = any single changed block triggers it; 100 = never. On a scene change the consumer leaves the frame unprocessed (passes the source through). |
| prefix | str | ("MVUtensils") | Prefix of the frame properties read/written by this function. Must match between the producer and consumer of a vector/super clip. |

> **Porting:** `thscd2` is now a 0–100 float **percentage** of changed blocks instead of the old
> 0–256 integer count. The default went from `130` (≈51% of 256) to the equivalent `51`; multiply
> an old value by `100/256` to convert. An out-of-range value (such as the old `130`) is now rejected
> rather than silently disabling scene detection.

> **Validation:** numeric arguments are checked when the filter is created — values outside the range
> shown here, and non-finite ones (`nan`/`inf`), are rejected with an error instead of silently
> producing wrong output. `Degrain`'s `limit` is the deliberate exception: there `inf`/`nan` disables
> limiting.

## Super

Prepares a clip for motion estimation: it pads the frame, optionally generates sub-pixel
(`pel`) planes, and builds the hierarchical pyramid used by `Analyse`.

```py
core.mvu.Super(vnode clip, int[] blksize, int[] overlap[, int[] pad=[16, 16], int pel=2, int sharp=2, int rfilter=1, bint onelevel=False, vnode pelclip=None, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to prepare. |
| blksize | int[] | (required) | Block size `[h, v]` (a single value sets both). Used to pad the frame so the right/bottom edges are fully covered. Must match the block size you intend to use in `Analyse`. |
| overlap | int[] | (required) | Block overlap `[h, v]`, must be ≤ blksize/2. Used together with `blksize` for edge padding. |
| pad | int[] | ([16, 16]) | Border padding `[h, v]` in pixels. One value applies to both axes. It must be divisible by the chroma subsampling: even horizontally for 4:2:0 and 4:2:2, even vertically for 4:2:0 and 4:4:0. |
| pel | int | 1, 2, 4 (2) | Sub-pixel accuracy: 1 = full-pixel, 2 = half-pixel, 4 = quarter-pixel. Higher needs more memory and time. |
| sharp | int | 0–2 (2) | Sub-pixel interpolation for `pel` > 1: 0 = bilinear, 1 = bicubic, 2 = Wiener (sharpest). |
| rfilter | int | 0–2 (1) | Pyramid downscale filter: 0 = simple average, 1 = bilinear, 2 = cubic. |
| onelevel | bint | (False) | Generate every hierarchical level. Only `Analyse` uses more than one level, if the super clip is only passed to other functions set it to `True` to save memory and a small speedup. |
| pelclip | vnode | (None) | Optional externally-supplied sub-pixel clip instead of generating one internally. |

> **Porting:** `blksize`/`overlap` were optional in mvtools and are now **mandatory** because the
> super clip pads itself to make edge blocks valid. `hpad`/`vpad` became `pad=[h, v]`, and `levels`
> became the boolean `onelevel`.

## Analyse

Estimates a field of motion vectors for one temporal direction/distance.

```py
core.mvu.Analyse(vnode super[, int[] blksize=<from super>, int[] overlap=<from super>, int levels=0, int search=2, int searchparam=2, int pelsearch=<pel>, int mvlambda=1000, bint chroma=True, int delta=1, int lsad=400, int plevel=1, bint globalmv=True, int pnew=25, int pzero=<pnew>, int pglobal=0, int badsad=10000, int badrange=24, bint meander=True, int trymany=0, bint satd=False, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| super | vnode | (required) | Super clip from `Super` (built with `onelevel=False`). |
| blksize | int[] | (super's value) | Block size `[h, v]`. Smaller = more accurate but slower. |
| overlap | int[] | (super's value) | Block overlap `[h, v]`, ≤ blksize/2. More overlap = smoother field, slower. |
| levels | int | (0 = all) | Number of hierarchical levels to use. 0 uses all available: every level that still fits at least one block, as in mvtools. Asking for more levels than fit, or than the super clip has, is an error. |
| search | int | 0–5 (2) | Search algorithm used at the finest level: 0 = logarithmic/diamond, 1 = exhaustive, 2 = hexagon, 3 = uneven multi-hexagon (UMH), 4 = horizontal, 5 = vertical. The coarser levels always search exhaustively, except with horizontal/vertical, which search along their axis at every level. |
| searchparam | int | (2) | Search radius at every level except the finest, in pixels of that level. One pixel at level *n* is 2ⁿ source pixels, so a small radius still covers large motion; raising it helps little and costs a lot (16 roughly doubles `Analyse` time). Not used with `levels=1`. |
| pelsearch | int | (super's pel) | Radius of the `search` algorithm at the finest level, in sub-pixel units (the default searches ±1 pixel). For logarithmic/diamond it is the initial step size, for hexagon/UMH the search range. |
| mvlambda | int | (1000) | Weight of the motion-coherence penalty (see [below](#motion-coherence-tuning-mvlambda-lsad-plevel)), **given per 8×8 block**: the value is multiplied internally by `blksizeh·blksizev/64` so the same setting behaves consistently at any block size (the same normalisation `lsad` and `badsad` use), then scaled for bit depth, `pel` and level. Higher favours smooth, spatially consistent fields; `0` disables the penalty and takes the pure lowest-SAD match per block. |
| chroma | bint | (True) | Include chroma planes in the SAD/SATD metric. |
| delta | int | (1) | Temporal distance **and direction** of the reference frame. **Positive = backward (past), negative = forward (future).** |
| lsad | int | (400) | SAD "knee" that throttles `mvlambda` per block (see [below](#motion-coherence-tuning-mvlambda-lsad-plevel)): the penalty is relaxed on blocks that already match poorly and kept at full strength where the match is good. Given per 8×8 block (scaled by `blksizeh·blksizev/64`), like `mvlambda`. Has **no effect when `mvlambda=0`**. |
| plevel | int | 0–2 (1) | How `mvlambda` scales across pyramid levels: 0 = constant, 1 = linear with scale, 2 = quadratic. Higher = more smoothing at the coarse levels. |
| globalmv | bint | (True) | Estimate a global (pan) motion vector and try it as an extra predictor for every block. |
| pnew | int | (25) | Extra penalty (relative to 256) added to the SAD of a freshly searched vector before it may replace the predictor. Higher = stickier to the predicted vector. |
| pzero | int | (pnew) | As `pnew`, but for accepting the zero vector. |
| pglobal | int | (0) | Penalty for the global-motion predictor. |
| badsad | int | (10000) | SAD above which a block gets a second, wider search. Given per 8×8 block (scaled by `blksizeh·blksizev/64`). |
| badrange | int | (24) | Radius of that wider search. |
| meander | bint | (True) | Scan block rows alternately left-to-right / right-to-left for better predictor reuse. |
| trymany | int | 0–2 (0) | Try multiple motion-vector candidates per block instead of only the single best predictor (slower, occasionally better). `0` = off; `1` = on every pyramid level **except** the finest; `2` = on **all** levels including the finest. |
| satd | bint | (False) | Use SATD instead of plain SAD as the block metric. Equivalent to mvtools `dct=5`. |

### Motion-coherence tuning: mvlambda, lsad, plevel

For every candidate vector Analyse minimises a *combined* cost rather than the block-matching error
(SAD) alone:

```
cost(v) = mvlambda · |v − predictor|²  +  SAD(v)
```

where `predictor` is the vector predicted from already-processed neighbours (their median). The first
term is a smoothness penalty that pulls each block toward its neighbours; the second is the
photometric match. The three parameters shape that trade-off:

* **`mvlambda`** sets how hard coherence is enforced. High values produce smooth, physically plausible
  "true motion" fields — what `Flow*` interpolation wants — while `mvlambda=0` ignores coherence and
  returns the lowest-SAD vector for every block, giving the smallest residual but a noisier field
  (often preferable for `Degrain`). It is the master switch: the other two only matter when it is
  non-zero. `mvlambda` is given in units of an 8×8 block — it is multiplied internally by
  `blksizeh·blksizev/64` so a value tuned at one block size carries over to another (the same
  normalisation `lsad` and `badsad` use) — and is also rescaled for bit depth, `pel` and pyramid
  level so a given value behaves the same across formats.

* **`lsad`** makes the penalty *adaptive to local match quality*. Per block the effective `mvlambda`
  is multiplied by roughly `(lsad / (lsad + blockSAD/2))²`, where `blockSAD` is an estimate of how
  well the neighbourhood is matching. Where a block already matches well (`blockSAD` ≪ `lsad`) the
  full penalty applies and coherence is enforced; where it matches badly (`blockSAD` ≫ `lsad` — e.g.
  occlusions, newly revealed detail) the penalty rolls off quadratically so the search is free to
  find the genuinely best vector instead of over-trusting a bad predictor. `lsad` is the SAD level at
  which that roll-off sets in: raise it to keep enforcing coherence even on hard blocks, lower it to
  surrender sooner. Because it only ever scales `mvlambda`, **`lsad` does nothing when `mvlambda=0`.**

* **`plevel`** controls how `mvlambda` grows toward the coarse pyramid levels, where each vector
  covers more pixels and benefits from extra smoothing.

> **Porting:** `isb` is gone — direction is now the sign of `delta` (negative = forward).
> `dct` became the boolean `satd` (`dct=0`→`satd=False`, `dct=5`→`satd=True`). `search`
> modes lost the old 0 and 1, so subtract 2 from your old value (`pelsearch` is a radius and stays as it was). `lambda`→`mvlambda`,
> `global`→`globalmv`. `trymany` is now a 0–2 int instead of a bool — the old `False`/`True` map to
> `0`/`1`, and the new `2` also tries multiple candidates on the finest level. `fields`/`tff` were removed:
> process `SeparateFields` output as is, vectors between fields of opposite parity already contain the half-line offset.
> The `truemotion` preset was removed in favour of fixed defaults: `mvlambda=1000`
> (per 8×8 block, scaled by block area), `lsad=400`, `plevel=1`, `globalmv=True`, `pnew=25`. These are mostly the old
> `truemotion=True` values, except `lsad` (now 400, the old `truemotion=False` value — it was 1200
> under `truemotion=True`) and `pnew` (now 25, midway between the old 50 / 0). The `search_coarse` and
> `divide` arguments were removed.

## AnalyseMany

Convenience wrapper that produces the whole list of vector clips `Degrain`, `FlowInter`, `FlowFPS`
and `FlowBlur` expect, in the right order. Takes every `Analyse` argument plus `radius`.

```py
core.mvu.AnalyseMany(vnode super[, <all Analyse args>, int delta=1, int radius=1, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| radius | int | (1) | Number of forward/backward vector pairs to produce. |
| delta | int | (1) | Step size between successive vectors (always positive here). |

With `delta=1, radius=2` the result is `[Analyse(delta=1), Analyse(delta=-1), Analyse(delta=2), Analyse(delta=-2)]`.

```py
vectors = core.mvu.AnalyseMany(super, radius=3)   # 6 clips, ready for Degrain(..., radius 3)
```

## Recalculate

Re-estimates an existing vector field at (typically) a finer block size, refining the vectors you
already have instead of searching from scratch. Pair it with a halved `blksize`/`overlap`.

```py
core.mvu.Recalculate(vnode super, vnode[] vectors[, int thsad=200, bint smooth=True, int[] blksize=<from super>, int search=2, int searchparam=2, int mvlambda=1000, bint chroma=True, int pnew=25, int[] overlap=<from super>, bint meander=True, bint satd=False, str prefix="MVUtensils"])
```

`vectors` takes a single vector clip or a whole list, and the result is the list of refined clips in
the same order so an entire `AnalyseMany` set can be refined in one call:

```py
super = core.mvu.Super(clip, blksize=8, overlap=4)
vectors = core.mvu.AnalyseMany(super, radius=2)                       # [bw1, fw1, bw2, fw2]
vectors = core.mvu.Recalculate(super, vectors, blksize=4, overlap=2)  # refine all at half block size
out = core.mvu.Degrain(clip, super, vectors)
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| super | vnode | (required) | Super clip. Only one level is needed. Unlike the filters that merely consume vectors, its bit depth must equal the one the vectors were analysed at — see [below](#recalculate-and-bit-depth). Its `pel` may differ from the vectors': they are rescaled, and the result has the super's `pel`. |
| vectors | vnode[] | (required) | Vector clip(s) to refine — a single clip or a whole list (e.g. an `AnalyseMany` set). The recalculated clips are returned as a list in the same order, as long as `super`; as in `Analyse`, a frame whose reference lies outside `super` gets no vectors. |
| thsad | int | (200) | Blocks whose SAD is below this keep their vector; worse blocks are re-searched. |
| smooth | bint | (True) | Interpolate the new (finer) vector field from neighbours (True) or take the nearest old vector (False). `smooth=False` roughly matches the old `divide=1` behaviour, `smooth=True` ≈ `divide=2`. |
| blksize | int[] | (super's value) | Finer block size `[h, v]`. Usually half of the original. |
| overlap | int[] | (super's value) | Finer overlap `[h, v]`. |

`search` picks the algorithm as in [Analyse](#analyse), but since `Recalculate` works on a single level,
`searchparam` is directly that algorithm's radius, in sub-pixel units (the role `pelsearch` plays in
`Analyse`). Other parameters (`mvlambda`, `chroma`, `pnew`, `meander`, `satd`) behave exactly as in
[Analyse](#analyse).

`Recalculate` deliberately has **no `lsad`** (nor `plevel`, `globalmv`, `pzero`, `pglobal`,
`badsad`/`badrange` or `trymany`). It is not a from-scratch hierarchical search: it works on a single
level, takes each existing vector as the sole predictor, and only re-searches a block when that
predictor's SAD exceeds `thsad`. So `thsad` already does what `lsad` does in [Analyse](#analyse) —
decide where the current motion is too poor to keep — making a separate adaptive-`mvlambda` knee
redundant, and the multi-level / multi-predictor machinery those other penalties tune simply isn't
present here. `mvlambda` is still honoured: it biases each refined vector to stay near the one it
started from.

> **Porting:** `Recalculate` takes a **list** of vector clips (and returns a list), so a whole
> `AnalyseMany` set is refined in one call instead of recalculating each clip separately. It will also
> raise an error if the chosen `blksize`/`overlap` can't cover the whole frame (unlike `Super`, which
> pads). Halving `blksize`+`overlap` and reusing the existing super usually works, unusual splits may
> need a new super clip. `fields`/`tff` were removed; they never changed its vectors.

### Recalculate and bit depth

Every other filter that takes a vector clip only reads it, so the vectors may come from an analysis at
a different bit depth (see [Notes](#notes)). `Recalculate` is different because it *writes* vectors:
it keeps the blocks whose SAD is already below `thsad` and replaces the rest, and the replacements get
SADs measured against the super it was handed. Allowing a mismatch would therefore produce a single
vector clip whose blocks carry SADs on two different scales, silently breaking every `thsad`/`thscd1`
comparison downstream. So `Recalculate` requires the super and the vectors to share a bit depth.

If you want to analyse cheaply at 8-bit and refine, do both steps at 8-bit and apply the final vectors
to the deeper clip:

```py
clip8 = core.resize.Bilinear(clip, format=vs.YUV420P8)
super8 = core.mvu.Super(clip8, blksize=8, overlap=4, pel=2)
vectors = core.mvu.AnalyseMany(super8, radius=3)
vectors = core.mvu.Recalculate(super8, vectors, blksize=4, overlap=2)

super16 = core.mvu.Super(clip, blksize=8, overlap=4, pel=2)   # clip is 16-bit
out = core.mvu.Degrain(clip, super16, vectors)                # vectors reused across depths
```

## Degrain

Motion-compensated temporal denoiser. Averages each block with its motion-compensated counterparts
from `radius` previous and `radius` following frames, weighted by how well they match.

`Degrain` auto-selects the radius from the number of vector clips; `Degrain1`…`Degrain25` are explicit
variants that take the same arguments. The radius may be 1–25, i.e. 2–50 vector clips.

```py
core.mvu.Degrain(vnode clip, vnode super, vnode[] vectors[, int[] thsad=[400, 400], int[] thsad2=thsad, int[] planes=[0, 1, 2], float[] limit=[inf, inf], int thscd1=400, float thscd2=51, int[] weights=None, vnode centersuper=None, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to denoise. |
| super | vnode | (required) | Super clip. Supplies the reference frames, and also the centre frame unless `centersuper` is given. |
| vectors | vnode[] | (required) | Vector clips in `AnalyseMany` order: `[bw1, fw1, bw2, fw2, …]`. Their count selects the radius; an even count of 2–50 clips (radius 1–25). |
| thsad | int[] | ([400, 400]) | SAD `[luma, chroma]` at which a reference block's weight reaches zero. Higher = stronger denoising. Chroma defaults to the luma value. Applies to the **nearest** references (temporal distance 1); more distant references interpolate towards `thsad2`. |
| thsad2 | int[] | (= `thsad`) | SAD `[luma, chroma]` for the **furthest** references (temporal distance `radius`). Each reference at distance `d` in `1…radius` uses a raised-cosine interpolation between `thsad` (at `d=1`) and `thsad2` (at `d=radius`). Setting `thsad2` below `thsad` keeps good compensation for low-SAD blocks at large radii while cutting the blur the far frames would otherwise add. Defaults to `thsad` (a flat threshold, the classic behaviour); has no effect at radius 1, which has no distant reference. |
| planes | int[] | ([0, 1, 2]) | Which planes to process; unprocessed planes are copied. |
| limit | float[] | ([inf, inf]) | Maximum absolute change per pixel `[luma, chroma]`. Non-finite (`inf`/`nan`) or a value above the format maximum disables limiting. |
| weights | int[] | (None) | Optional per-frame bias applied on top of the SAD-derived weights, in temporal order `[bw_radius, …, bw_1, centre, fw_1, …, fw_radius]` — exactly `2·radius + 1` non-negative values. Each reference's (and the source's) weight is multiplied by its entry before the weights are normalised, so only the ratios matter — the upper limit (≈2,800,000 at radius 1, falling to ≈164,000 at radius 25) exists purely to keep the internal weight sum inside a 32-bit int and is far beyond any real use. Omitted (or all-equal) leaves the default SAD weighting unchanged. |
| centersuper | vnode | (None) | Optional super clip that supplies the centre frame, and the reference for `limit`, instead of `super`, which then only supplies the references. It must be created with the same `blksize`, `overlap`, `pel` and `pad` as `super` and have at least as many frames as `clip`; only its first level is used, so `onelevel=True` is enough. See [the centre frame](#the-centre-frame). |

> **Porting:** the per-direction `mvbw*/mvfw*` arguments are now the single `vectors` list, the
> `thsad`/`thsadc` pair became `thsad=[luma, chroma]`, the `thsad2`/`thsadc2` pair became
> `thsad2=[luma, chroma]`, and `limit`/`limitc` became the float `limit=[luma, chroma]` (defaulting
> to no limit instead of 255). As in MDegrainN, `thsad2` defaults to `thsad`. The centre frame comes
> from `super`, not `clip`: scripts that build `super` from a different clip need `centersuper` to match
> mvtools, see [the centre frame](#the-centre-frame).

```py
# MVTools:    core.mv.Degrain2(clip, super, mvbw1, mvfw1, mvbw2, mvfw2, thsad=400, thsadc=300)
# MVUtensils:
v = core.mvu.AnalyseMany(super, radius=2)
out = core.mvu.Degrain(clip, super, v, thsad=[400, 300])
```

### The centre frame

Degrain blends the current frame (the centre) with the motion compensated references. mvtools takes that
centre from `clip`, while mvutensils takes it from `super` so that the partial blocks at the right and
bottom edges can be denoised too (mvtools leaves them untouched). The two are the same whenever `super` is
built from the clip being denoised, which covers QTGMC, MCTemporalDenoise, TemporalDegrain and SMDegrain
with its default settings.

Scripts that deliberately build the render super from a *different* clip are affected: MCDegrainSharp
(a soft clip is denoised with the super of a sharpened copy, so good matches pull the result towards the
sharp references while the centre stays soft), SMDegrain with `mfilter` and MLDegrain with `soft` > 0. Here
the centre silently becomes the super's clip. Pass a super of `clip` as `centersuper` to get mvtools'
behaviour:

```py
vectors = core.mvu.AnalyseMany(core.mvu.Super(clip, blksize=8, overlap=4), radius=2)
references = core.mvu.Super(sharpened, blksize=8, overlap=4, onelevel=True)
centre = core.mvu.Super(clip, blksize=8, overlap=4, onelevel=True)
out = core.mvu.Degrain(clip, references, vectors, centersuper=centre)
```

Degrain is the only filter that needs this. Compensate fills poorly matched blocks from the super's
current frame in mvtools too, the Flow filters render from the super in both, and FlowInter and FlowFPS
already blend from `clip` at scene changes.

## Compensate

Builds a single motion-compensated frame: each block is copied from the reference frame at its
motion vector.

```py
core.mvu.Compensate(vnode clip, vnode super, vnode vectors[, int thsad=10000, float time=100.0, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to compensate. |
| super | vnode | (required) | Super clip. |
| vectors | vnode | (required) | A single vector clip (one direction). |
| thsad | int | (10000) | Blocks whose SAD exceeds this are taken from the source instead of the compensated reference. |
| time | float | 0–100 (100.0) | Temporal position of the compensation, as a percentage toward the reference frame. |

> **Porting:** the `scbehavior`, `fields` and `tff` arguments were removed; vectors between fields of opposite parity already contain the half-line offset,
> which `fields=True` added a second time.

## Flow

Pixel-accurate motion compensation: instead of copying whole blocks it warps the reference frame
using a per-pixel vector field interpolated from the block vectors.

```py
core.mvu.Flow(vnode clip, vnode super, vnode vectors[, float time=100.0, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to warp. |
| super | vnode | (required) | Super clip. |
| vectors | vnode | (required) | A single vector clip. |
| time | float | 0–100 (100.0) | How far toward the reference frame to warp, in percent. |

> **Porting:** the `mode` argument was removed (only the former `mode=0` remains), and so were `fields` and
> `tff`: vectors between fields of opposite parity already contain the half-line offset, which `fields=True` added a second time.

## FlowInter

Interpolates a new frame *between* two existing frames by warping both halves of the motion field
toward the requested time.

```py
core.mvu.FlowInter(vnode clip, vnode super, vnode[] vectors[, float time=50.0, float ml=100.0, bint blend=True, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | vnode | (required) | Super clip. |
| vectors | vnode[] | (required) | `[mvbw, mvfw]` (e.g. from `AnalyseMany`). |
| time | float | 0–100 (50.0) | Position of the interpolated frame between the two source frames (50 = midpoint). |
| ml | float | (100.0) | Mask scale: the motion length that maps to full occlusion masking. Lower = stronger masking. |
| blend | bint | (True) | Blend occluded regions instead of hard-selecting one direction. |

> **Porting:** the `mvbw`/`mvfw` arguments became `vectors=[mvbw, mvfw]`.

## FlowFPS

Motion-compensated frame-rate conversion, building each output frame with `FlowInter`-style warping.

```py
core.mvu.FlowFPS(vnode clip, vnode super, vnode[] vectors[, int num=25, int den=1, bint extramask=True, float ml=100.0, bint blend=True, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | vnode | (required) | Super clip. |
| vectors | vnode[] | (required) | `[mvbw, mvfw]`. |
| num | int | (25) | Output frame-rate numerator. Must be non-negative; if `num` or `den` is `0`, the output rate defaults to twice the input rate. |
| den | int | (1) | Output frame-rate denominator. Must be non-negative. |
| extramask | bint | (True) | Use a second occlusion-mask pass. `extramask=False` matches the old `mask=1`; `extramask=True` (default) matches `mask=2`. |
| ml | float | (100.0) | Occlusion mask scale (see `FlowInter`). |
| blend | bint | (True) | Blend occluded regions. |

> **Porting:** the `mask` integer became the boolean `extramask`, and `mvbw`/`mvfw` became
> `vectors=[mvbw, mvfw]`.

## FlowBlur

Creates motion blur by smearing each pixel along its motion vector.

```py
core.mvu.FlowBlur(vnode clip, vnode super, vnode[] vectors[, float blur=50.0, int prec=1, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | vnode | (required) | Super clip. |
| vectors | vnode[] | (required) | `[mvbw, mvfw]`. |
| blur | float | 0–200 (50.0) | Blur strength as a percentage of the motion-vector length. |
| prec | int | (1) | Sampling step precision along the motion path; lower = more samples = smoother and slower. |

> **Porting:** the `mvbw`/`mvfw` arguments became `vectors=[mvbw, mvfw]`.

## VectorLengthMask / SADMask / OcclusionMask

These three functions replace the old `Mask(kind=…)`. Each produces a **full-range grayscale** mask
derived from a vector clip; the output bit depth/format is taken from the vector clip (float masks
are clamped to 0–1).

* **VectorLengthMask** — brightness proportional to motion-vector magnitude (old `kind=0`).
* **SADMask** — brightness proportional to per-block SAD (old `kind=1`).
* **OcclusionMask** — brightness proportional to occlusion / divergence of the field (old `kind=2`).

```py
core.mvu.VectorLengthMask(vnode vectors[, float ml=100.0, float gamma=1.0, float time=100.0, float scval=0.0, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
core.mvu.SADMask(vnode vectors[, ...same...])
core.mvu.OcclusionMask(vnode vectors[, ...same...])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| vectors | vnode | (required) | Vector clip; its format determines the mask format. |
| ml | float | (100.0) | Scale: the motion length / SAD that maps to the maximum mask value. Lower = stronger mask. Must be > 0. |
| gamma | float | (1.0) | Gamma curve applied to the mask. Must be ≥ 0. |
| time | float | 0–100 (100.0) | Temporal position used when projecting the vector field. |
| scval | float | (0.0) | The exact value written for frames detected as scene changes. For float you may pass any value (e.g. `inf`/`nan`), which can break the 0–1 guarantee. |

> **Porting:** `Mask(kind=…)` is split into the three named functions, the pointless `clip`
> argument was removed, and `ysc` became the float `scval`. Output is always a single grayscale
> plane rather than the original's UV trickery.

## SCDetection

Marks scene-change frames (using the vector clip's SAD statistics) by setting the
`_SceneChangePrev`/`_SceneChangeNext` frame properties.

```py
core.mvu.SCDetection(vnode clip, vnode vectors[, int thscd1=400, float thscd2=51, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | vnode | (required) | Clip to tag. |
| vectors | vnode | (required) | Vector clip used for the decision. |

## Depan functions

The Depan filters deal with **global** (whole-frame) motion — camera pan, zoom and rotation —
rather than the per-block local motion the rest of MVUtensils estimates. They are the only filters
**without** float support, and are otherwise carried over from mvtools largely unchanged, with fixed
edge handling and minor speedups. Fine-grained parameter meanings also match the mvtools `MDepan*`
filters in the [mvtools2 documentation](https://github.com/pinterf/mvtools/blob/mvtools-pfmod/Documentation/mvtools2.html).

They work in two stages:

1. **Measure** the global motion of every frame, producing a small *data* clip that carries the
   transform (dx, dy, zoom, rotation) as frame properties. Use **DepanEstimate** to measure it
   directly from the image, or **DepanAnalyse** to fit it from an existing MVUtensils vector clip.
2. **Apply** that data clip: **DepanStabilise** smooths the camera trajectory to remove shake (while
   keeping deliberate motion), and **DepanCompensate** shifts/zooms frames by the (optionally scaled)
   global motion — e.g. to undo, re-apply or align a camera move.

```py
# Stabilise shaky footage, estimating motion using whole frame correlation
data = core.mvu.DepanEstimate(clip)
out  = core.mvu.DepanStabilise(clip, data)

# Or derive the global motion from block vectors instead
sup  = core.mvu.Super(clip, blksize=8, overlap=4)
data = core.mvu.DepanAnalyse(clip, core.mvu.Analyse(sup))
out  = core.mvu.DepanStabilise(clip, data)
```

### DepanEstimate

Measures each frame's global motion (pan, optional zoom) directly from the image content and stores
it as frame properties. Outputs a *data* clip for `DepanStabilise`/`DepanCompensate`.

```py
core.mvu.DepanEstimate(vnode clip[, float trust=4.0, int winx=0, int winy=0, int wleft=-1, int wtop=-1, int dxmax=-1, int dymax=-1, float zoommax=1.0, float stab=1.0, float pixaspect=1.0, bint info=False, bint show=False, bint fields=False, bint tff=False])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer, GRAY/YUV | | Clip to measure. |
| trust | float | (4.0) | Trust limit for accepting an estimate; higher is more permissive. |
| winx / winy | int | (0 = auto) | Size of the analysis window; 0 picks a sensible size. |
| wleft / wtop | int | (-1 = centre) | Top-left corner of the analysis window; -1 centres it. |
| dxmax / dymax | int | (-1 = auto) | Maximum expected horizontal / vertical motion in pixels. |
| zoommax | float | (1.0) | Maximum expected zoom factor; 1.0 = pan only, no zoom. |
| stab | float | (1.0) | Stability / smoothing factor of the estimate. |
| pixaspect | float | (1.0) | Pixel aspect ratio. |
| info | bint | (False) | Overlay the numeric estimate on the frame. |
| show | bint | (False) | Visualise the estimated motion. |
| fields / tff | bint | (False) | Field-based handling and top-field-first. |

### DepanAnalyse

Fits a single global transform (pan, optional zoom and rotation) to the per-block motion vectors of
an MVUtensils `vectors` clip, producing the same kind of *data* clip as `DepanEstimate`.

```py
core.mvu.DepanAnalyse(vnode clip, vnode vectors[, vnode mask=None, bint zoom=True, bint rot=True, float pixaspect=1.0, float error=15.0, bint info=False, float wrong=10.0, float zerow=0.05, int thscd1=400, float thscd2=51, bint fields=False, bint tff=False])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer, GRAY/YUV | | Source clip (passed through; carries the data). |
| vectors | vnode | (required) | MVUtensils vector clip whose block vectors are fitted. |
| mask | vnode | (None) | Optional mask selecting which regions of the field to trust. |
| zoom | bint | (True) | Also estimate global zoom. |
| rot | bint | (True) | Also estimate global rotation. |
| pixaspect | float | (1.0) | Pixel aspect ratio. |
| error | float | (15.0) | Error limit for accepting the global fit. |
| wrong | float | (10.0) | Weight/penalty for vectors that disagree with the global motion. |
| zerow | float | (0.05) | Weight given to the zero vector in the fit. |
| info | bint | (False) | Overlay the numeric estimate on the frame. |
| fields / tff | bint | (False) | Field-based handling and top-field-first. |

### DepanStabilise

Smooths the global-motion trajectory from a `data` clip (inertial high-pass filtering) and applies
the inverse transform to remove camera shake while preserving intentional motion. Outputs the
stabilised video.

```py
core.mvu.DepanStabilise(vnode clip, vnode data[, float cutoff=1.0, float damping=0.9, float initzoom=1.0, bint addzoom=False, int prev=0, int next=0, int mirror=0, int blur=0, float dxmax=60.0, float dymax=30.0, float zoommax=1.05, float rotmax=1.0, int subpixel=2, float pixaspect=1.0, int fitlast=0, float tzoom=3.0, bint info=False, int method=0, bint fields=False])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer, GRAY/YUV | | Clip to stabilise. |
| data | vnode | (required) | Global-motion data from `DepanEstimate`/`DepanAnalyse`. |
| cutoff | float | (1.0) | Cutoff frequency (Hz) of the stabilisation filter; lower = smoother, slower correction. |
| damping | float | (0.9) | Damping ratio of the filter. |
| initzoom | float | (1.0) | Constant zoom applied to help hide exposed borders. |
| addzoom | bint | (False) | Automatically add zoom to hide exposed borders. |
| prev / next | int | (0) | Number of previous / following frames used in the trajectory fit. |
| mirror | int | 0–15 (0) | Fill exposed borders by mirroring (bitmask of edges: 1=top, 2=bottom, 4=left, 8=right). |
| blur | int | (0) | Blur width applied to the mirrored border region. |
| dxmax / dymax | float | (60.0 / 30.0) | Maximum correction in pixels. |
| zoommax | float | (1.05) | Maximum zoom correction. |
| rotmax | float | (1.0) | Maximum rotation correction in degrees. |
| subpixel | int | 0–2 (2) | Subpixel interpolation: 0=none, 1=bilinear, 2=bicubic. |
| pixaspect | float | (1.0) | Pixel aspect ratio. |
| fitlast | int | (0) | Number of trailing frames specially fitted for the clip end. |
| tzoom | float | (3.0) | Time in seconds over which the adaptive zoom (`addzoom`) is smoothed. With `addzoom` it must be at least 4/fps, i.e. at least one frame. |
| info | bint | (False) | Overlay diagnostic info on the frame. |
| method | int | (0) | Stabilisation method variant. |
| fields | bint | (False) | Field-based handling. |

### DepanCompensate

Warps each frame by the global motion in a `data` clip (scaled by `offset`) — e.g. to compensate for
or reproduce a camera move, or to align frames. Outputs the warped video.

```py
core.mvu.DepanCompensate(vnode clip, vnode data[, float offset=0.0, int subpixel=2, float pixaspect=1.0, bint matchfields=True, int mirror=0, int blur=0, bint info=False, bint fields=False, bint tff=False])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer, GRAY/YUV | | Clip to warp. |
| data | vnode | (required) | Global-motion data from `DepanEstimate`/`DepanAnalyse`. |
| offset | float | (0.0) | Temporal offset/scale of the motion to apply; 0 = none, fractional values compensate by a fraction of the move. |
| subpixel | int | 0–2 (2) | Subpixel interpolation: 0=none, 1=bilinear, 2=bicubic. |
| pixaspect | float | (1.0) | Pixel aspect ratio. |
| matchfields | bint | (True) | Match fields for interlaced content. |
| mirror | int | 0–15 (0) | Fill exposed borders by mirroring (bitmask of edges: 1=top, 2=bottom, 4=left, 8=right). |
| blur | int | (0) | Blur width applied to the mirrored border region. |
| info | bint | (False) | Overlay diagnostic info on the frame. |
| fields / tff | bint | (False) | Field-based handling and top-field-first. |
