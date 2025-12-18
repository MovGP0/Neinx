# Neinx

.NET library for parsing Universal Tensor Operations in Einstein-Inspired Notation,
similar to [einops](https://einops.rocks/) and [einx](https://github.com/fferflo/einx/).

## Projects

Neinx is split into a small core parser plus optional backend implementations:

- `Neinx`: pattern parser (`Parser.parseEinx`) + AST types (no numeric backend dependencies).
- `Neinx.TorchSharp`: tensor ops for `torch.Tensor` (TorchSharp backend).
- `Neinx.MathNet`: CPU/reference tensor ops backed by `float[]` + helpers for MathNet matrices.

## Pattern language (quick reference)

Patterns are strings of the form `"<inputs> -> <outputs>"`, where inputs/outputs are comma-separated expressions.

Supported syntax (parser):

- Named axes: `a`, `batch`, `height`
- Numeric axes: `16` (fixed-size axis)
- Placeholder axis: `_` (size inferred from context, if possible)
- Grouping: `(h w)` (flatten or split axes)
- Ellipsis: `...` (unspecified axes passthrough)
- Brackets: `[a b]` (used by `reduce` to mark axes to reduce)
- Star axis: `*` (used by `pack`/`unpack` to pack “all middle axes”)
- Concat: `a+b` (parsed, but not all backends support it yet)

Backend notes:

- TorchSharp backend currently does **not** support concat (`+`) in transforms.
- Some edge cases (e.g. `*` inside parentheses) may be rejected by design.

## Public API

### Core parser (`Neinx`)

- `Parser.parseEinx : string -> Expression list * Expression list`

This is mainly used internally by the backends, but is public and useful for debugging patterns.

### TorchSharp backend (`Neinx.TorchSharp`)

All functions are in the `EinxTorch` module and operate on `torch.Tensor`:

- `rearrange : string -> torch.Tensor -> (string * int64) list -> torch.Tensor`
- `repeat    : string -> torch.Tensor -> (string * int64) list -> torch.Tensor`
- `reduce    : string -> torch.Tensor -> Reduction -> bool -> (string * int64) list -> torch.Tensor`
- `einsum    : string -> torch.Tensor list -> torch.Tensor`
- `pack      : string -> torch.Tensor -> torch.Tensor * int64[]`
- `unpack    : string -> torch.Tensor -> int64[] -> torch.Tensor`

`Reduction` is a DU with:

- `Reduction.Sum`
- `Reduction.Mean`

### MathNet backend (`Neinx.MathNet`)

All functions are in the `EinxMathNet` module and operate on `EinxMathNetTypes.Tensor`:

- `rearrange : string -> Tensor -> (string * int64) list -> Tensor`
- `repeat    : string -> Tensor -> (string * int64) list -> Tensor`
- `reduce    : string -> Tensor -> Reduction -> bool -> (string * int64) list -> Tensor`
- `einsum    : string -> Tensor list -> Tensor`
- `pack      : string -> Tensor -> Tensor * int64[]`
- `unpack    : string -> Tensor -> int64[] -> Tensor`

`EinxMathNetTypes` also provides basic tensor utilities like `tensor`, `zeros`, `ones`, `reshape`, `permute`, `ofMatrix`, `toMatrix`.

## Usage examples

### Axis sizes (`axisSizes` argument)

Some patterns need explicit axis sizes:

- Splitting a grouped input axis: `"(h w) c -> h w c"` needs at least one of `h` or `w` provided.
- Creating a new axis in `repeat`: `"a b -> a b c"` needs `c` provided.

In F#, `axisSizes` is just a list: `[ "h", 3L; "c", 4L ]`.

In C#, it’s an `FSharpList<Tuple<string,long>>`. A simple helper is:

```csharp
using System;
using System.Linq;
using Microsoft.FSharp.Collections;

static FSharpList<Tuple<string, long>> AxisSizes(params (string name, long size)[] xs) =>
    ListModule.OfSeq(xs.Select(x => Tuple.Create(x.name, x.size)));
```

### TorchSharp (F#)

```fsharp
open TorchSharp
open type TorchSharp.torch

open EinxTorch

let x =
    torch.arange(0, 24, dtype = torch.ScalarType.Float32)
        .reshape([| 2L; 3L; 4L |])

// Permute axes
let y = rearrange "a b c -> b c a" x []

// Flatten
let z = rearrange "a b c -> a (b c)" x []

// Split (infer w = 4 from (h*w)=12)
let s =
    rearrange "(h w) c -> h w c"
        (torch.arange(0, 24, dtype = torch.ScalarType.Float32).reshape([| 12L; 2L |]))
        [ "h", 3L ]

// Repeat (add/broadcast axis)
let r = repeat "a b -> a b c" (torch.ones([| 2L; 3L |])) [ "c", 4L ]

// Reduce (sum/mean over bracketed axes)
let sumAc = reduce "a [b] c -> a c" (torch.ones([| 2L; 3L; 4L |])) Reduction.Sum false []
let meanAc = reduce "a [b] c -> a c" (torch.ones([| 2L; 3L; 4L |])) Reduction.Mean false []

// Einsum (matmul example)
let a = torch.arange(1, 7, dtype = torch.ScalarType.Float32).reshape([| 2L; 3L |])
let b = torch.arange(1, 13, dtype = torch.ScalarType.Float32).reshape([| 3L; 4L |])
let mm = einsum "x y, y z -> x z" [ a; b ]

// Pack / unpack
let packed, packedShape = pack "a * b" (torch.ones([| 2L; 3L; 4L; 5L |]))
let unpacked = unpack "a * b" packed packedShape
```

### TorchSharp (C#)

```csharp
using System;
using System.Linq;
using Microsoft.FSharp.Collections;
using TorchSharp;
using static EinxTorch;
using static TorchSharp.torch;

static FSharpList<Tuple<string, long>> AxisSizes(params (string name, long size)[] xs) =>
    ListModule.OfSeq(xs.Select(x => Tuple.Create(x.name, x.size)));

var x = torch.arange(0, 24, dtype: ScalarType.Float32).reshape(2, 3, 4);

var y = rearrange("a b c -> b c a", x, AxisSizes());
var r = repeat("a b -> a b c", torch.ones(2, 3), AxisSizes(("c", 4L)));

var sumAc = reduce("a [b] c -> a c", torch.ones(2, 3, 4), Reduction.Sum, false, AxisSizes());

var a = torch.arange(1, 7, dtype: ScalarType.Float32).reshape(2, 3);
var b = torch.arange(1, 13, dtype: ScalarType.Float32).reshape(3, 4);
var mm = einsum("x y, y z -> x z", ListModule.OfSeq(new[] { a, b }));
```

### MathNet (F#)

```fsharp
open EinxMathNet
open EinxMathNetTypes

let x = tensor [| 2L; 3L; 4L |] (Array.init (2 * 3 * 4) (fun i -> float i))

let y = rearrange "a b c -> b c a" x []
let z = rearrange "a b c -> a (b c)" x []

let r = repeat "a b -> a b c" (ones [| 2L; 3L |]) [ "c", 4L ]
let sumAc = reduce "a [b] c -> a c" (ones [| 2L; 3L; 4L |]) Reduction.Sum false []

let packed, packedShape = pack "a * b" (ones [| 2L; 3L; 4L; 5L |])
let unpacked = unpack "a * b" packed packedShape
```

### MathNet (C#)

```csharp
using System;
using System.Linq;
using Microsoft.FSharp.Collections;
using static EinxMathNet;
using static EinxMathNetTypes;

static FSharpList<Tuple<string, long>> AxisSizes(params (string name, long size)[] xs) =>
    ListModule.OfSeq(xs.Select(x => Tuple.Create(x.name, x.size)));

var x = EinxMathNetTypes.tensor(
    new long[] { 2, 3, 4 },
    Enumerable.Range(0, 2 * 3 * 4).Select(i => (double)i).ToArray());

var y = rearrange("a b c -> b c a", x, AxisSizes());
var r = repeat("a b -> a b c", ones(new long[] { 2, 3 }), AxisSizes(("c", 4L)));
var sumAc = reduce("a [b] c -> a c", ones(new long[] { 2, 3, 4 }), Reduction.Sum, false, AxisSizes());
```
