module EinxMathNetTypes

open System
open MathNet.Numerics.LinearAlgebra

type Tensor =
    {
        Data: float[]
        Shape: int64[]
        Strides: int64[]
    }

    member x.data = x.Data
    member x.shape = x.Shape
    member x.strides = x.Strides
    member x.rank = x.Shape.Length
    member x.length = x.Data.Length

let notSupported (message: string) : 'a =
    raise (NotSupportedException(message))

let elementCount (shape: int64[]) =
    shape
    |> Array.fold (fun acc v -> acc * int v) 1

let computeStrides (shape: int64[]) =
    let strides = Array.zeroCreate<int64> shape.Length
    let mutable stride = 1L

    for i = shape.Length - 1 downto 0 do
        strides[i] <- stride
        stride <- stride * shape[i]

    strides

let tensor (shape: int64[]) (data: float[]) =
    let expected = elementCount shape

    if data.Length <> expected then
        invalidOp (sprintf "Data length %i does not match shape element count %i." data.Length expected)

    {
        Data = data
        Shape = Array.copy shape
        Strides = computeStrides shape
    }

let zeros (shape: int64[]) =
    tensor shape (Array.zeroCreate<float> (elementCount shape))

let ones (shape: int64[]) =
    tensor shape (Array.create<float> (elementCount shape) 1.0)

let reshape (newShape: int64[]) (t: Tensor) =
    let expected = elementCount newShape

    if expected <> t.length then
        invalidOp (sprintf "Cannot reshape tensor of length %i to shape with %i elements." t.length expected)

    {
        t with
            Shape = Array.copy newShape
            Strides = computeStrides newShape
    }

let offset (strides: int64[]) (index: int64[]) =
    let mutable off = 0L

    for i = 0 to index.Length - 1 do
        off <- off + strides[i] * index[i]

    off

let nextIndexInPlace (shape: int64[]) (index: int64[]) =
    let mutable i = shape.Length - 1
    let mutable carry = true

    while i >= 0 && carry do
        let v = index[i] + 1L

        if v < shape[i] then
            index[i] <- v
            carry <- false
        else
            index[i] <- 0L
            i <- i - 1

    not carry

let permute (permutation: int64[]) (t: Tensor) =
    if permutation.Length <> t.rank then
        invalidOp "Permutation length must match tensor rank."

    let seen = System.Collections.Generic.HashSet<int64>()

    for p in permutation do
        if p < 0L || p >= int64 t.rank then
            invalidOp "Permutation index out of range."

        if not (seen.Add p) then
            invalidOp "Permutation contains duplicates."

    let newShape = permutation |> Array.map (fun p -> t.shape[int p])
    let result = zeros newShape

    let srcIndex = Array.zeroCreate<int64> t.rank
    let dstIndex = Array.zeroCreate<int64> t.rank

    let mutable hasMore = true

    while hasMore do
        for i = 0 to t.rank - 1 do
            dstIndex[i] <- srcIndex[int permutation[i]]

        let srcOffset = offset t.strides srcIndex |> int
        let dstOffset = offset result.strides dstIndex |> int
        result.data[dstOffset] <- t.data[srcOffset]

        hasMore <- nextIndexInPlace t.shape srcIndex

    result

let ofMatrix (m: Matrix<float>) =
    let rows = m.RowCount
    let cols = m.ColumnCount

    let data =
        Array.init (rows * cols) (fun idx ->
            let i = idx / cols
            let j = idx % cols
            m[i, j])

    tensor [| int64 rows; int64 cols |] data

let toMatrix (t: Tensor) =
    if t.rank <> 2 then
        invalidOp "Tensor is not rank-2."

    let rows = int t.shape[0]
    let cols = int t.shape[1]

    Matrix<float>.Build.Dense(rows, cols, fun i j -> t.data[i * cols + j])
