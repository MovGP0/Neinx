module EinxMathNet

open System
open MathNet.Numerics.LinearAlgebra
open Parser

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

type AxisId = string

type OutputItem =
    | OutputAxis of AxisId
    | OutputGroup of AxisId list
    | OutputConst of int64

type AxisSizes = Map<string, int64>

let toAxisSizes (axisSizes: (string * int64) list) : AxisSizes =
    axisSizes
    |> List.fold (fun acc (name, value) -> acc.Add(name, value)) Map.empty

let getAtoms (expr: Expression) : Expression list =
    match expr with
    | Composition parts -> parts
    | _ -> [ expr ]

let unwrapBracket (expr: Expression) : Expression =
    match expr with
    | Bracket inner -> inner
    | _ -> expr

let flattenConcatForbidden (expr: Expression) =
    match expr with
    | Concat _ -> notSupported "Concat ('+') is not supported by MathNet backend yet."
    | _ -> ()

let tryGetAxisSize (axisSizes: AxisSizes) (axis: Axis) : int64 option =
    match axis with
    | AxisName name -> axisSizes |> Map.tryFind name
    | AxisNumber n -> Some (int64 n)
    | AxisPlaceholder -> None
    | AxisStar -> None

let inferGroupSizes (axisSizes: AxisSizes) (group: Expression) (groupDimSize: int64) (placeholderStart: int) =
    let innerAtoms = getAtoms group |> List.map unwrapBracket

    if innerAtoms |> List.exists (function | Ellipsis | Empty | Concat _ -> true | _ -> false) then
        notSupported "Ellipsis/Empty/Concat inside parentheses is not supported yet."

    let axisSpecs =
        innerAtoms
        |> List.map (fun atom ->
            match atom with
            | Axis axis -> axis
            | _ -> notSupported "Only axes are supported inside parentheses for now.")

    let knownSizes = axisSpecs |> List.map (tryGetAxisSize axisSizes)
    let unknownCount = knownSizes |> List.filter Option.isNone |> List.length

    if unknownCount > 1 then
        notSupported "Parenthesis groups may have at most one unknown axis size."

    let productKnown =
        knownSizes
        |> List.choose id
        |> List.fold (fun acc x -> acc * x) 1L

    if productKnown = 0L then
        invalidOp "Invalid axis size 0."

    if groupDimSize % productKnown <> 0L then
        invalidOp (sprintf "Cannot infer group sizes: dimension %d is not divisible by known product %d." groupDimSize productKnown)

    let inferredSize =
        if unknownCount = 1 then
            groupDimSize / productKnown
        else
            if productKnown <> groupDimSize then
                invalidOp (sprintf "Group sizes mismatch: expected product %d but dimension is %d." productKnown groupDimSize)

            0L

    let mutable placeholderIndex = placeholderStart

    let ids, sizes =
        (axisSpecs, knownSizes)
        ||> List.map2 (fun axis known ->
            match axis with
            | AxisName name ->
                let size = known |> Option.defaultValue inferredSize
                name, size
            | AxisNumber n ->
                sprintf "__n%i" n, int64 n
            | AxisPlaceholder ->
                let id = sprintf "__p%i" placeholderIndex
                placeholderIndex <- placeholderIndex + 1
                id, inferredSize
            | AxisStar ->
                notSupported "Star inside parentheses is not supported.")
        |> List.unzip

    ids, sizes, placeholderIndex

let countConsumesOneDim (atom: Expression) =
    match atom with
    | Axis _ -> true
    | Bracket _ -> true
    | Composition _ -> true
    | Empty -> false
    | Ellipsis -> false
    | Concat _ -> true

let matchInput (axisSizes: AxisSizes) (inputExpr: Expression) (x: Tensor) : Tensor * AxisId list * int64 list * AxisId list =
    flattenConcatForbidden inputExpr

    let atoms = getAtoms inputExpr

    let shape = x.shape |> Array.toList
    let rank = shape.Length

    let fixedCount =
        atoms
        |> List.filter (fun a -> a <> Ellipsis)
        |> List.sumBy (fun a -> if countConsumesOneDim a then 1 else 0)

    let ellipsisCount = atoms |> List.filter ((=) Ellipsis) |> List.length

    if ellipsisCount > 1 then
        notSupported "Only a single ellipsis ('...') is supported."

    if fixedCount > rank then
        invalidOp (sprintf "Pattern requires at least %i dims but tensor rank is %i." fixedCount rank)

    let ellipsisDims = rank - fixedCount

    let mutable dimIndex = 0
    let mutable placeholderIndex = 0
    let mutable ellipsisAxisIds = []
    let mutable expandedAxisIds: AxisId list = []
    let mutable expandedSizes: int64 list = []

    for atom in atoms do
        match atom with
        | Empty ->
            expandedAxisIds <- expandedAxisIds @ [ "__empty" ]
            expandedSizes <- expandedSizes @ [ 1L ]

        | Ellipsis ->
            let ids = [ 0 .. ellipsisDims - 1 ] |> List.map (fun i -> sprintf "__e%i" i)
            let sizes = shape |> List.skip dimIndex |> List.take ellipsisDims
            dimIndex <- dimIndex + ellipsisDims
            ellipsisAxisIds <- ids
            expandedAxisIds <- expandedAxisIds @ ids
            expandedSizes <- expandedSizes @ sizes

        | Bracket _ ->
            notSupported "Brackets are not supported in input for rearrange/repeat; use reduce."

        | Axis axis ->
            let dimSize = shape[dimIndex]
            dimIndex <- dimIndex + 1

            match axis with
            | AxisName name ->
                expandedAxisIds <- expandedAxisIds @ [ name ]
                expandedSizes <- expandedSizes @ [ dimSize ]

            | AxisNumber n ->
                let expected = int64 n

                if dimSize <> expected then
                    invalidOp (sprintf "Axis size mismatch: expected %d but got %d." expected dimSize)

                expandedAxisIds <- expandedAxisIds @ [ sprintf "__n%i" n ]
                expandedSizes <- expandedSizes @ [ dimSize ]

            | AxisPlaceholder ->
                let id = sprintf "__p%i" placeholderIndex
                placeholderIndex <- placeholderIndex + 1
                expandedAxisIds <- expandedAxisIds @ [ id ]
                expandedSizes <- expandedSizes @ [ dimSize ]

            | AxisStar ->
                notSupported "Star axis is not supported in rearrange/repeat input; use pack/unpack."

        | Composition _ ->
            let groupDimSize = shape[dimIndex]
            dimIndex <- dimIndex + 1

            let groupIds, groupSizes, nextPlaceholderIndex = inferGroupSizes axisSizes atom groupDimSize placeholderIndex
            placeholderIndex <- nextPlaceholderIndex

            expandedAxisIds <- expandedAxisIds @ groupIds
            expandedSizes <- expandedSizes @ groupSizes

        | Concat _ ->
            notSupported "Concat is not supported in input."

    if dimIndex <> rank then
        invalidOp (sprintf "Pattern did not consume all dimensions (consumed %i, rank %i)." dimIndex rank)

    let xExpanded = reshape (expandedSizes |> List.toArray) x
    xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds

let buildOutputItems (outputExpr: Expression) (ellipsisAxisIds: AxisId list) : OutputItem list =
    flattenConcatForbidden outputExpr

    let atoms = getAtoms outputExpr

    let mutable outputItems: OutputItem list = []
    let mutable placeholderIndex = 0

    let rec addAtom (atom: Expression) =
        match atom with
        | Empty ->
            outputItems <- outputItems @ [ OutputConst 1L ]

        | Ellipsis ->
            outputItems <- outputItems @ (ellipsisAxisIds |> List.map OutputAxis)

        | Bracket inner ->
            addAtom inner

        | Axis axis ->
            match axis with
            | AxisName name ->
                outputItems <- outputItems @ [ OutputAxis name ]

            | AxisNumber n ->
                outputItems <- outputItems @ [ OutputConst (int64 n) ]

            | AxisPlaceholder ->
                let id = sprintf "__p%i" placeholderIndex
                placeholderIndex <- placeholderIndex + 1
                outputItems <- outputItems @ [ OutputAxis id ]

            | AxisStar ->
                outputItems <- outputItems @ [ OutputAxis "__star" ]

        | Composition parts ->
            let groupAxisIds =
                parts
                |> List.collect (fun p ->
                    match p with
                    | Axis axis ->
                        match axis with
                        | AxisName name -> [ name ]
                        | AxisPlaceholder ->
                            let id = sprintf "__p%i" placeholderIndex
                            placeholderIndex <- placeholderIndex + 1
                            [ id ]
                        | AxisNumber _ -> []
                        | AxisStar -> [ "__star" ]
                    | Ellipsis -> ellipsisAxisIds
                    | Bracket inner ->
                        getAtoms inner
                        |> List.choose (function | Axis (AxisName n) -> Some n | _ -> None)
                    | _ -> notSupported "Unsupported expression inside output group.")

            let groupConsts =
                parts
                |> List.choose (function | Axis (AxisNumber n) -> Some (int64 n) | _ -> None)

            if groupConsts.Length > 0 then
                notSupported "Numeric constants inside output groups are not supported yet."

            outputItems <- outputItems @ [ OutputGroup groupAxisIds ]

        | Concat _ ->
            notSupported "Concat is not supported in output."

    atoms |> List.iter addAtom

    if
        outputItems
        |> List.exists (function
            | OutputAxis "__star" -> true
            | OutputGroup ids -> ids |> List.contains "__star"
            | _ -> false)
    then
        notSupported "Star axis is reserved for pack/unpack and is not supported in rearrange/repeat output."

    outputItems

let permuteAndReshape (xExpanded: Tensor) (expandedAxisIds: AxisId list) (expandedSizes: int64 list) (outputItems: OutputItem list) =
    let axisToIndex =
        expandedAxisIds
        |> List.mapi (fun i id -> id, i)
        |> Map.ofList

    let axisToSize =
        (expandedAxisIds, expandedSizes)
        ||> List.zip
        |> Map.ofList

    let permuteAxisIds =
        outputItems
        |> List.collect (function
            | OutputAxis id -> [ id ]
            | OutputGroup ids -> ids
            | OutputConst _ -> [])

    if permuteAxisIds.Length <> expandedAxisIds.Length then
        invalidOp "Output axes must be a permutation of input axes."

    let duplicates =
        permuteAxisIds
        |> List.groupBy id
        |> List.choose (fun (id, xs) -> if xs.Length > 1 then Some id else None)

    if duplicates.Length > 0 then
        invalidOp (sprintf "Duplicate axes in output are not supported for rearrange/repeat: %s" (String.Join(", ", duplicates)))

    let permuteIndices =
        permuteAxisIds
        |> List.map (fun id ->
            axisToIndex
            |> Map.tryFind id
            |> Option.defaultWith (fun () -> invalidOp (sprintf "Axis '%s' not found in input." id)))
        |> List.map int64
        |> List.toArray

    let xPermuted = permute permuteIndices xExpanded

    let finalShape =
        outputItems
        |> List.map (function
            | OutputConst c -> c
            | OutputAxis id ->
                axisToSize
                |> Map.tryFind id
                |> Option.defaultWith (fun () -> invalidOp (sprintf "Axis '%s' missing size." id))
            | OutputGroup ids ->
                ids
                |> List.map (fun id ->
                    axisToSize
                    |> Map.tryFind id
                    |> Option.defaultWith (fun () -> invalidOp (sprintf "Axis '%s' missing size." id)))
                |> List.fold (fun acc x -> acc * x) 1L)
        |> List.toArray

    reshape finalShape xPermuted

type Reduction =
    | Sum
    | Mean

let rearrange (pattern: string) (x: Tensor) (axisSizes: (string * int64) list) : Tensor =
    let axisSizesMap = toAxisSizes axisSizes
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 1 || outputs.Length <> 1 then
        invalidOp "rearrange expects exactly one input and one output expression."

    let xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds = matchInput axisSizesMap inputs[0] x
    let outputItems = buildOutputItems outputs[0] ellipsisAxisIds

    if outputItems |> List.exists (function | OutputConst c when c <> 1L -> true | _ -> false) then
        invalidOp "rearrange can only insert constant axes of size 1."

    permuteAndReshape xExpanded expandedAxisIds expandedSizes outputItems

let private broadcastRepeat (input: Tensor) (targetShape: int64[]) (axisMap: int option[]) =
    if axisMap.Length <> targetShape.Length then
        invalidOp "Invalid axis map."

    let result = zeros targetShape

    let outIndex = Array.zeroCreate<int64> targetShape.Length
    let inIndex = Array.zeroCreate<int64> input.rank

    let mutable hasMore = true

    while hasMore do
        for i = 0 to axisMap.Length - 1 do
            match axisMap[i] with
            | Some j -> inIndex[j] <- outIndex[i]
            | None -> ()

        let srcOffset = offset input.strides inIndex |> int
        let dstOffset = offset result.strides outIndex |> int
        result.data[dstOffset] <- input.data[srcOffset]

        hasMore <- nextIndexInPlace targetShape outIndex

    result

let repeat (pattern: string) (x: Tensor) (axisSizes: (string * int64) list) : Tensor =
    let axisSizesMap = toAxisSizes axisSizes
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 1 || outputs.Length <> 1 then
        invalidOp "repeat expects exactly one input and one output expression."

    let xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds = matchInput axisSizesMap inputs[0] x
    let outputItems = buildOutputItems outputs[0] ellipsisAxisIds

    let existingAxisIds = expandedAxisIds |> Set.ofList

    let axisToIndex =
        expandedAxisIds
        |> List.mapi (fun i id -> id, i)
        |> Map.ofList

    let flattenedOutputAxisIds =
        outputItems
        |> List.collect (function
            | OutputAxis id -> [ id ]
            | OutputGroup ids -> ids
            | OutputConst _ -> [])

    let targetFlatSizes =
        flattenedOutputAxisIds
        |> List.map (fun id ->
            if existingAxisIds.Contains id then
                let idx = axisToIndex[id]
                expandedSizes[idx]
            else
                axisSizesMap
                |> Map.tryFind id
                |> Option.defaultWith (fun () -> invalidOp (sprintf "repeat requires size for new axis '%s'." id)))
        |> List.toArray

    let axisMap =
        flattenedOutputAxisIds
        |> List.map (fun id -> if existingAxisIds.Contains id then Some axisToIndex[id] else None)
        |> List.toArray

    let repeatedFlat = broadcastRepeat xExpanded targetFlatSizes axisMap

    let finalShape =
        outputItems
        |> List.map (function
            | OutputConst c -> c
            | OutputAxis id ->
                if existingAxisIds.Contains id then
                    let idx = axisToIndex[id]
                    expandedSizes[idx]
                else
                    axisSizesMap
                    |> Map.tryFind id
                    |> Option.defaultWith (fun () -> invalidOp (sprintf "repeat requires size for new axis '%s'." id))
            | OutputGroup ids ->
                ids
                |> List.map (fun id ->
                    if existingAxisIds.Contains id then
                        let idx = axisToIndex[id]
                        expandedSizes[idx]
                    else
                        axisSizesMap
                        |> Map.tryFind id
                        |> Option.defaultWith (fun () -> invalidOp (sprintf "repeat requires size for new axis '%s'." id)))
                |> List.fold (fun acc v -> acc * v) 1L)
        |> List.toArray

    reshape finalShape repeatedFlat

let private reduceOverDims (dims: int[]) (keepDims: bool) (t: Tensor) (op: float -> float -> float) (finalize: float -> float) =
    let dimsSet = dims |> Set.ofArray

    let outShape =
        if keepDims then
            t.shape
            |> Array.mapi (fun i s -> if dimsSet.Contains i then 1L else s)
        else
            t.shape
            |> Array.mapi (fun i s -> i, s)
            |> Array.choose (fun (i, s) -> if dimsSet.Contains i then None else Some s)

    let result = zeros outShape

    let outIndex = Array.zeroCreate<int64> outShape.Length
    let inIndex = Array.zeroCreate<int64> t.rank

    let outToIn =
        if keepDims then
            [| 0 .. t.rank - 1 |]
        else
            [| for i in 0 .. t.rank - 1 do if not (dimsSet.Contains i) then yield i |]

    let reduceAxes = dims |> Array.sort
    let reduceShape = reduceAxes |> Array.map (fun d -> t.shape[d])
    let reduceIndex = Array.zeroCreate<int64> reduceAxes.Length

    let mutable outHasMore = true

    while outHasMore do
        Array.Clear(inIndex, 0, inIndex.Length)

        for outAxis = 0 to outToIn.Length - 1 do
            let inAxis = outToIn[outAxis]
            inIndex[inAxis] <- outIndex[outAxis]

        let mutable acc = 0.0
        let mutable first = true

        Array.Clear(reduceIndex, 0, reduceIndex.Length)
        let mutable redHasMore = true

        while redHasMore do
            for ri = 0 to reduceAxes.Length - 1 do
                inIndex[reduceAxes[ri]] <- reduceIndex[ri]

            let srcOffset = offset t.strides inIndex |> int

            if first then
                acc <- t.data[srcOffset]
                first <- false
            else
                acc <- op acc t.data[srcOffset]

            redHasMore <- nextIndexInPlace reduceShape reduceIndex

        let dstOffset = offset result.strides outIndex |> int
        result.data[dstOffset] <- finalize acc

        outHasMore <- nextIndexInPlace outShape outIndex

    result

let reduce (pattern: string) (x: Tensor) (reduction: Reduction) (keepDims: bool) (axisSizes: (string * int64) list) : Tensor =
    let axisSizesMap = toAxisSizes axisSizes
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 1 || outputs.Length <> 1 then
        invalidOp "reduce expects exactly one input and one output expression."

    let inputExpr = inputs[0]
    let outputExpr = outputs[0]

    let atoms = getAtoms inputExpr

    let reduceAxisNames =
        atoms
        |> List.collect (fun atom ->
            match atom with
            | Bracket inner ->
                let innerAtoms = getAtoms inner |> List.map unwrapBracket

                innerAtoms
                |> List.choose (function
                    | Axis (AxisName name) -> Some name
                    | _ -> None)
            | _ -> [])
        |> Set.ofList

    if reduceAxisNames.IsEmpty then
        invalidOp "reduce requires bracketed axes in input expression (e.g., 'b [c] h -> b h')."

    let strippedInputExpr =
        let strippedAtoms =
            atoms
            |> List.collect (fun atom ->
                match atom with
                | Bracket inner -> getAtoms inner
                | _ -> [ atom ])

        Composition strippedAtoms

    let xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds = matchInput axisSizesMap strippedInputExpr x

    let reduceDims =
        expandedAxisIds
        |> List.mapi (fun i id -> i, id)
        |> List.choose (fun (i, id) -> if reduceAxisNames.Contains id then Some i else None)
        |> List.toArray

    let reduced =
        match reduction with
        | Reduction.Sum ->
            reduceOverDims reduceDims keepDims xExpanded (fun a b -> a + b) id
        | Reduction.Mean ->
            let count =
                reduceDims
                |> Array.map (fun d -> xExpanded.shape[d])
                |> Array.fold (fun acc v -> acc * v) 1L

            reduceOverDims reduceDims keepDims xExpanded (fun a b -> a + b) (fun s -> s / float count)

    let remainingAxisIds, remainingSizes =
        if keepDims then
            let updatedSizes =
                expandedAxisIds
                |> List.map (fun id -> if reduceAxisNames.Contains id then 1L else (expandedSizes[expandedAxisIds |> List.findIndex ((=) id)]))

            expandedAxisIds, updatedSizes
        else
            (expandedAxisIds, expandedSizes)
            ||> List.zip
            |> List.filter (fun (id, _) -> not (reduceAxisNames.Contains id))
            |> List.unzip

    let outputItems = buildOutputItems outputExpr ellipsisAxisIds

    let outputAxisIds =
        outputItems
        |> List.collect (function
            | OutputAxis id -> [ id ]
            | OutputGroup ids -> ids
            | OutputConst _ -> [])

    if outputAxisIds |> List.exists reduceAxisNames.Contains then
        invalidOp "Output expression references reduced axes."

    permuteAndReshape reduced remainingAxisIds remainingSizes outputItems

let einsum (pattern: string) (tensors: Tensor list) : Tensor =
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 2 || outputs.Length <> 1 then
        notSupported "MathNet einsum currently supports only two inputs and one output."

    let a = tensors[0]
    let b = tensors[1]

    if a.rank <> 2 || b.rank <> 2 then
        notSupported "MathNet einsum currently supports only rank-2 tensors."

    let am = toMatrix a
    let bm = toMatrix b
    let cm = am * bm
    ofMatrix cm

let pack (pattern: string) (x: Tensor) : Tensor * int64[] =
    let inputs, outputs = parseEinx pattern

    if outputs.Length <> 0 || inputs.Length <> 1 then
        invalidOp "pack expects a single expression without '->'."

    let expr = inputs[0]
    let atoms = getAtoms expr |> List.map unwrapBracket

    let starCount = atoms |> List.sumBy (function | Axis AxisStar -> 1 | _ -> 0)

    if starCount <> 1 then
        invalidOp "pack expects exactly one '*' in the pattern."

    if atoms |> List.exists (function | Ellipsis -> true | _ -> false) then
        notSupported "pack does not support ellipsis yet."

    let before = atoms |> List.takeWhile (function | Axis AxisStar -> false | _ -> true) |> List.length
    let after = atoms |> List.rev |> List.takeWhile (function | Axis AxisStar -> false | _ -> true) |> List.length

    let rank = x.rank

    if before + after + 1 > rank then
        invalidOp (sprintf "pack pattern requires at least %i dims but tensor rank is %i." (before + after + 1) rank)

    let starDims = rank - before - after
    let packedShape = x.shape |> Array.skip before |> Array.take starDims
    let packedSize = packedShape |> Array.fold (fun acc v -> acc * v) 1L

    let newShape =
        Array.concat
            [ x.shape |> Array.take before
              [| packedSize |]
              x.shape |> Array.skip (before + starDims) ]

    reshape newShape x, packedShape

let unpack (pattern: string) (xPacked: Tensor) (packedShape: int64[]) : Tensor =
    let inputs, outputs = parseEinx pattern

    if outputs.Length <> 0 || inputs.Length <> 1 then
        invalidOp "unpack expects a single expression without '->'."

    let expr = inputs[0]
    let atoms = getAtoms expr |> List.map unwrapBracket

    let starCount = atoms |> List.sumBy (function | Axis AxisStar -> 1 | _ -> 0)

    if starCount <> 1 then
        invalidOp "unpack expects exactly one '*' in the pattern."

    if atoms |> List.exists (function | Ellipsis -> true | _ -> false) then
        notSupported "unpack does not support ellipsis yet."

    let before = atoms |> List.takeWhile (function | Axis AxisStar -> false | _ -> true) |> List.length
    let after = atoms |> List.rev |> List.takeWhile (function | Axis AxisStar -> false | _ -> true) |> List.length

    let rank = xPacked.rank

    if before + after + 1 <> rank then
        invalidOp "unpack expects packed tensor rank to match the pattern (with '*' as one axis)."

    let newShape =
        Array.concat
            [ xPacked.shape |> Array.take before
              packedShape
              xPacked.shape |> Array.skip (before + 1) ]

    reshape newShape xPacked
