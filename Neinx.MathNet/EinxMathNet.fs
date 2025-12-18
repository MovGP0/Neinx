module EinxMathNet

open System
open Parser

open EinxMathNetPattern
open EinxMathNetTypes

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
