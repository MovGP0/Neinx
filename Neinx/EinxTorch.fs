module EinxTorch

open System
open TorchSharp

open Parser

type private AxisId = string

type private OutputItem =
    | OutputAxis of AxisId
    | OutputGroup of AxisId list
    | OutputConst of int64

type private AxisSizes = Map<string, int64>

let private toAxisSizes (axisSizes: (string * int64) list) : AxisSizes =
    axisSizes
    |> List.fold (fun acc (name, value) -> acc.Add(name, value)) Map.empty

let private getAtoms (expr: Expression) : Expression list =
    match expr with
    | Composition parts -> parts
    | _ -> [ expr ]

let private unwrapBracket (expr: Expression) : Expression =
    match expr with
    | Bracket inner -> inner
    | _ -> expr

let private notSupported (message: string) : 'a =
    raise (NotSupportedException(message))

let private flattenConcatForbidden (expr: Expression) =
    match expr with
    | Concat _ -> notSupported "Concat ('+') is not supported by TorchSharp transforms yet."
    | _ -> ()

let private axisIdFromAxis (axis: Axis) (placeholderIndex: int) : AxisId * int =
    match axis with
    | AxisName name -> name, placeholderIndex
    | AxisNumber _ -> notSupported "Numeric axes should be handled separately."
    | AxisPlaceholder -> sprintf "__p%i" placeholderIndex, placeholderIndex + 1
    | AxisStar -> notSupported "Star axis should be handled separately."

let private tryGetAxisSize (axisSizes: AxisSizes) (axis: Axis) : int64 option =
    match axis with
    | AxisName name -> axisSizes |> Map.tryFind name
    | AxisNumber n -> Some (int64 n)
    | AxisPlaceholder -> None
    | AxisStar -> None

let private inferGroupSizes (axisSizes: AxisSizes) (group: Expression) (groupDimSize: int64) (placeholderStart: int) =
    let innerAtoms = getAtoms group |> List.map unwrapBracket

    if innerAtoms |> List.exists (function | Ellipsis | Empty | Concat _ -> true | _ -> false) then
        notSupported "Ellipsis/Empty/Concat inside parentheses is not supported yet."

    let axisSpecs =
        innerAtoms
        |> List.map (fun atom ->
            match atom with
            | Axis axis -> axis
            | _ -> notSupported "Only axes are supported inside parentheses for now.")

    let knownSizes =
        axisSpecs
        |> List.map (tryGetAxisSize axisSizes)

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

let private countConsumesOneDim (atom: Expression) =
    match atom with
    | Axis _ -> true
    | Bracket _ -> true
    | Composition _ -> true
    | Empty -> false
    | Ellipsis -> false
    | Concat _ -> true

let private matchInput
    (axisSizes: AxisSizes)
    (inputExpr: Expression)
    (x: torch.Tensor)
    : torch.Tensor * AxisId list * int64 list * AxisId list =

    flattenConcatForbidden inputExpr

    let atoms = getAtoms inputExpr

    let shape = x.shape |> Array.map int64 |> Array.toList
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
            let ids =
                [ 0 .. ellipsisDims - 1 ]
                |> List.map (fun i -> sprintf "__e%i" i)
            let sizes =
                shape
                |> List.skip dimIndex
                |> List.take ellipsisDims
            dimIndex <- dimIndex + ellipsisDims
            ellipsisAxisIds <- ids
            expandedAxisIds <- expandedAxisIds @ ids
            expandedSizes <- expandedSizes @ sizes
        | Bracket inner ->
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

            let groupIds, groupSizes, nextPlaceholderIndex =
                inferGroupSizes axisSizes atom groupDimSize placeholderIndex
            placeholderIndex <- nextPlaceholderIndex

            expandedAxisIds <- expandedAxisIds @ groupIds
            expandedSizes <- expandedSizes @ groupSizes
        | Concat _ ->
            notSupported "Concat is not supported in input."

    if dimIndex <> rank then
        invalidOp (sprintf "Pattern did not consume all dimensions (consumed %i, rank %i)." dimIndex rank)

    let xExpanded = x.reshape(expandedSizes |> List.toArray)
    xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds

let private buildOutputItems
    (axisSizes: AxisSizes)
    (outputExpr: Expression)
    (ellipsisAxisIds: AxisId list)
    (existingAxisIds: Set<AxisId>)
    : OutputItem list =

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
                    | Bracket inner -> getAtoms inner |> List.choose (function | Axis (AxisName n) -> Some n | _ -> None)
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

    // Validate: star must not appear (reserved for pack/unpack).
    if outputItems |> List.exists (function | OutputAxis "__star" -> true | OutputGroup ids -> ids |> List.contains "__star" | _ -> false) then
        notSupported "Star axis is reserved for pack/unpack and is not supported in rearrange/repeat output."

    // Validate: all output axes that are supposed to come from input exist, new axes handled elsewhere.
    outputItems

let private permuteAndReshape (xExpanded: torch.Tensor) (expandedAxisIds: AxisId list) (expandedSizes: int64 list) (outputItems: OutputItem list) =
    let axisToIndex =
        expandedAxisIds
        |> List.mapi (fun i id -> id, i)
        |> Map.ofList

    let axisToSize =
        expandedAxisIds
        |> List.zip expandedSizes
        |> List.map (fun (size, id) -> id, size)
        |> Map.ofList

    let permuteAxisIds =
        outputItems
        |> List.collect (function
            | OutputAxis id -> [ id ]
            | OutputGroup ids -> ids
            | OutputConst _ -> [])

    if permuteAxisIds.Length <> expandedAxisIds.Length then
        invalidOp "Output axes must be a permutation of input axes."

    // Rearrange doesn't support duplicating existing axes.
    let duplicates =
        permuteAxisIds
        |> List.groupBy id
        |> List.choose (fun (id, xs) -> if xs.Length > 1 then Some id else None)
    if duplicates.Length > 0 then
        let duplicatesText = String.Join(", ", duplicates)
        invalidOp (sprintf "Duplicate axes in output are not supported for rearrange/repeat: %s" duplicatesText)

    let permuteIndices =
        permuteAxisIds
        |> List.map (fun id ->
            axisToIndex
            |> Map.tryFind id
            |> Option.defaultWith (fun () -> invalidOp (sprintf "Axis '%s' not found in input." id)))
        |> List.map int64
        |> List.toArray

    if permuteIndices.Length <> xExpanded.shape.Length then
        invalidOp "permute requires a permutation for every tensor dimension."

    let xPermuted = torch.permute(xExpanded, permuteIndices)

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

    xPermuted.reshape(finalShape)

let rearrange (pattern: string) (x: torch.Tensor) (axisSizes: (string * int64) list) : torch.Tensor =
    let axisSizesMap = toAxisSizes axisSizes
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 1 || outputs.Length <> 1 then
        invalidOp "rearrange expects exactly one input and one output expression."

    let xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds = matchInput axisSizesMap inputs[0] x

    let outputItems =
        buildOutputItems axisSizesMap outputs[0] ellipsisAxisIds (expandedAxisIds |> Set.ofList)

    // Rearrange allows inserting only constant singleton dimensions.
    if outputItems |> List.exists (function | OutputConst c when c <> 1L -> true | _ -> false) then
        invalidOp "rearrange can only insert constant axes of size 1."

    permuteAndReshape xExpanded expandedAxisIds expandedSizes outputItems

let repeat (pattern: string) (x: torch.Tensor) (axisSizes: (string * int64) list) : torch.Tensor =
    let axisSizesMap = toAxisSizes axisSizes
    let inputs, outputs = parseEinx pattern

    if inputs.Length <> 1 || outputs.Length <> 1 then
        invalidOp "repeat expects exactly one input and one output expression."

    let xExpanded, expandedAxisIds, expandedSizes, ellipsisAxisIds = matchInput axisSizesMap inputs[0] x

    let existingAxisIds = expandedAxisIds |> Set.ofList

    let outputItems =
        buildOutputItems axisSizesMap outputs[0] ellipsisAxisIds existingAxisIds

    let outputAxisIds =
        outputItems
        |> List.collect (function
            | OutputAxis id -> [ id ]
            | OutputGroup ids -> ids
            | OutputConst _ -> [])

    let newAxes =
        outputAxisIds
        |> List.filter (fun id -> not (existingAxisIds.Contains id))

    // Permute existing axes into the order they appear in output (flatten groups).
    let existingOrder =
        outputAxisIds
        |> List.filter (fun id -> existingAxisIds.Contains id)

    let axisToIndex =
        expandedAxisIds
        |> List.mapi (fun i id -> id, i)
        |> Map.ofList

    if existingOrder.Length <> expandedAxisIds.Length then
        invalidOp "repeat currently requires all input axes to be present in the output."

    let permuteIndices =
        existingOrder
        |> List.map (fun id -> axisToIndex[id] |> int64)
        |> List.toArray

    let xPermuted = torch.permute(xExpanded, permuteIndices)

    let mutable current = xPermuted
    let mutable currentAxisIds = existingOrder
    let mutable currentSizes =
        existingOrder
        |> List.map (fun id ->
            let idx = axisToIndex[id]
            expandedSizes[idx])

    // Insert new axes in order.
    for item in outputItems do
        match item with
        | OutputConst c ->
            let dim = currentAxisIds.Length
            current <- current.unsqueeze(dim).expand(Array.append (current.shape |> Array.map int64) [| c |])
            currentAxisIds <- currentAxisIds @ [ sprintf "__c%i" dim ]
            currentSizes <- currentSizes @ [ c ]
        | OutputAxis id when existingAxisIds.Contains id ->
            ()
        | OutputAxis id ->
            let size =
                axisSizesMap
                |> Map.tryFind id
                |> Option.defaultWith (fun () -> invalidOp (sprintf "repeat requires size for new axis '%s'." id))

            let insertPos =
                outputAxisIds
                |> List.findIndex ((=) id)
            // Insert at end and then permute later to exact order.
            let dim = currentAxisIds.Length
            current <- current.unsqueeze(dim).expand(Array.append (current.shape |> Array.map int64) [| size |])
            currentAxisIds <- currentAxisIds @ [ id ]
            currentSizes <- currentSizes @ [ size ]
        | OutputGroup _ ->
            ()

    // Now permute+reshape using the same pipeline but with the newly created axes present.
    let finalOutputItems =
        outputItems
        |> List.map (function
            | OutputAxis id -> OutputAxis id
            | OutputGroup ids -> OutputGroup ids
            | OutputConst c -> OutputConst c)

    permuteAndReshape current currentAxisIds currentSizes finalOutputItems

type Reduction =
    | Sum
    | Mean

let reduce (pattern: string) (x: torch.Tensor) (reduction: Reduction) (keepDims: bool) (axisSizes: (string * int64) list) : torch.Tensor =
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
                |> List.choose (function | Axis (AxisName name) -> Some name | _ -> None)
            | _ -> [])
        |> Set.ofList

    if reduceAxisNames.IsEmpty then
        invalidOp "reduce requires bracketed axes in input expression (e.g., 'b [c] h -> b h')."

    // Build an input expression without brackets for matching.
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
        |> List.map int64
        |> List.toArray

    let reduced =
        match reduction with
        | Sum -> xExpanded.sum(reduceDims, keepdim = keepDims)
        | Mean -> xExpanded.mean(reduceDims, keepDims)

    // Remove reduced axes from expanded axis lists if keepDims = false.
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

    let outputItems =
        buildOutputItems axisSizesMap outputExpr ellipsisAxisIds (remainingAxisIds |> Set.ofList)

    // Validate: output must not reference reduced axes.
    let outputAxisIds =
        outputItems
        |> List.collect (function | OutputAxis id -> [ id ] | OutputGroup ids -> ids | OutputConst _ -> [])

    if outputAxisIds |> List.exists reduceAxisNames.Contains then
        invalidOp "Output expression references reduced axes."

    permuteAndReshape reduced remainingAxisIds remainingSizes outputItems

let einsum (pattern: string) (tensors: torch.Tensor list) : torch.Tensor =
    let inputs, outputs = parseEinx pattern

    if outputs.Length <> 1 then
        invalidOp "einsum expects exactly one output expression."

    if inputs.Length <> tensors.Length then
        invalidOp (sprintf "einsum expects %i tensors but got %i." inputs.Length tensors.Length)

    let axisNames =
        (inputs @ outputs)
        |> List.collect (fun expr ->
            let rec collect (e: Expression) =
                match e with
                | Axis (AxisName n) -> [ n ]
                | Axis (AxisNumber _) -> notSupported "Numeric axes are not supported in einsum."
                | Axis AxisPlaceholder -> notSupported "Placeholder axes are not supported in einsum."
                | Axis AxisStar -> notSupported "Star axes are not supported in einsum."
                | Ellipsis -> []
                | Empty -> []
                | Bracket inner -> collect inner
                | Composition parts
                | Concat parts -> parts |> List.collect collect
            collect expr)
        |> List.distinct

    let letters =
        [ 'a' .. 'z' ] @ [ 'A' .. 'Z' ]
        |> List.map string

    if axisNames.Length > letters.Length then
        notSupported "Too many distinct axes for einsum label mapping."

    let axisToLabel =
        (axisNames, letters |> List.take axisNames.Length)
        ||> List.zip
        |> Map.ofList

    let rec exprToSubscript (expr: Expression) =
        let rec toTokens (e: Expression) =
            match e with
            | Ellipsis -> [ "..." ]
            | Axis (AxisName n) -> [ axisToLabel[n] ]
            | Bracket inner -> toTokens inner
            | Composition parts -> parts |> List.collect toTokens
            | Concat _ -> notSupported "Concat is not supported in einsum."
            | Axis _ -> notSupported "Unsupported axis kind in einsum."
            | Empty -> []
        String.Concat(toTokens expr)

    let inputSubscripts = inputs |> List.map exprToSubscript
    let outputSubscript = exprToSubscript outputs[0]

    let inputSubscriptsText = String.Join(",", inputSubscripts)
    let equation = sprintf "%s->%s" inputSubscriptsText outputSubscript
    torch.einsum(equation, tensors |> List.toArray)

let pack (pattern: string) (x: torch.Tensor) : torch.Tensor * int64[] =
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

    let before =
        atoms
        |> List.takeWhile (function | Axis AxisStar -> false | _ -> true)
        |> List.length

    let after =
        atoms
        |> List.rev
        |> List.takeWhile (function | Axis AxisStar -> false | _ -> true)
        |> List.length

    let rank = x.shape.Length
    if before + after + 1 > rank then
        invalidOp (sprintf "pack pattern requires at least %i dims but tensor rank is %i." (before + after + 1) rank)

    let starDims = rank - before - after
    let packedShape =
        x.shape
        |> Array.map int64
        |> Array.skip before
        |> Array.take starDims

    let packedSize =
        packedShape
        |> Array.fold (fun acc v -> acc * v) 1L

    let newShape =
        Array.concat
            [ x.shape |> Array.map int64 |> Array.take before
              [| packedSize |]
              x.shape |> Array.map int64 |> Array.skip (before + starDims) ]

    x.reshape(newShape), packedShape

let unpack (pattern: string) (xPacked: torch.Tensor) (packedShape: int64[]) : torch.Tensor =
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

    let before =
        atoms
        |> List.takeWhile (function | Axis AxisStar -> false | _ -> true)
        |> List.length

    let after =
        atoms
        |> List.rev
        |> List.takeWhile (function | Axis AxisStar -> false | _ -> true)
        |> List.length

    let rank = xPacked.shape.Length
    if before + after + 1 <> rank then
        invalidOp "unpack expects packed tensor rank to match the pattern (with '*' as one axis)."

    let newShape =
        Array.concat
            [ xPacked.shape |> Array.map int64 |> Array.take before
              packedShape
              xPacked.shape |> Array.map int64 |> Array.skip (before + 1) ]

    xPacked.reshape(newShape)
