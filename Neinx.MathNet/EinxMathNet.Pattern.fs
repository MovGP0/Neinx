module EinxMathNetPattern

open System
open Parser
open EinxMathNetTypes

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
