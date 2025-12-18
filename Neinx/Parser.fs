module Parser

open System
open FParsec

/// AST types representing the einx expression syntax
type Axis =
    | AxisName of string         // named axis (e.g., 'a', 'height')
    | AxisNumber of int          // unnamed axis with fixed size (e.g., 16)

type Expression =
    | Axis of Axis               // a single axis (named or numeric)
    | Ellipsis                   // ellipsis "..." representing unspecified axes
    | Composition of Expression list   // composition of multiple sub-expressions (axes in sequence)
    | Concat of Expression list        // concatenation of axes (with '+')
    | Bracket of Expression      // bracket notation grouping axes for an operation

/// Parse the einx operation string into input and output expression ASTs.
let parseEinx (opString: string) : Expression list * Expression list =
    // Helper: split a string by top-level commas (not inside brackets/parentheses).
    let splitTopLevelCommas (s:string) : string list =
        let segments = ResizeArray<string>()
        let mutable depthParen = 0
        let mutable depthBracket = 0
        let sb = Text.StringBuilder()
        for ch in s do
            match ch with
            | '(' -> 
                depthParen <- depthParen + 1
                sb.Append(ch) |> ignore
            | ')' ->
                if depthParen > 0 then depthParen <- depthParen - 1
                sb.Append(ch) |> ignore
            | '[' ->
                depthBracket <- depthBracket + 1
                sb.Append(ch) |> ignore
            | ']' ->
                if depthBracket > 0 then depthBracket <- depthBracket - 1
                sb.Append(ch) |> ignore
            | ',' when depthParen = 0 && depthBracket = 0 ->
                // top-level comma: split here
                segments.Add(sb.ToString().Trim())
                sb.Clear() |> ignore
            | _ ->
                sb.Append(ch) |> ignore
        // add final segment if any
        let lastSeg = sb.ToString().Trim()
        if lastSeg.Length > 0 then segments.Add(lastSeg)
        List.ofSeq segments

    // Helper: expand a bracket expression containing '->' (and possibly commas) internally
    let expandBracketExpression (expr:string) : string list * string list =
        // Find the bracket bounds that contain an '->'
        let startIdx = expr.IndexOf('[')
        let endIdx = expr.LastIndexOf(']')
        if startIdx = -1 || endIdx = -1 || startIdx > endIdx then
            failwith "Bracket expansion error: no matching '[' ']'"
        let prefix = expr.Substring(0, startIdx)
        let suffix = expr.Substring(endIdx + 1)
        let content = expr.Substring(startIdx + 1, endIdx - startIdx - 1)
        // Split content by '->'
        let arrowIdx = content.IndexOf("->")
        let leftContent, rightContent =
            if arrowIdx >= 0 then
                content.Substring(0, arrowIdx), content.Substring(arrowIdx + 2)
            else content, ""  // no arrow in content
        // Split left and right parts by commas
        let leftParts = splitTopLevelCommas leftContent
        let rightParts =
            if String.IsNullOrWhiteSpace(rightContent) then []
            else splitTopLevelCommas rightContent
        // Build new input expressions for each left part
        let inputExprs = 
            leftParts 
            |> List.map (fun part -> 
                let partTrim = part.Trim()
                if partTrim = "" then (prefix + suffix).Trim()
                else (prefix + "[" + partTrim + "]" + suffix).Trim())
        // Build new output expressions (if any)
        let outputExprs =
            if rightParts.IsEmpty then 
                // If arrow was present but right side empty (e.g., "->" at end), output is prefix+suffix
                if arrowIdx >= 0 then [ (prefix + suffix).Trim() ] else []
            else
                rightParts |> List.map (fun part ->
                    let partTrim = part.Trim()
                    if partTrim = "" then (prefix + suffix).Trim()
                    else (prefix + "[" + partTrim + "]" + suffix).Trim())
        inputExprs, outputExprs

    // Stage 0: Split the operation string by top-level '->' and commas.
    let arrowPos =
        let idx = opString.IndexOf("->")
        if idx >= 0 then
            // Check that this '->' is at top-level (not inside any brackets/parentheses)
            let before = opString.Substring(0, idx)
            let mutable depthP = 0
            let mutable depthB = 0
            for ch in before do
                match ch with
                | '(' -> depthP <- depthP + 1
                | ')' -> if depthP > 0 then depthP <- depthP - 1
                | '[' -> depthB <- depthB + 1
                | ']' -> if depthB > 0 then depthB <- depthB - 1
                | _ -> ()
            if depthP = 0 && depthB = 0 then idx else -1
        else -1
    let inputStrings, outputStrings =
        if arrowPos >= 0 then
            let left = opString.Substring(0, arrowPos)
            let right = opString.Substring(arrowPos + 2)
            (splitTopLevelCommas left), (splitTopLevelCommas right)
        else
            (splitTopLevelCommas opString), []

    // Handle any internal bracket expansions (arrow inside brackets) in each input expression.
    let mutable finalInputStrs = []
    let mutable finalOutputStrs = outputStrings
    for expr in inputStrings do
        if expr.Contains("->") then
            let inExprs, outExprs = expandBracketExpression expr
            finalInputStrs <- finalInputStrs @ inExprs
            if outExprs <> [] then
                finalOutputStrs <- outExprs  // override outputs if this defines output
        else
            finalInputStrs <- finalInputStrs @ [expr]

    // Parser for a single expression (Stage 1).
    let ws = spaces  // whitespace parser
    // Primitive lexers for axis name and number
    let pAxisName =
        many1Satisfy2L isLetter (fun c -> isLetter c || isDigit c) "axis name"
        |>> AxisName
    let pAxisNumber =
        many1SatisfyL isDigit "axis size"
        |>> (fun digits -> AxisNumber (int digits))
    // Forward declaration for recursive expression parser
    let pExpr, pExprRef = createParserForwardedToRef<Expression, unit>()

    // Parser for parenthesized expressions (can contain composition or concatenation).
    let pParenContent =
        // Parse either a concatenation (with '+') or just a composition of expressions.
        let pConcatParts = sepBy1 (pExpr .>> ws) (pchar '+' .>> ws)
        pConcatParts |>> (fun parts -> if List.length parts > 1 then Concat parts else List.head parts)
    let pParens = between (pchar '(' .>> spaces) (spaces >>. pchar ')') pParenContent

    // Parser for bracketed expressions (no internal arrow at this stage).
    let pBracketExpr = between (pchar '[' .>> spaces) (spaces >>. pchar ']') pExpr |>> Bracket

    // Atom parsers (smallest units).
    // Allow ellipsis directly after axis name, bracket, or parenthesis without space.
    let pEllipsis = pstring "..." |>> fun _ -> Ellipsis
    let pAxisWithEllipsis =
        attempt (pAxisName .>>. pstring "..." |>> fun (AxisName name, _) -> 
                    Composition [ Axis(AxisName name); Ellipsis ])
    let pBracketWithEllipsis =
        attempt (pBracketExpr .>>. pstring "..." |>> fun (Bracket inner, _) ->
                    Composition [ Bracket inner; Ellipsis ])
    let pParenWithEllipsis =
        attempt (pParens .>>. pstring "..." |>> fun (inner, _) ->
                    Composition [ inner; Ellipsis ])
    let pAtom =
        choice [
            pAxisWithEllipsis |>> (fun e -> e : Expression)
            pBracketWithEllipsis |>> (fun e -> e : Expression)
            pParenWithEllipsis |>> (fun e -> e : Expression)
            pBracketExpr
            pParens
            pEllipsis
            pAxisNumber |>> Axis
            pAxisName  |>> Axis
        ]

    // Parser for a sequence of atoms (composition), separated by whitespace.
    let pComposition =
        pAtom .>>. many (many1 (pchar ' ' <|> pchar '\t') >>. pAtom)
        |>> fun (first, rest) ->
            if rest.IsEmpty then first
            else Composition (first :: rest)

    pExprRef := pComposition

    // Function to parse a single expression string into an Expression AST.
    let parseExpressionString (exprStr:string) : Expression =
        match run (pExpr .>> eof) exprStr with
        | Success(result, _, _) -> result
        | Failure(msg, _, _) -> failwithf "Failed to parse expression '%s': %s" exprStr msg

    // Parse all input and output expression strings into ASTs.
    let inputASTs = List.map parseExpressionString finalInputStrs
    let outputASTs = List.map parseExpressionString finalOutputStrs

    // Semantic check: ensure no axis name appears more than once per expression.
    let ensureUniqueAxes (expr:Expression) =
        let seen = System.Collections.Generic.HashSet<string>()
        let rec check expr =
            match expr with
            | Axis (AxisName name) ->
                if seen.Contains(name) then failwithf "Duplicate axis name '%s' in expression" name
                else seen.Add(name) |> ignore
            | Axis (AxisNumber _) -> ()   // numeric axes are unnamed
            | Ellipsis -> ()             // ellipsis is not a named axis
            | Composition parts
            | Concat parts ->
                List.iter check parts
            | Bracket subexpr ->
                check subexpr
        check expr
    inputASTs |> List.iter ensureUniqueAxes
    outputASTs |> List.iter ensureUniqueAxes

    (inputASTs, outputASTs)