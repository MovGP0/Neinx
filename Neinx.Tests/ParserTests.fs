namespace einx.Parser.Tests

open System
open Parser
open Xunit

type ParseEinxTests() =
    [<Fact(DisplayName = "Should parse a single axis when invoked")>]
    member _.ShouldParseSingleAxisWhenInvoked() =
        let result = parseEinx "a"
        let expected = ([ Axis(AxisName "a") ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse a numeric axis when invoked")>]
    member _.ShouldParseNumericAxisWhenInvoked() =
        let result = parseEinx "16"
        let expected = ([ Axis(AxisNumber 16) ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse ellipsis when invoked")>]
    member _.ShouldParseEllipsisWhenInvoked() =
        let result = parseEinx "..."
        let expected = ([ Ellipsis ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse composition when invoked")>]
    member _.ShouldParseCompositionWhenInvoked() =
        let result = parseEinx "a b c"
        let expected = ([ Composition [ Axis(AxisName "a"); Axis(AxisName "b"); Axis(AxisName "c") ] ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should split multiple inputs by top-level commas when invoked")>]
    member _.ShouldSplitInputsByTopLevelCommasWhenInvoked() =
        let result = parseEinx "a, b c"
        let expected = ([ Axis(AxisName "a"); Composition [ Axis(AxisName "b"); Axis(AxisName "c") ] ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse top-level arrow into input and output when invoked")>]
    member _.ShouldParseTopLevelArrowWhenInvoked() =
        let result = parseEinx "a b -> b a"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Axis(AxisName "b") ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "a") ] ] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse bracketed expressions with ellipsis inside when invoked")>]
    member _.ShouldParseBracketedEllipsisInsideWhenInvoked() =
        let result = parseEinx "b [s...] c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Bracket (Composition [ Axis(AxisName "s"); Ellipsis ])
                      Axis(AxisName "c") ] ],
              [] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse ellipsis after a bracket when invoked")>]
    member _.ShouldParseEllipsisAfterBracketWhenInvoked() =
        let result = parseEinx "b [s]... c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Composition [ Bracket (Axis(AxisName "s")); Ellipsis ]
                      Axis(AxisName "c") ] ],
              [] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse ellipsis after parentheses when invoked")>]
    member _.ShouldParseEllipsisAfterParensWhenInvoked() =
        let result = parseEinx "b (s [s2])... c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Composition
                          [ Composition [ Axis(AxisName "s"); Bracket (Axis(AxisName "s2")) ]
                            Ellipsis ]
                      Axis(AxisName "c") ] ],
              [] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse concatenation using plus when invoked")>]
    member _.ShouldParseConcatWhenInvoked() =
        let result = parseEinx "a + b"
        let expected = ([ Concat [ Axis(AxisName "a"); Axis(AxisName "b") ] ], [])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should expand bracket arrow into output when invoked")>]
    member _.ShouldExpandBracketArrowWhenInvoked() =
        let result = parseEinx "a [b->c]"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "b")) ] ],
              [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "c")) ] ] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should expand bracket arrow with ellipsis when invoked")>]
    member _.ShouldExpandBracketArrowWithEllipsisWhenInvoked() =
        let result = parseEinx "b [s...->s2] c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Bracket (Composition [ Axis(AxisName "s"); Ellipsis ])
                      Axis(AxisName "c") ] ],
              [ Composition
                    [ Axis(AxisName "b")
                      Bracket (Axis(AxisName "s2"))
                      Axis(AxisName "c") ] ] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should expand bracket arrow with empty right side when invoked")>]
    member _.ShouldExpandBracketArrowWithEmptyRightSideWhenInvoked() =
        let result = parseEinx "a [b->]"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "b")) ] ],
              [ Axis(AxisName "a") ] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should expand bracket arrow with empty left side when invoked")>]
    member _.ShouldExpandBracketArrowWithEmptyLeftSideWhenInvoked() =
        let result = parseEinx "a [->c]"
        let expected =
            ( [ Axis(AxisName "a") ],
              [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "c")) ] ] )
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should parse fully empty bracket arrow as empty expressions when invoked")>]
    member _.ShouldParseFullyEmptyBracketArrowWhenInvoked() =
        let result = parseEinx "[->]"
        let expected = ([ Empty ], [ Empty ])
        Assert.Equal(expected, result)

    [<Fact(DisplayName = "Should expand bracket arrow with multiple left parts when invoked")>]
    member _.ShouldExpandBracketArrowWithMultipleLeftPartsWhenInvoked() =
        let result = parseEinx "b p [i,->]"
        let expectedInputs =
            [ Composition [ Axis(AxisName "b"); Axis(AxisName "p"); Bracket (Axis(AxisName "i")) ]
              Composition [ Axis(AxisName "b"); Axis(AxisName "p") ] ]
        let expectedOutputs =
            [ Composition [ Axis(AxisName "b"); Axis(AxisName "p") ] ]
        Assert.Equal((expectedInputs, expectedOutputs), result)

    [<Fact(DisplayName = "Should find top-level arrow after bracket arrow when invoked")>]
    member _.ShouldFindTopLevelArrowAfterBracketArrowWhenInvoked() =
        let result = parseEinx "a [b->c], d -> a [c]"
        let expectedInputs =
            [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "b")) ]
              Axis(AxisName "d") ]
        let expectedOutputs =
            [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "c")) ] ]
        Assert.Equal((expectedInputs, expectedOutputs), result)

    [<Fact(DisplayName = "Should reject duplicate axis names in a single expression when invoked")>]
    member _.ShouldRejectDuplicateAxesWhenInvoked() =
        Assert.Throws<Exception>(fun () -> parseEinx "a a" |> ignore) |> ignore

