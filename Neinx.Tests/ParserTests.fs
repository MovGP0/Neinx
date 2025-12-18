namespace einx.Parser.Tests

open Parser
open Xunit
open Shouldly

type ParseEinxTests() =
    [<Fact(DisplayName = "Should parse a single axis when invoked")>]
    member _.ShouldParseSingleAxisWhenInvoked() =
        let result = parseEinx "a"
        let expected = ([ Axis(AxisName "a") ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse a numeric axis when invoked")>]
    member _.ShouldParseNumericAxisWhenInvoked() =
        let result = parseEinx "16"
        let expected = ([ Axis(AxisNumber 16) ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse an axis name containing underscores when invoked")>]
    member _.ShouldParseAxisNameWithUnderscoreWhenInvoked() =
        let result = parseEinx "c_in c_out"
        let expected = ([ Composition [ Axis(AxisName "c_in"); Axis(AxisName "c_out") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse placeholder axis when invoked")>]
    member _.ShouldParsePlaceholderAxisWhenInvoked() =
        let result = parseEinx "_"
        let expected = ([ Axis AxisPlaceholder ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should allow repeated placeholder axes when invoked")>]
    member _.ShouldAllowRepeatedPlaceholderAxesWhenInvoked() =
        let result = parseEinx "_ _"
        let expected = ([ Composition [ Axis AxisPlaceholder; Axis AxisPlaceholder ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse packed star axis when invoked")>]
    member _.ShouldParseStarAxisWhenInvoked() =
        let result = parseEinx "*"
        let expected = ([ Axis AxisStar ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse ellipsis when invoked")>]
    member _.ShouldParseEllipsisWhenInvoked() =
        let result = parseEinx "..."
        let expected = ([ Ellipsis ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse composition when invoked")>]
    member _.ShouldParseCompositionWhenInvoked() =
        let result = parseEinx "a b c"
        let expected = ([ Composition [ Axis(AxisName "a"); Axis(AxisName "b"); Axis(AxisName "c") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse composition containing a placeholder axis when invoked")>]
    member _.ShouldParseCompositionWithPlaceholderAxisWhenInvoked() =
        let result = parseEinx "batch _ h w"
        let expected =
            ([ Composition [ Axis(AxisName "batch"); Axis AxisPlaceholder; Axis(AxisName "h"); Axis(AxisName "w") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should split multiple inputs by top-level commas when invoked")>]
    member _.ShouldSplitInputsByTopLevelCommasWhenInvoked() =
        let result = parseEinx "a, b c"
        let expected = ([ Axis(AxisName "a"); Composition [ Axis(AxisName "b"); Axis(AxisName "c") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse top-level arrow into input and output when invoked")>]
    member _.ShouldParseTopLevelArrowWhenInvoked() =
        let result = parseEinx "a b -> b a"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Axis(AxisName "b") ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "a") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse an einops-style rearrange pattern when invoked")>]
    member _.ShouldParseEinopsStyleRearrangeWhenInvoked() =
        let result = parseEinx "b c h w -> b h w c"
        let expected =
            ( [ Composition [ Axis(AxisName "b"); Axis(AxisName "c"); Axis(AxisName "h"); Axis(AxisName "w") ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "h"); Axis(AxisName "w"); Axis(AxisName "c") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse grouped axes using parentheses when invoked")>]
    member _.ShouldParseGroupedAxesWhenInvoked() =
        let result = parseEinx "b c h w -> b (h w) c"
        let expected =
            ( [ Composition [ Axis(AxisName "b"); Axis(AxisName "c"); Axis(AxisName "h"); Axis(AxisName "w") ] ],
              [ Composition [ Axis(AxisName "b"); Composition [ Axis(AxisName "h"); Axis(AxisName "w") ]; Axis(AxisName "c") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse empty parentheses as empty expression when invoked")>]
    member _.ShouldParseEmptyParenthesesWhenInvoked() =
        let result = parseEinx "b c () ()"
        let expected =
            ([ Composition [ Axis(AxisName "b"); Axis(AxisName "c"); Empty; Empty ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse decomposing a grouped axis when invoked")>]
    member _.ShouldParseDecomposingGroupedAxisWhenInvoked() =
        let result = parseEinx "(b1 b2) h w c -> b1 b2 h w c"
        let expected =
            ( [ Composition
                    [ Composition [ Axis(AxisName "b1"); Axis(AxisName "b2") ]
                      Axis(AxisName "h")
                      Axis(AxisName "w")
                      Axis(AxisName "c") ] ],
              [ Composition [ Axis(AxisName "b1"); Axis(AxisName "b2"); Axis(AxisName "h"); Axis(AxisName "w"); Axis(AxisName "c") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse bracketed expressions with ellipsis inside when invoked")>]
    member _.ShouldParseBracketedEllipsisInsideWhenInvoked() =
        let result = parseEinx "b [s...] c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Bracket (Composition [ Axis(AxisName "s"); Ellipsis ])
                      Axis(AxisName "c") ] ],
              [] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse ellipsis after a bracket when invoked")>]
    member _.ShouldParseEllipsisAfterBracketWhenInvoked() =
        let result = parseEinx "b [s]... c"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Composition [ Bracket (Axis(AxisName "s")); Ellipsis ]
                      Axis(AxisName "c") ] ],
              [] )
        result.ShouldBe(expected)

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
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse concatenation using plus when invoked")>]
    member _.ShouldParseConcatWhenInvoked() =
        let result = parseEinx "a + b"
        let expected = ([ Concat [ Axis(AxisName "a"); Axis(AxisName "b") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse a reduce-style pattern with numeric axes when invoked")>]
    member _.ShouldParseReduceStylePatternWithNumericAxesWhenInvoked() =
        let result = parseEinx "b c (h1 2) (w1 2) -> b c h1 w1"
        let expected =
            ( [ Composition
                    [ Axis(AxisName "b")
                      Axis(AxisName "c")
                      Composition [ Axis(AxisName "h1"); Axis(AxisNumber 2) ]
                      Composition [ Axis(AxisName "w1"); Axis(AxisNumber 2) ] ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "c"); Axis(AxisName "h1"); Axis(AxisName "w1") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse a repeat-style pattern adding new axes when invoked")>]
    member _.ShouldParseRepeatStylePatternWhenInvoked() =
        let result = parseEinx "b c -> b c h w"
        let expected =
            ( [ Composition [ Axis(AxisName "b"); Axis(AxisName "c") ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "c"); Axis(AxisName "h"); Axis(AxisName "w") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse an einsum-style two-input pattern when invoked")>]
    member _.ShouldParseEinsumStyleTwoInputWhenInvoked() =
        let result = parseEinx "b c, c d -> b d"
        let expected =
            ( [ Composition [ Axis(AxisName "b"); Axis(AxisName "c") ]
                Composition [ Axis(AxisName "c"); Axis(AxisName "d") ] ],
              [ Composition [ Axis(AxisName "b"); Axis(AxisName "d") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse an einsum-style pattern with ellipsis when invoked")>]
    member _.ShouldParseEinsumStyleWithEllipsisWhenInvoked() =
        let result = parseEinx "... in_dim, out_dim in_dim -> ... out_dim"
        let expected =
            ( [ Composition [ Ellipsis; Axis(AxisName "in_dim") ]
                Composition [ Axis(AxisName "out_dim"); Axis(AxisName "in_dim") ] ],
              [ Composition [ Ellipsis; Axis(AxisName "out_dim") ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse an einsum-style pattern with empty output when invoked")>]
    member _.ShouldParseEinsumStyleWithEmptyOutputWhenInvoked() =
        let result = parseEinx "i i ->"
        let expected =
            ( [ Composition [ Axis(AxisName "i"); Axis(AxisName "i") ] ],
              [ Empty ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse multiple outputs separated by commas when invoked")>]
    member _.ShouldParseMultipleOutputsSeparatedByCommasWhenInvoked() =
        let result = parseEinx "a b -> a, b"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Axis(AxisName "b") ] ],
              [ Axis(AxisName "a"); Axis(AxisName "b") ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse pack-style pattern using star in the middle when invoked")>]
    member _.ShouldParsePackStyleStarInMiddleWhenInvoked() =
        let result = parseEinx "i j * k"
        let expected = ([ Composition [ Axis(AxisName "i"); Axis(AxisName "j"); Axis AxisStar; Axis(AxisName "k") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse packed star axis in an einops-style pack pattern when invoked")>]
    member _.ShouldParsePackPatternWithStarWhenInvoked() =
        let result = parseEinx "h w *"
        let expected = ([ Composition [ Axis(AxisName "h"); Axis(AxisName "w"); Axis AxisStar ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse pack pattern with star prefix when invoked")>]
    member _.ShouldParsePackPatternWithStarPrefixWhenInvoked() =
        let result = parseEinx "* h w c"
        let expected = ([ Composition [ Axis AxisStar; Axis(AxisName "h"); Axis(AxisName "w"); Axis(AxisName "c") ] ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse bracketed placeholder axis when invoked")>]
    member _.ShouldParseBracketedPlaceholderAxisWhenInvoked() =
        let result = parseEinx "[_]"
        let expected = ([ Bracket (Axis AxisPlaceholder) ], [])
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should expand bracket arrow into output when invoked")>]
    member _.ShouldExpandBracketArrowWhenInvoked() =
        let result = parseEinx "a [b->c]"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "b")) ] ],
              [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "c")) ] ] )
        result.ShouldBe(expected)

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
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should expand bracket arrow with empty right side when invoked")>]
    member _.ShouldExpandBracketArrowWithEmptyRightSideWhenInvoked() =
        let result = parseEinx "a [b->]"
        let expected =
            ( [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "b")) ] ],
              [ Axis(AxisName "a") ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should expand bracket arrow with empty left side when invoked")>]
    member _.ShouldExpandBracketArrowWithEmptyLeftSideWhenInvoked() =
        let result = parseEinx "a [->c]"
        let expected =
            ( [ Axis(AxisName "a") ],
              [ Composition [ Axis(AxisName "a"); Bracket (Axis(AxisName "c")) ] ] )
        result.ShouldBe(expected)

    [<Fact(DisplayName = "Should parse fully empty bracket arrow as empty expressions when invoked")>]
    member _.ShouldParseFullyEmptyBracketArrowWhenInvoked() =
        let result = parseEinx "[->]"
        let expected = ([ Empty ], [ Empty ])
        result.ShouldBe(expected)

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

    [<Fact(DisplayName = "Should parse duplicate axis names in a single expression when invoked")>]
    member _.ShouldParseDuplicateAxesWhenInvoked() =
        let result = parseEinx "a a"
        let expected = ([ Composition [ Axis(AxisName "a"); Axis(AxisName "a") ] ], [])
        result.ShouldBe(expected)
