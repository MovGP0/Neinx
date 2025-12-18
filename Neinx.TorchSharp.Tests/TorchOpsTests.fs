namespace einx.Parser.Tests

open Shouldly
open TorchSharp
open Xunit

open EinxTorch

type TorchOpsTests() =
    [<Fact(DisplayName = "Should rearrange by permuting dimensions when invoked")>]
    member _.ShouldRearrangeByPermutingDimensionsWhenInvoked() =
        // Arrange
        use x = torch.arange(0, 2 * 3 * 4 * 5, dtype = torch.int64).reshape([| 2L; 3L; 4L; 5L |])

        // Act
        use result = rearrange "b c h w -> b h w c" x []

        // Assert
        use expected = torch.permute(x, [| 0L; 2L; 3L; 1L |])
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape[1].ShouldBe(4L)),
            (fun () -> result.shape[2].ShouldBe(5L)),
            (fun () -> result.shape[3].ShouldBe(3L)),
            (fun () -> torch.allclose(result, expected).ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should rearrange by flattening grouped dimensions when invoked")>]
    member _.ShouldRearrangeByFlatteningGroupedDimensionsWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L; 5L |])

        // Act
        use result = rearrange "b c h w -> b (h w) c" x []

        // Assert
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape[0].ShouldBe(2L)),
            (fun () -> result.shape[1].ShouldBe(20L)),
            (fun () -> result.shape[2].ShouldBe(3L))
        )

    [<Fact(DisplayName = "Should rearrange by splitting grouped input dimension when invoked")>]
    member _.ShouldRearrangeBySplittingGroupedInputDimensionWhenInvoked() =
        // Arrange
        use x = torch.randn([| 20L; 3L |])

        // Act (infer w = 5 from 20 / h=4)
        use result = rearrange "(h w) c -> h w c" x [ "h", 4L ]

        // Assert
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape[0].ShouldBe(4L)),
            (fun () -> result.shape[1].ShouldBe(5L)),
            (fun () -> result.shape[2].ShouldBe(3L))
        )

    [<Fact(DisplayName = "Should repeat by inserting and expanding new axes when invoked")>]
    member _.ShouldRepeatByInsertingAndExpandingNewAxesWhenInvoked() =
        // Arrange
        use x = torch.arange(0, 6, dtype = torch.int64).reshape([| 2L; 3L |])

        // Act
        use result = repeat "b c -> b c h w" x [ "h", 2L; "w", 3L ]

        // Assert
        use expected = x.unsqueeze(2).unsqueeze(3).expand([| 2L; 3L; 2L; 3L |])
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape[0].ShouldBe(2L)),
            (fun () -> result.shape[1].ShouldBe(3L)),
            (fun () -> result.shape[2].ShouldBe(2L)),
            (fun () -> result.shape[3].ShouldBe(3L)),
            (fun () -> torch.allclose(result, expected).ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should reduce bracketed axes using mean when invoked")>]
    member _.ShouldReduceBracketedAxesUsingMeanWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L |])

        // Act
        use result = reduce "b [c] h -> b h" x Reduction.Mean false []

        // Assert
        use expected = x.mean([| 1L |], false)
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape[0].ShouldBe(2L)),
            (fun () -> result.shape[1].ShouldBe(4L)),
            (fun () -> torch.allclose(result, expected).ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should compute einsum like matrix multiplication when invoked")>]
    member _.ShouldComputeEinsumMatmulWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L |])
        use y = torch.randn([| 3L; 4L |])

        // Act
        use result = einsum "b c, c d -> b d" [ x; y ]

        // Assert
        use expected = torch.matmul(x, y)
        torch.allclose(result, expected).ShouldBeTrue()

    [<Fact(DisplayName = "Should pack and unpack with star axis when invoked")>]
    member _.ShouldPackAndUnpackWithStarAxisWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L; 5L |])

        // Act
        let packed, packedShape = pack "b * c" x
        use packedTensor = packed
        use unpacked = unpack "b * c" packedTensor packedShape

        // Assert
        packedTensor.ShouldSatisfyAllConditions(
            (fun () -> packedTensor.shape[0].ShouldBe(2L)),
            (fun () -> packedTensor.shape[1].ShouldBe(12L)),
            (fun () -> packedTensor.shape[2].ShouldBe(5L)),
            (fun () -> packedShape.ShouldBe([| 3L; 4L |])),
            (fun () -> torch.allclose(unpacked, x).ShouldBeTrue())
        )
