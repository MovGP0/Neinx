namespace einx.Parser.Tests

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
        Assert.Equal<int64>(4L, result.shape[1])
        Assert.Equal<int64>(5L, result.shape[2])
        Assert.Equal<int64>(3L, result.shape[3])

        use expected = torch.permute(x, [| 0L; 2L; 3L; 1L |])
        Assert.True(torch.allclose(result, expected))

    [<Fact(DisplayName = "Should rearrange by flattening grouped dimensions when invoked")>]
    member _.ShouldRearrangeByFlatteningGroupedDimensionsWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L; 5L |])

        // Act
        use result = rearrange "b c h w -> b (h w) c" x []

        // Assert
        Assert.Equal<int64>(2L, result.shape[0])
        Assert.Equal<int64>(20L, result.shape[1])
        Assert.Equal<int64>(3L, result.shape[2])

    [<Fact(DisplayName = "Should rearrange by splitting grouped input dimension when invoked")>]
    member _.ShouldRearrangeBySplittingGroupedInputDimensionWhenInvoked() =
        // Arrange
        use x = torch.randn([| 20L; 3L |])

        // Act (infer w = 5 from 20 / h=4)
        use result = rearrange "(h w) c -> h w c" x [ "h", 4L ]

        // Assert
        Assert.Equal<int64>(4L, result.shape[0])
        Assert.Equal<int64>(5L, result.shape[1])
        Assert.Equal<int64>(3L, result.shape[2])

    [<Fact(DisplayName = "Should repeat by inserting and expanding new axes when invoked")>]
    member _.ShouldRepeatByInsertingAndExpandingNewAxesWhenInvoked() =
        // Arrange
        use x = torch.arange(0, 6, dtype = torch.int64).reshape([| 2L; 3L |])

        // Act
        use result = repeat "b c -> b c h w" x [ "h", 2L; "w", 3L ]

        // Assert
        Assert.Equal<int64>(2L, result.shape[0])
        Assert.Equal<int64>(3L, result.shape[1])
        Assert.Equal<int64>(2L, result.shape[2])
        Assert.Equal<int64>(3L, result.shape[3])

        use expected = x.unsqueeze(2).unsqueeze(3).expand([| 2L; 3L; 2L; 3L |])
        Assert.True(torch.allclose(result, expected))

    [<Fact(DisplayName = "Should reduce bracketed axes using mean when invoked")>]
    member _.ShouldReduceBracketedAxesUsingMeanWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L |])

        // Act
        use result = reduce "b [c] h -> b h" x Reduction.Mean false []

        // Assert
        Assert.Equal<int64>(2L, result.shape[0])
        Assert.Equal<int64>(4L, result.shape[1])

        use expected = x.mean([| 1L |], false)
        Assert.True(torch.allclose(result, expected))

    [<Fact(DisplayName = "Should compute einsum like matrix multiplication when invoked")>]
    member _.ShouldComputeEinsumMatmulWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L |])
        use y = torch.randn([| 3L; 4L |])

        // Act
        use result = einsum "b c, c d -> b d" [ x; y ]

        // Assert
        use expected = torch.matmul(x, y)
        Assert.True(torch.allclose(result, expected))

    [<Fact(DisplayName = "Should pack and unpack with star axis when invoked")>]
    member _.ShouldPackAndUnpackWithStarAxisWhenInvoked() =
        // Arrange
        use x = torch.randn([| 2L; 3L; 4L; 5L |])

        // Act
        let packed, packedShape = pack "b * c" x
        use packedTensor = packed
        use unpacked = unpack "b * c" packedTensor packedShape

        // Assert
        Assert.Equal<int64>(2L, packedTensor.shape[0])
        Assert.Equal<int64>(12L, packedTensor.shape[1])
        Assert.Equal<int64>(5L, packedTensor.shape[2])
        Assert.Equal<int64[]>([| 3L; 4L |], packedShape)
        Assert.True(torch.allclose(unpacked, x))
