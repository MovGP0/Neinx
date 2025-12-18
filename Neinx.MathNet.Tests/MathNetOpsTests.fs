namespace einx.MathNet.Tests

open Shouldly
open Xunit

open EinxMathNet
open EinxMathNetTypes

[<Sealed>]
type MathNetOpsTests() =
    [<Fact(DisplayName = "Should rearrange by permuting dimensions when invoked")>]
    member _.ShouldRearrangeByPermutingDimensionsWhenInvoked() =
        // Arrange
        let x =
            tensor
                [| 2L; 3L; 4L |]
                (Array.init (2 * 3 * 4) (fun i -> float i))

        // Act
        let result = rearrange "a b c -> b c a" x []

        // Assert
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape.ShouldBe([| 3L; 4L; 2L |])),
            (fun () -> result.data[0].ShouldBe(x.data[0])),
            (fun () -> result.data[result.data.Length - 1].ShouldBe(x.data[x.data.Length - 1]))
        )

    [<Fact(DisplayName = "Should rearrange by flattening grouped dimensions when invoked")>]
    member _.ShouldRearrangeByFlatteningGroupedDimensionsWhenInvoked() =
        // Arrange
        let x = tensor [| 2L; 3L; 4L |] (Array.init (2 * 3 * 4) (fun i -> float i))

        // Act
        let result = rearrange "a b c -> a (b c)" x []

        // Assert
        result.shape.ShouldBe([| 2L; 12L |])

    [<Fact(DisplayName = "Should rearrange by splitting grouped input dimension when invoked")>]
    member _.ShouldRearrangeBySplittingGroupedInputDimensionWhenInvoked() =
        // Arrange
        // (h w) = 12, c = 2  =>  h = 3, w inferred = 4
        let x = tensor [| 12L; 2L |] (Array.init (12 * 2) (fun i -> float i))

        // Act
        let result = rearrange "(h w) c -> h w c" x [ "h", 3L ]

        // Assert
        result.shape.ShouldBe([| 3L; 4L; 2L |])

    [<Fact(DisplayName = "Should repeat by inserting and broadcasting new axes when invoked")>]
    member _.ShouldRepeatByInsertingAndBroadcastingNewAxesWhenInvoked() =
        // Arrange
        let x = tensor [| 2L; 3L |] (Array.init (2 * 3) (fun i -> float i))

        // Act
        let result = repeat "a b -> a b c" x [ "c", 4L ]

        // Assert
        let isRepeatedCorrectly =
            let aSize = int x.shape[0]
            let bSize = int x.shape[1]
            let cSize = 4

            seq {
                for a in 0 .. (aSize - 1) do
                    for b in 0 .. (bSize - 1) do
                        let original = x.data[(a * bSize) + b]

                        for c in 0 .. (cSize - 1) do
                            let idx = (((a * bSize) + b) * cSize) + c
                            yield result.data[idx] = original
            }
            |> Seq.forall id

        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape.ShouldBe([| 2L; 3L; 4L |])),
            (fun () -> isRepeatedCorrectly.ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should reduce bracketed axes using sum when invoked")>]
    member _.ShouldReduceBracketedAxesUsingSumWhenInvoked() =
        // Arrange
        let x = ones [| 2L; 3L; 4L |]

        // Act
        let result = reduce "a [b] c -> a c" x Reduction.Sum false []

        // Assert
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape.ShouldBe([| 2L; 4L |])),
            (fun () -> result.data |> Array.forall (fun v -> v = 3.0) |> fun b -> b.ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should reduce bracketed axes using mean when invoked")>]
    member _.ShouldReduceBracketedAxesUsingMeanWhenInvoked() =
        // Arrange
        let x = ones [| 2L; 3L; 4L |]

        // Act
        let result = reduce "a [b] c -> a c" x Reduction.Mean false []

        // Assert
        result.ShouldSatisfyAllConditions(
            (fun () -> result.shape.ShouldBe([| 2L; 4L |])),
            (fun () -> result.data |> Array.forall (fun v -> v = 1.0) |> fun b -> b.ShouldBeTrue())
        )

    [<Fact(DisplayName = "Should compute einsum as matrix multiplication when invoked")>]
    member _.ShouldComputeEinsumAsMatrixMultiplicationWhenInvoked() =
        // Arrange
        let a = tensor [| 2L; 3L |] (Array.init (2 * 3) (fun i -> float (i + 1)))
        let b = tensor [| 3L; 4L |] (Array.init (3 * 4) (fun i -> float (i + 1)))

        // Act
        let result = einsum "x y, y z -> x z" [ a; b ]

        // Assert
        result.shape.ShouldBe([| 2L; 4L |])

    [<Fact(DisplayName = "Should pack and unpack star axis when invoked")>]
    member _.ShouldPackAndUnpackStarAxisWhenInvoked() =
        // Arrange
        let x = tensor [| 2L; 3L; 4L; 5L |] (Array.init (2 * 3 * 4 * 5) (fun i -> float i))

        // Act
        let packed, packedShape = pack "a * b" x
        let unpacked = unpack "a * b" packed packedShape

        // Assert
        unpacked.ShouldSatisfyAllConditions(
            (fun () -> packed.shape.ShouldBe([| 2L; 12L; 5L |])),
            (fun () -> packedShape.ShouldBe([| 3L; 4L |])),
            (fun () -> unpacked.shape.ShouldBe(x.shape)),
            (fun () -> unpacked.data.ShouldBe(x.data))
        )
