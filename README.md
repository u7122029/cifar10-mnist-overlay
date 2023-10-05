# cifar10-mnist-overlay
Overlay mnist images over cifar10.
## Why?
Adding noise to MNIST images tests or trains a classifier over data that is not part of the training distribution.

Extending the training data with these synthesised images can act as a form of regularisation to avoid overfitting while
maintaining high accuracy on a corresponding testing dataset.

Extending the testing data with these synthesised images allows a better understanding of how a classifier will perform 
on data that is outside the training distribution, thus permitting better judgement on any future actions.
## Methods
### CIFAR-10 Background, Inverse Pixels
Let $`M \in (\mathbb{N} \cap [0,255])^{28 \times 28}`$ be a MNIST image, and let 
$`C \in (\mathbb{N} \cap [0,255])^{32 \times 32 \times 3}`$ be CIFAR-10 image.

Note that for our purposes, we resize $`M`$ so that its dimensions match those of $`C`$. This is done through `torchvision.transforms.Resize`
and the `ToNumpyRGB` which repeats the grayscale channel 3 times to emulate an RGB image.

We let the transformed MNIST image be $`M' \in (\mathbb{N} \cap [0,255])^{32 \times 32 \times 3}`$.

Next, let $`M'_{i,j}`$ be an RGB pixel in image $`M'`$ and $`C_{i,j}`$ be the corresponding pixel in image $`C`$.

The resulting pixel $`R_{i,j}`$ can be written as

```math
R_{i,j} = \frac{M'_{i,j}}{255}(255 - C_{i,j}) + \left(1 - \frac{M'_{i,j}}{255}\right)C_{i,j}
```

Here we are imagining $`C_{i,j}`$ and its RGB inverse $`255 - C_{i,j}`$ as vectors joined together by a line, parameterised by some $`t \in [0,1]`$.
If $`t = 0`$, then the resulting pixel is $`C_{i,j}`$, and if $`t = 1`$, then it is $`255 - C_{i,j}`$. Choosing $`t`$ in between these
limits will give a pixel that is between the original and its RGB inverse, and we decide this choice of $`t`$ through the value of 
$`M'_{i,j}/255`$.

### CIFAR-10 Background, Random Colour Pixels

### Random Coloured Background, Random Coloured Digit
