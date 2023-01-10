# difflogic - A Library for Differentiable Logic Gate Networks

![difflogic_logo](difflogic_logo.png)

This repository includes the official implementation of our NeurIPS 2022 Paper "Deep Differentiable Logic Gate Networks"
(Paper @ [ArXiv](https://arxiv.org/abs/2210.08277)).

The goal behind differentiable logic gate networks is to solve machine learning tasks by learning combinations of logic
gates, i.e., so-called logic gate networks. As logic gate networks are conventionally non-differentiable, they can 
conventionally not be trained with methods such as gradient descent. Thus, differentiable logic gate networks are a 
differentiable relaxation of logic gate networks which allows efficiently learning of logic gate networks with gradient
descent. Specifically, `difflogic` combines real-valued logics and a continuously parameterized relaxation of
the network. This allows learning which logic gate (out of 16 possible) is optimal for each neuron.
The resulting discretized logic gate networks achieve fast inference speeds, e.g., beyond a million images
of MNIST per second on a single CPU core.

`difflogic` is a Python 3.6+ and PyTorch 1.9.0+ based library for training and inference with logic gate networks.
The library can be installed with:
```shell
pip install difflogic
```
> ⚠️ Note that `difflogic` requires CUDA, the CUDA Toolkit (for compilation), and `torch>=1.9.0` (matching the CUDA version).

For additional installation support, see [INSTALLATION_SUPPORT.md](INSTALLATION_SUPPORT.md).

## 🌱 Intro and Training

This library provides a framework for both training and inference with logic gate networks.
The following gives an example of a definition of a differentiable logic network model for the MNIST data set:

```python
from difflogic import LogicLayer, GroupSum
import torch

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LogicLayer(784, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    GroupSum(k=10, tau=30)
)
```

This model receives a `784` dimensional input and returns `k=10` values corresponding to the 10 classes of MNIST.
The model may be trained, e.g., with a `torch.nn.CrossEntropyLoss` similar to how other neural networks models are trained in PyTorch.
Notably, the Adam optimizer (`torch.optim.Adam`) should be used for training and the recommended default learning rate is `0.01` instead of `0.001`.
Finally, it is also important to note that the number of neurons in each layer is much higher for logic gate networks compared to 
conventional MLP neural networks because logic gate networks are very sparse.

To go into details, for each of these modules, in the following we provide more in-depth examples:

```python
layer = LogicLayer(
    in_dim=784,             # number of inputs
    out_dim=16_000,         # number of outputs
    device='cuda',          # the device (cuda / cpu)
    implementation='cuda',  # the implementation to be used (native cuda / vanilla pytorch)
    connections='random',   # the method for the random initialization of the connections
    grad_factor=1.1,        # for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
)
```

At this point, it is important to discuss the options for `device` and the provided implementations. Specifically,
`difflogic` provides two implementations (both of which work with PyTorch):

* **`python`** the Python implementation is a substantially slower implementation that is easy to understand as it is implemented directly in Python with PyTorch and does not require any C++ / CUDA extensions. It is compatible with `device='cpu'` and `device='cuda'`.
* **`cuda`** is a well-optimized implementation that runs natively on CUDA via custom extensions. This implementation is around 50 to 100 times faster than the python implementation (for large models). It only supports `device='cuda'`. 

To aggregate output neurons into a lower dimensional output space, we can use `GroupSum`, which aggregates a number of output neurons into 
a `k` dimensional output, e.g., `k=10` for a 10-dimensional classification setting. 
It is important to set the parameter `tau`, which the sum of neurons is divided by to keep the range reasonable.
As each neuron has a value between 0 and 1 (or in inference a value of 0 or 1), assuming `n` output neurons of the last `LogicLayer`,
the range of outputs is `[0, n / k / tau]`.

## 🖥 Model Inference

During training, the model should remain in the PyTorch training mode (`.train()`), which keeps the model differentiable.
However, we can easily switch the model to a hard / discrete / non-differentiable model by calling `model.eval()`, i.e., for inference.
Typically, this will simply discretize the model but not make it faster per se.

However, there are two modes that allow for fast inference:

### `PackBitsTensor`

The first option is to use a `PackBitsTensor`. 
`PackBitsTensor`s allow efficient dynamic execution of trained logic gate networks on GPU.

A `PackBitsTensor` can package a tensor (of shape `b x n`) with boolean 
data type in a way such that each boolean entry requires only a single bit (in contrast to the full byte typically
required by a bool) by packing the bits along the batch dimension. If we choose to pack the bits into the `int32` data 
type (the options are 8, 16, 32, and 64 bits), we would receive a tensor of shape `ceil(b/32) x n` of dtype `int32`.
To create a `PackBitsTensor` from a boolean tensor `data`, simply call:
```python
data_bits = difflogic.PackBitsTensor(data)
```
To apply a model to the `PackBitsTensor`, simply call:
```python
output = model(data_bits)
```
This requires that the `model` is in `.eval()` mode, and if supplied with a `PackBitsTensor`, will automatically use 
a logic gate-based inference on the tensor. This also requires that `model.implementation = 'cuda'` as the mode is only
implemented in CUDA.
It is notable that, while the model is in `.eval()` mode, we can still also feed float tensors through the model, in 
which case it will simply use a hard variant of the real-valued logics.

### `CompiledLogicNet`

The second option is to use a `CompiledLogicNet`.
This allows especially efficient static execution of a fixed trained logic gate network on CPU.
Specifically, `CompiledLogicNet` converts a model into efficient C code and can compile this code into a binary that
can then be efficiently run or exported for applications.
The following is an example for creating `CompiledLogicNet` from a trained `model`:

```python
compiled_model = difflogic.CompiledLogicNet(
    model=model,            # the trained model (should be a `torch.nn.Sequential` with `LogicLayer`s)
    num_bits=64,            # the number of bits of the datatype used for inference (typically 64 is fastest, should not be larger than batch size)
    cpu_compiler='gcc',     # the compiler to use for the c code (alternative: clang)
    verbose=True            
)
compiled_model.compile(
    save_lib_path='my_model_binary.so',  # the (optional) location for storing the binary such that it can be reused
    verbose=True
)

# to apply the model, we need a 2d numpy array of dtype bool, e.g., via  `data = data.bool().numpy()`
output = compiled_model(data)
```

This will compile a model into a shared object binary, which is then automatically imported.
To export this to other applications, one may either call the shared object binary from another program or export 
the model into C code via `compiled_model.get_c_code()`.
A limitation of the current `CompiledLogicNet` is that the compilation time can become long for large models.

We note that between publishing the paper and the publication of `difflogic`, we have substantially improved the implementations.
Thus, the model inference modes have some deviation from the implementations for the original paper as we have 
focussed on making it more scalable, efficient, and easier to apply in applications.
We have especially focussed on modularity and efficiency for larger models and have opted to polish the presented 
implementations over publishing a plethora of different competing implementations.

## 🧪 Experiments

In the following, we present a few example experiments which are contained in the `experiments` directory.
`main.py` executes the experiments for difflogic and `main_baseline.py` contains regular neural network baselines.

### ☄️ Adult / Breast Cancer

```shell
python experiments/main.py  -eid 526010           -bs 100 -t 20 --dataset adult         -ni 100_000 -ef 1_000 -k 256 -l 5 --compile_model
python experiments/main.py  -eid 526020 -lr 0.001 -bs 100 -t 20 --dataset breast_cancer -ni 100_000 -ef 1_000 -k 128 -l 5 --compile_model
```

### 🔢 MNIST

```shell
python experiments/main.py  -bs 100 -t  10 --dataset mnist20x20 -ni 200_000 -ef 1_000 -k  8_000 -l 6 --compile_model
python experiments/main.py  -bs 100 -t  30 --dataset mnist      -ni 200_000 -ef 1_000 -k 64_000 -l 6 --compile_model
# Baselines:
python experiments/main_baseline.py  -bs 100 --dataset mnist    -ni 200_000 -ef 1_000 -k  128 -l 3
python experiments/main_baseline.py  -bs 100 --dataset mnist    -ni 200_000 -ef 1_000 -k 2048 -l 7
```

### 🐶 CIFAR-10

```shell
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 200_000 -ef 1_000 -k    12_000 -l 4 --compile_model
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 200_000 -ef 1_000 -k   128_000 -l 4 --compile_model
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k   256_000 -l 5
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k   512_000 -l 5
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k 1_024_000 -l 5
```

## 📖 Citing

```bibtex
@inproceedings{petersen2022difflogic,
  title={{Deep Differentiable Logic Gate Networks}},
  author={Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## 📜 License

`difflogic` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it. 

Patent pending.
