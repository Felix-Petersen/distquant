# Distributional Quantization

![distquant_logo](https://github.com/Felix-Petersen/distquant/blob/main/images/distquant_logo.png?raw=true)

**Distributional quantization provides an information theoretically optimal and efficient quantization of data following a distributional prior.**

This repository includes a [pre-print of an excerpt of the distributional quantization paper](preprint.pdf).

Simulation plots can be found [here](SIMULATIONS.md).

The `distquant` package requires PyTorch and can be installed via 
```shell
pip install distquant
```


## 💻 Usage

The `distquant` package can be used to efficiently and effectively quantize tensors (e.g., embeddings to be stored or weights of a neural network) using distributional quantization.
In particular, the API of the package provides two core functions:
* `distquant.quantize(data, k=k)` for quantizing a Tensor `data` of floating point values.
* `distquant.dequantize(quantized_data, **info)` for dequantizing `quantized_data`.

In detail, the API is specified as
```python
import torch
from distquant import quantize, dequantize

data = torch.randn(10_000)

q_data, info = quantize(
    data,
    k=4,              # number of bits to quantize to; alternatively specify `n`. 
    # n=15,           # number of quantization points; mutually exclusive to `k`.
    dist='gaussian',  # distributional assumption
    packbits=True,    # Packs the results into int64, e.g., for `k=4`, there will be 16 values in each element 
                      # for storing. If set to `False`, the indices are returned in raw format.
    beta=1.0,         # Factor hyperparameter for adjusting the range of values (esp. important for small `k`).
    zero=None         # Can enforce a particular zero (by setting it to a resp. float, e.g., `0.`)
)

deq_data = dequantize(data, **info)
```

* Bitpacking (`packbits=True`) allows storing the data efficiently, e.g., 21 3-bit values can be stored in a single int64.
* `beta` may be set `None` (default), which will cause an automatic search for the optimal choice. This can be more expensive but is valuable for smaller `k` / `n`.
* Setting `dist=None` searches for the best distribution to quantize the provided data.

Supported options for `dist`: 
```python
['gaussian' 'gaussian_cdf', 'logistic', 'logistic_cdf', 'gumbel', 'gumbel_cdf', 'exponential', 'exponential_cdf', 'cauchy_cdf']
```


## 📖 Citing

```bibtex
@article{petersen2023distributional,
  title={{Distributional Quantization}},
  author={Petersen, Felix and Sutter, Tobias},
  year={2021--2023}
}
```


## 📜 License

`distquant` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it. 

