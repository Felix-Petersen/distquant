import torch
import math
import numpy as np


gaussian    = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
exponential = torch.distributions.exponential.Exponential(torch.tensor(1.0))
gumbel      = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))
cauchy      = torch.distributions.cauchy.Cauchy(torch.tensor(0.0), torch.tensor(1.0))
euler_mascheroni_constant = 0.5772156649

BETA_GRID_LARGE = sorted(np.logspace(-1, -.1, 10).round(3).tolist() + np.logspace(-.07, .07, 15).round(3).tolist() + [0.99, 1.01] + np.logspace(.1, 1., 10).round(3).tolist())
BETA_GRID_CAUCHY = np.logspace(-3, 3, 31).round(3).tolist()


def encode_gaussian(tensor, n, beta=1, p=1, zero=None):
    if zero is None:
        mean = tensor.mean().item()
    else:
        mean = zero
    std = tensor.std().item()
    tensor = (tensor - mean) / std
    if p == 1:
        tensor = gaussian.cdf(tensor / math.sqrt(2) / beta)
    elif p is None:
        tensor = gaussian.cdf(tensor / beta)
    else:
        raise ValueError(p)
    tensor = tensor * n
    tensor = tensor.floor().long()
    return tensor, std, mean


def decode_gaussian(tensor, std, mean, n, beta=1, p=1):
    tensor = (tensor.float() + .5) / n
    if p == 1:
        tensor = gaussian.icdf(tensor) * math.sqrt(2) * beta
    elif p is None:
        tensor = gaussian.icdf(tensor) * beta
    else:
        raise ValueError(p)
    tensor = (tensor * std) + mean
    return tensor


def encode_exponential(tensor, n, beta=1, p=1, zero=None):
    loc = zero
    std = tensor.std().item()
    tensor = (tensor - loc) / std
    if p == 1:
        tensor = exponential.cdf((torch.relu(tensor - 1e-8) + 1e-8) / 2 / beta)
    elif p is None:
        tensor = exponential.cdf((torch.relu(tensor - 1e-8) + 1e-8) / beta)
    else:
        raise ValueError(p)
    tensor = tensor * n
    tensor = tensor.floor().long()
    return tensor, std, loc


def decode_exponential(tensor, std, loc, n, beta=1, p=1):
    tensor = (tensor.float() + .5) / n
    if p == 1:
        tensor = exponential.icdf(tensor) * 2 * beta
    elif p is None:
        tensor = exponential.icdf(tensor) * beta
    else:
        raise ValueError(p)
    tensor = (tensor * std) + loc
    return tensor


def encode_logistic(tensor, n, beta=1, p=1, zero=None):
    if zero is None:
        mean = tensor.mean().item()
    else:
        mean = zero
    std = (tensor.std() / math.pi * math.sqrt(3)).item()
    tensor = (tensor - mean) / std
    if p == 1:
        tensor = 2 * torch.arctan(torch.exp(tensor / beta / 2)) / math.pi
    elif p is None:
        tensor = torch.sigmoid(tensor / beta)
    else:
        raise ValueError(p)
    tensor = tensor * n
    tensor = tensor.floor().long()
    return tensor, std, mean


def decode_logistic(tensor, std, mean, n, beta=1, p=1):
    tensor = (tensor.float() + .5) / n
    if p == 1:
        tensor = torch.log(torch.tan(tensor / 2 * math.pi)) * 2 * beta
    elif p is None:
        tensor = torch.logit(tensor) * beta
    else:
        raise ValueError(p)
    tensor = (tensor * std) + mean
    return tensor


def encode_gumbel(tensor, n, beta=1, p=1, zero=None):
    scale = tensor.std() * math.sqrt(6) / math.pi
    if zero is None:
        loc = (tensor.mean() - euler_mascheroni_constant * scale).item()
    else:
        loc = zero
    tensor = (tensor - loc) / scale
    if p == 1:
        tensor = 1 - torch.erf(torch.exp(- tensor / 2 / beta) / math.sqrt(2) )
    elif p is None:
        tensor = gumbel.cdf((tensor / beta).clamp(min=-4.46975, max=15.9424))
    else:
        raise ValueError(p)
    tensor = tensor * n
    tensor = tensor.floor().long()
    return tensor, scale, loc


def decode_gumbel(tensor, scale, loc, n, beta=1, p=1):
    tensor = (tensor.float() + .5) / n
    if p == 1:
        tensor = torch.log(1 / (2 * torch.special.erfinv(1 - tensor)**2)) * beta
    elif p is None:
        tensor = gumbel.icdf(tensor) * beta
    else:
        raise ValueError(p)
    tensor = (tensor * scale) + loc
    return tensor


def encode_cauchy(tensor, n, beta=1, zero=None):
    if zero is None:
        loc = 0.
    else:
        loc = zero
    scale = 1
    tensor = (tensor - loc) / scale
    tensor = cauchy.cdf(tensor / beta)
    tensor = tensor * n
    tensor = tensor.floor().long()
    return tensor, scale, loc


def decode_cauchy(tensor, scale, loc, n, beta=1):
    tensor = (tensor.float() + .5) / n
    tensor = cauchy.icdf(tensor) * beta
    tensor = (tensor * scale) + loc
    return tensor


########################################################################################################################


def encode(dist, tensor, n, beta, zero):
    if dist == 'gaussian':
        return encode_gaussian(tensor, n, beta=beta, p=1, zero=zero)
    elif dist == 'gaussian_cdf':
        return encode_gaussian(tensor, n, beta=beta, p=None, zero=zero)
    elif dist == 'logistic':
        return encode_logistic(tensor, n, beta=beta, p=1, zero=zero)
    elif dist == 'logistic_cdf':
        return encode_logistic(tensor, n, beta=beta, p=None, zero=zero)
    elif dist == 'cauchy_cdf':
        return encode_cauchy(tensor, n, beta=beta, zero=zero)
    elif dist == 'gumbel':
        return encode_gumbel(tensor, n, beta=beta, p=1, zero=zero)
    elif dist == 'gumbel_cdf':
        return encode_gumbel(tensor, n, beta=beta, p=None, zero=zero)
    elif dist == 'exponential':
        return encode_exponential(tensor, n, beta=beta, p=1, zero=zero)
    elif dist == 'exponential_cdf':
        return encode_exponential(tensor, n, beta=beta, p=None, zero=zero)
    else:
        raise ValueError(dist)


def decode(dist, tensor, scale, loc, n, beta):
    if dist == 'gaussian':
        return decode_gaussian(tensor, scale, loc, n, beta=beta, p=1)
    elif dist == 'gaussian_cdf':
        return decode_gaussian(tensor, scale, loc, n, beta=beta, p=None)
    elif dist == 'logistic':
        return decode_logistic(tensor, scale, loc, n, beta=beta, p=1)
    elif dist == 'logistic_cdf':
        return decode_logistic(tensor, scale, loc, n, beta=beta, p=None)
    elif dist == 'cauchy_cdf':
        return decode_cauchy(tensor, scale, loc, n, beta=beta)
    elif dist == 'gumbel':
        return decode_gumbel(tensor, scale, loc, n, beta=beta, p=1)
    elif dist == 'gumbel_cdf':
        return decode_gumbel(tensor, scale, loc, n, beta=beta, p=None)
    elif dist == 'exponential':
        return decode_exponential(tensor, scale, loc, n, beta=beta, p=1)
    elif dist == 'exponential_cdf':
        return decode_exponential(tensor, scale, loc, n, beta=beta, p=None)
    else:
        raise ValueError(dist)


########################################################################################################################


def quantize(
        tensor,
        k=None,
        n=None,
        dist='gaussian',
        packbits=True,
        beta=None,
        zero=None,
):
    if dist is None:
        return auto_quantize(tensor, k=k, n=n, packbits=packbits, zero=zero)

    assert n is not None or k is not None, '`k` bits or `n` values needs to be specified.'
    assert not(n is not None and k is not None), 'only one of `k` bits or `n` values can be specified.'
    assert dist in [
        'gaussian', 'gaussian_cdf', 'logistic', 'logistic_cdf', 'cauchy_cdf',
        'gumbel', 'gumbel_cdf', 'exponential', 'exponential_cdf'
    ], dist

    shape = list(tensor.shape)
    tensor = tensor.view(-1)

    if n is None:
        n = 2**k

    if dist in ['exponential', 'exponential_cdf']:
        if zero is None:
            zero = 0.
    else:
        if zero is not None:
            assert n % 2 == 1, '`n` must be odd to account for the zero.'

    if beta is None:
        best_error = 1e10
        best_beta = 1.
        for beta_ in BETA_GRID_LARGE if dist != 'cauchy_cdf' else BETA_GRID_CAUCHY:
            q_tensor_, scale, loc = encode(dist, tensor, n, beta=beta_, zero=zero)
            abs_error_ = (decode(dist, q_tensor_, scale, loc, n, beta_) - tensor).abs().mean()
            if abs_error_ < best_error:
                best_error = abs_error_
                best_beta = beta_
        beta = best_beta

    q_tensor, scale, loc = encode(dist, tensor, n, beta=beta, zero=zero)
    abs_error = (decode(dist, q_tensor, scale, loc, n, beta) - tensor).abs().mean().item()
    info = {
        'dist': dist,
        'scale': scale,
        'loc': loc,
        'n': n,
        'packbits': packbits,
        'beta': beta,
        'abs_error': abs_error,
        'shape': shape,
        'n_elems': q_tensor.shape[0],
    }

    if packbits:
        if k is None:
            k = math.ceil(math.log2(n))
        num_per_uint64 = 64 // k
        q_tensor = torch.cat([q_tensor, torch.zeros((num_per_uint64 - (q_tensor.shape[0] % num_per_uint64)) % num_per_uint64, device=q_tensor.device, dtype=torch.int64)])
        q_tensor = q_tensor.reshape(-1, num_per_uint64)
        packed_tensor = torch.zeros(q_tensor.shape[0], device=q_tensor.device, dtype=torch.int64)
        for i in range(num_per_uint64):
            packed_tensor <<= k
            packed_tensor |= q_tensor[:, i]
        q_tensor = packed_tensor
    else:
        q_tensor = q_tensor.view(*shape)

    return q_tensor, info



########################################################################################################################


def auto_quantize(tensor, k=None, n=None, packbits=True, zero=None):
    best_dist = None
    best_error = 1e10

    for dist in [
        'gaussian', 'gaussian_cdf', 'logistic', 'logistic_cdf', 'cauchy_cdf',
        'gumbel', 'gumbel_cdf', 'exponential', 'exponential_cdf'
    ]:
        _, info = quantize(tensor, k=k, n=n, dist=dist, packbits=False, zero=zero)
        if info['abs_error'] < best_error:
            best_error = info['abs_error']
            best_dist = dist

    return quantize(tensor, k=k, n=n, dist=best_dist, packbits=packbits, zero=zero)


########################################################################################################################


def dequantize(q_tensor, **info):
    q_tensor = q_tensor.view(-1)

    if info['packbits']:
        k = math.ceil(math.log2(info['n']))
        num_per_uint64 = 64 // k
        unpacked_tensor = torch.zeros(q_tensor.shape[0], num_per_uint64, device=q_tensor.device, dtype=torch.int64)
        mask_tensor = torch.zeros(1, device=q_tensor.device, dtype=torch.int64) | (2**k - 1)
        for i in range(num_per_uint64-1, -1, -1):
            unpacked_tensor[:, i] = q_tensor & mask_tensor
            q_tensor >>= k
        q_tensor = unpacked_tensor.view(-1)[:info['n_elems']]

    deq_tensor = decode(info['dist'], q_tensor, info['scale'], info['loc'], info['n'], info['beta'])

    deq_tensor = deq_tensor.view(*info['shape'])

    return deq_tensor


########################################################################################################################


if __name__ == '__main__':
    torch.manual_seed(0)
    data = torch.randn(10_000)

    print('distquant-Tests:')
    print('\n1) validity of packbits')
    q_data, info = quantize(data, n=33, packbits=True, zero=0.)
    print(info, q_data.shape)
    print((data - dequantize(q_data, **info)).abs().mean())

    print('\n2) randn + rand')
    data2 = data + 2 * torch.rand_like(data)
    print(quantize(data2, n=15, zero=0)[1])
    print(auto_quantize(data2, n=15, zero=0)[1])

    print('\n3) relu')
    data3 = torch.relu(data2)
    print(quantize(data3, n=15, zero=0)[1])
    print(auto_quantize(data3, n=15, zero=0)[1])

    print('\n3) cauchy')
    data4 = cauchy.rsample(data.shape)
    print(quantize(data4, n=15, zero=0)[1])
    print(auto_quantize(data4, n=15, zero=0)[1])

