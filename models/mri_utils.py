from typing import List, Optional

import numpy as np
import torch
import torch.fft


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()

class FFT_NN_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(FFT_NN_Wrapper, self).__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        im = ifft2c_new(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.model(im, *args, **kwargs)
        im_out = fft2c_new(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return im_out

try:
    import sigpy as sp
    class L1WaveletRecon(sp.app.App):
        def __init__(self, ksp, mask, mps, lamda, max_iter):
            img_shape = mps.shape[1:]

            S = sp.linop.Multiply(img_shape, mps)
            F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
            P = sp.linop.Multiply(ksp.shape, mask)
            self.W = sp.linop.Wavelet(img_shape)
            A = P * F * S * self.W.H

            proxg = sp.prox.L1Reg(A.ishape, lamda)

            self.wav = np.zeros(A.ishape, complex)
            alpha = 1

            def gradf(x):
                return A.H * (A * x - ksp)

            alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg,
                                        max_iter=max_iter)
            super().__init__(alg, show_pbar=False)

        def _output(self):
            return self.W.H(self.wav)

    def wavelet_restore(pinv_y_0, mask):
        visual_mask = torch.zeros_like(pinv_y_0).reshape(pinv_y_0.shape[0], -1)
        visual_mask[:, mask] = 1
        visual_mask = visual_mask.reshape(pinv_y_0.shape)
        ksp = torch.view_as_complex(fft2c_new(pinv_y_0.permute(0, 2, 3, 1))).cpu()
        x = L1WaveletRecon(ksp.numpy(), visual_mask[:, 0].bool().cpu().numpy(), torch.ones_like(ksp).numpy(), 0.0005,
                           max_iter=100).run()
        return torch.view_as_real(torch.from_numpy(x).unsqueeze(0)).permute(0, 3, 1, 2)

except ModuleNotFoundError as e:
    print(e)
