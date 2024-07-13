"""
MRI Reconstruction model as described in `Zhang, Zizhao, et al. "Reducing uncertainty in
undersampled mri reconstruction with active acquisition." Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. 2019.`
"""
import abc
import functools
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from datasets.singlecoil_knee_data import MICCAI2020Data
from models.mri_utils import fftshift


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_func(m):
    init_type = "normal"
    gain = 0.02
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, gain)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_probability, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, dropout_probability, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, dropout_probability, use_bias
    ):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if dropout_probability > 0:
            conv_block += [nn.Dropout(dropout_probability)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ReconstructorNetwork(nn.Module):
    """Reconstructor network used in Zhang et al., CVPR'19.

    Args:
        number_of_encoder_input_channels(int): Number of input channels to the
                reconstruction model.
        number_of_decoder_output_channels(int): Number of output channels
                of the reconstruction model.
        number_of_filters(int): Number of convolutional filters.\n
        dropout_probability(float): Dropout probability.
        number_of_layers_residual_bottleneck (int): Number of residual
                blocks in each model between two consecutive down-
                or up-sampling operations.
        number_of_cascade_blocks (int): Number of times the entire architecture is
                replicated.
        mask_embed_dim(int): Dimensionality of the mask embedding.
        padding_type(str): Convolution operation padding type.
        n_downsampling(int): Number of down-sampling operations.
        img_width(int): The width of the image.
        use_deconv(binary): Whether to use deconvolution in the up-sampling.
    """

    def __init__(
        self,
        number_of_encoder_input_channels=2,
        number_of_decoder_output_channels=3,
        number_of_filters=128,
        dropout_probability=0.0,
        number_of_layers_residual_bottleneck=6,
        number_of_cascade_blocks=3,
        mask_embed_dim=6,
        padding_type="reflect",
        n_downsampling=3,
        img_width=128,
        use_deconv=True,
    ):
        super(ReconstructorNetwork, self).__init__()
        self.number_of_encoder_input_channels = number_of_encoder_input_channels
        self.number_of_decoder_output_channels = number_of_decoder_output_channels
        self.number_of_filters = number_of_filters
        self.use_deconv = use_deconv
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.number_of_cascade_blocks = number_of_cascade_blocks
        self.use_mask_embedding = True if mask_embed_dim > 0 else False

        if self.use_mask_embedding:
            number_of_encoder_input_channels += mask_embed_dim
            print("[Reconstructor Network] -> use masked embedding condition")

        # Lists of encoder, residual bottleneck and decoder blocks for all cascade blocks
        self.encoders_all_cascade_blocks = nn.ModuleList()
        self.residual_bottlenecks_all_cascade_blocks = nn.ModuleList()
        self.decoders_all_cascade_blocks = nn.ModuleList()

        # Architecture for the Cascade Blocks
        for iii in range(1, self.number_of_cascade_blocks + 1):

            # Encoder for iii_th cascade block
            encoder = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    number_of_encoder_input_channels,
                    number_of_filters,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=use_bias,
                ),
                norm_layer(number_of_filters),
                nn.ReLU(True),
            ]

            for i in range(1, n_downsampling):
                mult = 2 ** i
                encoder += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        number_of_filters * mult // 2,
                        number_of_filters * mult,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        bias=use_bias,
                    ),
                    norm_layer(number_of_filters * mult),
                    nn.ReLU(True),
                ]

            self.encoders_all_cascade_blocks.append(nn.Sequential(*encoder))

            # Bottleneck for iii_th cascade block
            residual_bottleneck = []
            mult = 2 ** (n_downsampling - 1)
            for i in range(number_of_layers_residual_bottleneck):
                residual_bottleneck += [
                    ResnetBlock(
                        number_of_filters * mult,
                        padding_type=padding_type,
                        norm_layer=norm_layer,
                        dropout_probability=dropout_probability,
                        use_bias=use_bias,
                    )
                ]

            self.residual_bottlenecks_all_cascade_blocks.append(
                nn.Sequential(*residual_bottleneck)
            )

            # Decoder for iii_th cascade block
            decoder = []
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - 1 - i)
                if self.use_deconv:
                    decoder += [
                        nn.ConvTranspose2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
                else:
                    decoder += [nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1)] + [
                        nn.Conv2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
            decoder += [
                nn.Conv2d(
                    number_of_filters // 2,
                    number_of_decoder_output_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            ]  # better

            self.decoders_all_cascade_blocks.append(nn.Sequential(*decoder))

        if self.use_mask_embedding:
            self.mask_embedding_layer = nn.Sequential(
                nn.Conv2d(img_width, mask_embed_dim, 1, 1)
            )

        self.apply(init_func)

    def data_consistency(self, x, input, mask):
        ksp = torch.view_as_real(
                torch.fft.fftn(  # type: ignore
                    torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()), dim=(-2, -1), norm="backward"
                )
            ).permute(0, 3, 1, 2).contiguous()
        masked = torch.where((1 - mask).bool(), ksp, torch.tensor(0.0).to(ksp.device))
        fuse = (
            torch.view_as_real(
                torch.fft.ifftn(  # type: ignore
                    torch.view_as_complex(masked.permute(0, 2, 3, 1).contiguous()), dim=(-2, -1), norm="backward"
                )
            ).permute(0, 3, 1, 2).contiguous()
            + input
        )
        return fuse

    def embed_mask(self, mask):
        b, c, h, w = mask.shape
        mask = mask.view(b, w, 1, 1)
        cond_embed = self.mask_embedding_layer(mask)
        return cond_embed

    # noinspection PyUnboundLocalVariable
    def forward(self, zero_filled_input, mask):
        """Generates reconstructions given images with partial k-space info.

        Args:
            zero_filled_input(torch.Tensor): Image obtained from zero-filled reconstruction
                of partial k-space scans.
            mask(torch.Tensor): Mask used in creating the zero filled image from ground truth
                image.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): Contains:\n
                * Reconstructed high resolution image.
                * Uncertainty map.
                * Mask_embedding.
        """
        if self.use_mask_embedding:
            mask_embedding = self.embed_mask(mask)
            mask_embedding = mask_embedding.repeat(
                1, 1, zero_filled_input.shape[2], zero_filled_input.shape[3]
            )
            encoder_input = torch.cat([zero_filled_input, mask_embedding], 1)
        else:
            encoder_input = zero_filled_input
            mask_embedding = None

        residual_bottleneck_output = None
        for cascade_block, (encoder, residual_bottleneck, decoder) in enumerate(
            zip(
                self.encoders_all_cascade_blocks,
                self.residual_bottlenecks_all_cascade_blocks,
                self.decoders_all_cascade_blocks,
            )
        ):
            encoder_output = encoder(encoder_input)
            if cascade_block > 0:
                # Skip connection from previous residual block
                encoder_output = encoder_output + residual_bottleneck_output

            residual_bottleneck_output = residual_bottleneck(encoder_output)

            decoder_output = decoder(residual_bottleneck_output)

            reconstructed_image = self.data_consistency(
                decoder_output[:, :-1, ...], zero_filled_input, mask
            )
            uncertainty_map = decoder_output[:, -1:, :, :]

            if self.use_mask_embedding:
                encoder_input = torch.cat([reconstructed_image, mask_embedding], 1)
            else:
                encoder_input = reconstructed_image

        return reconstructed_image, uncertainty_map, mask_embedding

    def init_from_checkpoint(self, checkpoint):

        if not isinstance(self, nn.DataParallel):
            self.load_state_dict(
                {
                    # This assumes that environment code runs in a single GPU
                    key.replace("module.", ""): val
                    for key, val in checkpoint["reconstructor"].items()
                }
            )
        else:
            self.load_state_dict(checkpoint["reconstructor"])
        return checkpoint["options"]

class Reconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def init_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Optional[Any]:
        pass

class CVPR19Reconstructor(Reconstructor):
    def __init__(
        self,
        number_of_encoder_input_channels: int = 2,
        number_of_decoder_output_channels: int = 3,
        number_of_filters: int = 128,
        dropout_probability: float = 0.0,
        number_of_layers_residual_bottleneck: int = 6,
        number_of_cascade_blocks: int = 3,
        mask_embed_dim: int = 6,
        padding_type: str = "reflect",
        n_downsampling: int = 3,
        img_width: int = 128,
        use_deconv: bool = True,
    ):
        super().__init__()
        self.reconstructor = ReconstructorNetwork(
            number_of_encoder_input_channels=number_of_encoder_input_channels,
            number_of_decoder_output_channels=number_of_decoder_output_channels,
            number_of_filters=number_of_filters,
            dropout_probability=dropout_probability,
            number_of_layers_residual_bottleneck=number_of_layers_residual_bottleneck,
            number_of_cascade_blocks=number_of_cascade_blocks,
            mask_embed_dim=mask_embed_dim,
            padding_type=padding_type,
            n_downsampling=n_downsampling,
            img_width=img_width,
            use_deconv=use_deconv,
        )

    def forward(  # type: ignore
        self, zero_filled_input: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, Any]:
        reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
            zero_filled_input, mask
        )
        reconstructed_image = reconstructed_image.permute(0, 2, 3, 1)
        uncertainty_map = uncertainty_map.permute(0, 2, 3, 1)

        return {
            "reconstruction": reconstructed_image,
            "uncertainty_map": uncertainty_map,
            "mask_embedding": mask_embedding,
        }

    def init_from_checkpoint(self, checkpoint: Dict[str, Any]):
        return self.reconstructor.init_from_checkpoint(checkpoint)


def raw_transform_miccai2020(kspace=None, mask=None, **_kwargs):
    """Transform to produce input for reconstructor used in `Pineda et al. MICCAI'20 <https://arxiv.org/pdf/2007.10469.pdf>`_.

    Produces a zero-filled reconstruction and a mask that serve as a input to models of type
    :class:`activemri.models.cvpr10_reconstructor.CVPR19Reconstructor`. The mask is almost
    equal to the mask passed as argument, except that high-frequency padding columns are set
    to 1, and the mask is reshaped to be compatible with the reconstructor.

    Args:
        kspace(``np.ndarray``): The array containing the k-space data returned by the dataset.
        mask(``torch.Tensor``): The masks to apply to the k-space.

    Returns:
        tuple: A tuple containing:
            - ``torch.Tensor``: The zero-filled reconstructor that will be passed to the
              reconstructor.
            - ``torch.Tensor``: The mask to use as input to the reconstructor.
    """
    # alter mask to always include the highest frequencies that include padding
    mask[
        :,
        :,
        166 : 202,
    ] = 1
    mask = mask.unsqueeze(1)

    all_kspace = []
    for ksp in kspace:
        all_kspace.append(torch.from_numpy(ksp).permute(2, 0, 1))
    k_space = torch.stack(all_kspace)

    masked_true_k_space = torch.where(
        mask.bool(),
        k_space,
        torch.tensor(0.0).to(mask.device),
    )

    reconstructor_input = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(masked_true_k_space.permute(0, 2, 3, 1).contiguous()), dim=(-2, -1), norm="backward"
        )
    )
    reconstructor_input = fftshift(reconstructor_input, dim=[-3, -2]).permute(0, 3, 1, 2)
    return reconstructor_input, mask

def get_reconstructor():
    rec = CVPR19Reconstructor(**{'number_of_filters': 256,
                                 'dropout_probability': 0.0000001,
                                 'number_of_layers_residual_bottleneck': 3,
                                 'number_of_cascade_blocks': 4,
                                 'mask_embed_dim': 0,
                                 'n_downsampling': 3,
                                 'img_width': 368,
                                 'use_deconv': True})
    print(rec.load_state_dict(torch.load("checkpoints/reconstructor.pth", map_location='cpu')))
    return rec.eval()

def get_reconstructor_extreme():
    rec = CVPR19Reconstructor(**{'number_of_filters': 256,
                                 'dropout_probability': 0.0000001,
                                 'number_of_layers_residual_bottleneck': 3,
                                 'number_of_cascade_blocks': 4,
                                 'mask_embed_dim': 0,
                                 'n_downsampling': 3,
                                 'img_width': 368,
                                 'use_deconv': True})
    print(rec.load_state_dict(torch.load("checkpoints/reconstructor_extreme.pth", map_location='cpu')))
    return rec.eval()

def restore(rec_model, ksp, all_indices):
    current_mask = torch.zeros((ksp.shape[0], 1, ksp.shape[-1]))
    current_mask[:, :, all_indices] = 1
    inputs = raw_transform_miccai2020(kspace=[ksp.permute(0, 2, 3, 1)[0].cpu().numpy()],
                                      mask=torch.roll(current_mask, 184, -1))
    with torch.no_grad():
        output = rec_model(inputs[0].to(ksp.device), inputs[1].to(ksp.device))
    norm_rec = output['reconstruction'].permute(0, 3, 1, 2) / MICCAI2020Data.NORMALIZING_FACTOR
    return norm_rec


