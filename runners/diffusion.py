import os

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform, center_crop
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random
try:
    import skimage
except:
    pass

def show(x):
    import matplotlib.pyplot as plt
    plt.imshow((0.5 + 0.5*x[0]).detach().cpu().permute(1, 2, 0).numpy())
    plt.show()

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            elif self.config.data.dataset == 'MRI':
                name = 'fmri'
            elif self.config.data.dataset == 'CT':
                name = 'ct'
            else:
                raise ValueError
            if name not in ['celeba_hq', 'celeba_64', 'kmri', 'fmri', 'ct']:
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs", "celeba", "celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            elif name == 'fmri':
                ckpt = os.path.join(self.args.exp, "checkpoint", "fmri.ckpt")
            elif name == 'ct':
                ckpt = os.path.join(self.args.exp, "checkpoint", "CT_ema.pth")
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size[0], self.config.data.image_size[1]))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size[0], self.config.data.image_size[1]), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size[0], self.config.data.image_size[1]))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        ## get degradation matrix ##
        startmask = None
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size[1], compress_by, torch.randperm(self.config.data.image_size[0] * self.config.data.image_size[1], device=self.device), self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                # missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 - 64].to(self.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size[1], missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size[1], self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size[1], self.device, stride = factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size[1], self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size[1], self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size[1], self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size[1], blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'adasense':
            if self.config.data.dataset == 'MRI':
                fmri_crop_mask = torch.zeros((1, 640, 368, 2)).to(self.device)
                # adding frequencies outside the valid range
                fmri_crop_mask[:, 160:360 + 160, 24:344, :] = 1
                from functions.svd_replacement import MRIInpainting
                if config.data.vertical_mask:
                    if config.data.L30:
                        all_indices = torch.cat([torch.arange(18).long(), torch.arange(169, 169+30).long(), torch.arange(368 - 18, 368).long()])
                    elif config.data.L2:
                        all_indices = torch.cat([torch.arange(18).long(), torch.arange(183, 185).long(), torch.arange(368 - 18, 368).long()])
                    else:
                        all_indices = torch.cat([torch.arange(18).long(), torch.arange(368 - 18, 368).long()])

                    startmask = all_indices
                    mask = torch.cat([all_indices + l * config.data.image_size[1] +
                                      k * config.data.image_size[0] * config.data.image_size[1] for l in
                                     range(config.data.image_size[0])
                                     for k in range(config.data.channels)], dim=0)
                else:
                    all_indices = torch.cat([torch.arange(18).long(), torch.arange(368 - 18, 368).long()])
                    all_indices = torch.cat([all_indices + l * config.data.image_size[1] for l in
                                      range(config.data.image_size[0])], dim=0)
                    startmask = all_indices
                    mask = torch.cat([all_indices + k * config.data.image_size[0] * config.data.image_size[1]
                                      for k in range(config.data.channels)], dim=0)
                H_funcs = MRIInpainting(mask, self.device,
                                        (config.data.channels, config.data.image_size[0], config.data.image_size[1]))
            elif self.config.data.dataset == 'CT':
                from torch_radon import ParallelBeam
                radon = ParallelBeam(256, np.linspace(0, np.pi, 256, endpoint=False))
                normalizing_matrix = torch.from_numpy(np.load(os.path.join(f"CT_radon", "normalizing.npy")))
                from functions.svd_replacement import CTRows
                from runners.create_circle import draw_circle
                all_indices = torch.zeros((0,)).long()
                H_funcs = CTRows(all_indices, self.device,
                                 (config.data.channels, config.data.image_size[0], config.data.image_size[1]))
            else:
                from functions.svd_replacement import GeneralH
                vecs = torch.randn((0, config.data.image_size[0] * config.data.image_size[1] * config.data.channels))
                H_funcs = GeneralH(vecs, self.device)
        elif "general-h" in deg:
            from functions.svd_replacement import GeneralH
            vecs = torch.from_numpy(np.load(args.h_path)).to(self.device)
            H_funcs = GeneralH(vecs, self.device)
        elif "mri-sub-indices" in deg:
            from functions.svd_replacement import MRIInpainting
            loaded = np.load(args.h_path)
            all_indices = torch.from_numpy(loaded).reshape(-1).nonzero().squeeze(1).long().to(self.device)
            mask = torch.cat([all_indices + k * config.data.image_size[0] * config.data.image_size[1]
                              for k in range(config.data.channels)], dim=0)
            H_funcs = MRIInpainting(mask, self.device,
                                    (config.data.channels, config.data.image_size[0], config.data.image_size[1]))

        elif "ct-sub-rand" in deg:
            number = int(deg[len('ct-sub-rand'):])
            from functions.svd_replacement import CTRows
            from runners.create_circle import draw_circle
            all_indices = torch.multinomial(torch.ones((256,)), number).long()
            H_funcs = CTRows(all_indices, self.device,
                             (config.data.channels, config.data.image_size[0], config.data.image_size[1]))

        elif "ct-sub-equispaced" in deg:
            number = int(deg[len('ct-sub-equispaced'):])
            from functions.svd_replacement import CTRows
            from runners.create_circle import draw_circle
            all_indices = torch.from_numpy(np.around(np.arange(torch.randint(256 // number, (1,)).item(), 256, 256 / number))).int()
            assert (len(all_indices) == number)
            H_funcs = CTRows(all_indices, self.device,
                             (config.data.channels, config.data.image_size[0], config.data.image_size[1]))

        else:
            print("ERROR: degradation type not supported")
            quit()

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_psnr_of_mean = 0.0
        ssim = 0.0
        ssim_of_mean = 0.0
        if args.wavelet:
            wavelet_psnr = 0.0
            wavelet_ssim = 0.0
        if args.rec:
            from models.reconstruction import get_reconstructor, get_reconstructor_extreme, restore
            rec_getter = get_reconstructor_extreme if config.data.L2 else get_reconstructor
            rec_model = rec_getter().to(self.device)
            rec_psnr = 0.0
            rec_ssim = 0.0
        pbar = tqdm.tqdm(val_loader)
        h, w, ch = config.data.image_size[0], config.data.image_size[1], config.data.channels
        num_samples = args.samples * (args.stack_steps if args.stack_steps else 1)
        for x_orig, ksp in pbar:
            samples = args.samples
            timesteps = args.timesteps
            if deg == "adasense":
                reconstructions = []
                if self.config.data.dataset == 'MRI':
                    from functions.svd_replacement import MRIInpainting
                    if config.data.vertical_mask:
                        if startmask is not None:
                            all_indices = startmask
                        else:
                            all_indices = torch.cat([torch.arange(18).long(), torch.arange(368 - 18, 368).long()])
                        mask = torch.cat(
                            [all_indices + l * w + k * h * w for l in
                             range(h)
                             for k in range(config.data.channels)], dim=0)
                    else:
                        if startmask is not None:
                            all_indices = startmask
                        else:
                            all_indices = torch.cat([torch.arange(18).long(), torch.arange(368 - 18, 368).long()])
                            all_indices = torch.cat([all_indices + l * config.data.image_size[1] for l in
                                                     range(config.data.image_size[0])], dim=0)
                        mask = torch.cat([all_indices + k * config.data.image_size[0] * config.data.image_size[1]
                                          for k in range(config.data.channels)], dim=0)
                    H_funcs = MRIInpainting(mask, self.device,
                                            (
                                            config.data.channels, config.data.image_size[0], config.data.image_size[1]))
                elif self.config.data.dataset == 'CT':
                    all_indices = torch.zeros((0,)).long()
                    H_funcs = CTRows(all_indices, self.device,
                                     (config.data.channels, config.data.image_size[0], config.data.image_size[1]))
                else:
                    vecs = torch.randn((0, h * w * ch))
                    H_funcs = GeneralH(vecs, self.device)
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)
            if self.config.data.dataset == 'MRI':
                ksp = ksp.to(self.device)
                from models.mri_utils import complex_abs
                x_ = center_crop(complex_abs(x_orig.permute(0, 2, 3, 1)), (320, 320))
                _min, _max = 0, x_.max().item()
            elif self.config.data.dataset == 'CT':
                from datasets import DeepLesion
                x_ = (x_orig * DeepLesion.STD) + DeepLesion.Mean
                _min, _max = x_.min().item(), x_.max().item()
                classes = classes[0]
            else:
                _min, _max = None, None
                classes = classes[0]

            for i in tqdm.tqdm(range(args.iters + 1), leave=False):
                y_0 = H_funcs.H(x_orig)

                pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], ch, h, w)
                if deg[:6] == 'deblur': pinv_y_0 = y_0.view(y_0.shape[0], ch, h, w)
                elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, h, w).repeat(1, 3, 1, 1)
                elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

                for k in range(len(pinv_y_0)):
                    tvu.save_image(
                        inverse_data_transform(config, pinv_y_0[k], _min, _max), os.path.join(self.args.image_folder, f"y0_{idx_so_far + k}.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[k], _min, _max), os.path.join(self.args.image_folder, f"orig_{idx_so_far + k}.png")
                    )
                    if self.config.data.dataset == 'MRI':
                        visual_mask = torch.zeros_like(x_orig).reshape(x_orig.shape[0], -1)
                        visual_mask[:, mask] = 1
                        visual_mask = visual_mask.reshape(x_orig.shape)
                        tvu.save_image(
                            visual_mask[k],
                            os.path.join(self.args.image_folder, f"mask_{idx_so_far + k}.png")
                        )
                    if self.config.data.dataset == 'CT':
                        tvu.save_image(
                            torch.from_numpy(draw_circle(H_funcs.angles.cpu().numpy()))[:, :, 0] / 255,
                            os.path.join(self.args.image_folder, f"mask_{idx_so_far + k}.png")
                        )

                if i == args.iters:
                    samples = args.uncertainty if args.uncertainty is not None else samples

                ##Begin DDIM
                x = torch.randn(
                    y_0.shape[0] * samples,
                    ch,
                    h,
                    w,
                    device=self.device,
                    )

                with torch.no_grad():
                    x, _ = self.sample_image(x, model, H_funcs, y_0.repeat([samples, 1]), timesteps=timesteps, last=False, classes=classes.repeat([samples, 1]))
                    x = x[-1]
                    reconstructions = [x]
                    if deg == "adasense":
                        if i < args.iters:
                            assert y_0.shape[0] == 1
                            if self.config.data.dataset == 'MRI':
                                from models.mri_utils import fft2c_new, complex_abs
                                if args.abs_crop:
                                    x_flat = complex_abs(fmri_crop_mask * x.to(self.device).permute(0, 2, 3, 1))
                                    x_flat = fft2c_new(torch.stack([x_flat, torch.zeros_like(x_flat)], -1))
                                else:
                                    x_flat = fft2c_new(x.to(self.device).permute(0, 2, 3, 1))
                                x_centered = x_flat - x_flat.mean(0, keepdim=True)
                                if config.data.vertical_mask:
                                    x_averaged = x_centered.pow(2).mean((0, 1, 3)).reshape(-1)
                                    x_averaged[all_indices] = -1
                                    indices = torch.topk(x_averaged, k=args.rank, dim=-1).indices.cpu()
                                    all_indices = torch.cat([all_indices, indices], dim=0)
                                    mask = torch.cat([all_indices + l * w + k * h * w for l in range(h)
                                                      for k in range(ch)], dim=0)
                                else:
                                    x_averaged = x_centered.pow(2).mean((0, 3)).reshape(-1)
                                    x_averaged[all_indices] = 0
                                    indices = torch.topk(x_averaged, k=args.rank, dim=-1).indices.cpu()
                                    all_indices = torch.cat([all_indices, indices], dim=0)
                                    mask = torch.cat([all_indices, all_indices + h * w], dim=0)
                                H_funcs = MRIInpainting(mask, self.device, (ch, h, w))
                            elif self.config.data.dataset == 'CT':
                                x_radon = radon.forward(x.to(self.device)).detach().cpu()
                                x_centered = x_radon - x_radon.mean(0, keepdim=True)
                                x_averaged = (x_centered * (normalizing_matrix[None, None, :, :, :] @ x_centered.unsqueeze(-1)).squeeze(-1)).mean((0, 3)).reshape(-1)
                                x_averaged[H_funcs.angles] = 0
                                angles = torch.topk(x_averaged, k=args.rank, dim=-1).indices
                                H_funcs.add_rows(angles)
                            else:
                                x_flat = x.reshape(num_samples, -1)
                                x_centered = x_flat - x_flat.mean(0, keepdim=True)
                                new_vectors = torch.linalg.svd(x_centered, full_matrices=False)[-1][:args.rank]
                                vecs = torch.cat([vecs, new_vectors], dim=0)
                                H_funcs = GeneralH(vecs, device=self.device)



            reconstructions = [inverse_data_transform(config, y, _min, _max) for y in reconstructions]

            for i in [-1]: #range(len(x)):
                for j in range(reconstructions[i].size(0) // samples):
                    tvu.save_image(
                        reconstructions[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(reconstructions)-1 or i == -1:
                        if args.mean is not None and args.mean:
                            tvu.save_image(
                                inverse_data_transform(config, x.mean(0), _min, _max),
                                os.path.join(self.args.image_folder, f"mean_{idx_so_far + j}.png")
                            )
                        if args.uncertainty is not None:
                            tvu.save_image(
                                inverse_data_transform(config, ((x - x.mean(0)).pow(2).mean(0).sqrt()), _min, _max), os.path.join(self.args.image_folder, f"uncertainty_{idx_so_far + j}.png")
                            )
                        if self.config.data.dataset == 'MRI':
                            gt = center_crop(complex_abs((x_orig[j].unsqueeze(0).permute(0, 2, 3, 1))), (320, 320)).squeeze(0).cpu().numpy()
                            cur_image_max = gt.max()
                            cur_x = center_crop(complex_abs((x.permute(0, 2, 3, 1))), (320, 320))[0].squeeze(0).cpu().numpy()
                            mean_x = center_crop(complex_abs((x.permute(0, 2, 3, 1))), (320, 320)).mean(0).squeeze(0).cpu().numpy()
                            avg_psnr += skimage.metrics.peak_signal_noise_ratio(cur_x, gt, data_range=cur_image_max)
                            avg_psnr_of_mean += skimage.metrics.peak_signal_noise_ratio(mean_x, gt, data_range=cur_image_max)
                            ssim += skimage.metrics.structural_similarity(cur_x, gt, data_range=cur_image_max)
                            ssim_of_mean += skimage.metrics.structural_similarity(mean_x, gt, data_range=cur_image_max)
                            if args.rec:
                                rec_x = restore(rec_model, ksp.to(self.device), all_indices)
                                tvu.save_image(
                                    inverse_data_transform(config, rec_x[0], _min, _max),
                                    os.path.join(self.args.image_folder, f"rec_{idx_so_far + j}.png")
                                )
                                rec_x = center_crop(complex_abs(rec_x.permute(0, 2, 3, 1)), (320, 320))[0].squeeze(0).cpu().numpy()
                                rec_psnr += skimage.metrics.peak_signal_noise_ratio(rec_x, gt, data_range=cur_image_max)
                                rec_ssim += skimage.metrics.structural_similarity(rec_x, gt, data_range=cur_image_max)
                            if args.wavelet:
                                from models.mri_utils import wavelet_restore
                                wavelet_x = wavelet_restore(pinv_y_0, mask)
                                tvu.save_image(
                                    inverse_data_transform(config, wavelet_x[0], _min, _max),
                                    os.path.join(self.args.image_folder, f"wavelet_{idx_so_far + j}.png")
                                )
                                wavelet_x = center_crop(complex_abs((wavelet_x.permute(0, 2, 3, 1))), (320, 320))[0].squeeze(0).cpu().numpy()
                                wavelet_psnr += skimage.metrics.peak_signal_noise_ratio(wavelet_x, gt, data_range=cur_image_max)
                                wavelet_ssim += skimage.metrics.structural_similarity(wavelet_x, gt, data_range=cur_image_max)
                            if args.save_indices:
                                np.save(os.path.join(self.args.image_folder, f"indices_{idx_so_far + j}.npy"), all_indices.int().cpu().numpy())
                        else:
                            orig = inverse_data_transform(config, x_orig[j], _min, _max)
                            mse = torch.mean((reconstructions[i][j].to(self.device) - orig) ** 2)
                            psnr = 10 * torch.log10(1 / mse)
                            avg_psnr += psnr
                            mse = torch.mean((reconstructions[i].mean(0).to(self.device) - orig) ** 2)
                            psnr = 10 * torch.log10(1 / mse)
                            avg_psnr_of_mean += psnr
                            ssim += skimage.metrics.structural_similarity(reconstructions[i][j].squeeze().cpu().numpy(), orig.squeeze().cpu().numpy(), data_range=1.0)
                            ssim_of_mean += skimage.metrics.structural_similarity(reconstructions[i].mean(0).squeeze().cpu().numpy(), orig.squeeze().cpu().numpy(), data_range=1.0)


            idx_so_far += y_0.shape[0]

            if args.rec:
                pbar.set_description(f"PSNR: {(avg_psnr / (idx_so_far - idx_init)):.2f}, PSNR - mean: {(avg_psnr_of_mean / (idx_so_far - idx_init)):.2f}, PSNR - rec: {(rec_psnr / (idx_so_far - idx_init)):.2f}, SSIM: {ssim / (idx_so_far - idx_init):.4f}")
            else:
                pbar.set_description(f"PSNR: {(avg_psnr / (idx_so_far - idx_init)):.2f}, PSNR - mean: {(avg_psnr_of_mean / (idx_so_far - idx_init)):.2f}, SSIM: {ssim / (idx_so_far - idx_init):.4f}, SSIM - Mean: {ssim_of_mean / (idx_so_far - idx_init):.4f}")

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_psnr_of_mean = avg_psnr_of_mean / (idx_so_far - idx_init)
        ssim = ssim / (idx_so_far - idx_init)
        ssim_of_mean = ssim_of_mean / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average SSIM: %.4f" % ssim)
        print("Total Average PSNR of mean: %.2f" % avg_psnr_of_mean)
        print("Total Average SSIM of mean: %.4f" % ssim_of_mean)
        if args.rec:
            rec_psnr = rec_psnr / (idx_so_far - idx_init)
            rec_ssim = rec_ssim / (idx_so_far - idx_init)
            print("Rec PSNR: %.2f" % rec_psnr)
            print("Rec SSIM: %.4f" % rec_ssim)
        if args.wavelet:
            wavelet_psnr = wavelet_psnr / (idx_so_far - idx_init)
            wavelet_ssim = wavelet_ssim / (idx_so_far - idx_init)
            print("Wavelet PSNR: %.2f" % wavelet_psnr)
            print("Wavelet SSIM: %.4f" % wavelet_ssim)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, timesteps, last=True, classes=None):
        skip = self.num_timesteps // timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, etaA=self.args.eta, classes=classes)
        if last:
            x = x[0][-1]
        return x