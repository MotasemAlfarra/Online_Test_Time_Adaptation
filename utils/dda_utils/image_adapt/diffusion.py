import math
import torch

from .guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from .resizer import Resizer

class diffusion_args(object):
    def __init__(self) -> None:
        self.attention_resolutions = '32,16,8'
        self.class_cond = False
        self.diffusion_steps = 1000
        self.image_size = 256
        self.learn_sigma = True
        self.noise_schedule = 'linear'
        self.num_channels = 256
        self.num_head_channels = 64
        self.num_res_blocks = 2
        self.resblock_updown = True
        self.use_fp16 = True
        self.use_scale_shift_norm = True
        self.D = 8
        self.M = 20
        self.N = 50
        self.channel_mult = ''
        self.class_cond=False
        self.clip_denoised=True
        self.diffusion_steps=1000
        self.dropout=0.0
        self.image_size=256
        self.learn_sigma=True
        self.model_path='./utils/dda_utils/ckpt/256x256_diffusion_uncond.pt'
        self.noise_schedule='linear'
        self.num_channels=256
        self.num_head_channels=64
        self.num_heads=4
        self.num_heads_upsample=-1
        self.num_res_blocks=2
        self.num_samples=4
        self.predict_xstart=False
        self.resblock_updown=True
        self.rescale_learned_sigmas=False
        self.rescale_timesteps=False
        self.save_dir='dataset/generated/'
        self.save_latents=False
        self.severity=5
        self.timestep_respacing='100'
        self.use_checkpoint=False
        self.use_ddim=False
        self.use_fp16=True
        self.use_kl=False
        self.use_new_attention_order=False
        self.use_scale_shift_norm=True


def diffusion(batch_size=4):
    # args = create_argparser().parse_args()
    args = diffusion_args()
    # args = load_args(args)
    args.batch_size = batch_size
    # th.manual_seed(0)

    # dist_util.setup_dist()
    # logger.configure(dir=args.save_dir)

    # logger.log("creating model...")
    print("creating model...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to('cuda')
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    # logger.log("creating resizers...")
    print("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.D), int(args.image_size / args.D))
    down = Resizer(shape, 1 / args.D).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.D).to(next(model.parameters()).device)
    resizers = (down, up)
    
    diffusion_to_return = my_diffusion(
        diffusion, model, args.batch_size, args.image_size, args.clip_denoised, resizers, args.M, args.N
    )
    return diffusion_to_return

class my_diffusion(torch.nn.Module):
    def __init__(self, diffusion, model, batch_size, image_size, clip_denoised, resizers, M, N) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size
        self.clip_denoised = clip_denoised
        self.resizers = resizers
        self.M, self.N = M, N
    
    def forward(self, x):
        model_kwargs = {"ref_img": x}
        sample = self.diffusion.p_sample_loop(
            self.model,
            (self.batch_size, 3, self.image_size, self.image_size),
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            resizers=self.resizers,
            M=self.M,
            N=self.N,
        )
        return sample