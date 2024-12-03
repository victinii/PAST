# This code partially references https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
# and https://github.com/gligen/diffusers/blob/0e09db9d7150126e327ff93cf91857b00f624ee0/examples/research_projects/mulit_token_textual_inversion/textual_inversion.py
# and https://github.com/CrystalNeuro/visual-concept-translator/blob/main/main.py

import pivot_turning_inversion
import argparse
import logging
import math
import os
import copy
import random
from pathlib import Path
from typing import Optional
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from multi_token_clip import MultiTokenCLIPTokenizer
import datasets
import diffusers
import PIL
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
import shutil
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from new_scheduling_ddpm import DDPMScheduler
import time


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


logger = get_logger(__name__)


# def save_progress(tokenizer, text_encoder, accelerator, args, global_step):
#     for placeholder_token in tokenizer.token_map:
#         placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
#         learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]
#         if len(placeholder_token_ids) == 1:
#                 learned_embeds = learned_embeds[None]
#         learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
#         safe_placeholder_token = re.sub(r'[^\w_. -]', '_', placeholder_token)
#         torch.save(learned_embeds_dict, os.path.join(args.output_dir, f'{safe_placeholder_token}_{global_step}.bin'))
def save_progress(tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2, accelerator, args, global_step, file_names):
    folder1 = os.path.join(args.output_dir, file_names[0])
    folder2 = os.path.join(args.output_dir, file_names[1])

    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    tokenizers = [(tokenizer_1, text_encoder_1, folder1), (tokenizer_2, text_encoder_2, folder2)]

    for tokenizer, text_encoder, folder in tokenizers:
        for placeholder_token in tokenizer.token_map:
            placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]
            if len(placeholder_token_ids) == 1:
                learned_embeds = learned_embeds[None]
            learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
            safe_placeholder_token = re.sub(r'[^\w_. -]', '_', placeholder_token)

            torch.save(learned_embeds_dict, os.path.join(folder, f'{safe_placeholder_token}_{global_step}.bin'))

def load_multitoken_tokenizer(tokenizer, text_encoder, learned_embeds_dict):
    for placeholder_token in learned_embeds_dict:
        num_vec_per_token = learned_embeds_dict[placeholder_token].shape[0]
        add_tokens(tokenizer, text_encoder, placeholder_token, num_vec_per_token=num_vec_per_token)
        placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] =token_embeds[i]

def add_tokens(tokenizer, text_encoder, placeholder_token, num_vec_per_token=1, initializer_token=None, use_neg=False):
    """
    Add tokens to the tokenizer and set the initial value of token embeddings
    """
    # tokenizer.add_placeholder_tokens(placeholder_token, num_vec_per_token=num_vec_per_token)
    for placeholder_token_0 in placeholder_token:
        tokenizer.add_placeholder_tokens(placeholder_token_0, num_vec_per_token=num_vec_per_token)

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    placeholder_token_ids = []
    for placeholder_token_0 in placeholder_token:
        token_id = tokenizer.encode(placeholder_token_0, add_special_tokens=False)
        placeholder_token_ids.append(token_id)

    print(f"number of placeholder tokens are: {len(placeholder_token_ids)}")
    if initializer_token:
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        for placeholder_token_id_0 in placeholder_token_ids:
            for i, placeholder_token_id in enumerate(placeholder_token_id_0):

                token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids)//num_vec_per_token]]
                if use_neg:
                    token_embeds[placeholder_token_id] += torch.randn_like(token_embeds[placeholder_token_id])*1e-3
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
                token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])

def get_mask(tokenizer, accelerator):
    # Get the mask of the weights that won't change
    mask = torch.ones(len(tokenizer)).to(accelerator.device, dtype=torch.bool)
    for placeholder_token in tokenizer.token_map:
        placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
        for i in range(len(placeholder_token_ids)):
            mask = mask & (torch.arange(len(tokenizer)) != placeholder_token_ids[i]).to(accelerator.device)
    return mask



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--concept_image_dir", 
        type=str, 
        default=None, 
        required=True, 
        help="A folder containing the training concept images."
    )

    parser.add_argument(
        "--content_image_dir", 
        type=str, 
        default=None, 
        required=True, 
        help="A folder containing the test content images."
    )

    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output_images",
        help="The path to save the translated images.",
    )

    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=True,
        help="Save only the embeddings for the new concept.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for inference.",
    )
    
    parser.add_argument(
        "--cross_attention_injection_ratio",
        type=float,
        default=0.2,
        help="Larger self-attention injection ratio or cross-attention injection ratio means more source contents preserved and less target concepts transferred.",
    )
    
    parser.add_argument(
        "--self_attention_injection_ratio",
        type=float,
        default=0.9,
        help="Larger self-attention injection ratio or cross-attention injection ratio means more source contents preserved and less target concepts transferred.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--concept_embedding_num",
        type=int,
        default=3,
        help=(
            "The number of vectors used to represent the target concept."
        ),
    )
    
    parser.add_argument(
        "--placeholder_token",
        type=list,
        default=[f'<s{i}>' for i in range(10)],
        # default='<s0>',
        help="A list of tokens to use as a placeholder for each concept position.",
    )
    parser.add_argument(
        "--initializer_token", 
        type=str, 
        default=None, 
        required=True, 
        help="A token to use as initializer word, this should be close to the target concept"
    )
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir", # model checkpoint save path
        type=str,
        default="output",
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=500)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5.0e-04,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        type=bool,
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--cos_sim_weight", type=list, default=[0.002 for i in range(10)])

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.concept_image_dir is None:
        raise ValueError("You must specify a data directory with one or several concept images.")

    return args

class InversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer_1,
        tokenizer_2,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        
        self.size = size
        self.placeholder_token = placeholder_token
        
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = sorted([os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)])
        assert len(self.image_paths) == 2
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats
            # self._length = 100

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_1 = Image.open(self.image_paths[0])
        image_2 = Image.open(self.image_paths[1])

        if not image_1.mode == "RGB":
            image_1 = image_1.convert("RGB")
        if not image_2.mode == "RGB":
            image_2 = image_2.convert("RGB")

        # text = "%s"%(self.placeholder_token)

        timesteps = torch.randint(0, 1000, (), device='cuda:0')
        timesteps = timesteps.long()
        example['time'] = timesteps
        
        example["input_ids"] = self.tokenizer_1(
            self.placeholder_token[timesteps//100],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        assert (example['input_ids']==self.tokenizer_2(
            self.placeholder_token[timesteps//100],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]).all()
        

        img_1 = np.array(image_1).astype(np.uint8)
        img_2 = np.array(image_2).astype(np.uint8)

        image_1 = Image.fromarray(img_1)
        image_1 = image_1.resize((self.size, self.size), resample=self.interpolation)

        image_2 = Image.fromarray(img_2)
        image_2 = image_2.resize((self.size, self.size), resample=self.interpolation)

        image_1 = np.array(image_1).astype(np.uint8)
        image_2 = np.array(image_2).astype(np.uint8)

        image_1 = (image_1 / 127.5 - 1.0).astype(np.float32)
        image_2 = (image_2 / 127.5 - 1.0).astype(np.float32)

        example["pixel_values_1"] = torch.from_numpy(image_1).permute(2, 0, 1)
        example["pixel_values_2"] = torch.from_numpy(image_2).permute(2, 0, 1)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def style_inversion():
    args = parse_args()
    file_names = [os.path.splitext(f)[0] for f in os.listdir(args.concept_image_dir) if
                  os.path.isfile(os.path.join(args.concept_image_dir, f))]

    if file_names:
        concept_file_name = file_names[0]
    else:
        concept_file_name = "default_name"

    current_timestamp = int(time.time())
    formatted_time = time.strftime('%m_%d_%Y_%H%M', time.localtime(current_timestamp))

    concept_folder_name = f"{concept_file_name}_{args.max_train_steps}_{formatted_time}"

    # # obtain the concept folder name form its path
    # concept_folder_name = os.path.split(os.path.normpath(args.concept_image_dir))[-1] or os.path.splitdrive(os.path.normpath(args.concept_image_dir))[-1]
    #
    #
    # current_timestamp = int(time.time())
    # formatted_time = time.strftime('%m_%d_%Y_%H%M', time.localtime(current_timestamp))
    #
    # # add time stamp
    # concept_folder_name = concept_folder_name + "_" + formatted_time

    args.output_dir = os.path.join(args.output_dir, concept_folder_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_image_path, exist_ok=True)

    if os.path.exists(os.path.join(args.concept_image_dir, ".ipynb_checkpoints")):
        shutil.rmtree(os.path.join(args.concept_image_dir, ".ipynb_checkpoints"))

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = MultiTokenCLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = MultiTokenCLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_1 = copy.deepcopy(tokenizer)
    tokenizer_2 = copy.deepcopy(tokenizer)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type='epsilon')
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_1 = copy.deepcopy(text_encoder)
    text_encoder_2 = copy.deepcopy(text_encoder)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Add the placeholder token in tokenizer
    #num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    # if num_added_tokens == 0:
    #     raise ValueError(
    #         f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
    #         " `placeholder_token` that is not already in the tokenizer."
    #     )

    # Convert the initializer_token, placeholder_token to ids


    # add_tokens(tokenizer, text_encoder, args.placeholder_token, args.concept_embedding_num, args.initializer_token)
    add_tokens(tokenizer_1, text_encoder_1, args.placeholder_token, args.concept_embedding_num, args.initializer_token)
    add_tokens(tokenizer_2, text_encoder_2, args.placeholder_token, args.concept_embedding_num, args.initializer_token)



    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    # text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_1.resize_token_embeddings(len(tokenizer_2))


    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder_2.text_model.encoder.requires_grad_(False)
    text_encoder_2.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_2.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        list(text_encoder_1.get_input_embeddings().parameters()) + list(
            text_encoder_2.get_input_embeddings().parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = InversionDataset(
        data_root=args.concept_image_dir,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        center_crop=args.center_crop,
        set="train",
    )
    data_name = train_dataset.image_paths
    file_names = [os.path.splitext(os.path.basename(path))[0] for path in data_name]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder_1,text_encoder_2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_1,text_encoder_2, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params_1 = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()
    orig_embeds_params_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate([text_encoder_1, text_encoder_2]):
                # Convert images to latent space
                latents_1 = vae.encode(batch["pixel_values_1"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents_1 = latents_1 * 0.18215
                latents_2 = vae.encode(batch["pixel_values_2"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents_2 = latents_2 * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents_1)
                bsz = latents_1.shape[0]
                timesteps = batch['time']

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents_1 = noise_scheduler.add_noise(latents_1, noise, timesteps)
                noisy_latents_2 = noise_scheduler.add_noise(latents_2, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states_1 = text_encoder_1(batch["input_ids"])[0].to(dtype=weight_dtype)
                encoder_hidden_states_2 = text_encoder_2(batch["input_ids"])[0].to(dtype=weight_dtype)

                cos_sim_weight = torch.tensor(args.cos_sim_weight).to(device = accelerator.device)
                placeholder_token_ids = tokenizer_1.encode(args.placeholder_token[timesteps//100], add_special_tokens=False)
                learned_embeds_1 = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[placeholder_token_ids]
                learned_embeds_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[placeholder_token_ids]
                loss_sim = -torch.nn.functional.cosine_similarity(learned_embeds_1, learned_embeds_2,dim=-1).mean().squeeze()*cos_sim_weight[timesteps//100]
                # Predict the noise residual
                model_pred_1 = unet(noisy_latents_1, timesteps, encoder_hidden_states_1).sample
                model_pred_2 = unet(noisy_latents_2, timesteps, encoder_hidden_states_2).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred_1.float(), target.float(), reduction="mean")+F.mse_loss(model_pred_2.float(), target.float(), reduction="mean")+loss_sim
                # loss = F.mse_loss(model_pred_1.float(), target.float(), reduction="mean") + F.mse_loss(
                #     model_pred_2.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = get_mask(tokenizer_1, accelerator)
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params_1[index_no_updates]
                    accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params_2[index_no_updates]
                del index_no_updates

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")

        save_progress(tokenizer_1, text_encoder_1, tokenizer_2,text_encoder_2, accelerator, args, global_step, file_names=file_names)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    return args


if __name__ == "__main__":
    
    args = style_inversion()
    torch.cuda.empty_cache()
    pivot_turning_inversion.inversion(args.output_dir, args.content_image_dir, args.output_image_path, args.concept_image_dir, args.max_train_steps, args.cross_attention_injection_ratio, args.self_attention_injection_ratio, args.pretrained_model_name_or_path)




