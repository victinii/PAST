# PAST
Style transfer modifies a content image to reflect the artistic style of a style image while preserving its original content. However, traditional methods often suffer from "style confusion," where the artistic style is conflated with other visual features like color and layout, leading to artifacts and inconsistencies. To address this, we propose Pure Artistic Style Transfer (PAST), which separates artistic style from specific visual features. Our method enlarges the embedding space of the style image, enabling targeted training during the denoising process, and introduces auxiliary style images to constrain embedding distances, eliminating feature-based influences. A parallel denoising structure ensures the synthesized image maintains structural consistency with the content image. Extensive experiments demonstrate PAST's effectiveness, showcasing its ability to inject artistic style without the negative impact of color and structural features. This work contributes to the evolution of style transfer techniques, offering a more interpretable and controllable approach to artistic applications.
![abstract-new](https://github.com/user-attachments/assets/511478db-0d78-4d10-b1cf-727e97aedc0a)

## Setup
```
conda create -n past python=3.8
conda activate past
pip install -r requirements.txt
```
Our method should work in any GPU with at least 24G memory. We tested out method on NVIDIA GeForce RTX 4090.

## Usage 
To test the color immunization experiment, please run 
```
accelerate launch main.py \
    --concept_image_dir="./examples/concept_image" \
    --content_image_dir="./examples/content_image" \
    --pretrained_model_name_or_path="/put/your/downloaded/stable_diffusion/model"
    --output_image_path="./outputs" \
    --initializer_token="painting" \
    --max_train_steps=500 \
    --concept_embedding_num=3 \
    --self_attention_injection_ratio=0.9 
```
Put content images into `content_image_dir`, also put one style image and one auxiliary style image into `content_image_dir`. The `initializer_token` is used as the beginning of concept embeddings.

It is recommended to download the pre-trained stable-diffusion model(we use v1-5) to `pretrained_model_name_or_path`.
