# MagicScroll: Enhancing Immersive Storytelling with Controllable Scroll Image Generation (VR 2025)
## [<a href="https://magicscroll.github.io/" target="_blank">Project Page</a>]

![teaser](imgs/teaser.png)

​					**MagicScroll** is designed to generate coherent, controllable, and engaging scroll images from story texts.

## Abstract

> Scroll images are a unique medium commonly used in virtual reality (VR) providing an immersive visual storytelling experience. Despite rapid advances in diffusion-based image generation, it re mains an open research question to generate scroll images suit able for immersive, coherent, and controllable storytelling in VR. This paper proposes a multi-layered, diffusion-based scroll image generation framework with a novel semantic-aware denoising process. We incorporate layout prediction and style control modules to generate coherent scroll images of any aspect ratio. Based on the scroll image generation framework, we use different multi window strategies to render diverse visual forms such as chains, rings, and forks for VR storytelling. Quantitative and qualitative evaluations demonstrate that our techniques can significantly enhance text-image consistency and visual coherence in scroll image generation, as well as the level of immersion and engagement of VR storytelling. We will release our source code to facilitate better collaborations on immersive storytelling between AI researchers and creative practitioners.

For more see the [project webpage](https://magicscroll.github.io).

## Diffusers Integration

```
import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "a photo of the dolomites"
image = pipe(prompt).images[0]
```

## Gradio Demo 
We provide a gradio UI for our method:
```
python app_gradio.py
```
This demo is also hosted on HuggingFace [here](changec the url!)

## Citation
```
@misc{wang2023magicscrollnontypicalaspectratioimage,
      title={MagicScroll: Nontypical Aspect-Ratio Image Generation for Visual Storytelling via Multi-Layered Semantic-Aware Denoising}, 
      author={Bingyuan Wang and Hengyu Meng and Zeyu Cai and Lanjiong Li and Yue Ma and Qifeng Chen and Zeyu Wang},
      year={2023},
      eprint={2312.10899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.10899}, 
}
```
