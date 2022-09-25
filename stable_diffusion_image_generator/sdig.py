import yaml
import numpy as np
import torch

from typing import List
from argparse import ArgumentParser

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from super_resolution import cartoon_upsampling_8x
from PIL import Image

from pathlib import Path

config_path = str(Path(__file__).parent / "config.yml")


def dummy(images, **kwargs):
    return images, False


def load_model_path():
    return yaml.load(open(config_path, "r"), yaml.SafeLoader)["model_path"]


def set_model_path(model_path: str):
    print("Configuring path to...")
    print(model_path)
    with open(config_path, "w+") as f:
        f.write(f"---\n    model_path: {model_path}")


class SDIG:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.device = device
        self.pipe_text2img = None
        self.pipe_img2img = None

    def assert_text2img_pipeline(self):
        if self.pipe_text2img is not None:
            return self.pipe_text2img

        print("Loading models...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, use_auth_token=False
        )
        pipe.to(self.device)
        pipe.safety_checker = dummy
        self.pipe_text2img = pipe
        return pipe

    def assert_img2img_pipeline(self):
        if self.pipe_img2img is not None:
            return self.pipe_img2img

        print("Loading models...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_path, use_auth_token=False
        )
        pipe.to(self.device)
        pipe.safety_checker = dummy
        self.pipe_img2img = pipe
        return pipe

    def generate_from_text(
        self,
        prompt: str,
        num: int = 1,
        strength: float = 0.5,
        latents: np.array = None,
    ) -> List[Image.Image]:
        self.assert_text2img_pipeline()
        pipe = self.pipe_text2img

        if latents is None:
            latents = self.create_latents()

        print("Generating images...")
        images = []
        for i in range(num):
            images.append(
                pipe(
                    prompt=prompt,
                    strength=strength,
                    guidance_scale=7.5,
                    latents=latents,
                ).images[0]
            )

        print("Done.")
        return images

    def generate_from_img(
        self,
        prompt: str,
        source_img: Image,
        num: int,
        strength: float,
    ) -> List[Image.Image]:
        pipe = self.assert_img2img_pipeline()
        images = []
        for i in range(num):
            images.append(
                pipe(
                    prompt=prompt,
                    init_image=source_img,
                    strength=strength,
                    guidance_scale=7.5,
                ).images[0]
            )

        print("Done.")
        return images

    def save_images(self, images: List, output_name: str):
        filenames = [f"{output_name}_{i}.png" for i in range(len(images))]
        for i, image in enumerate(images):
            image.save(filenames[i])

    def animate_image_with_img2img(
        self,
        prompt: str,
        image: Image.Image,
        duration: int = 100,
        output: str = "animated.gif",
        strength: float = 0.1,
        num_frames: int = 10,
    ) -> List[Image.Image]:
        generated_imgs = []
        current_img = image
        for i in range(num_frames):
            img = self.generate_from_img(
                prompt=prompt,
                source_img=current_img,
                num=1,
                strength=strength,
            )[0]
            current_img = img
            generated_imgs.append(img)

        frames = generated_imgs.copy()

        images_reverse = frames.copy()
        images_reverse.reverse()
        frames[0].save(
            output,
            format="GIF",
            append_images=frames[1:] + images_reverse,
            save_all=True,
            duration=duration,
            loop=0,
        )

        return generated_imgs

    def animate_image_with_text2img(
        self,
        prompt: str,
        image: Image.Image,
        latents: torch.Tensor,
        duration: int = 50,
        output: str = "animated.gif",
        strength: float = 0.1,
        num_frames: int = 10,
        latent_offset_size: float = 0.1,
    ) -> List[Image.Image]:
        generated_imgs = []
        for i in range(num_frames):
            img = self.generate_from_text(
                prompt=prompt,
                num=1,
                strength=strength,
                latents=latents,
            )[0]
            generated_imgs.append(img)

            latents = latents + self.create_latents().clamp(
                -latent_offset_size, latent_offset_size
            )

        frames = generated_imgs.copy()

        images_reverse = frames.copy()
        images_reverse.reverse()
        frames[0].save(
            output,
            format="GIF",
            append_images=frames[1:] + images_reverse,
            save_all=True,
            duration=duration,
            loop=0,
        )

        return generated_imgs

    def upscale(self, images: List[Image.Image]) -> List[Image.Image]:
        upscaled_imgs = []
        for image in images:
            img_array = np.array(image.convert("RGB"))
            upscaled_array = cartoon_upsampling_8x(img_array)
            upscaled_image = Image.fromarray(upscaled_array)
            upscaled_imgs.append(upscaled_image)
        return upscaled_imgs

    def create_latents(self):
        generator = torch.Generator(device=self.device)
        latents = None
        seed = generator.seed()
        generator = generator.manual_seed(seed)

        image_latents = torch.randn(
            (1, 4, 512 // 8, 512 // 8), generator=generator, device=self.device
        )
        latents = (
            image_latents
            if latents is None
            else torch.cat((latents, image_latents))
        )
        return latents


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="txt2img",
        choices=["txt2img", "img2img"],
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text to generate from"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Source image path to be used as starter.",
    )
    parser.add_argument(
        "--animate",
        type=bool,
        default=False,
        help="Animate provided image or text caption.",
    )
    parser.add_argument(
        "--num", type=int, default=1, help="Number of images generated"
    )
    parser.add_argument(
        "--output_name", type=str, default="output", help="Output image name"
    )
    parser.add_argument(
        "--strength", type=float, default=0.5, help="Image generation strength"
    )
    parser.add_argument(
        "--animation_strength",
        type=float,
        default=0.0125,
        help="Animation strength",
    )
    parser.add_argument(
        "--animation_latent_offset",
        type=float,
        default=0.0125,
        help="Animation latent offset",
    )
    parser.add_argument(
        "--animation_duration",
        type=int,
        default=100,
        help="Animation duration",
    )
    parser.add_argument(
        "--animation_num_frames",
        type=int,
        default=10,
        help="Number of frames in animation",
    )
    parser.add_argument("--upscale", type=bool, default=False)
    parser.add_argument(
        "--set_model_path",
        type=str,
        default=None,
        help="Path to stable diffusion model. Only necessary once.",
    )
    arguments = parser.parse_args()

    if arguments.text is None and arguments.image is None:
        parser.print_help()
        return

    if arguments.set_model_path is not None:
        set_model_path(arguments.set_model_path)
        return

    images = []
    sdig = SDIG(load_model_path())
    if arguments.mode == "txt2img":
        latents = sdig.create_latents()
        images = sdig.generate_from_text(
            prompt=arguments.text,
            num=arguments.num,
            strength=arguments.strength,
            latents=latents,
        )

        if arguments.animate:
            images = sdig.animate_image_with_text2img(
                arguments.text,
                images[0],
                latents=latents,
                output=f"{arguments.output_name}.gif",
                duration=arguments.animation_duration,
                strength=arguments.strength,
                latent_offset_size=arguments.animation_latent_offset,
                num_frames=arguments.animation_num_frames,
            )

    elif arguments.mode == "img2img":
        source_img = Image.open(arguments.image)
        images = sdig.generate_from_img(
            prompt=arguments.text,
            source_img=source_img,
            num=arguments.num,
            strength=arguments.strength,
        )

        if arguments.animate:
            images = sdig.animate_image_with_img2img(
                arguments.text,
                source_img,
                output=f"{arguments.output_name}.gif",
                num_frames=arguments.animation_num_frames,
                duration=arguments.animation_duration,
                strength=arguments.animation_strength,
            )

    if arguments.upscale:
        images = sdig.upscale(images)

    sdig.save_images(images, arguments.output_name)


if __name__ == "__main__":
    main()
