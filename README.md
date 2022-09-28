```
usage: sdig.py [-h] [--mode {txt2img,img2img}] [--text TEXT] [--image IMAGE] [--animate ANIMATE] [--num NUM] [--output_name OUTPUT_NAME] [--strength STRENGTH]
               [--animation_strength ANIMATION_STRENGTH] [--animation_latent_offset ANIMATION_LATENT_OFFSET] [--animation_duration ANIMATION_DURATION]
               [--animation_num_frames ANIMATION_NUM_FRAMES] [--upscale UPSCALE] [--set_model_path SET_MODEL_PATH]

options:
  -h, --help            show this help message and exit
  --mode {txt2img,img2img}
  --text TEXT           Text to generate from
  --image IMAGE         Source image path to be used as starter.
  --animate ANIMATE     Animate provided image or text caption.
  --num NUM             Number of images generated
  --output_name OUTPUT_NAME
                        Output image name
  --strength STRENGTH   Image generation strength
  --animation_strength ANIMATION_STRENGTH
                        Animation strength
  --animation_latent_offset ANIMATION_LATENT_OFFSET
                        Animation latent offset
  --animation_duration ANIMATION_DURATION
                        Animation duration
  --animation_num_frames ANIMATION_NUM_FRAMES
                        Number of frames in animation
  --upscale UPSCALE
  --set_model_path SET_MODEL_PATH
                        Path to stable diffusion model. Only necessary once.
```