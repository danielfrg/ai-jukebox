import os
import time

import torch
import torchaudio

from diffrhythm.infer.infer_utils import (
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


from diffrhythm.infer.infer import inference

if __name__ == "__main__":
    # Determine device
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    print("Device:", device)

    # Hard-coded configuration
    audio_length = 95
    max_frames = 2048  # set for audio_length 95

    # Load models and utilities
    cfm, tokenizer, muq, vae = prepare_model(device)

    # Hard-coded text prompt (lyrics) for generation
    lrc = "These are sample lyrics to generate a creative audio piece."
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)

    # Hard-coded style prompt (using text prompt variant)
    style_prompt = get_style_prompt(muq, prompt="A calm and inspiring musical vibe")

    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt = get_reference_latent(device, max_frames)

    # Run inference
    s_t = time.time()
    generated_song = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        chunked=False,  # hard-coded to False
    )
    e_t = time.time() - s_t
    print(f"Inference cost {e_t} seconds")

    # Save the generated audio
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)
