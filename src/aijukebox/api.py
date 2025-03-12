import os
import pathlib
import subprocess

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from diffrhythm.infer.infer_utils import (
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)
from diffrhythm.infer.infer import inference

app = FastAPI()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

my_dir = pathlib.Path(__file__).parent.resolve()
hls_dir = my_dir / "hls"
static_dir = my_dir / "static"

# Mount static directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/hls", StaticFiles(directory=hls_dir), name="hls")


@app.get("/test/")
async def test_page():
    """Serve the test player HTML page."""
    return FileResponse("src/aijukebox/static/test.html")


# # Load models
# cfm, tokenizer, muq, vae = prepare_model(device)


OUTPUT_WAV = "mozart.wav"
HLS_DIR = hls_dir
HLS_PLAYLIST = HLS_DIR / "stream.m3u8"


def generate_and_segment_audio():
    """Generates an AI song and converts it into HLS format for streaming."""
    print("Generating AI audio...")

    # max_frames = 2048
    # lyrics = "This is an AI-generated song that loops forever."
    # style = "A calm and inspiring musical vibe"
    #
    # lrc_prompt, start_time = get_lrc_token(lyrics, tokenizer, device)
    # style_prompt = get_style_prompt(muq, prompt=style)
    # negative_style_prompt = get_negative_style_prompt(device)
    # latent_prompt = get_reference_latent(device, max_frames)
    #
    # generated_song = inference(
    #     cfm_model=cfm,
    #     vae_model=vae,
    #     cond=latent_prompt,
    #     text=lrc_prompt,
    #     duration=max_frames,
    #     style_prompt=style_prompt,
    #     negative_style_prompt=negative_style_prompt,
    #     start_time=start_time,
    #     chunked=False,
    # )

    # Save audio as WAV
    # torchaudio.save(OUTPUT_WAV, generated_song, sample_rate=44100)

    # Convert to HLS format using FFmpeg
    os.makedirs(HLS_DIR, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg",
        "-stream_loop",
        "-1",  # Loop input infinitely
        "-re",  # Read input at native rate (simulate live)
        "-i",
        "src/aijukebox/mozart.wav",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-f",
        "hls",
        "-hls_time",
        "5",
        "-hls_list_size",
        "6",  # Maintain a sliding window of segments
        "-hls_flags",
        "delete_segments+omit_endlist",  # Delete old segments and omit the end marker
        "-hls_allow_cache",
        "0",
        "-hls_base_url",
        "/hls/",  # Ensures correct segment URLs
        "src/aijukebox/hls/stream.m3u8",
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_command)

    # subprocess.run(ffmpeg_command, check=True)

    print("AI audio generated and converted to HLS format.")


@app.get("/stream/")
async def stream_audio():
    """Serve the HLS playlist."""
    return FileResponse(HLS_PLAYLIST, media_type="application/x-mpegURL")


@app.get("/hls/{segment}")
async def stream_hls(segment: str):
    """Serve individual HLS segments."""
    return FileResponse(os.path.join(HLS_DIR, segment), media_type="video/MP2T")


generate_and_segment_audio()  # Generate the AI song on startup


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
