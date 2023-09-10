from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion

USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()


model.set_generation_params(
    use_sampling=True,
    top_k=5, # 250
    duration=5 # 30
)

import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio

def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

from audiocraft.utils.notebook import display_audio
from pydub import AudioSegment

output = model.generate(
    descriptions=[
        #'80s pop track with bassy drums and synth',
        #'90s rock song with loud guitars and heavy drums',
        #'Progressive rock drum and bass solo',
        #'Punk Rock song with loud drum and power guitar',
        #'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
        #'Jazz Funk song with slap bass and powerful saxophone',
        'drum and bass beat with intense percussions'
    ],
    progress=True, return_tokens=True
)
display_audio(output[0], sample_rate=32000)

try:
    import IPython.display as ipd  # type: ignore
except ImportError:
    # Note in a notebook...
    pass


import torch
import torchaudio


def dustin(samples: torch.Tensor, sample_rate: int):
    """Renders an audio player for the given audio samples.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
    """
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for audio in samples:
        torchaudio.save('demos/output.mp3', audio, sample_rate)
        # ipd.display(ipd.Audio(audio, rate=sample_rate))

dustin(output[0], sample_rate=32000)