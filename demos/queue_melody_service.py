import redis
import multiprocessing
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from audiocraft.utils.notebook import display_audio
from pydub import AudioSegment
import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio
import json

USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()


model.set_generation_params(
    use_sampling=True,
    top_k=250, # 250
    duration=15 # 30
)

def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope


redis_client = redis.Redis(host='localhost', port=6379, db=0)

def process_item(item):
    print("Processing item")
    data = json.loads(item)
    print (data['lyrics'])
    descriptions = data['lyrics']
    # descriptions = [
    #     # '80s pop track with bassy drums and synth',
    #     # '90s rock song with loud guitars and heavy drums',
    #     # 'Progressive rock drum and bass solo',
    #     # 'Punk Rock song with loud drum and power guitar',
    #     # 'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
    #     # 'Jazz Funk song with slap bass and powerful saxophone',
    #     # 'drum and bass beat with intense percussions'
    #     'Classic 80s song with guitar and bass solo'
    # ]

    output = model.generate(
        descriptions=descriptions,
        progress=True, return_tokens=True
    )
    display_audio(output[0], sample_rate=32000)

    try:
        import IPython.display as ipd  # type: ignore
    except ImportError:
        # Note in a notebook...
        pass
    def create_melody(samples: torch.Tensor, sample_rate: int):
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
            torchaudio.save('../voice-generation/mp3_gen/best_music/melody/' + descriptions[0] + '.mp3', audio, sample_rate)
            # ipd.display(ipd.Audio(audio, rate=sample_rate))
    create_melody(output[0], sample_rate=32000)

# Function to pull and process items from the Redis queue
def worker(queue_name):
    while True:
        # Blocking pop operation from the queue
        item = redis_client.blpop(queue_name, timeout=0)
        
        if item:
            item = item[1].decode('utf-8')  # Convert bytes to string
            process_item(item)

if __name__ == "__main__":
    queue_name = "my_melody"  # Replace with your Redis queue name
    num_processes = 1  # Number of worker processes
    
    # Create and start worker processes
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(queue_name,))
        processes.append(p)
        p.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()