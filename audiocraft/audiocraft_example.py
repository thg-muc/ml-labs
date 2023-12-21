from datetime import datetime

from tqdm import tqdm

from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen

# https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/models/audiogen.html

# Config
model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=5)  # in seconds
# model.set_generation_params(temperature=1.0)


descriptions = [
    "sirene of an emergency vehicle",
    "footsteps in a corridor",
    "a car passing by",
    "knocking on a door",
    "a baby crying",
    "dog barking",
    "a cat meowing",
    "a person sneezing",
    "a busy street",
    "a person walking on a wooden floor",
    "a person walking on gravel",
    "a person walking on grass",
    "Heavy rain and thunderstorm.",
    "Light smooth rain in the mountains.",
    "Rain on a car roof",
]

print(f"\nStarting at: {datetime.now()}\n")


for idx, description in tqdm(enumerate(descriptions)):
    wav = model.generate(description)
    export_description = (
        f"{description.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    audio_write(
        export_description,
        wav.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )

print(f"\nFinished at: {datetime.now()}\n")
