import os
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from magenta.models.shared import sequence_proto_to_midi
from magenta.music import midi
from pydub import AudioSegment
from gtts import gTTS

# Function to generate music based on Magenta
def generate_music():
    model_name = 'cat-mel_2bar_big'  # You can change to other models if desired
    config = configs.CONFIG_MAP[model_name]
    checkpoint = os.path.join(magenta.models.music_vae.__path__[0], 'checkpoints', model_name)
    model = TrainedModel(config, batch_size=1, checkpoint_dir_or_path=checkpoint)

    # Generate music sequence
    z = model.sample(n=1, length=80)  # Length is 80 time steps
    midi_sequence = sequence_proto_to_midi(z[0])  # Convert to MIDI format
    
    # Save MIDI file
    output_path = '/content/generated_song.mid'
    midi.sequence_proto_to_midi_file(midi_sequence, output_path)
    
    return output_path

# Function to generate vocals from lyrics using gTTS (Google Text-to-Speech)
def generate_vocals(lyrics):
    tts = gTTS(text=lyrics, lang='en')
    vocal_output_path = "/content/vocals.mp3"
    tts.save(vocal_output_path)
    
    return vocal_output_path

# Function to combine generated music and vocals
def combine_audio(music_path, vocals_path):
    # Convert MIDI to audio using pydub (needs external tools for real-time conversion)
    music_audio = AudioSegment.from_file(music_path, format="mid")
    
    # Load the vocals
    vocals = AudioSegment.from_mp3(vocals_path)
    
    # Overlay vocals on music
    combined = music_audio.overlay(vocals, position=0)
    combined_path = '/content/final_song.wav'
    combined.export(combined_path, format="wav")
    
    return combined_path

# Generate a song with lyrics
def generate_song(lyrics):
    music_path = generate_music()
    vocals_path = generate_vocals(lyrics)
    final_song_path = combine_audio(music_path, vocals_path)
    
    return final_song_path
