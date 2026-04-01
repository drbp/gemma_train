from miditok import REMI, TokenizerConfig
from pathlib import Path

# 1. Base Config: REMI with the structural settings we discussed
config = TokenizerConfig(
    num_velocities=16,          # Saves VRAM
    use_chords=True,            # High musical logic
    use_programs=True,          # Multi-instrument support
    use_rests=True,             # Saves context space
    one_token_stream_for_programs=True
)
midi_tokenizer = REMI(config)

# 2. Get your 4,000 MIDI file paths
midi_paths = list(Path("./orig_midi_files").glob("**/*.mid"))

# 3. Train BPE
# 8,000 is the 'sweet spot' for a 12B model: 
# Big enough for complex rock chords, small enough to save ~3GB VRAM.
midi_tokenizer.train(
    vocab_size=8000, 
    files_paths=midi_paths,
)

# 4. Save the 'Musical Dictionary'
midi_tokenizer.save_params("rock_midi_tokenizer.json")
