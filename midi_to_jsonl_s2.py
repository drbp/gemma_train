
import json
from pathlib import Path
from miditok import REMI
from tqdm import tqdm
from multiprocessing import Pool

# 1. Load your specialized Rock Tokenizer
tokenizer = REMI(params="rock_midi_tokenizer.json")

def process_one_file(midi_path):
    """Converts a single MIDI file into a string of BPE tokens."""
    try:
        # Convert MIDI to BPE tokens
        tok_seqs = tokenizer.midi_to_tokens(midi_path)
        # Convert token strings to space-separated sequence
        # tok_seqs[0] is the single stream (one_token_stream_for_programs=True)
        token_str = " ".join(tok_seqs[0].tokens)
        
        # Add the BOS/EOS markers for the model
        return {"text": f"<bos> {token_str} <eos>"}
    except Exception as e:
        # Skip corrupted MIDI files common in the Lakh dataset
        return None

def main():
    # 2. Collect all 4,000 MIDI paths
    dataset_path = Path("path/to/your/midi_folder")
    midi_files = list(dataset_path.glob("**/*.mid")) + list(dataset_path.glob("**/*.midi"))
    
    print(f"Found {len(midi_files)} files. Starting conversion...")

    # 3. Use all 16 threads of your i7-10700
    with Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(process_one_file, midi_files), total=len(midi_files)))

    # 4. Filter out None (failed files) and save to JSONL
    final_data = [r for r in results if r is not None]
    
    with open("midi_training_data.jsonl", "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Done! Saved {len(final_data)} sequences to midi_training_data.jsonl")

if __name__ == "__main__":
    main()
