import torch
import torch.nn.functional as F
import argparse
import os
from pdf2image import convert_from_path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from . import config
from .tokenizer import StTokenizer
from .model import ImageToStModel

def beam_search_predict(model, image_tensor, tokenizer, beam_width=5, max_len=500):
    """
    Generates an ST string prediction using a batched beam search for faster inference.
    """
    model.eval()
    
    sos_token_id = tokenizer.token_to_id['<sos>']
    eos_token_id = tokenizer.token_to_id['<eos>']
    device = config.DEVICE

    # --- Encoder Step ---
    src_image = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # The memory is calculated once and expanded to match the beam width.
        memory = model.encode(src_image)  # Shape: [1, H*W, d_model]
        memory = memory.expand(beam_width, -1, -1)  # Shape: [beam_width, H*W, d_model]

    # --- Batched Beam Search Decoder ---
    # We'll store the top k sequences and their scores.
    # sequences shape: [beam_width, 1] -> [beam_width, max_len]
    sequences = torch.full((beam_width, 1), sos_token_id, dtype=torch.long, device=device)
    
    # top_k_scores shape: [beam_width, 1]
    top_k_scores = torch.zeros(beam_width, 1, device=device)

    # Keep track of completed sequences
    complete_sequences = []
    complete_sequence_scores = []

    pbar = tqdm(range(max_len - 1), desc="Beam Search", leave=False)
    for step in pbar:
        if sequences.size(0) == 0: # Stop if all beams are complete
            break

        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = model.decode(sequences, memory) # Shape: [current_beam_size, seq_len, vocab_size]
            
        # Get the logits for the last token and calculate log probabilities
        last_token_logits = output[:, -1, :] # Shape: [current_beam_size, vocab_size]
        log_probs = F.log_softmax(last_token_logits, dim=-1)
        
        # Add the current scores to the log probabilities to get candidate scores
        # log_probs shape: [current_beam_size, vocab_size]
        # top_k_scores.view(-1, 1) shape: [current_beam_size, 1]
        candidate_scores = top_k_scores.view(-1, 1) + log_probs
        
        # Get the top k candidates from all beams combined
        # We want top `beam_width` scores from `current_beam_size * vocab_size` possibilities
        num_candidates = min(beam_width, candidate_scores.numel())
        top_candidate_scores, top_candidate_indices = torch.topk(
            candidate_scores.view(-1), num_candidates
        )

        # Decode the top candidate indices to find the beam and token IDs
        prev_beam_indices = top_candidate_indices // config.VOCAB_SIZE
        next_token_ids = top_candidate_indices % config.VOCAB_SIZE

        # Create the new sequences and scores
        new_sequences = sequences[prev_beam_indices]
        new_sequences = torch.cat([new_sequences, next_token_ids.unsqueeze(1)], dim=1)
        
        # Separate completed sequences from ongoing ones
        is_complete = (next_token_ids == eos_token_id)
        
        # Store completed sequences
        if torch.any(is_complete):
            complete_sequences.extend(new_sequences[is_complete].tolist())
            complete_sequence_scores.extend(top_candidate_scores[is_complete].tolist())
            
            # Reduce the beam width by the number of completed sequences
            beam_width -= is_complete.sum().item()
            if beam_width == 0:
                break

        # Update sequences, scores, and memory for ongoing beams
        is_ongoing = ~is_complete
        sequences = new_sequences[is_ongoing]
        memory = memory[prev_beam_indices[is_ongoing]]
        top_k_scores = top_candidate_scores[is_ongoing].unsqueeze(1)

    # If no sequences were completed, use the current best one
    if not complete_sequences:
        complete_sequences = sequences.tolist()
        complete_sequence_scores = top_k_scores.squeeze().tolist()

    # Find the best sequence among all completed ones
    best_score_index = complete_sequence_scores.index(max(complete_sequence_scores))
    best_seq = complete_sequences[best_score_index]
    
    # Decode, skipping the <sos> token
    predicted_st = tokenizer.decode(best_seq[1:])
    return predicted_st


def main():
    parser = argparse.ArgumentParser(description="Run inference on a PDF file to generate an SMT string.")
    parser.add_argument('--pdf-path', type=str, required=True, help='Path to the input PDF file.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the generated SMT text file.')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Full path to the model checkpoint file to use.')
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for beam search decoding.')
    args = parser.parse_args()

    # --- 1. Setup ---
    print(f"Using device: {config.DEVICE}")
    tokenizer = StTokenizer()
    tokenizer.load(config.TOKENIZER_VOCAB_PATH)
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor()
    ])
    
    # --- 2. Load Model ---
    model = ImageToStModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    )
    
    checkpoint_path = args.checkpoint_path
    print(f"Loading model from: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
        
    model.to(config.DEVICE)

    # --- 3. Process PDF and Predict ---
    print(f"Converting PDF to images: {args.pdf_path}")
    try:
        images = convert_from_path(args.pdf_path)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return

    all_smt_strings = []
    print(f"Generating SMT for {len(images)} page(s)...")
    for i, image in enumerate(tqdm(images, desc="Processing pages")):
        # Convert to grayscale and apply transformations
        image = image.convert('L')
        image_tensor = transform(image)
        
        predicted_st = beam_search_predict(model, image_tensor, tokenizer, beam_width=args.beam_width)
        all_smt_strings.append(predicted_st)
    
    # --- 4. Save Output ---
    final_smt = "\n".join(all_smt_strings)
    with open(args.output_path, 'w') as f:
        f.write(final_smt)
        
    print(f"\nPrediction complete. SMT string saved to: {args.output_path}")
    print("\n--- First 200 characters of generated SMT ---")
    print(final_smt[:200] + "...")


if __name__ == '__main__':
    main()