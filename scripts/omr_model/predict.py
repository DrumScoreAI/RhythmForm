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
    Generates an ST string prediction using beam search.
    """
    model.eval()
    
    sos_token_id = tokenizer.token_to_id['<sos>']
    eos_token_id = tokenizer.token_to_id['<eos>']
    
    # The transform is now applied outside, so the image_tensor is ready
    src_image = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    # --- Encoder Step ---
    # The image is processed by the encoder only once.
    with torch.no_grad():
        memory = model.encode(src_image)

    # --- Decoder (Beam Search) Step ---
    # A beam is a tuple of (log_probability, sequence)
    beams = [(0.0, [sos_token_id])]
    
    for _ in range(max_len):
        all_candidates = []
        
        for log_prob, seq in beams:
            if seq[-1] == eos_token_id:
                all_candidates.append((log_prob, seq))
                continue

            with torch.no_grad():
                tgt_sequence = torch.tensor([seq], dtype=torch.long).to(config.DEVICE)
                output = model.decode(tgt_sequence, memory)

            last_token_logits = output[:, -1, :]
            log_probs = F.log_softmax(last_token_logits, dim=-1)
            
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token_id = top_indices[0, i].item()
                new_log_prob = log_prob + top_log_probs[0, i].item()
                new_seq = seq + [next_token_id]
                all_candidates.append((new_log_prob, new_seq))

        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        beams = ordered[:beam_width]
        
        if beams[0][1][-1] == eos_token_id:
            break
            
    best_seq = beams[0][1]
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
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
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