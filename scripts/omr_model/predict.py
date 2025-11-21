import torch
import torch.nn.functional as F
import argparse

from . import config
from .dataset import ScoreDataset
from .tokenizer import SmtTokenizer
from .model import ImageToSmtModel

def beam_search_predict(model, image_tensor, tokenizer, beam_width=5, max_len=200):
    """
    Generates an SMT string prediction using beam search.
    """
    model.eval()
    
    sos_token_id = tokenizer.token_to_id['<sos>']
    eos_token_id = tokenizer.token_to_id['<eos>']
    
    src_image = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    # --- Encoder Step ---
    # The image is processed by the encoder only once.
    with torch.no_grad():
        # This is a simplified way to get the encoder's output.
        # We manually pass the image through the encoder part of the model.
        src_embedded = model.patch_embedding(src_image).flatten(2).permute(0, 2, 1)
        src_embedded = src_embedded.permute(1, 0, 2)
        src_embedded = model.encoder_pos_encoder(src_embedded)
        src_embedded = src_embedded.permute(1, 0, 2)
        memory = model.transformer.encoder(src_embedded)

    # --- Decoder (Beam Search) Step ---
    # A beam is a tuple of (log_probability, sequence)
    # We start with one beam: an empty sequence with a log probability of 0.
    beams = [(0.0, [sos_token_id])]
    
    for _ in range(max_len):
        all_candidates = []
        
        for log_prob, seq in beams:
            # If a beam has already reached the end, keep it as a candidate.
            if seq[-1] == eos_token_id:
                all_candidates.append((log_prob, seq))
                continue

            # Get the model's prediction for the next token
            with torch.no_grad():
                tgt_sequence = torch.tensor([seq], dtype=torch.long).to(config.DEVICE)
                
                # Manually pass data through the decoder part
                tgt_embedded = model.decoder_embedding(tgt_sequence) * torch.tensor(config.D_MODEL).sqrt()
                tgt_embedded = tgt_embedded.permute(1, 0, 2)
                tgt_embedded = model.decoder_pos_encoder(tgt_embedded)
                tgt_embedded = tgt_embedded.permute(1, 0, 2)
                tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(config.DEVICE)
                
                output = model.transformer.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                output = model.output_layer(output)

            # Get the log probabilities of the next possible tokens
            last_token_logits = output[:, -1, :]
            log_probs = F.log_softmax(last_token_logits, dim=-1)
            
            # Get the top `beam_width` next tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            # Create new candidate beams
            for i in range(beam_width):
                next_token_id = top_indices[0, i].item()
                new_log_prob = log_prob + top_log_probs[0, i].item()
                new_seq = seq + [next_token_id]
                all_candidates.append((new_log_prob, new_seq))

        # Sort all candidates by their log probability and keep the top `beam_width`
        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        beams = ordered[:beam_width]
        
        # If the top beam has ended, we can stop.
        if beams[0][1][-1] == eos_token_id:
            break
            
    # The best sequence is the one with the highest log probability
    best_seq = beams[0][1]
    predicted_smt = tokenizer.decode(best_seq)
    return predicted_smt


def main():
    parser = argparse.ArgumentParser(description="Generate a prediction for a sample from the dataset.")
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number of the model checkpoint to load.')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of the sample in the full dataset to predict.')
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for beam search decoding.')
    args = parser.parse_args()

    # --- 1. Setup ---
    print(f"Using device: {config.DEVICE}")
    tokenizer = SmtTokenizer()
    tokenizer.load(config.TOKENIZER_VOCAB_PATH)
    full_dataset = ScoreDataset(manifest_path=config.DATASET_JSON_PATH)
    
    # --- 2. Load Model ---
    model = ImageToSmtModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        image_height=config.IMG_HEIGHT,
        image_width=config.IMG_WIDTH,
        patch_size=config.PATCH_SIZE
    )
    
    checkpoint_path = config.CHECKPOINT_DIR / f"model_epoch_{args.epoch}.pth"
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    # --- 3. Predict ---
    sample = full_dataset[args.sample_idx]
    image_tensor = sample['image']
    ground_truth_smt = sample['smt_string']
    
    print(f"\n--- Generating Prediction (Beam Width: {args.beam_width}) ---")
    predicted_smt = beam_search_predict(model, image_tensor, tokenizer, beam_width=args.beam_width)
    
    print("\n--- Ground Truth SMT ---")
    print(ground_truth_smt)
    
    print("\n--- Predicted SMT ---")
    print(predicted_smt)


if __name__ == '__main__':
    main()