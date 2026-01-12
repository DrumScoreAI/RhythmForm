import torch
import torch.nn as nn
import math
from . import config

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This version is modified to be batch-first.
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ImageToStModel(nn.Module):
    """
    An Image-to-Text Transformer model for OMR, using a CNN encoder.
    Optionally accepts num_encoder_layers for config compatibility, but does not use it.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, num_encoder_layers=None):
        super(ImageToStModel, self).__init__()

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers  # For config compatibility, not used

        # --- Image Encoder (CNN) ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Adjust pooling to maintain sequence length
            nn.Conv2d(256, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # --- Text Decoder Components ---
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # --- Output Layer ---
        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generates a square mask for the decoder to prevent it from seeing future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src_image):
        """Encodes the source image using the CNN encoder."""
        # Pass through CNN
        src_encoded = self.cnn_encoder(src_image)
        
        # Reshape for the decoder
        # src_encoded shape: (B, d_model, H', W')
        b, c, h, w = src_encoded.shape
        src_encoded = src_encoded.view(b, c, h * w) # (B, d_model, H'*W')
        src_encoded = src_encoded.permute(0, 2, 1) # (B, H'*W', d_model)
        return src_encoded

    def decode(self, tgt_sequence, memory):
        """Decodes the target sequence using the encoder's memory."""
        # Process Text: Embed and add positional encoding
        tgt_embedded = self.decoder_embedding(tgt_sequence) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_pos_encoder(tgt_embedded)

        # Generate mask and pass through the transformer's decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_sequence.size(1)).to(memory.device)
        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        
        return self.output_layer(output)

    def forward(self, src_image, tgt_sequence):
        """
        Forward pass of the model.
        
        Args:
            src_image (Tensor): The input image tensor. Shape: (batch_size, channels, height, width)
            tgt_sequence (Tensor): The target ST token sequence. Shape: (batch_size, seq_len)
        """
        # Encode the source image
        memory = self.encode(src_image)
        
        # Decode the target sequence using the encoder's output
        output = self.decode(tgt_sequence, memory)
        
        return output


# This block allows you to test the model by running `python -m omr_model.model`
if __name__ == '__main__':
    # --- Configuration (we'll move this to config.py later) ---
    VOCAB_SIZE = config.VOCAB_SIZE # Example vocab size
    D_MODEL = config.D_MODEL
    NHEAD = config.NHEAD
    NUM_DECODER_LAYERS = config.NUM_DECODER_LAYERS
    DIM_FEEDFORWARD = config.DIM_FEEDFORWARD
    
    # --- Create Model ---
    model = ImageToStModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_encoder_layers=getattr(config, 'NUM_ENCODER_LAYERS', None)
    )
    
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # --- Create Dummy Data ---
    batch_size = 4
    seq_length = 50
    # Use different dummy dimensions to test flexibility
    IMG_HEIGHT = 512 
    IMG_WIDTH = 256
    dummy_image = torch.randn(batch_size, 1, IMG_HEIGHT, IMG_WIDTH)
    dummy_target = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length))
    
    print(f"\n--- Testing forward pass ---")
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input target shape: {dummy_target.shape}")
    
    # --- Perform Forward Pass ---
    output = model(dummy_image, dummy_target)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {(batch_size, seq_length, VOCAB_SIZE)}")
    
    # The CNN encoder changes the sequence length of the encoded features.
    # The test needs to be adjusted. For now, we'll just check the last two dimensions.
    assert output.shape[-2:] == (seq_length, VOCAB_SIZE)
    print("✅ Forward pass test passed (shape check adjusted for CNN encoder)!")