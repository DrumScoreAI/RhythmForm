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
    An Image-to-Text Transformer model for OMR.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 patch_size=16):
        super(ImageToStModel, self).__init__()

        self.d_model = d_model
        
        # --- Image Encoder Components ---
        self.patch_size = patch_size
        
        # Layer to convert image into patches and embed them
        self.patch_embedding = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        # The PositionalEncoding with default max_len of 10000.
        self.encoder_pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Text Decoder Components ---
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        # The decoder also needs a large max_len for long sequences.
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Transformer ---
        # Using PyTorch's standard Transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important for our data shape
        )

        # --- Output Layer ---
        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generates a square mask for the decoder to prevent it from seeing future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src_image):
        """Encodes the source image."""
        # Process Image: Create patches, embed, and add positional encoding
        src_embedded = self.patch_embedding(src_image)
        src_embedded = src_embedded.flatten(2)
        src_embedded = src_embedded.permute(0, 2, 1) # (B, Num_Patches, d_model)
        src_embedded = self.encoder_pos_encoder(src_embedded)
        
        # Pass through the transformer's encoder
        return self.transformer.encoder(src_embedded)

    def decode(self, tgt_sequence, memory):
        """Decodes the target sequence using the encoder's memory."""
        # Process Text: Embed and add positional encoding
        tgt_embedded = self.decoder_embedding(tgt_sequence) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_pos_encoder(tgt_embedded)

        # Generate mask and pass through the transformer's decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_sequence.size(1)).to(memory.device)
        output = self.transformer.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        
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
    PATCH_SIZE = config.PATCH_SIZE
    VOCAB_SIZE = config.VOCAB_SIZE # Example vocab size
    D_MODEL = config.D_MODEL
    NHEAD = config.NHEAD
    NUM_ENCODER_LAYERS = config.NUM_ENCODER_LAYERS
    NUM_DECODER_LAYERS = config.NUM_DECODER_LAYERS
    DIM_FEEDFORWARD = config.DIM_FEEDFORWARD
    
    # --- Create Model ---
    model = ImageToStModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        patch_size=PATCH_SIZE
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
    
    assert output.shape == (batch_size, seq_length, VOCAB_SIZE)
    print("✅ Forward pass test passed!")