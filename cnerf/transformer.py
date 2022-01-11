import torch
th = torch
nn = torch.nn
F = nn.functional

B = 32

transformer_model = RFtransformer()
plenoxel_seq = getNeighborhood()
PlenoxelEmbedder(plenoxels)
posenc = addPositionalEncoding()
src = posenc(src)
src = torch.rand((10, B, 512))
tgt = torch.rand((20, B, 512))
out = transformer_model(src, tgt)

class RFtransformer(nn.Transformer):
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def autoencode_field(self, plenoxels, masked_frac=.2, enc_seq_len=25):
        """
        Args:
            plenoxels: [N, 7].
        """
        N = plenoxels.size(0)
        mask_N = round(masked_frac * N)
        masked_plenoxels, unmasked_plenoxels = torch.random.split(plenoxels, mask_N)
        dec_seq_len = round(enc_seq_len / (1-masked_frac))
        mask_per_seq = dec_seq_len - enc_seq_len
        self.estimate_coords(masked_plenoxels[:,:3], unmasked_plenoxels)
        self.encoder()
        return

    def estimate_coords(self, unseen_coords, seen_coords):
        """
        Args:
            unseen_coords: [M, 3]. world coords to determine
            seen_coords: [N, 7]. plenoxels
        Return:
            imputed_values: [M, 4]. ?
        """
        src = self.embed_field(seen_coords)
        imputed_values = self(src, tgt)
        return imputed_values

    def forward(self, src, tgt, src_mask=None,
        tgt_mask=None, memory_mask=None,
        src_key_padding_mask=None, tgt_key_padding_mask=None,
        memory_key_padding_mask=None):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


def addPositionalEncoding(sequence):
    return sequence

