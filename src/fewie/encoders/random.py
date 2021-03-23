import torch

from fewie.encoders.encoder import Encoder, EncoderOutput


class RandomEncoder(Encoder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> EncoderOutput:
        shape = input_ids.shape + (self.embedding_dim,)
        embeddings = torch.rand(shape, device=input_ids.device)
        return EncoderOutput(embeddings=embeddings)
