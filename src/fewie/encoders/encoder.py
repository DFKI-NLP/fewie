from typing import Optional

import torch
from transformers.file_utils import ModelOutput


class EncoderOutput(ModelOutput):
    embeddings: Optional[torch.FloatTensor] = None


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> EncoderOutput:
        pass
