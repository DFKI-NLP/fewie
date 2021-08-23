import torch
from transformers import AutoModel

from fewie.encoders.encoder import Encoder, EncoderOutput


class AlbertEncoder(Encoder):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        #        encoder_hidden_states=None,
        #       encoder_attention_mask=None,
        output_attentions=None,
        return_dict=None,
    ) -> EncoderOutput:

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            #           encoder_hidden_states=encoder_hidden_states,
            #          encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
            return_dict=True,
        )

        return EncoderOutput(embeddings=output.last_hidden_state)
