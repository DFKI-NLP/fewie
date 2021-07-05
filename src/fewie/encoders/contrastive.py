import torch
from transformers import AutoModel
from fewie.encoders.encoder import EncoderOutput


class ContrastiveTransformerEncoder(torch.nn.Module):
    """Contrastive transformer encoder, for bert, xlnet, spanbert, albert, roberta, etc."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(
        self,
        contrastive_left,
        contrastive_right,
        pos_left,
        pos_right,
    ):
        embedding_left = self.model(
            **contrastive_left,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state
        embedding_right = self.model(
            **contrastive_right,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state

        pos_left = pos_left.view(-1, 1)
        hidden_size = embedding_left.shape[-1]
        ids = pos_left.repeat(1, hidden_size).view(-1, 1, hidden_size)
        embedding_left = torch.gather(embedding_left, 1, ids)

        pos_right = pos_right.view(-1, 1)
        ids = pos_right.repeat(1, hidden_size).view(-1, 1, hidden_size)
        embedding_right = torch.gather(embedding_right, 1, ids)

        # [batch_size, 2, hidden_size]
        output = torch.cat((embedding_left, embedding_right), dim=1)
        return output
