
import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        result = []

        for i in range(new_chars):

            # Crop context to context_length
            context = context[:, -context_length:]

            # Get logits for the current context
            with torch.no_grad():
                logits = model(context)

            # Take logits from the last position
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Reset generator state as required by the fixed code
            generator.set_state(initial_state)

            # Sample next token
            next_token = torch.multinomial(
                probs,
                1,
                generator=generator
            )

            # Append token to context
            context = torch.cat(
                (context, next_token),
                dim=1
            )

            # Convert token ID to character
            token_id = next_token.item()
            result.append(int_to_char[token_id])

        return "".join(result)