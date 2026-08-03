import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        
        # Keep track of generated tokens to decode at the end
        generated_indices = []

        for i in range(new_chars):

            # Crop context to max context_length
            context_cond = context[:, -context_length:]
            
            # Forward pass to get logits for the sequence
            logits = model(context_cond)
            
            # Get logits for the last token position: shape (1, vocab_size)
            last_logits = logits[:, -1, :]
            
            # Convert logits to probability distribution
            probs = torch.softmax(last_logits, dim=-1)

            # The line where you call torch.multinomial(). Pass in the generator as well.
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
            generator.set_state(initial_state)

            # Append sampled token to the running context tensor along sequence dimension
            context = torch.cat((context, next_token), dim=1)
            
            # Store integer token for output decoding
            generated_indices.append(next_token.item())

        # Map generated token IDs back to string characters
        generated_text = "".join([int_to_char[idx] for idx in generated_indices])
        return generated_text

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!