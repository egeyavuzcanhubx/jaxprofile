import jax
import jax.numpy as jnp
import flaxmodels as fm

class GPT2Model:
    def __init__(self, prompt: str, model_name: str = "gpt2-xl", num_tokens: int = 20):
        print("[GPT2Model] Initializing GPT-2 model with prompt:", prompt)
        self.prompt = prompt
        self.model_name = model_name
        self.num_tokens = num_tokens

        self.key = jax.random.PRNGKey(0)
        print("[GPT2Model] Loading tokenizer for model:", self.model_name)
        self.tokenizer = fm.gpt2.get_tokenizer()
        self.generated = self.tokenizer.encode(self.prompt)
        print("[GPT2Model] Encoded prompt:", self.generated)
        # context: shape (batch_size, sequence_length)
        self.context = jnp.array([self.generated])
        self.past = None

        print("[GPT2Model] Initializing model parameters...")
        self.model = fm.gpt2.GPT2LMHeadModel(pretrained=self.model_name)
        self.params = self.model.init(self.key, input_ids=self.context, past_key_values=self.past)
        print("[GPT2Model] Model parameters initialized.")

    def generate(self):
        print("[GPT2Model] Starting generation of", self.num_tokens, "tokens.")
        for i in range(self.num_tokens):
            print(f"[GPT2Model] Generation step {i+1}/{self.num_tokens}")
            output = self.model.apply(
                self.params,
                input_ids=self.context,
                past_key_values=self.past,
                use_cache=True
            )
            # Force computation to complete before moving on (optional)
            output['logits'].block_until_ready()

            # Get the most probable next token
            token = int(jnp.argmax(output['logits'][..., -1, :]))
            print(f"[GPT2Model] Generated token: {token}")
            self.generated += [token]

            # For efficient generation, only feed the last generated token
            self.context = jnp.expand_dims(jnp.array([token]), axis=0)
            self.past = output['past_key_values']

        decoded = self.tokenizer.decode(self.generated)
        print("[GPT2Model] Generation completed. Decoded sequence:")
        print(decoded)
        return decoded
