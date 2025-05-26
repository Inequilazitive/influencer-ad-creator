from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LlamaGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # uses GPU if available
            torch_dtype="auto"
        )
        self.max_tokens = max_tokens
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt):
        outputs = self.generator(prompt, max_new_tokens=self.max_tokens, do_sample=True, temperature=0.7)
        return outputs[0]['generated_text'].replace(prompt, "").strip()
