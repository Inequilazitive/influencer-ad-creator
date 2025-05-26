def build_prompt(persona, prompt_template_path):
    with open(prompt_template_path, 'r') as f:
        template = f.read()

    keywords = ", ".join(persona["keywords"])
    return template.replace("{{keywords}}", keywords)

def generate_ad(persona, llama, prompt_path):
    prompt = build_prompt(persona, prompt_path)
    ad_text = llama.generate(prompt)
    return ad_text
