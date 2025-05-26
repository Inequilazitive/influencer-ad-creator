import gradio as gr
from scraping.twitter_scraper import scrape_twitter
from analysis.persona_builder import build_persona_from_text
from generation.llama_wrapper import LlamaGenerator
from generation.ad_generator import generate_ad
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

llama = LlamaGenerator()

def run_pipeline(handles: str):
    usernames = [u.strip() for u in handles.split(",") if u.strip()]
    all_tweets = []

    for user in usernames:
        tweets = scrape_twitter(user, config['scraping']['num_posts'])
        all_tweets.extend(tweets)

    persona = build_persona_from_text(all_tweets, config['persona']['recent_weight'])
    ad_text = generate_ad(persona, llama, config['generation']['prompt_template'])

    return ad_text

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Influencer Ad Generator")
    inp = gr.Textbox(label="Twitter Handles (comma-separated)")
    out = gr.Textbox(label="Generated Ad Copy")
    btn = gr.Button("Generate Ad")
    btn.click(fn=run_pipeline, inputs=inp, outputs=out)

demo.launch(share=True)
