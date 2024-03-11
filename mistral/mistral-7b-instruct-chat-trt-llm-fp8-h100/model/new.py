import time
from openai import OpenAI
import random
from transformers import AutoTokenizer
import requests
import tiktoken

MAX_TOKENS = 512

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = tiktoken.encoding_for_model("gpt-4")


def make_openai_request(api_key, base_url, prompt):
    client = OpenAI(api_key=api_key, base_url=base_url)
    start_time = time.time()
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=MAX_TOKENS,
    )
    first_response_time = None
    response_txt = ""
    for chunk in response:
        if first_response_time is None:
            if chunk.choices[0].delta.content is not None:
                first_response_time = time.time()
                first_chunk_txt = chunk.choices[0].delta.content
        delta = chunk.choices[0].delta
        if delta.content is not None:
            response_txt += delta.content
    last_response_time = time.time()
    return start_time, first_response_time, last_response_time, response_txt, first_chunk_txt

def make_non_openai_request(api_key, base_url, prompt):
    start_time = time.time()
    response = requests.post(
        f"{base_url}",
        headers={"Authorization": f"Api-Key {api_key}"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS},
        stream=True
    )
    first_response_time = None
    response_txt = ""
    for chunk in response.iter_content(chunk_size=None):
        if first_response_time is None:
            first_response_time = time.time()
            first_chunk_txt = chunk.decode("utf-8")
        response_txt += chunk.decode("utf-8")
    last_response_time = time.time()
    return start_time, first_response_time, last_response_time, response_txt, first_chunk_txt

def measure_client_ttft(api_key, base_url, prompts, make_request):
    ttft_list = []
    tps_list = []
    for prompt in prompts:
        start_time, first_response_time, last_response_time, response_txt, first_chunk_txt = make_request(api_key, base_url, prompt)
        ttft_time = first_response_time - start_time
        ttft_list.append(ttft_time)
        encoded = tokenizer.encode(response_txt)
        first_chunk_encoded = tokenizer.encode(first_chunk_txt)
        tps = (len(encoded) - len(first_chunk_encoded)) / (last_response_time - start_time)
        tps_list.append(tps)

    for ttft, tps in zip(ttft_list, tps_list):
        print(f"TTFT: {ttft:.2f} seconds, TPS: {tps:.2f}")

    average_ttft = sum(ttft_list) / len(ttft_list)
    average_tps = sum(tps_list) / len(tps_list)
    print(f"\nAverage TTFT: {average_ttft:.2f} seconds"
            f"\nAverage TPS: {average_tps:.2f}")

# Sample prompts of varying lengths
all_prompts = [
    "What are the potential benefits and risks of artificial intelligence?",
    "How does climate change affect global biodiversity?",
    "Explain the significance of the discovery of the Higgs boson particle.",
    "What role did the Silk Road play in shaping ancient civilizations?",
    "How do vaccines work to protect the body from disease?",
    "Discuss the cultural impact of the Renaissance period in Europe.",
    "What are the challenges of deep space exploration?",
    "How do quantum computers differ from traditional computers?",
    "What are the ethical implications of cloning technology?",
    "Describe the process of photosynthesis in plants.",
    "How does the stock market influence the global economy?",
    "What is the importance of the Rosetta Stone in understanding ancient languages?",
    "Explain the principle behind nuclear fusion as an energy source.",
    "What were the key factors leading to the fall of the Roman Empire?",
    "How does the human brain process and store memories?",
    "Discuss the impact of social media on modern communication.",
    "What are the main challenges in achieving sustainable development?",
    "How do black holes form and what are their properties?",
    "Explain the cultural significance of the Pyramids of Giza.",
    "What is the role of gut microbiota in human health?",
    "Discuss the implications of gene editing technologies like CRISPR.",
    "How does the theory of relativity challenge classical physics?",
    "What were the major outcomes of the Industrial Revolution?",
    "How do renewable energy sources impact the environment?",
    "What is the significance of the Great Barrier Reef and its current threats?",
    "Discuss the historical impact of the printing press.",
    "How do autonomous vehicles navigate and make decisions?",
    "What are the implications of extending human lifespan through technology?",
    "Explain the process of natural selection in evolution.",
    "How does the blockchain technology work and what are its uses?",
    "What were the causes and effects of the American Revolution?",
    "How do antidepressants work in the brain?",
    "Discuss the cultural and historical significance of Machu Picchu.",
    "What are the challenges of Mars colonization?",
    "How does the internet work at a technical level?",
    "What are the principles of effective leadership?",
    "Explain the impact of deforestation on the global ecosystem.",
    "How do satellites contribute to modern communication and navigation?",
    "What are the health benefits and risks of intermittent fasting?",
    "Discuss the role of artificial intelligence in healthcare.",
    "How do tectonic plates shape the Earth's surface?",
    "What are the fundamental concepts of quantum mechanics?",
    "Explain the significance of the Mona Lisa in art history.",
    "How does encryption ensure data security?",
    "What is the impact of overfishing on marine ecosystems?",
    "Discuss the evolution of video game technology.",
    "How do solar panels convert sunlight into electricity?",
    "What are the psychological effects of social isolation?",
    "Explain the process of water cycle and its importance to life on Earth.",
    "What are the key challenges in the field of robotics?",
    "Discuss the significance of biodiversity conservation."
]

# MODEL_ID = "rwnp7d13"
API_KEY = "90mInCv9.UWggVtvVDdjskNthuQ8uZYMB75F1bEAI"

# print(f"Running non-OpenAI requests")
# selected_prompts = random.sample(all_prompts, 10)
# measure_client_ttft(API_KEY, f"https://model-{MODEL_ID}.api.baseten.co/production/predict", selected_prompts, make_non_openai_request)

# print(f"\nRunning OpenAI-client requests")
selected_prompts = random.sample(all_prompts, 10)
# measure_client_ttft(API_KEY, f"https://bridge.baseten.co/{MODEL_ID}/v1", selected_prompts, make_openai_request)
measure_client_ttft(API_KEY, "http://localhost:8081/v1/models/model:predict", selected_prompts, make_non_openai_request)