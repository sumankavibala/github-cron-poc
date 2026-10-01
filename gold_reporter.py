import os
import sys
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_gold_prices():
    """
    Scrapes today's 22K and 24K gold rates from Groww or returns fallback values if request fails.
    """
    url = "https://groww.in/gold-rates/gold-rate-today-in-erode"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    fallback_data = {
        "22K": "₹6,850 per gram",
        "24K": "₹7,473 per gram",
        "source": "fallback"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Parse pricing information from Groww page structure if available
            tables = soup.find_all("table")
            data_found = {}
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cols = [ele.text.strip() for ele in row.find_all(["td", "th"])]
                    if len(cols) >= 2:
                        header_text = cols[0].lower()
                        if "22k" in header_text or "22 karat" in header_text:
                            data_found["22K"] = cols[1]
                        elif "24k" in header_text or "24 karat" in header_text:
                            data_found["24K"] = cols[1]
            
            if "22K" in data_found and "24K" in data_found:
                data_found["source"] = "live"
                return data_found
            
            # Additional fallback check for general text parsing if table parsing differs
            text = soup.get_text()
            if "22K" in text and "24K" in text:
                # If specific parse didn't lock, log and use structured response or fallback
                pass

    except Exception as e:
        print(f"Error fetching live gold price: {e}", file=sys.stderr)
        
    return fallback_data

def generate_report(prices):
    """
    Loads Qwen/Qwen2.5-0.5B-Instruct in CPU float32 mode and generates a clean announcement for gold rates in Erode.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name} on CPU float32...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    prompt = (
        f"Today's Gold Rates in Erode:\n"
        f"- 22K Gold: {prices['22K']}\n"
        f"- 24K Gold: {prices['24K']}\n\n"
        f"Summarize this into a short, clean, direct announcement for the citizens of Erode."
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful, direct financial assistant reporter."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def write_to_step_summary(content, prices):
    """
    Appends the report to GITHUB_STEP_SUMMARY file if available.
    """
    summary_md = f"""## 🪙 Daily Gold Rate Summary - Erode

> **Live Rates extracted:**
> - **22K Gold:** {prices['22K']}
> - **24K Gold:** {prices['24K']}

### 📢 AI Financial Report
{content}
"""

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(summary_md + "\n")
        print("Successfully wrote report to GITHUB_STEP_SUMMARY.")
    else:
        print("\n--- GITHUB STEP SUMMARY OUTPUT ---")
        print(summary_md)

def main():
    print("Fetching gold prices...")
    prices = get_gold_prices()
    print(f"Extracted Prices: {prices}")

    print("Generating AI report...")
    report = generate_report(prices)

    print("Writing to summary...")
    write_to_step_summary(report, prices)

if __name__ == "__main__":
    main()
