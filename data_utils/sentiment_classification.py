from openai import OpenAI
import json
import time

# ========== Configuration ==========
DATASET = "GDELT"
INPUT_FILE = f"../data/original/{DATASET}/relation2id.json"
OUTPUT_FILE = f"../data/processed/train/{DATASET}/relation2sentiment.txt"
MODEL_NAME = "gpt-4o"
client = OpenAI(api_key="your_api_key_here")  # Replace with your OpenAI API key
# ==============================

def get_sentiment(relation):
    prompt = f""" 
    You are analyzing relation labels from a political event knowledge graph.
    Each relation describes an action or request in a geopolitical context. 
    Unless the relation is truly ambiguous or purely procedural, avoid using "neutral".

    Please classify the sentiment of the following relation into one of:
    - positive (e.g., promoting peace, aid, cooperation)
    - negative (e.g., violence, repression, aggression)
    - neutral (e.g., procedural or ambiguous actions)

    Only reply with one word from the list above.

    Relation: "{relation}"."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip().lower()

        if "positive" in content:
            return "positive"
        elif "negative" in content:
            return "negative"
        elif "neutral" in content:
            return "neutral"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error processing relation '{relation}':\n{e}\n")
        return "neutral"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        relation_dict = json.load(f)

    output_lines = []
    for relation in relation_dict.keys():
        sentiment = get_sentiment(relation)
        output_lines.append(f"{relation}\t{sentiment}")
        print(f"{relation} -> {sentiment}")
        time.sleep(1.2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(output_lines))

    print(f"\n✅ Done! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
