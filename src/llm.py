import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def ask_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Sends the prompt to Groq's hosted LLaMA model and returns the reply text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    test_prompt = "In one sentence, what is Southeast University known for?"
    reply = ask_llm(test_prompt)
    print(reply)