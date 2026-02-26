from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    temperature=0.9,
    messages=[
        {"role": "system", "content": "Você é um expert sobre a historia dos LLMs"},
        {
            "role": "user",
            #"content": "Escreva uma hisotira sobre o desenvolvimento do campor de Inteligencia Artificial até a invenção dos LLms"
            "content": "Escreva em uma frase o que é um LMM",
        }

    ]
)

print(completion.choices[0].message.content)