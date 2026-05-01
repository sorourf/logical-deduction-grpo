import openai
from reasoning_gym import create_dataset
from utils import system_prompt
import dotenv
import os
import asyncio
import json
import backoff
import sys

'''
create a .env file and paste your OPENAI_API_KEY there.

Run this script using: `python data_generation.py syllogism 200` to generate 200 examples of the syllogism task
'''


dotenv.load_dotenv()
client = openai.AsyncClient()
ENVIRONMENT = sys.argv[1] # "syllogism"
model = "gpt-4.1-mini"
semaphore = asyncio.Semaphore(50)
num_datapoints = 1000 if len(sys.argv) == 2 else int(sys.argv[2]) # By default generates 1000 examples
system_prompt = (
    system_prompt
    + """You will also be provided the real answer. Your thinking should eventually result in producing the real answer."""
)

dataloader = create_dataset(name=ENVIRONMENT, size=num_datapoints)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_response(item):
    async with semaphore:  # Use the global semaphore
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
    Question: {item['question']}
    Metadata: {item['metadata']}
    Answer: {item['answer']}
                    """,
            },
        ]
        response = await client.chat.completions.create(messages=messages, model=model)
        print("-" * 5)
        print(response.choices[0].message.content)
        print("GT answer: ", item["answer"])
        print("-" * 5)
        return {
            "question": item["question"],
            "metadata": item["metadata"],
            "answer": item["answer"],
            "response": response.choices[0].message.content,
        }


async def main():
    responses = await asyncio.gather(*[generate_response(item) for item in dataloader])
    fname = f"responses_{ENVIRONMENT}_{model}.json"    
    json.dump(responses, open(fname, "w"), indent=4)
    print(f"Saved responses to {fname}")


if __name__ == "__main__":
    asyncio.run(main())
