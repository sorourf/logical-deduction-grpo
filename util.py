
import reasoning_gym
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from peft import AutoPeftModelForCausalLM
import torch
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
import json

from torch.nn.utils.rnn import pad_sequence

console = Console(markup=True)

def left_pad(tensors, padding_value=0):
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")

def pprint(content, title=None, is_json=False, **kwargs):
    if is_json:
        content = JSON(json.dumps(content))
    if title is not None:
        content = Panel(content, title=title)
    console.print(content, **kwargs)

batch_size = 2
n_rollouts = 2
max_new_tokens = 100

system_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.

Do not generate new code. Do not write python code.

You may also be given examples by the user telling you the expected response format.
Follow the format of the examples, but solve the specific problem asked by the user, not the examples.

Very important - Remember again, your output format should be:
<think> reasoning process here </think>
<answer> answer here </answer>

Your response will be scored by extracting the substring between the <answer>...</answer> tags.
It is critical to follow the above format.
feature_extraction_utilsling to follow the response format will result in a penalty.
"""

def load_model(model_name):
    # load model and make the whole thing trainable
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    for param in llm.parameters():
        param.requires_grad = True
    return llm
  
def load_peft_model(model_name):
    # Load LORA model and only make them trainable
    llm = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )
    llm.print_trainable_parameters()
    return llm

  
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_data_loader(env, tokenizer):
    dataloader = get_dataloader(
        env,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    return dataloader



class ReasoningDataset(Dataset):
    def __init__(
        self,
        environment_name,
        tokenizer,
        system_prompt=None,
    ):
        self.data = reasoning_gym.create_dataset(environment_name, seed=42)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        if self.tokenizer is not None:
            assert (
                self.system_prompt is not None
            ), "System prompt is required for tokenization"

    def __len__(self):
        return len(self.data)

    def create_prompt(self, x):
        chat_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": x},
        ]
        chat_template = self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return chat_template

    def __getitem__(self, i):
        x = self.data[i]
        data = {"inputs": {}, "validator": x}
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                self.create_prompt(x["question"]),
                return_tensors="pt",
            )
            data["inputs"]["input_ids"] = tokenized["input_ids"]
            data["inputs"]["attention_mask"] = tokenized["attention_mask"]
        return data


def collate_fn(batch, pad_token_id):
    return {
        "validator": [item["validator"] for item in batch],
        "inputs": {
            "input_ids": pad_sequence(
                [item["inputs"]["input_ids"][0] for item in batch],
                batch_first=True,
                padding_value=pad_token_id,
                padding_side="left",
            ),
            "attention_mask": pad_sequence(
                [item["inputs"]["attention_mask"][0] for item in batch],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )
        }
    }


def get_dataloader(
    dataset,
    tokenizer,
    batch_size=batch_size,
    system_prompt=system_prompt
):
    dataset = ReasoningDataset(
        dataset, tokenizer, system_prompt
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.eos_token_id),
    )
    return dataloader
