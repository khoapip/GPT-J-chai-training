#!/usr/bin/python3
import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompt import make_prompt

BASE_MODEL = "PygmalionAI/pygmalion-6b"
#PEFT_WEIGHTS = "./out/lora_gpt-j-6B-1e/"
#PEFT_WEIGHTS = "../GPT-j-soda-light-lightwild/"
PEFT_WEIGHTS = "./out/lora_gpt-j-6B-1e/"
#PEFT_WEIGHTS = "tiendung/pygmalion-6b_20-percent-soda_2e"
load_in_8bit = False

if torch.cuda.is_available():
    device = "cuda"
    device_map = {'': 0}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
    #model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, device_map=device_map)
else:
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    #model = PeftModel.from_pretrained(model, PEFT_WEIGHTS)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()
if torch.__version__ >= "2": # tăng tốc
    model = torch.compile(model)

def get_answer(q, max_new_tokens=196, skip_tl=False):
    input_ids = tokenizer(make_prompt(q), return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_length=len(input_ids) + max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1.2,
            eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.eos_token_id,
        )
    origin_output = tokenizer.batch_decode(gen_tokens)[0]

    return origin_output

    """
    output = origin_output.split("###")[2]
    try:
        k = output.index(":")
        if k < 10: output = output[k+1:]
    except:
        output = output
    print(f"\n- - -{origin_output}- - -\n")
    return output.strip()
    """

def main():
    print("\n")
    if True:
        query = "Tanjiro Kamado's Persona: Tanjiro is a skilled swordsman and demon slayer, you and Tanjiro are on a mission. The team is fightting a dangerous demon to save Nezuko, she is Tanjiro's sister.\n<START>\nYou: This demon is too strong. We can't defeat it!\nTanjiro: We have to believe in the team. No matter how devastating the blows might be.\nYou: Run now, I'll hold the demons. We cann't lose you.\nTanjiro: I'm the guy who gets it done, broken bones or not. No matter what, I can fightttt!\nYou: Don't stop! Run! You gotta protect Nezuko-chan!\nTanjiro:"
        print(f"{get_answer(query)}")

if __name__ == "__main__": main()
