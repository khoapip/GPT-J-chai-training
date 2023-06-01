import json


TEMPLATE_PROMPT = "character's persona: [DESC]\n<START>[HISTORY]\nCharacter Reply: "

data = []


with open("gpteacher.json", "r") as f:
    sample_list = json.load(f)


for sample in sample_list:
    if "Chat History:" in sample["instruction"]:
        desc, history = sample["instruction"].split("Chat History:")
        history = "\n" + history
    else:
        desc = sample["instruction"]
        history = ""



    if sample["input"] == "":
        sample["input"] = desc

    prompt = TEMPLATE_PROMPT.replace("[DESC]", desc)
    prompt = prompt.replace("[HISTORY]", history)
    response = sample["output"]

    data.append({"prompt": prompt, "response": response})

with open("gpteacher_processed.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
