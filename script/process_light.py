import json
from tqdm import tqdm

TEMPLATE_PROMPT = "[CHARACTER]'s persona: [DESC]\n<START>\n[HISTORY]\n[USER]: "

with open("light_wild.jsonl", "r") as f:
    json_list = list(f)

data = []

for sample in json_list:
    result = json.loads(sample)

    dialog = {"con" : [], "_self_name": "", "_partner_name":"", "_setting_desc": "", "_self_persona": ""}

    for i  in range(len(result['dialog'])):
        dia = result['dialog'][i]
        cons = dia[0]["text"].split("\n")

        for con in cons:
            if con.startswith("_partner_say") or con.startswith("_self_say"):
                dialog["con"].append(con)
            else:
                if con == "":
                    continue
                dialog[con.split()[0]] = con.strip(con.split()[0])

    if len(dialog["con"]) <= 1:
        continue

    if  dialog["_partner_name"] == "":
        dialog["_partner_name"] = "You"


    dialog["output"] = dialog["con"][-1]
    dialog["con"].pop()

    history = ("\n".join(dialog["con"]).replace("_self_say", dialog["_self_name"] + ": ").replace("_partner_say", dialog["_partner_name"] + ": "))
    prompt = TEMPLATE_PROMPT.replace("[CHARACTER]", dialog["_self_name"])
    prompt = prompt.replace("[DESC]", dialog["_setting_desc"] + ". " + dialog["_self_persona"])
    prompt = prompt.replace("[HISTORY]", history)
    prompt = prompt.replace("[USER]", dialog["_partner_name"])
    output = dialog["output"].strip(dialog["output"].split()[0])

    data.append({"prompt": prompt, "response": output})

with open("light_wild_processed.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

