import os
import gzip
import json
import time
import random
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


R2R_VALUNSEEN_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
DIR_NAME = "data/datasets/LLM_REPLYS/"
FILE_NAME = "llm_reply_valunseen"
TEMP_SAVE_PATH = '/data/ckh/Zero-Shot-VLN-FusionMap/tests/llm_reply_valunseen_temp.json'
dones = 0

prompt_template = f"""Parse a navigation instruction delimited by triple quotes and your task is to perform the following actions:
1. Extract Destination: Understand the entire instruction and summarize a description of the destination. The description should be a sentence containing landmark and roomtype.
2. Split instructions: Split the instruction into a series of sub-instructions according to the execution steps. Each sub-instruction should contain a landmark.
3. Infer agent's state constraints: Infer the state constraints that the agent should satisfy for each sub-instruction. There're thee constraint types: location constraints, diretion constraints, object constraints. You need to select an appropriate constraint type and give the corresponding constraint object. Direction constraint object has two types: left, right. Constraints can format as a tuple: (constraint type, constraint object)
4. Make a decision: Analyze the landmarks, actions, and directions in each sub-instruction to determine how the agent should act. For a landmark, the agent has three options: approach, move away, or approach and then move away. For direction, the agent has three options: turn left, turn right, or go forward
Provide your answer in JSON format with the following details:
1. use the following keys: destination, sub-instructions, state-constraints, decisions
2. the value of destination is a string
3. the value of sub-instructions is a list of all sub-instructions
4. the value of state-constraints is a JSON. The key is index start from zero and the value is a list of all constraints, each constraint is a tuple
5. the value of decisions is a nested JSON. The first level JSON's key is index start from zero and it;s value is second level JONS with keys: landmarks, directions. The value of landmarks is a list of tuples, each tuple contains (landmark, action). The value of directions is a list of direction choice for each sub-instruction.
An Example:
User: "exit exercise room to living room, turn slight left, walk behind couch, turn right and walk behind couch, turn left into dining room. Stop next to 2 chairs at glass table."
You: {{"destination": "two chairs near glass table in dining room","sub-instructions":["exit exercise room and go to living room","turn left and walk behind couch", "turn right and walk behind couch", "turn left and go to dining room", "stop next to two chairs near glass table"],"state-constraints":{{0: [("location constraint", "living room")], 1: [("direction constraint", "left"),("object constraint", "couch")], 2: [("direction constraint", "right"),("object constraint", "couch")], 3: [("direction constraint", "left"), ("location constraint", "dining room")], 4: [("object constraint", "chairs"),("object constraint", "table")]}},"decisions":{{0: {{"landmarks":[("exercise room", "move away"), ("living room", "approach")],"directions":["forward"]}}, 1:{{"landmarks":[("couch", "approach")],"directions":["left"]}}, 2:{{"landmarks":[("couch", "approach")],"directions":["right"]}}, 3:{{"landmarks":[("dining room", "approach")],"directions":["left"]}}, 4:{{"landmarks":[("chair", "approach")],"directions":["forward"]}}}}}}
ATTENTION:
1. constraint type: location constraint, object constraint, directions constraint
2. landmark choice: approach, move away, approach then move away
3. direction choice: left, right, forward
"""


def get_reply(client, id, prompt, max_retry_times=3, retry_interval_initial=1):
    global dones
    retry_interval = retry_interval_initial
    reply = None
    for _ in range(max_retry_times):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # model="gpt-3.5-turbo",
                model="gpt-4",
                temperature=0
            )
            reply = eval(chat_completion.choices[0].message.content.strip())
            res = {str(id): reply}
            with open(TEMP_SAVE_PATH, 'a') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            
            dones += 1
            print(id, dones)
            
            return id, reply
        except:
            print("Error, retrying...")
            time.sleep(retry_interval)
            retry_interval *= 2

    dones += 1
    print(id, dones)
    return id, reply


def check_llm_replys(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing_data = json.load(f)
        keys = list(existing_data.keys())
        keys = [int(k) for k in keys]
        max_key = max(keys)
        
        return max_key
    else:
        return 0


def generate_prompts(start_idx, num=None):
    with gzip.open(R2R_VALUNSEEN_PATH, 'r') as f:
        eps_data = json.loads(f.read().decode('utf-8'))
    eps_data = eps_data["episodes"]
    random.shuffle(eps_data)
    if num is None:
        episodes = eps_data[start_idx : ]
    else:
        episodes = eps_data[start_idx : start_idx + num]
    prompts = {}
    for episode in episodes:
        id = episode["episode_id"]
        instruction = episode["instruction"]["instruction_text"]
        prompts[id] = prompt_template + f"\"\"\"{instruction}\"\"\""
    
    return prompts
        

def main():
    client = OpenAI(
        api_key="dc4e8ca91bb9509727662d60ff4ad16b",
        base_url="https://flag.smarttrot.com/v1/"
    )
    
    if os.path.exists(DIR_NAME):
        all_exist_files = sorted(os.listdir(DIR_NAME), reverse=True)
        if len(all_exist_files) > 0:
            current_file = all_exist_files[0]
            file_path = os.path.join(DIR_NAME, current_file)
        else:
            file_path = ''
    else:
        os.makedirs(DIR_NAME, exist_ok=True)
        file_path = ''
    
    start_idx = check_llm_replys(file_path)
    prompts = generate_prompts(start_idx, num=10)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = [executor.submit(get_reply, client, id, prompt) for id, prompt in prompts.items()]
        query2res = {job.result()[0]: job.result()[1] for job in as_completed(results)}
        
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        for key, value in query2res.items():
            if str(key) not in existing_data:
                existing_data[str(key)] = value
        sorted_data = {k: existing_data[k] for k in sorted(existing_data, key=int)}
        
        # avoid overwrite
        length = len(sorted_data)
        new_filename = FILE_NAME + str(length) + ".json"
        with open(os.path.join(DIR_NAME, new_filename), 'w') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    else:
        sorted_data = {k: query2res[k] for k in sorted(query2res, key=int)}
        length = len(sorted_data)
        new_filename = FILE_NAME + str(length) + ".json"
        with open(os.path.join(DIR_NAME, new_filename), 'w') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()