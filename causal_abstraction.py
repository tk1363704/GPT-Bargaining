import time
from retry.api import retry_call
import openai
import random
import keys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

THRESHOLD = 0.7


def get_response(prompt):
    chatgpt_query_time = time.time()
    # print(prompt)
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model":"gpt-3.5-turbo",
            "messages":[{"role": "user", "content": prompt}],
            "api_key":random.choice(keys.keys),
            "n":1,"temperature":0.25, "request_timeout":30}, tries=3, delay=1, jitter=1)
        #completion = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo",
        #    messages=[{"role": "user", "content": prompt}],
        #    api_key=random.choice(keys),
        #    n=4,
        #    temperature=0.25,
        #)
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices':[{'message':{'content':'Error'}}]}
    """
    nocontext_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": no_context_prompt}],
        api_key=random.choice(keys),
        n=3,
        temperature=0.25,
    )
    """
    print("chatgpt query time is : {}".format(str(time.time()-chatgpt_query_time)))
    return completion

def get_input(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    list_ = []

    for (a,b,c) in zip(df['Level_1'], df['Level_2'], df['Level_3']):
        dict_ = {'Level_1':a, 'Level_2':b, 'Level_3':c}
        list_.append(dict_)
    return list_

def compare_references(refers, cands):
    if len(refers) != len(cands):
        print('ERROR! The length for references and candidates are not the same!')
    else:
        references, candidates = [], []
        correct_count = 0
        for ref, cand in zip(refers, cands):
            if ref.strip() == cand.strip():
                correct_count += 1
            # Tokenize sentences
            references.append(ref.split())
            candidates.append(cand.split())

        # Calculate BLEU for each pair
        scores = [sentence_bleu([ref], cand) for ref, cand in zip(references, candidates)]

        # Calculate average BLEU
        average_bleu = sum(scores) / len(scores)
        print(f"Average BLEU Score: {average_bleu}")
        correctness = float(correct_count / len(references))
        print(f"Perfect Match Rate: {correctness}")

def extract(content):
    lists = [x.strip() for x in content.strip().split('\n') if x.strip() != '']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result


def main():
    prompt = ""
    # Open the file in read mode
    with open('causal_prompt.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()
    for line in lines:
        prompt += line
    prompt = prompt.strip()

    file = 'causal_abstraction.csv'
    inputs = get_input(file)

    column_headers = ['Ref_Level_1', 'Can_Level_1', 'Ref_Level_2', 'Can_Level_2', 'Ori_Level_3']

    # Create an empty DataFrame with these headers
    df_ = pd.DataFrame(columns=column_headers)
    references, candidates = [], []
    len_ = len(inputs)

    for i, input in enumerate(inputs):
        print('--------------- {} in {} ---------------'.format(i+1, len_))
        temp_prompt = prompt.replace("$$$$$", input['Level_3'])
        completion = get_response(temp_prompt)
        content = completion['choices'][0]['message']['content']
        if content != 'Error':
            level_2_result, level_1_result = extract(content)
            if level_1_result != '':
                candidates.append(level_1_result)
                references.append(input['Level_1'])
        else:
            level_2_result, level_1_result = '', ''
        new_row = {'Ref_Level_1': input['Level_1'],
                   'Can_Level_1': level_1_result,
                   'Ref_Level_2': input['Level_2'],
                   'Can_Level_2': level_2_result,
                   'Ori_Level_3': input['Level_3']}
        df_ = pd.concat([df_, pd.DataFrame([new_row])], ignore_index=True)
    compare_references(references, candidates)
    # Write the DataFrame to a CSV file
    df_.to_csv('causal_abstraction_output.csv', index=False)

def retrieve_the_gold(input_):
    df = pd.read_csv('causal_abstraction_output.csv')
    ref1 = list(set(df['Ref_Level_1'].tolist()))

    prompt = ""
    # Open the file in read mode
    with open('causal_prompt.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()
    for line in lines:
        prompt += line
    prompt = prompt.strip()
    temp_prompt = prompt.replace("$$$$$", input_)
    completion = get_response(temp_prompt)
    content = completion['choices'][0]['message']['content']
    max_sim = (-1, '')
    if content != 'Error':
        level_2_result, level_1_result = extract(content)
        if level_1_result != '':
            abstract = level_1_result
            for ref in ref1:
                sim_ = sentence_bleu([ref.split()], abstract.split())
                max_sim = (sim_, ref) if max_sim[0] < sim_ else max_sim
            max_sim = max_sim if max_sim[0] > THRESHOLD else (-1, '')
    return max_sim


if __name__ == "__main__":
    # main()
    print(retrieve_the_gold("I'd like a hamburger"))

