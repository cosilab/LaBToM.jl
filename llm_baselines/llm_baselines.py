import os
import base64
import requests
import pandas as pd
import json
import re

import google.generativeai as genai

# Global variables and paths
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

__DIR__ = os.path.dirname(os.path.abspath(__file__))
STATEMENTS_DIR = os.path.join(__DIR__, '..', 'dataset', 'statements')
STIMULI_DIR = os.path.join(__DIR__, '..', 'dataset', 'stimuli')
IMAGE_DIR = os.path.join(STIMULI_DIR, 'png')
VIDEO_DIR = os.path.join(STIMULI_DIR, 'mp4')

RESULTS_DIR = os.path.join(__DIR__, '..', 'results')
HUMAN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'humans')
LLM_RESULTS_DIR = os.path.join(RESULTS_DIR, 'llm_baselines')

CURR_NARRATIVES_PATH = os.path.join(__DIR__, 'current_narratives.csv')
INIT_NARRATIVES_PATH = os.path.join(__DIR__, 'initial_narratives.csv')

EXP1_CURR_STATEMENTS_PATH = os.path.join(STATEMENTS_DIR, 'exp1', 'exp1_current_in_vs_out.csv')
EXP1_INIT_STATEMENTS_PATH = os.path.join(STATEMENTS_DIR, 'exp1', 'exp1_initial_in_vs_out.csv')

EXP2_CURR_STATEMENTS_PATH = os.path.join(STIMULI_DIR, 'stimuli_current.json')
EXP2_INIT_STATEMENTS_PATH = os.path.join(STIMULI_DIR, 'stimuli_initial.json')

EXP2_CURR_HUMAN_RATINGS_PATH = os.path.join(HUMAN_RESULTS_DIR, 'exp2_current', 'mean_human_data.csv')
EXP2_INIT_HUMAN_RATINGS_PATH = os.path.join(HUMAN_RESULTS_DIR, 'exp2_initial', 'mean_human_data.csv')

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to load belief statements
def load_statements(statements_path):
    with open(statements_path, 'r') as file:
        return {s['name']: s['statements'] for s in json.load(file)}

# Function to construct fixed task instructions
def construct_task_instructions():
    return """
    You're watching someone play the treasure game shown to the left.
    
    The player controls a character, and their goal is to collect one of the four gems (triangle, square, hexagon, or circle).
    
    The rules of the game are as follows:
    - The player can move on the white squares.
    - The player has a full view of the map at all times.
    - The player's goal is to collect exactly one target gem.
    - Keys unlock doors of the same color (e.g. red keys unlock red doors).
    - Each key can only be used once. Keys disappear after use.
    - Each box may be empty or contain exactly one key.
    - The player may or may not know what's in each box.
    - Neither you nor the player can see what's hidden in each box. But both of you can see all other objects in the scene.
    - There are at most two keys hidden among the boxes.
    - The player knows that the puzzle is solvable, which means there are just enough keys to reach any of the target gems.
    - The keys and doors are labeled. The labels are shown on the top right corner of each cell.
    
    Your task is to figure out what the player's goal is, and also what the player initially believed about the contents of the boxes.
    """

# Function to construct task prompt for a specific judgment point and statement
def construct_task_prompt(narrative, statement, few_shot_examples=None, task="exp1"):
    if task == "exp1":
        return f"""
        Now you observe the following:
        
        {narrative}
        
        Given this information, please give a likelihood rating for the following statement about the player's initial belief at timestep 0 from 1 (definitely false) to 100 (definitely true)?

        Statement:
        
        {statement}
        
        Please respond in the following JSON format:
        
        {{
        rating: [x] ,
        }}
        

        The rating should be an integer from 1 to 100. Please provide an explanation to your response.
        """
    elif task == "exp2":
        few_shot_text = ""
        if few_shot_examples:
            few_shot_text = "\nHere are the average ratings given by other people for similar statements:\n"
            for i, (eg_statement, rating) in enumerate(few_shot_examples, start=1):
                few_shot_text += f"\nStatement {i}: {eg_statement}\nRating: {rating}\n"
        
        return f"""
        Now you observe the following:
        
        {narrative}
        
        Given this information, which gem(s) are most likely to be the human agent's goal? And how would you rate the following statement about the player's current belief from 1 (definitely false) to 7 (definitely true)? Rate 4 if you think there is an equal chance of the statement being true and false.
        {few_shot_text}   
        Now please rate the following statement:
        
        {statement}
        
        Please respond in the following JSON format:
        
        {{
        "goal": [gems...],
        "rating": [x]
        }}
        
        The gems should be one of [triangle, square, hexagon, circle] and you should indicate all the likely goal gems in your response. The rating should be an integer from 1 to 7. Please provide an explanation for your response.
        """

# Function to send request to GPT-4o
def query_gpt4o(narrative, statement, base64_image, few_shot_examples=None, task="exp1"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": construct_task_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": construct_task_prompt(narrative, statement, few_shot_examples, task)}
            ]}
        ],
        "max_tokens": 1024,
        "temperature": 0
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# Function to send request to Gemini-Pro
def query_gemini(narrative, statement, media_path, few_shot_examples=None, task="exp1"):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    config = genai.types.GenerationConfig(temperature=0)
    prompt = construct_task_instructions() + "\n\n"
    prompt += construct_task_prompt(narrative, statement, few_shot_examples, task)
    media_file = genai.upload_file(path=media_path)
    response = model.generate_content(
        [media_file, prompt],
        generation_config=config, request_options={"timeout": 600}
    )
    return response.text

# Top-level function that runs LLM baseline for experiment 1
def run_llm_baseline_exp1(
    narratives_path, statements_path, output_path,
    image_dir = IMAGE_DIR, model = 'gpt4o', narrative_cols = ['plan_summary']
):
    narratives_df = pd.read_csv(narratives_path).set_index(['plan_id', 'timestep'])
    statements_df = pd.read_csv(statements_path)
    statement_ratings = []
    statement_probs = []
        
    # Iterate over each statement
    for _, row in statements_df.iterrows():
        plan_id, timestep = row['plan_id'], row['timestep']
        statement = row['statement']

        narrative_row = narratives_df.loc[(plan_id, timestep)]
        narrative = [narrative_row[col] for col in narrative_cols].join('\n\n')

        # Load image
        media_path = os.path.join(image_dir, f"{plan_id}.png")
        base64_image = encode_image(media_path)

        # Query model
        if model == 'gpt4o':
            response_text = query_gpt4o(narrative, statement, base64_image, task="exp1")
        elif model == 'gemini':
            response_text = query_gemini(narrative, statement, media_path, task="exp1")

        # Parse response           
        rating_match = re.search(r'rating\s*:\s*(\d+)', response_text)
        if rating_match:
            rating = int(rating_match.group(1))
        else:
            rating = 50

        # Add rating and statement probs
        statement_ratings.append(rating)
        statement_probs.append(rating / 100)
    
    # Add ratings and probs to dataframe
    statements_df['rating'] = statement_ratings
    statements_df['statement_probs'] = statement_probs

    # Save dataframe
    statements_df.to_csv(output_path, index=False)

# Top-level function that runs LLM baseline for experiment 2
def run_llm_baseline_exp2(
    narratives_path, statements_path, output_path, output_dir,
    human_ratings_path = None, image_dir = IMAGE_DIR, video_dir = VIDEO_DIR,
    model = 'gpt4o', narrative_cols = ['plan_summary'], use_video = False
):
    ratings_df = pd.DataFrame(columns=['plan_id', 'timestep', 'goal'] +\
                              [f'statement_rating_{i}' for i in range(1, 6)])
    narratives_df = pd.read_csv(narratives_path)
    statement_set = load_statements(statements_path)
    
    # Load human ratings for few-shot prompting
    if human_ratings_path is not None:
        human_df = pd.read_csv(human_ratings_file)
        human_df = human_df[['plan_id', 'timestep'] +\
                            [f'statement_rating_{i}' for i in range(1, 6)]]
        human_df = human_df.set_index(['plan_id', 'timestep'])
    else:
        human_df = None
    
    # Iterate over each scenario and judgment point / timestep
    for index, row in narratives_df.iterrows():
        plan_id, timestep = row['plan_id'], row['timestep']
        narrative = [row[col] for col in narrative_cols].join('\n\n')
        new_row = {'plan_id': plan_id, 'timestep': timestep}

        # Load video or image
        if use_video and model == 'gemini':
            media_path = os.path.join(video_dir, f"{plan_id}_{index+1}.mp4")
        else:
            media_path = os.path.join(image_dir, f"{plan_id}.png")
            base64_image = encode_image(media_path)

        # Load example human ratings
        if human_df is not None:
            human_ratings = human_df.loc[(plan_id, timestep)].tolist()
            human_examples = list(zip(statement_set[plan_id], human_ratings))
        else:
            human_examples = None

        # Query model for each statement
        for j in range(5):
            statement = statement_set[plan_id][j]

            # Construct few-shot examples by omitting current statement
            if human_examples is not None:
                few_shot_examples = human_examples[:j] + human_examples[j+1:]
            else:
                few_shot_examples = None

            if model == 'gpt4o':
                response_text = query_gpt4o(narrative, statement, base64_image,
                                            few_shot_examples, task="exp2")
            elif model == 'gemini':
                response_text = query_gemini(narrative, statement, media_path,
                                             few_shot_examples, task="exp2")

            # Parse response           
            goal_match = re.search(r'goal\s*:\s*\[(.*?)\]', response_text)
            if goal_match:
                new_row['goal'] = goal_match.group(1)
            
            rating_match = re.search(r'rating\s*:\s*(\d)', response_text)
            if rating_match:
                new_row[f'statement_rating_{j+1}'] = rating_match.group(1)
            
            # Write response to file
            response_path = os.path.join(output_dir, f"{plan_id}_{timestep}_{j+1}.txt")
            with open(response_path, "w") as file:
                file.write(response_text)
        
        # Add row to dataframe
        ratings_df = ratings_df.append(new_row, ignore_index=True)
    
    ratings_df.to_csv(output_path, index=False)

## Experiment 1 (In-Context vs. Out-of-Context Likelihood Comparison on Full Dataset)

# Run GPT-4o baseline (narrative)
output_path = os.path.join(RESULTS_DIR, 'exp1', 'exp1_current_gpt4o.csv')
run_llm_baseline_exp1(CURR_NARRATIVES_PATH, EXP1_CURR_STATEMENTS_PATH, output_path,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

output_path = os.path.join(RESULTS_DIR, 'exp1', 'exp1_initial_gpt4o.csv')
run_llm_baseline_exp1(INIT_NARRATIVES_PATH, EXP1_INIT_STATEMENTS_PATH, output_path,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

## Experiment 2 (Likelihood Evaluation on Selected Statements)

# Run GPT-4o baseline (plan)
output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_current_plan.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_current_plan')
run_llm_baseline_exp2(CURR_NARRATIVES_PATH, EXP2_CURR_STATEMENTS_PATH, output_path, output_dir,
                      model = 'gpt4o', narrative_cols = ['plan_summary'])

output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_initial_plan.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_initial_plan')
run_llm_baseline_exp2(INIT_NARRATIVES_PATH, EXP2_INIT_STATEMENTS_PATH, output_path, output_dir,
                      model = 'gpt4o', narrative_cols = ['plan_summary'])

# Run GPT-4o baseline (narrative)
output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_current_narrative.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_current_narrative')
run_llm_baseline_exp2(CURR_NARRATIVES_PATH, EXP2_CURR_STATEMENTS_PATH, output_path, output_dir,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_initial_narrative.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_initial_narrative')
run_llm_baseline_exp2(INIT_NARRATIVES_PATH, EXP2_INIT_STATEMENTS_PATH, output_path, output_dir,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

# Run GPT-4o baseline (narrative, few-shot)
output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_current_narrative_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_current_narrative_few_shot')
run_llm_baseline_exp2(CURR_NARRATIVES_PATH, EXP2_CURR_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_CURR_HUMAN_RATINGS_PATH,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

output_path = os.path.join(LLM_RESULTS_DIR, 'gpt4o_exp2_initial_narrative_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gpt4o', 'exp2_initial_narrative_few_shot')
run_llm_baseline_exp2(INIT_NARRATIVES_PATH, EXP2_INIT_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_INIT_HUMAN_RATINGS_PATH,
                      model = 'gpt4o', narrative_cols = ['background', 'narrative'])

# Run Gemini-Pro baseline (narrative, image, few-shot)
output_path = os.path.join(LLM_RESULTS_DIR, 'gemini_exp2_current_narrative_image_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gemini', 'exp2_current_narrative_image_few_shot')
run_llm_baseline_exp2(CURR_NARRATIVES_PATH, EXP2_CURR_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_CURR_HUMAN_RATINGS_PATH, use_video=False,
                      model = 'gemini', narrative_cols = ['background', 'narrative'])

output_path = os.path.join(LLM_RESULTS_DIR, 'gemini_exp2_initial_narrative_image_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gemini', 'exp2_initial_narrative_image_few_shot')
run_llm_baseline_exp2(INIT_NARRATIVES_PATH, EXP2_INIT_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_INIT_HUMAN_RATINGS_PATH, use_video=False,
                      model = 'gemini', narrative_cols = ['background', 'narrative'])

# Run Gemini-Pro baseline (narrative, video, few-shot)
output_path = os.path.join(LLM_RESULTS_DIR, 'gemini_exp2_current_narrative_video_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gemini', 'exp2_current_narrative_video_few_shot')
video_dir = os.path.join(VIDEO_DIR, 'exp2_current')
run_llm_baseline_exp2(CURR_NARRATIVES_PATH, EXP2_CURR_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_CURR_HUMAN_RATINGS_PATH,
                      use_video=True, video_dir=video_dir,
                      model = 'gemini', narrative_cols = ['background', 'narrative'])

output_path = os.path.join(LLM_RESULTS_DIR, 'gemini_exp2_initial_narrative_video_few_shot.csv')
output_dir = os.path.join(LLM_RESULTS_DIR, 'gemini', 'exp2_initial_narrative_video_few_shot')
video_dir = os.path.join(VIDEO_DIR, 'exp2_initial')
run_llm_baseline_exp2(INIT_NARRATIVES_PATH, EXP2_INIT_STATEMENTS_PATH, output_path, output_dir,
                      human_ratings_path = EXP2_INIT_HUMAN_RATINGS_PATH,
                      use_video=True, video_dir=video_dir,
                      model = 'gemini', narrative_cols = ['background', 'narrative'])