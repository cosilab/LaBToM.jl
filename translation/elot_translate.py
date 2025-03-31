import os
import argparse

import pandas as pd
import numpy as np

from genparse import InferenceSetup
from genparse.cfglm import BoolCFGLM
from genparse.lark_interface import LarkStuff

import grammars
import prompts

# Directories
__DIR__ = os.path.dirname(os.path.abspath(__file__))

STATEMENTS_DIR = os.path.join(__DIR__, '..', 'dataset', 'statements')
CURR_STATEMENTS_PATH = os.path.join(STATEMENTS_DIR, 'exp2_current_gold_unique.csv')
INIT_STATEMENTS_PATH = os.path.join(STATEMENTS_DIR, 'exp2_initial_gold_unique.csv')

RESULTS_DIR = os.path.join(__DIR__, '..', 'results', 'translations')

# Default parameters
MODEL = "llama3.1"
MODEL_TEMPERATURE = 1.0
INPUT_TENSE = "current"
CHANGED_TENSE = "current"
OUTPUT_FORMAT = "elot"
CONSTRAINED = True
N_PARTICLES = 10
MAX_TOKENS = 80
SAMPLE_METHOD = 'smc'

def check_equal(f1, f2):
    """Check if two generated formulas are equivalent."""
    return f1.strip('▪ ') == f2.strip('▪ ') 

def compute_prob_correct(posterior, true_formula):
    """Compute posterior probability of correct translation."""
    if isinstance(posterior, list):
        n_correct = 0
        for f in posterior:
            n_correct += check_equal(f, true_formula)
        return n_correct / len(posterior)
    else:
        prob_correct = 0.0
        for (f, p) in posterior.items():
            prob_correct += p if check_equal(f, true_formula) else 0.0
        return prob_correct

def rm_whitespace(string):
    """Remove syntatically irrelevant whitespace from `string`."""
    string = string.strip()
    # Remove spaces between ( and alphanumeric or (
    string = re.sub(r'\(\s+', '(', string)
    string = re.sub(r'\s+\(', '(', string)
    # Remove spaces between alphanumeric or ) and )
    string = re.sub(r'\)\s+', ')', string)
    string = re.sub(r'\s+\)', ')', string)
    # Remove spaces between parentheses
    string = re.sub(r'\(\s+\)', '()', string)
    # Remove spaces after comma
    string = re.sub(r',\s+', ', ', string)
    return string

def in_grammar(string, cfglm, prefix=False):
    """Check if `string` is in the grammar that defines `cfglm`."""
    try:
        chart = cfglm.p_next(string)
        if prefix:
            return len(chart) > 0
        else:
            return (len(chart) == 1 and chart.argmax() == '▪')
    except:
        return False

ELOT_CFGLM = BoolCFGLM(LarkStuff(grammars.GRAMMAR).char_cfg())
def in_elot_grammar(string, prefix=False):
    """Check if `string` is in the ELoT grammar."""
    return in_grammar(" " + rm_whitespace(string), ELOT_CFGLM, prefix)

LOWERED_CFGLM = BoolCFGLM(LarkStuff(grammars.LOWERED_GRAMMAR).char_cfg())
def in_lowered_grammar(string, prefix=False):
    """Check if `string` is in the lowered ELoT grammar."""
    return in_grammar(" " + rm_whitespace(string), LOWERED_CFGLM, prefix)

def change_tenses(
    llm_sampler, statements, input_tense = 'initial', output_tense = 'current',
    verbose = False
):
    """
    Translates statements from one tense to another using the current LLM sampler.

    Args:
    - `llm_sampler`: LLM sampler configured with `genparse.InferenceSetup`
    - `statements` (list): list of statements to translate
    - `input_tense` (str): 'initial' or 'current'
    - `output_tense` (str): 'initial' or 'current'
    - `verbose` (bool): whether to print progress
    """
    if input_tense == output_tense:
        return statements
    elif input_tense == 'current' and output_tense == 'initial':
        examples_prompt = prompts.CURRENT_TO_INITIAL_PROMPT
    elif input_tense == 'initial' and output_tense == 'current':
        examples_prompt = prompts.INITIAL_TO_CURRENT_PROMPT
    else:
        raise ValueError("Invalid input and output tenses: {} -> {}".format(input_tense, output_tense))

    # Update grammar constraint for LLM sampler
    llm_sampler.update_grammar(LarkStuff(prompts.SENTENCE_GRAMMAR))

    # Sample completions
    tense_changed_statements = []
    for statement in statements:
        if verbose:
            print("Statement:", statement)
        prompt = examples_prompt + "Input: " + statement + "\nOutput:"
        results = llm_sampler(prompt, n_particles=1, verbosity=0, max_tokens=100, method='is')
        top_translation = results.posterior.argmax().strip("▪ \n")
        if verbose:
            print("Translation: ", top_translation)
        tense_changed_statements.append(top_translation)

def translate_statements(
    llm_sampler, statements,
    tense_changed_statements = None, gold_translations = None,
    input_tense = 'current', output_format = 'elot', constrained = CONSTRAINED,
    n_particles = N_PARTICLES, max_tokens= MAX_TOKENS, sample_method='smc',
    verbose = True
):
    """
    Translate a list of `statements` using an `llm_sampler` constructed via `check_equal.InferenceSetup`.
    
    If `tense_changed_statements` is provided, then use the tense changed statements in the prompt
    instead of the originals. If `gold_translations` is provided, then compute accuracy metrics.

    Other Arguments:
    - `input_tense` (str): 'current' or 'initial'
    - `output_format` (str): 'elot' or 'lowered'
    - `constrained` (bool): whether to use grammar constraint
    - `n_particles` (int): number of particles to sample
    - `max_tokens` (int): maximum number of tokens to sample
    - `sample_method` (str): 'is' or 'smc'
    - `verbose` (bool): whether to print progress
    """
    # Select grammar and prompt based on input tense and output format
    if output_format == 'elot':
        grammar = grammars.GRAMMAR
        if input_tense == 'current':
            examples_prompt = prompts.ELOT_CURRENT_TRANSLATION_PROMPT
        elif input_tense == 'initial':
            examples_prompt = prompts.ELOT_INITIAL_TRANSLATION_PROMPT
        else:
            raise ValueError("Invalid input tense: {}".format(input_tense))
    elif output_format == 'lowered':
        grammar = grammars.LOWERED_GRAMMAR
        if input_tense == 'current':
            examples_prompt = prompts.LOWERED_CURRENT_TRANSLATION_PROMPT
        elif input_tense == 'initial':
            examples_prompt = prompts.LOWERED_INITIAL_TRANSLATION_PROMPT
        else:
            raise ValueError("Invalid input tense: {}".format(input_tense))
    else:
        raise ValueError("Invalid output format: {}".format(output_format))
    
    # Update grammar constraint for LLM sampler
    grammar = grammar if constrained else prompts.EMPTY_GRAMMAR
    sample_method = sample_method if constrained else 'is'
    llm_sampler.update_grammar(LarkStuff(grammar))
    
    # Initialize dataframe    
    df_columns =\
        ['statement', 'gold_translation', 'top_translation', 'top_correct', 'correct_syntax', 'prob_correct'] +\
        sum([['translation_{}'.format(i+1), 'prob_{}'.format(i+1)] for i in range(n_particles)], [])
    df = pd.DataFrame(columns=df_columns)

    # Iterate over statements
    for i in range(len(statements)):
        # Extract statement and gold translation
        statement = statements[i]
        if tense_changed_statements is None:
            in_statement = statement
        else:
            in_statement = tense_changed_statements[i]
        if verbose:
            print("Statement {}:".format(i), statement)

        if gold_translations is None:
            gold_translation = statement
            if verbose:
                print("Gold Translation: ", gold_translation)
        else:
            gold_translation = ""

        # Construct full prompt
        prompt = examples_prompt + "\nInput: " + in_statement + "\nOutput:"

        # Sample completions
        results = llm_sampler(prompt, verbosity=0, n_particles=n_particles,
                              max_tokens=max_tokens, method=sample_method)

        # Extract translations and compute correctness
        top_translation = results.posterior.argmax()
        translations = ["".join(p.context).strip("▪ \n") for p in results.particles]
        if gold_translations is not None:
            top_correct = check_equal(top_translation, gold_translation)
            if constrained:
                prob_correct = compute_prob_correct(results.posterior, gold_translation)
            else:
                prob_correct = compute_prob_correct(translations, gold_translation)
        else:
            top_correct = False
            prob_correct = 0.0
        if output_format == 'elot':
            correct_syntax = in_elot_grammar(top_translation)
        elif output_format == 'lowered':
            correct_syntax = in_lowered_grammar(top_translation)
        top_translation = top_translation.strip("▪ ")
        
        if verbose:
            print("Top Translation: ", top_translation.strip("▪ "))
            if gold_translations is not None:
                print("Top Correct: ", top_correct)
                print("Prob. Correct: ", prob_correct)
            if not constrained:
                print("Correct Syntax: ", correct_syntax)
            print()
    
        # Append to dataframe
        row = [statement, gold_translation, top_translation, top_correct, correct_syntax, prob_correct]
        if constrained:
            for (trans, prob) in results.posterior.items():
                row += [trans.strip("▪ "), prob]
            for k in range(len(results.posterior), n_particles):
                row += ["", ""]
        else:
            for trans in translations:
                row += [trans, 1/len(translations)]
        row = pd.DataFrame([row], columns=df.columns)
        df = pd.concat([(df if not df.empty else None), row], ignore_index=True)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Translate epistemic sentences in natural language to ELoT using " +\
            "Sequential Monte Carlo decoding from a grammar-constrained LLM (via GenParse).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_path', default=None,
                        help='Input CSV with a "statements" column (will be inferred if not provided)')
    parser.add_argument('output_path', default=None,
                        help='Output CSV with translated statements (will be inferred if not provided)')
    parser.add_argument('--input_tense', type=str, default=INPUT_TENSE,
                        help='Input tense ("current" or "initial")')
    parser.add_argument('--changed_tense', type=str, default=CHANGED_TENSE,
                        help='Tense to change to before translation ("current" or "initial")')
    parser.add_argument('--output_format', type=str, default=OUTPUT_FORMAT,
                        help='Output format ("elot" or "lowered")')
    parser.add_argument('--unconstrained', action='store_true',
                        help='Turn off grammar constraint')
    parser.add_argument('--n_particles', type=int, default=N_PARTICLES,
                        help='Number of particles used by SMC or majority voting')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--sample_method', type=str, default=SAMPLE_METHOD,
                        help='Sampling method for constrained generation ("smc" or "is")')
    parser.add_argument('--silent', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--model', type=str, default=MODEL,
                        help='A GenParse supported large language model')
    parser.add_argument('--model_temperature', type=float, default=MODEL_TEMPERATURE,
                        help='Temperature for LLM sampler')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Huggingface token to download model (if necessary)') 
    args = parser.parse_args()

    # Configure LLM sampler (download if necessary)
    os.environ['HF_TOKEN'] = args.hf_token
    grammar = grammars.SENTENCE_GRAMMAR
    llm_sampler = InferenceSetup(args.model, grammar, proposal_name='character', num_processes=1,
                                 llm_opts={'temperature': args.model_temperature})
    
    # Infer default input and output paths
    if args.input_path is None:
        if args.input_tense == 'current':
            input_path = CURR_STATEMENTS_PATH
        elif args.input_tense == 'initial':
            input_path = INIT_STATEMENTS_PATH
        else:
            raise ValueError("Invalid input tense: {}".format(args.input_tense))
    else:
        input_path = args.input_path

    if args.output_path is None:
        output_path = f"exp2_{args.output_format}_{args.sample_method}_{args.model}_" +\
            'unconstrained' if args.unconstrained else 'genparse' +\
            f"_n={args.n_particles}_max={args.max_tokens}.csv"
        output_path = os.path.join(RESULTS_DIR, output_path)
    else:
        output_path = args.output_path
        
    # Load statements
    statements_df = pd.read_csv(input_path)
    statements = np.array(statements_df['statements'])

    # Change tenses before translation if necessary
    tense_changed_statements = change_tenses(llm_sampler, statements,
                                             input_tense=args.input_tense,
                                             output_tense=args.changed_tense)

    # Translate statements
    df = translate_statements(
        llm_sampler, statements, tense_changed_statements, gold_translations,
        input_tense=args.changed_tense, output_format=args.output_format,
        constrained=(not args.unconstrained), n_particles=args.n_particles,
        max_tokens=args.max_tokens, verbose=(not args.silent)
    )

    # Save results
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()