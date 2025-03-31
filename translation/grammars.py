## Grammar definitions

# Grammar for Doors, Keys, Gems specific predicates and objects
DKG_GRAMMAR = r"""
type_expr: type_name "(" var_arg ")"
type_name: "key" | "color" | "box"

pred_expr: iscolor_expr | inside_expr | empty_expr
iscolor_expr: "iscolor" "(" key_arg ", " color_arg ")"
inside_expr: "inside" "(" key_arg ", " box_arg ")"
empty_expr: "empty" "(" box_arg ")"

key_arg: key_const | var_arg
box_arg: box_const | var_arg
color_arg: color_const | var_arg
const_arg: box_const | key_const | color_const

color_const: "red" | "blue"
box_const: "box" /[1-3]/
key_const: "key" /[1-3]/
var_arg: /[A-Z]/
"""

# Grammar for first-order logic
FOL_GRAMMAR = r"""
fol_expr: and_expr | or_expr | imply_expr | not_expr | forall_expr | exists_expr | type_expr | pred_expr

not_and_expr: or_expr | imply_expr | not_expr | forall_expr | exists_expr | type_expr | pred_expr
not_or_expr: and_expr | imply_expr | not_expr | forall_expr | exists_expr | type_expr | pred_expr
not_not_expr: and_expr | or_expr | imply_expr | forall_expr | exists_expr | type_expr | pred_expr
not_forall_expr: and_expr | or_expr | imply_expr | not_expr | exists_expr | type_expr | pred_expr
not_exists_expr: and_expr | or_expr | imply_expr | not_expr | forall_expr | type_expr | pred_expr

and_expr: "and" "(" not_and_expr (", " not_and_expr)+ ")"
or_expr: "or" "(" not_or_expr (", " not_or_expr)+ ")"
imply_expr: "imply" "(" fol_expr ", " fol_expr ")"
not_expr: "not" "(" not_not_expr ")"

forall_expr: "forall" "(" not_forall_expr ", " not_forall_expr ")"
exists_expr: "exists" "(" not_exists_expr ", " not_exists_expr ")"
"""

# Grammar for epistemic language-of-thought
ELOT_GRAMMAR = r"""
start: " " + elot_expr

elot_expr: e_pred_expr | e_and_expr | e_or_expr | e_imply_expr | e_not_expr | e_forall_expr | e_exists_expr | fol_expr

e_pred_expr: believes_expr | knowledge_expr | hopes_expr | certainty_expr

e_and_expr: "and" "(" e_pred_expr (", " e_pred_expr)+ ")"
e_or_expr: "or" "(" e_pred_expr (", " e_pred_expr)+ ")"
e_imply_expr: "imply" "(" elot_expr ", " e_pred_expr ")"
e_not_expr: "not" "(" e_pred_expr ")"
e_forall_expr: "forall" "(" fol_expr ", " elot_expr ")"
e_exists_expr: "exists" "(" fol_expr ", " elot_expr ")"

believes_expr: "believes" "(" agent_arg (", " base_or_modal_expr)+ ")"

knowledge_expr: knows_that_expr | knows_if_expr | knows_about_expr | not_knows_that_expr | not_knows_if_expr | not_knows_about_expr
knows_that_expr: "knows_that" "(" agent_arg (", " base_expr)+ ")"
knows_if_expr: "knows_if" "(" agent_arg (", " base_expr)+ ")"
knows_about_expr: "knows_about" "(" agent_arg ", " fol_expr ", " fol_expr ")"
not_knows_that_expr: "not_knows_that" "(" agent_arg (", " base_expr)+ ")"
not_knows_if_expr: "not_knows_if" "(" agent_arg (", " base_expr)+ ")"
not_knows_about_expr: "not_knows_about" "(" agent_arg ", " fol_expr ", " fol_expr ")"

hopes_expr: "hopes" "(" agent_arg (", " base_or_modal_expr)+ ")"

certainty_expr: certain_expr | certain_about_expr | uncertain_if_expr | uncertain_if_qual_expr | uncertain_about_expr
certain_expr: "certain" "(" agent_arg (", " base_or_modal_expr)+ ")"
certain_about_expr: "certain_about" "(" agent_arg ", " fol_expr ", " fol_expr ")"
uncertain_if_expr: "uncertain_if" "(" agent_arg ", " base_expr (", " base_expr)? ")"
uncertain_if_qual_expr: "uncertain_if" "(" agent_arg ", " fol_expr ", " fol_expr ", " fol_expr ")"
uncertain_about_expr: "uncertain_about" "(" agent_arg ", " fol_expr ", " fol_expr ")"

agent_arg: "player"

base_or_modal_expr: base_expr | modal_expr | comp_expr | sup_expr | most_expr
base_expr: "formula" "(" fol_expr ")"
modal_expr: modal_name "(" fol_expr ")"
modal_name: "could" | "can" | "might" | "may" | "should" | "must" | adj_name
adj_name: "likely" | "unlikely" 
comp_expr: comp_name "(" adj_name ", " fol_expr ", " fol_expr (", " fol_expr)? ")"
comp_name: "more" | "less" | "equal"
sup_expr: sup_name "(" adj_name ", " const_arg ", " var_arg ", " fol_expr ", " fol_expr ")"
sup_name: "most" | "least"
most_expr: "most" "(" adj_name ", " fol_expr ")"
"""

# Grammar for lowered epistemic formulae
LOWERED_ELOT_GRAMMAR = r"""
start: " " + elot_expr

elot_expr: e_pred_expr | e_and_expr | e_or_expr | e_imply_expr | e_not_expr | e_forall_expr | e_exists_expr | fol_expr
e_and_expr: "and" "(" elot_expr (", " elot_expr)+ ")"
e_or_expr: "or" "(" elot_expr (", " elot_expr)+ ")"
e_imply_expr: "imply" "(" elot_expr ", " elot_expr ")"
e_not_expr: "not" "(" elot_expr ")"
e_forall_expr: "forall" "(" fol_expr ", " elot_expr ")"
e_exists_expr: "exists" "(" fol_expr ", " elot_expr ")"

e_pred_expr: comp_op "(" prob_of_literal ", " prob_valued_expr ")"
comp_op: ">" | ">=" | "<" | "<=" | "=="
prob_valued_expr: prob_of_literal | prob_quant_expr | thresh_expr
prob_quant_expr: quant_name "(" fol_expr ", " prob_of_literal ")"
quant_name: "maximum" | "minimum"
prob_of_literal: "-" "(" prob_of_expr ")" | prob_of_expr
prob_of_expr: "prob_of" "(" agent_arg ", " fol_expr ")"

thresh_expr: "threshold" "(" thresh_arg ")"
thresh_arg: thresh_const | multiplier_name "(" thresh_const ")"
thresh_const: "believes" | "certain" | "uncertain" | "hopes" | "likely" | "unlikely" | "could" | "can" | "might" | "may" | "should" | "must"
multiplier_name: "most"

agent_arg: "player"
"""

# Combine grammars
GRAMMAR = ELOT_GRAMMAR.strip() + "\n" + FOL_GRAMMAR.strip() + "\n" + DKG_GRAMMAR.strip()
LOWERED_GRAMMAR = LOWERED_ELOT_GRAMMAR.strip() + "\n" + FOL_GRAMMAR.strip() + "\n" + DKG_GRAMMAR.strip()

# Grammar for single line sentences that end with a period
SENTENCE_GRAMMAR = r"""start: " " /[^\n\.]+/ ".\n" """ 

# Empty single-line grammar
EMPTY_GRAMMAR = r"""start: " " /[^\n]+/ "\n" """
