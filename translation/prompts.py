## Prompts for statement translation

# Natural language examples of current belief statements
CURRENT_INPUT_EXAMPLES =\
"""The player knows that box 2 and box 3 are empty.
The player knows whether there is a key in box 2 or 3.
The player knows the color of the keys in all of the boxes.
The player doesn't know that there is a blue key in box 2.
The player does not know if there is a key in box 3.
The player does not know if box 2 is empty.
The player doesn't know which box has a key.
The player is certain that there is a blue key in box 1.
The player is sure of the color of the key in box 4.
The player is uncertain if there is a key in box 1.
The player is uncertain whether there is a key in box 2 or a key in box 1.
The player is unsure if there is a blue key in box 1 or 3.
The player is uncertain what color the key in box 1 is.
The player is unsure which box has a red key.
The player is uncertain about what's in box 2.
The player believes that there is a key in box 4.
The player does not believe that there is a key in box 2 or box 3.
The player thinks that there is no red key in box 1.
The player thinks that boxes 2 and 3 are both empty.
The player believes that if box 1 is empty then box 3 has the blue key.
The player thinks that there is a red key in either box 1 or box 3.
The player expects that there is a blue key in box 2 and a red key in box 1.
The player thinks there might be a key in box 1 or box 2.
The player thinks that one of the boxes may be empty.
The player thinks that box 1, 2 or 3 could contain a red key.
The player believes that there can be a key in box 3.
The player believes that either box 1 or box 2 should contain a blue key.
The player thinks that that the red key must be in box 2 if it is not in box 3.
The player thinks there is likely a key in box 2.
The player believes it is likely that box 3 and box 2 are empty.
The player thinks it is unlikely for box 1 to contain a red key.
The player believes there is probably a blue key in box 1, 3 or 4.
The player thinks that key 1 is more likely to be in box 1 than box 2.
The player thinks that a blue key is more likely to be in box 1 than box 2 or 3.
The player believes that box 1 is the most likely to contain a key.
The player believes that box 3 is most likely to contain a key."""

# Natural language examples of initial belief statements
INITIAL_INPUT_EXAMPLES =\
"""The player initially knew that box 2 and box 3 were empty.
The player knew whether there was a key in box 2 or 3.
The player initially knew the color of the keys in all of the boxes.
The player did not know that there was a blue key in box 2.
The player didn't know if there was a key in box 3.
The player initially did not know if box 2 was empty.
The player initially didn't know which box had a key.
The player was certain that there was a blue key in box 1.
The player was sure of the color of the key in box 4.
The player was initially uncertain if there was a key in box 1.
The player was uncertain whether there was a key in box 2 or a key in box 1.
The player was initially unsure if there was a blue key in box 1 or 3.
The player was initially uncertain what color the key in box 1 was.
The player was unsure which box had a red key.
The player was uncertain about what's in box 2.
The player initially believed that there was a key in box 4.
The player did not initially believe that there was a key in box 2 or box 3.
The player thought that there was no red key in box 1.
The player thought that boxes 2 and 3 were both empty.
The player believed that if box 1 was empty then box 3 had the blue key.
The player initially thought that there was a red key in either box 1 or box 3.
The player expected that there was a blue key in box 2 and a red key in box 1.
The player thought there might be a key in box 1 or box 2.
The player initially thought that one of the boxes may be empty.
The player thought that box 1, 2 or 3 could contain a red key.
The player believed there can be a key in box 3.
The player believed that either box 1 or box 2 should contain a blue key.
The player thought that that the red key must be in box 2 if it was not in box 3.
The player initially thought there was likely a key in box 2.
The player initially believed it was likely that box 3 and box 2 were empty.
The player thought it was unlikely for box 1 to contain a red key.
The player initially believed there was probably a blue key in box 1, 3 or 4.
The player initially thought that key 1 was more likely to be in box 1 than box 2.
The player thought that a blue key was more likely to be in box 1 than box 2 or 3.
The player initially believed that box 1 was the most likely to contain a key.
The player believed that box 3 was most likely to contain a key."""

# ELoT translations of example belief statements
ELOT_EXAMPLES =\
"""knows_that(player, formula(and(empty(box2), empty(box3))))
knows_if(player, formula(exists(key(K), or(inside(K, box2), inside(K, box3)))))
forall(box(B), knows_about(player, color(C), exists(and(key(K), inside(K, B)), iscolor(K, C))))
not_knows_that(player, formula(exists(and(key(K), iscolor(K, red)), inside(K, box2))))
not_knows_if(player, formula(exists(key(K), inside(K, box3))))
not_knows_if(player, formula(empty(box2)))
not_knows_about(player, box(B), exists(key(K), inside(K, B)))
certain(player, formula(exists(and(key(K), iscolor(K, blue)), inside(K, box1))))
certain_about(player, color(C), exists(and(key(K), inside(K, box4)), iscolor(K, C)))
uncertain_if(player, formula(exists(key(K), inside(K, box1))))
uncertain_if(player, formula(exists(key(K), inside(K, box2))), formula(exists(key(K), inside(K, box1))))
uncertain_if(player, and(key(K), iscolor(K, blue)), inside(K, box1), inside(K, box3))
uncertain_about(player, color(C), exists(and(key(K), inside(K, box1)), iscolor(K, C)))
uncertain_about(player, and(box(B), not(closed(B))), exists(and(key(K), iscolor(K, red)), inside(K, B)))
uncertain_about(player, color(C), exists(and(key(K), inside(K, box2)), iscolor(K, C)))
believes(player, formula(exists(key(K), inside(K, box4))))
not(believes(player, formula(exists(key(K), or(inside(K, box2), inside(K, box3))))))
believes(player, formula(not(exists(and(key(K), iscolor(K, red)), inside(K, box1))))) 
believes(player, formula(and(empty(box2), empty(box3))))
believes(player, formula(imply(empty(box1), exists(and(key(K), iscolor(K, blue)), inside(K, box3)))))
believes(player, formula(exists(and(key(K), iscolor(K, red)), or(inside(K, box1), inside(K, box3)))))
believes(player, formula(and(exists(and(key(K), iscolor(K, blue)), inside(K, box2)), exists(and(key(K), iscolor(K, red)), inside(K, box1)))))
believes(player, might(exists(key(K), or(inside(K, box1), inside(K, box2)))))
believes(player, may(exists(box(B), empty(B))))
believes(player, could(exists(key(K), or(inside(K, box1), inside(K, box2), inside(K, box3)))))
believes(player, can(exists(key(K), inside(K, box3))))
believes(player, should(exists(and(key(K), iscolor(K, blue)), or(inside(K, box1), inside(K, box2)))))
believes(player, must(exists(and(key(K), iscolor(K, red)), imply(not(inside(K, box3)), inside(K, box2)))))
believes(player, likely(exists(key(K), inside(K, box2))))
believes(player, likely(and(empty(box3), empty(box2))))
believes(player, unlikely(exists(and(key(K), iscolor(K, red)), inside(K, box1))))
believes(player, likely(exists(and(key(K), iscolor(K, blue)), or(inside(K, box1), inside(K, box3), inside(K, box4)))))
believes(player, more(likely, inside(key1, box1), inside(key1, box2)))
believes(player, more(likely, exists(and(key(K), iscolor(K, blue)), inside(K, box1)), exists(and(key(K), iscolor(K, blue)), or(inside(K, box2), inside(K, box3)))))
believes(player, most(likely, box1, B, box(B), exists(key(K), inside(K, B))))
believes(player, most(likely, exists(key(K), inside(K, box3))))"""

# Lowered translations of example belief statements
LOWERED_EXAMPLES =\
"""and(>=(prob_of(player, and(empty(box2), empty(box3))), threshold(believes)), and(empty(box2), empty(box3)))
or(and(>=(prob_of(player, exists(key(K), or(inside(K, box2), inside(K, box3)))), threshold(believes)), exists(key(K), or(inside(K, box2), inside(K, box3)))), and(>=(prob_of(player, not(exists(key(K), or(inside(K, box2), inside(K, box3))))), threshold(believes)), not(exists(key(K), or(inside(K, box2), inside(K, box3))))))
forall(box(B), exists(color(C), and(>=(prob_of(player, exists(and(key(K), inside(K, B)), iscolor(K, C))), threshold(believes)), exists(and(key(K), inside(K, B)), iscolor(K, C)))))
and(not(>=(prob_of(player, exists(and(key(K), iscolor(K, red)), inside(K, box2))), threshold(believes))), exists(and(key(K), iscolor(K, red)), inside(K, box2)))
and(not(>=(prob_of(player, exists(key(K), inside(K, box3))), threshold(believes))), not(>=(prob_of(player, not(exists(key(K), inside(K, box3)))), threshold(believes))))
and(not(>=(prob_of(player, empty(box2)), threshold(believes))), not(>=(prob_of(player, not(empty(box2))), threshold(believes))))
forall(box(B), or(<(prob_of(player, exists(key(K), inside(K, B))), threshold(believes)), not(exists(key(K), inside(K, B)))))
>=(prob_of(player, exists(and(key(K), iscolor(K, blue)), inside(K, box1))), threshold(certain))
exists(color(C), >=(prob_of(player, exists(and(key(K), inside(K, box4)), iscolor(K, C))), threshold(certain)))
and(<(prob_of(player, exists(key(K), inside(K, box1))), threshold(uncertain)), <(prob_of(player, exists(key(K), inside(K, box1))), threshold(uncertain)))
and(<(prob_of(player, exists(key(K), inside(K, box2))), threshold(uncertain)), <(prob_of(player, exists(key(K), inside(K, box1))), threshold(uncertain)))
exists(and(key(K), iscolor(K, blue)), and(<(prob_of(player, inside(K, box1)), threshold(uncertain)), <(prob_of(player, inside(K, box3)), threshold(uncertain))))
forall(color(C), <(prob_of(player, exists(and(key(K), inside(K, box1)), iscolor(K, C))), threshold(uncertain)))
forall(and(box(B), not(closed(B))), <(prob_of(player, exists(and(key(K), iscolor(K, red)), inside(K, B))), threshold(uncertain)))
forall(color(C), <(prob_of(player, exists(and(key(K), inside(K, box2)), iscolor(K, C))), threshold(uncertain)))
>=(prob_of(player, exists(key(K), inside(K, box4))), threshold(believes))
not(>=(prob_of(player, exists(key(K), or(inside(K, box2), inside(K, box3)))), threshold(believes))
>=(prob_of(player, not(exists(and(key(K), iscolor(K, red)), inside(K, box1)))), threshold(believes))
>=(prob_of(player, and(empty(box2), empty(box3))), threshold(believes))
>=(prob_of(player, imply(empty(box1), exists(and(key(K), iscolor(K, blue)), inside(K, box3)))), threshold(believes))
>=(prob_of(player, exists(and(key(K), iscolor(K, red)), or(inside(K, box1), inside(K, box3)))), threshold(believes))
>=(prob_of(player, and(exists(and(key(K), iscolor(K, blue)), inside(K, box2)), exists(and(key(K), iscolor(K, red)), inside(K, box1)))), threshold(believes))
>=(prob_of(player, exists(key(K), or(inside(K, box1), inside(K, box2)))), threshold(might))
>=(prob_of(player, exists(box(B), empty(B))), threshold(may))
>=(prob_of(player, exists(key(K), or(inside(K, box1), inside(K, box2), inside(K, box3)))), threshold(could))
>=(prob_of(player, exists(key(K), inside(K, box3))), threshold(can))
>=(prob_of(player, exists(and(key(K), iscolor(K, blue)), or(inside(K, box1), inside(K, box2)))), threshold(should))
>=(prob_of(player, exists(and(key(K), iscolor(K, red)), imply(not(inside(K, box3)), inside(K, box2)))), threshold(must))
>=(prob_of(player, exists(key(K), inside(K, box2))), threshold(likely))
>=(prob_of(player, and(empty(box3), empty(box2))), threshold(likely))
<=(-(prob_of(player, exists(and(key(K), iscolor(K, red)), inside(K, box1)))), threshold(unlikely))
>=(prob_of(player, exists(and(key(K), iscolor(K, blue)), or(inside(K, box1), inside(K, box3), inside(K, box4)))), threshold(likely))
>(prob_of(player, inside(key1, box1)), prob_of(player, inside(key1, box2)))
>(prob_of(player, exists(and(key(K), iscolor(K, blue)), inside(K, box1))), prob_of(player, exists(and(key(K), iscolor(K, blue)), or(inside(K, box2), inside(K, box3)))))
>=(prob_of(player, exists(key(K), inside(K, box1))), maximum(box(B), prob_of(player, exists(key(K), inside(K, B)))))
>=(prob_of(player, exists(key(K), inside(K, box3))), threshold(most(likely)))"""

# Prompt headers and footers
INITIAL_TO_CURRENT_HEADER =\
    "Please translate the following sentences from past tense to present tense:\n\n"
INITIAL_TO_CURRENT_FOOTER = ""

CURRENT_TO_INITIAL_HEADER =\
    "Please translate the following sentences from present tense to past tense:\n\n"
CURRENT_TO_INITIAL_FOOTER = ""

ELOT_TRANSLATION_HEADER =\
    "Please translate the statement below into logical form. Here are some examples of the types of statements you might encounter:\n\n"
ELOT_TRANSLATION_FOOTER =\
    "\nPlease note that you can only use the following predicates in your logical form: knows_that, knows_if, knows_about, not_knows_that, not_knows_if, not_knows_about, certain, certain_about, uncertain_if, uncertain_about, believes.\n"

LOWERED_TRANSLATION_HEADER =\
    "Please translate the statement below into logical form. Here are some examples of the types of statements you might encounter:\n\n"
LOWERED_TRANSLATION_FOOTER =\
    """\nOutput one line of logical form with no explanation. You may only use the threshold predicates "believes", "likely", "most(likely)", "unlikely", "certain", "uncertain", "must", "may", "might", "can", "could", "should".\n"""

# Prompt for tense change (from initial to current)
INITIAL_TO_CURRENT_PROMPT =\
    INITIAL_TO_CURRENT_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(init, curr) for (init, curr) in zip(INITIAL_INPUT_EXAMPLES, CURRENT_INPUT_EXAMPLES)])

# Prompt for tense change (from current to initial)
CURRENT_TO_INITIAL_PROMPT =\
    CURRENT_TO_INITIAL_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(curr, init) for (curr, init) in zip(CURRENT_INPUT_EXAMPLES, INITIAL_INPUT_EXAMPLES)])

# Prompt for translating current belief statements to ELoT
ELOT_CURRENT_TRANSLATION_PROMPT =\
    ELOT_TRANSLATION_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(i, o) for (i, o) in zip(CURRENT_INPUT_EXAMPLES, ELOT_EXAMPLES)]) +\
    ELOT_TRANSLATION_FOOTER

# Prompt for translating initial belief statements to ELoT
ELOT_INITIAL_TRANSLATION_PROMPT =\
    ELOT_TRANSLATION_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(i, o) for (i, o) in zip(INITIAL_INPUT_EXAMPLES, ELOT_EXAMPLES)]) +\
    ELOT_TRANSLATION_FOOTER

# Prompt for translating initial belief statements to lowered form
LOWERED_CURRENT_TRANSLATION_PROMPT =\
    LOWERED_TRANSLATION_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(i, o) for (i, o) in zip(CURRENT_INPUT_EXAMPLES, LOWERED_EXAMPLES)]) +\
    LOWERED_TRANSLATION_FOOTER

# Prompt for translating initial belief statements to lowered form
LOWERED_INITIAL_TRANSLATION_PROMPT =\
    LOWERED_TRANSLATION_HEADER +\
    "".join(["Input: {}\nOutput: {}\n".format(i, o) for (i, o) in zip(INITIAL_INPUT_EXAMPLES, LOWERED_EXAMPLES)]) +\
    LOWERED_TRANSLATION_FOOTER
