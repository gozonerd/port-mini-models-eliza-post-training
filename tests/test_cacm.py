"""The 1966 CACM conversation, replies as printed in the paper (and reproduced by Hay)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eliza import Eliza, load_script

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'third-party', 'ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt')
EXCHANGES = [
    ("Men are all alike.", "IN WHAT WAY"),
    ("They're always bugging us about something or other.", "CAN YOU THINK OF A SPECIFIC EXAMPLE"),
    ("Well, my boyfriend made me come here.", "YOUR BOYFRIEND MADE YOU COME HERE"),
    ("He says I'm depressed much of the time.", "I AM SORRY TO HEAR YOU ARE DEPRESSED"),
    ("It's true. I am unhappy.", "DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE UNHAPPY"),
    ("I need some help, that much seems certain.", "WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP"),
    ("Perhaps I could learn to get along with my mother.", "TELL ME MORE ABOUT YOUR FAMILY"),
    ("My mother takes care of me.", "WHO ELSE IN YOUR FAMILY TAKES CARE OF YOU"),
    ("My father.", "YOUR FATHER"),
    ("You are like my father in some ways.", "WHAT RESEMBLANCE DO YOU SEE"),
    ("You are not very aggressive but I think you don't want me to notice that.", "WHAT MAKES YOU THINK I AM NOT VERY AGGRESSIVE"),
    ("You don't argue with me.", "WHY DO YOU THINK I DON'T ARGUE WITH YOU"),
    ("You are afraid of me.", "DOES IT PLEASE YOU TO BELIEVE I AM AFRAID OF YOU"),
    ("My father is afraid of everybody.", "WHAT ELSE COMES TO MIND WHEN YOU THINK OF YOUR FATHER"),
    ("Bullies.", "DOES THAT HAVE ANYTHING TO DO WITH THE FACT THAT YOUR BOYFRIEND MADE YOU COME HERE"),
]


def test_cacm():
    e = Eliza(load_script(SCRIPT))
    assert e.hello() == "HOW DO YOU DO. PLEASE TELL ME YOUR PROBLEM"
    for u, expected in EXCHANGES:
        assert e.response(u) == expected, (u, e.response)


def test_sad_cycle():
    e = Eliza(load_script(SCRIPT))
    seen = [e.response("I am sad") for _ in range(5)]
    assert seen[0] == "I AM SORRY TO HEAR YOU ARE SAD"
    assert seen[1] == "DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE SAD"
    assert seen[2] == "I'M SURE ITS NOT PLEASANT TO BE SAD"
    assert seen[3] == "CAN YOU EXPLAIN WHAT MADE YOU SAD"
    assert seen[4] == seen[0]


if __name__ == '__main__':
    test_cacm(); test_sad_cycle(); print('ok')
