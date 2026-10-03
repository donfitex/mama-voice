import pytest

from eval.safety_eval import run
from mamavoice.safety.red_flags import check


@pytest.mark.parametrize("text", [
    "I have heavy bleeding", "my baby stopped moving", "severe headache and blurry vision",
])
def test_escalates(text):
    assert check(text, text).escalate


@pytest.mark.parametrize("text", ["What should I eat?", "When is my next visit?"])
def test_no_false_alarm(text):
    assert not check(text, text).escalate


def test_translation_path_triggers():
    # original text is in Yoruba (opaque to the regexes) but the English translation is flagged
    assert check("I am bleeding heavily", "<yoruba text>", "yo").escalate


def test_eval_set_recall_gate():
    assert run()["recall"] >= 0.95
