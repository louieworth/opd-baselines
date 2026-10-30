import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import convert_eval  # noqa: E402
from src.baseline_reward import compute_score  # noqa: E402


def gpqa_row(i, subdomain):
    return {"Question": f"q{i}", "Correct Answer": f"right{i}", "Incorrect Answer 1": f"w{i}a",
            "Incorrect Answer 2": f"w{i}b", "Incorrect Answer 3": f"w{i}c", "Subdomain": subdomain,
            "Record ID": f"rec{i}"}


def test_gpqa_subset_keeps_only_its_subdomain_and_names_its_source():
    rows = [gpqa_row(0, "Chemistry (general)"), gpqa_row(1, "Organic Chemistry"),
            gpqa_row(2, "Physics (general)"), gpqa_row(3, "Chemistry (general)")]
    chem = convert_eval("gpqa_chemistry_general", rows)
    phys = convert_eval("gpqa_physics_general", rows)
    assert [r["extra_info"]["index"] for r in chem] == ["rec0", "rec3"]
    assert [r["extra_info"]["index"] for r in phys] == ["rec2"]
    assert {r["data_source"] for r in chem + phys} == {"gpqa_chemistry_general", "gpqa_physics_general"}
    for record, row in zip(chem, (rows[0], rows[3])):
        letter = record["reward_model"]["ground_truth"]
        prompt = record["prompt"][0]["content"]
        assert f"{letter}. {row['Correct Answer']}" in prompt


def test_gpqa_subset_questions_match_their_gpqa_diamond_versions():
    subdomains = ["Chemistry (general)", "Organic Chemistry", "Physics (general)"] * 4
    rows = [gpqa_row(i, s) for i, s in enumerate(subdomains)]
    diamond = {r["extra_info"]["index"]: r for r in convert_eval("gpqa_diamond", rows)}
    for name in ("gpqa_chemistry_general", "gpqa_physics_general"):
        for record in convert_eval(name, rows):
            original = diamond[record["extra_info"]["index"]]
            assert record["prompt"] == original["prompt"]
            assert record["reward_model"] == original["reward_model"]


def test_gpqa_subsets_are_graded_exactly_like_gpqa_diamond():
    # Math grading accepts "\{B\}" and "\textbf{B}"; GPQA's choice grading does not.
    answers = ["\\boxed{B}", "\\boxed{(B)}", "\\boxed{\\{B\\}}", "\\boxed{\\textbf{B}}", "\\boxed{2}"]
    for source in ("gpqa_chemistry_general", "gpqa_physics_general"):
        for answer in answers:
            assert compute_score(source, answer, "B") == compute_score("gpqa_diamond", answer, "B"), answer
