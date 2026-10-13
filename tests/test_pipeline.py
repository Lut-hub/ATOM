"""Offline regression checks for the Atomic-QA pipeline and perturbations."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from atomqa.generation import generate, deduplicate
from atomqa import llm as utils
from atomqa.generation.leaves import expand_tree
from atomqa.prompts.atomic import OPERAND_GENERATION_PROMPT, VERIFICATION_PROMPT, PERTURBER_PROMPT
from atomqa.io import read_json, write_json
from atomqa.perturbation.entity import process_entity_data
from atomqa.perturbation.operator import perturb_item

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.entities = read_json(ROOT / "examples/atomic_qa.json")

    def leaves(self):
        result = expand_tree({}, read_json(ROOT / "examples/category_tree.json"))
        result["/Entities/Geography/Countries"].update(read_json(ROOT / "examples/operators.json")["/Entities/Geography/Countries"])
        return result

    def test_full_tree_preserves_all_leaf_paths(self):
        leaves = expand_tree({}, read_json(ROOT / "assets/trees/entity_tree.json"))
        self.assertEqual(len(leaves), 6931)
        self.assertEqual(len({tuple(v["path"]) for v in leaves.values()}), 6931)

    def test_seed_assignment_does_not_truncate_at_4000(self):
        entities = {f"country{i}": {"category": "Countries"} for i in range(4002)}
        leaves = expand_tree(entities, read_json(ROOT / "examples/category_tree.json"))
        self.assertEqual(len(next(iter(leaves.values()))["entities"]), 4002)

    def test_ambiguous_category_is_rejected(self):
        tree = {"A": {"definition": "a", "children": {"X": {"definition": "x"}}},
                "B": {"definition": "b", "children": {"X": {"definition": "x"}}}}
        with self.assertRaises(ValueError):
            expand_tree({"entity": {"category": "X"}}, tree)

    def test_last_qa_line_is_not_dropped(self):
        text = '<information>\n- Question: What is the capital of France? - Answer: Paris\n</information>'
        with patch.object(generate, "call_gpt", return_value=[text]):
            rows = generate.gen_entity_information("Countries", "countries", "France", "country", "CapitalOf", "capital")
        self.assertEqual([r["attribute"] for r in rows], ["Paris"])

    def test_exhausted_invalid_output_is_not_returned_as_valid(self):
        with self.assertLogs(level="ERROR"), patch.object(utils, "call_gpt", return_value=['```json\n{"wrong":1}\n```']):
            result = utils.get_gpt_output("system", "user", "Fix: {error_reason}", {"judgement": ""})
        self.assertIsNone(result)

    def test_paper_prompts_match_archived_appendix(self):
        archive = read_json(ROOT / "assets/prompts/paper_prompts.json")
        self.assertEqual(OPERAND_GENERATION_PROMPT,
                         archive["Operand-Generation"]["text"].strip().replace("{existing operands}", "{existing_entities}"))
        self.assertEqual(VERIFICATION_PROMPT, archive["Verification"]["text"].strip())
        self.assertEqual(PERTURBER_PROMPT, archive["Perturber"]["text"].strip())

    def test_operand_prompt_receives_operator(self):
        with patch.object(generate, "call_gpt", return_value=['```json\n{"France":"country"}\n```']) as call:
            generate.gen_entity_name("Countries", "countries", [], "CapitalOf", "national capital")
        prompt = call.call_args.args[0][1]["content"]
        self.assertIn("- Operator: CapitalOf", prompt)
        self.assertIn("- Operator Definition: national capital", prompt)

    def test_retry_keeps_single_operator_instructions(self):
        response = '<information>\n- Question: What is the capital of France? - Answer: Paris\n</information>'
        multiple = '<information>\n- Question: What is the capital of France? - Answer: Paris\n- Question: Where is France? - Answer: Europe\n</information>'
        with self.assertLogs(level="ERROR"), patch.object(generate, "call_gpt", side_effect=[[multiple], [response]]) as call:
            rows = generate.gen_entity_information("Countries", "countries", "France", "country", "CapitalOf", "capital")
        feedback = call.call_args.args[0][3]["content"]
        self.assertIn("exactly one QA for the given operator", feedback)
        self.assertNotIn("YYY", feedback)
        self.assertEqual(len(rows), 1)

    def test_missing_operator_cannot_fall_back_to_unconstrained_generation(self):
        with patch.object(generate, "call_gpt") as call:
            with self.assertRaises(ValueError):
                generate.gen_entity_name("Countries", "countries", [], None, None)
            with self.assertRaises(ValueError):
                generate.gen_entity_information("Countries", "countries", "France", "country", "CapitalOf", None)
        call.assert_not_called()

    def test_generation_rejects_unbound_taxonomy_before_api_call(self):
        write_json(self.path / "entities.json", {})
        write_json(self.path / "leaves.json", expand_tree({}, read_json(ROOT / "examples/category_tree.json")))
        with patch.object(generate, "call_gpt") as call, self.assertRaises(ValueError):
            generate.gen_entity(str(self.path / "entities.json"), str(self.path / "leaves.json"))
        call.assert_not_called()

    def test_generation_verification_filters_and_respects_cap(self):
        write_json(self.path / "entities.json", {})
        write_json(self.path / "leaves.json", self.leaves())
        with patch.object(generate, "gen_entity_name", return_value={"France": "country", "Japan": "country", "Brazil": "country"}), \
             patch.object(generate, "gen_entity_information", side_effect=lambda c,d,n,e,*a: deepcopy(self.entities[n]["items"])), \
             patch.object(generate, "get_gpt_output", side_effect=[{"judgement": "fail", "reason": "reject"}, {"judgement": "pass", "reason": "valid"}]):
            result = generate.gen_entity(str(self.path / "entities.json"), str(self.path / "leaves.json"), 1, max_rounds=1)
        self.assertEqual(list(result), ["Japan"])
        self.assertEqual(result["Japan"]["items"][0]["operator"], "CapitalOf")

    def test_generation_stops_at_max_rounds_and_saves_partial_data(self):
        write_json(self.path / "entities.json", {})
        write_json(self.path / "leaves.json", self.leaves())
        with self.assertLogs(level="ERROR"), patch.object(generate, "gen_entity_name", return_value=None) as call, self.assertRaises(RuntimeError):
            generate.gen_entity(str(self.path / "entities.json"), str(self.path / "leaves.json"), max_rounds=2)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(read_json(self.path / "entities.json"), {})

    def test_same_operand_can_have_distinct_operators(self):
        leaves = self.leaves()
        second = deepcopy(next(iter(leaves.values())))
        second.update({"name": "Population", "operator": "PopulationOf", "entities": []})
        leaves["/Entities/Geography/Population"] = second
        write_json(self.path / "entities.json", {})
        write_json(self.path / "leaves.json", leaves)
        def make_qa(category, definition, name, description, operator, operator_definition):
            return [{"prompt": f"{operator} of {name}?", "template": operator + " of {}?", "attribute": "value"}]
        with patch.object(generate, "gen_entity_name", return_value={"France": "country"}), \
             patch.object(generate, "gen_entity_information", side_effect=make_qa), \
             patch.object(generate, "get_gpt_output", return_value={"judgement": "pass", "reason": "mock"}):
            result = generate.gen_entity(str(self.path / "entities.json"), str(self.path / "leaves.json"), 1, max_rounds=1)
        self.assertEqual({item["operator"] for item in result["France"]["items"]}, {"CapitalOf", "PopulationOf"})

    def test_deduplication_preserves_operator_metadata_and_leaf_references(self):
        first = deepcopy(self.entities["France"])
        alias = deepcopy(first)
        alias["subject"] = "French Republic"
        alias["category"] = "/Entities/Geography/Countries"
        alias["items"] = [{"prompt": "Population of French Republic?", "template": "Population of {}?",
                           "attribute": "68 million", "operand": "French Republic", "operator": "PopulationOf",
                           "output": "68 million", "verification": {"judgement": "pass"}}]
        leaves = self.leaves()
        next(iter(leaves.values()))["entities"] = ["France", "French Republic"]
        write_json(self.path / "entities.json", {"France": first, "French Republic": alias})
        write_json(self.path / "leaves.json", leaves)
        with patch.object(deduplicate, "pre_dedup_entity", return_value={"same": ["France", "French Republic"]}), \
             patch.object(deduplicate, "get_gpt_output", return_value={"judgement": "same", "reason": "aliases"}):
            entities, result_leaves = deduplicate.dedup_entity(str(self.path / "entities.json"), str(self.path / "leaves.json"))
        self.assertEqual(list(entities), ["France"])
        self.assertEqual(entities["France"]["items"][1]["operand"], "France")
        self.assertEqual(next(iter(result_leaves.values()))["entities"], ["France"])

    def test_operand_relabeling_is_joint_consistent_and_same_length(self):
        write_json(self.path / "input.json", self.entities)
        random.seed(42)
        with redirect_stdout(io.StringIO()):
            process_entity_data(str(self.path / "input.json"), str(self.path / "output.json"))
        output = read_json(self.path / "output.json")
        mapping = read_json(str(self.path / "output.json") + ".mapping.json")
        for subject, original in self.entities.items():
            renamed = mapping[subject]
            item = output[renamed]["items"][0]
            self.assertEqual(len(renamed), len(subject))
            self.assertEqual(item["attribute"], mapping[original["items"][0]["attribute"]])
            self.assertEqual(len(item["attribute"]), len(original["items"][0]["attribute"]))
            self.assertEqual(item["operator"], "CapitalOf")
            self.assertIn(renamed, item["prompt"])
            self.assertEqual(item["operand"], renamed)

    def test_operator_perturbation_keeps_question_and_operand(self):
        item = self.entities["France"]["items"][0]
        with patch("atomqa.perturbation.operator.get_gpt_output", side_effect=[
            {"output": "Berlin", "reason": "wrong capital"}, {"judgement": "fail", "reason": "incorrect"}]):
            result = perturb_item(item)
        self.assertEqual(result["prompt"], item["prompt"])
        self.assertEqual(result["operand"], item["operand"])
        self.assertEqual(result["attribute"], "Berlin")
        self.assertEqual(item["attribute"], "Paris")

    def test_operator_perturbation_rejects_still_valid_output(self):
        with patch("atomqa.perturbation.operator.get_gpt_output", side_effect=[
            {"output": "Paris, France", "reason": "rephrase"}, {"judgement": "pass", "reason": "still correct"}]):
            with self.assertRaises(RuntimeError):
                perturb_item(self.entities["France"]["items"][0], attempts=1)

    def test_lfs_pointer_is_not_treated_as_data(self):
        p = self.path / "missing.json"
        p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS"):
            read_json(p)


if __name__ == "__main__":
    unittest.main()
