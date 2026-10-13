# Atomic-QA pipeline

An Atomic-QA record represents one operation, `f(x) = y`: an operator `f`, an operand `x`, and its output `y`. Generation accepts a record only when the verification model judges this operation correct.

## Workflow

1. Normalize seeds from FB15k, ZsRE, or wiki_recent.
2. Construct and expand a taxonomy using [EpiCoder](https://openreview.net/forum?id=RAxe7nF4Oz).
3. Prepare leaf categories with explicit operator bindings.
4. Generate operands, compute one QA pair per operation, and verify the output.
5. Deduplicate equivalent operands within neighboring categories.
6. Export QA data or apply operand/output perturbations.

The commands below run from the repository root. Classification, generation, deduplication, and operator perturbation require the API environment variables described in the [README](../README.md).

## Seed preparation

```bash
python -m atomqa.preprocessing.prepare_seeds \
  --fb15k examples/seeds_fb15k_format.json --output-dir outputs/seeds
```

All inputs are JSON arrays with `subject` and `prompt`. A `template` containing `{}` may also be supplied; otherwise it is derived by replacing the subject in the prompt. Answer fields depend on the source:

| Input option | Answer field |
| --- | --- |
| `--fb15k FILE` | `attribute`, a string |
| `--zsre FILE` | First element of `ground_truth`, an array of strings |
| `--wiki-recent FILE ...` | `target_new`, a string; select `ground_truth` with `--wiki-recent-answer ground_truth` when appropriate for the input |

The command writes `base_data.json` (flat QA records) and `entities.json` (QA records grouped by subject). Use the normalized seeds for representative selection and tree construction with EpiCoder. The tree-building workflow is not included in this package.

To assign seeds to an existing tree:

```bash
python -m atomqa.generation.classify \
  --tree examples/category_tree.json --entities outputs/seeds/entities.json \
  --output outputs/seeds/classified.json
```

Classification adds category paths. It does not verify QA records, so classified seeds cannot be used directly as a generation checkpoint.

## Trees and operator bindings

A tree is a nested JSON object. Each node has a `definition` and optional `children`; see [category_tree.json](../examples/category_tree.json). A leaf needs both `operator` and `operator_definition`, supplied on the node itself or in a separate mapping:

```json
{
  "/Entities/Geography/Countries": {
    "operator": "CapitalOf",
    "operator_definition": "Map a country to its national capital."
  }
}
```

Mapping keys are full leaf paths with JSON Pointer escaping (`~` becomes `~0`, `/` becomes `~1` within a node name). The included [entity tree](../assets/trees/entity_tree.json) has 6,931 leaves; a complete operator map for that tree is not bundled. Category names alone do not define an atomic operator.

```bash
python -m atomqa.generation.leaves \
  --tree examples/category_tree.json --operators examples/operators.json \
  --output-dir outputs/leaves --splits 1
```

This writes `leaves_0.json`. Increase `--splits` to partition a larger tree. Leaf records contain `name`, `path`, `definition`, `siblings`, `entities`, and the supplied operator fields. To inspect seed coverage, pass classified seeds with `--entities`; for fresh generation, prepare empty leaves as above.

## Generation and checkpoints

```bash
python -m atomqa.generation.generate \
  --leaves outputs/leaves/leaves_0.json --output-dir outputs/generated \
  --max-entities 25 --max-rounds 10 --checkpoint-every 1000
```

Every generation call receives an explicit operator and definition. Each candidate operand produces one QA pair, which is accepted only if Verification returns `pass`. The model defaults to `o3-mini`; set `ATOM_MODEL` to change it.

`--max-entities` sets the target count per leaf, including existing verified records. `--max-rounds` bounds attempts to fill the leaves. If the target is not reached, the command saves partial results and exits with an error.

Outputs are `entities.json` and `leaves.json`, periodic `entities_ckpt_*.json` / `leaves_ckpt_*.json` pairs, and `run.json` on successful completion. Resume with a matching verified entity/leaf pair:

```bash
python -m atomqa.generation.generate \
  --entities outputs/generated/entities.json --leaves outputs/generated/leaves.json \
  --output-dir outputs/resumed --max-entities 25 --max-rounds 10
```

After an interrupted API call, use the latest complete checkpoint pair. Run one writer per output directory. Model sampling is not guaranteed deterministic; `--seed` in the export and operand-perturbation commands controls local randomness only.

## Deduplication and export

```bash
python -m atomqa.generation.deduplicate \
  --entities outputs/generated/entities.json --leaves outputs/generated/leaves.json \
  --output-dir outputs/deduplicated

python -m atomqa.preprocessing.export \
  --entities outputs/deduplicated/entities.json --format alpaca \
  --output outputs/atomic_qa.json --seed 42
```

Deduplication groups candidate aliases across sibling leaves and uses model judgments to merge equivalent entities. It retains operator metadata and updates leaf references. For split runs, deduplicate with a combined entity dictionary and all neighboring leaves present.

Export accepts one or more entity dictionaries or flat QA lists with `--entities FILE ...`. It concatenates and shuffles records; it does not remove repeated shard records. `--format raw` preserves item fields. `--format alpaca` writes `instruction`, an empty `input`, and `output` for training.

## Atomic record format

Generation and perturbation use a dictionary keyed by subject:

```json
{
  "France": {
    "subject": "France",
    "category": "/Entities/Geography/Countries",
    "items": [{
      "prompt": "What is the capital of France?",
      "template": "What is the capital of {}?",
      "attribute": "Paris",
      "operand": "France",
      "operator": "CapitalOf",
      "operator_definition": "Map a country to its national capital.",
      "output": "Paris",
      "verification": {"judgement": "pass", "reason": "Illustrative example"}
    }]
  }
}
```

`attribute` and `output` contain the same answer; `operand` matches the subject. Preserve the metadata when passing records between generation and perturbation. Export to Alpaca format only after these operations.

## Perturbations

Both commands take verified atomic records. They write perturbed records separately and preserve the input judgment as `source_verification`.

### Operand relabeling

```bash
python -m atomqa.perturbation.entity \
  --input examples/atomic_qa.json --output outputs/operand.json --seed 42
```

Operands and outputs are jointly relabeled using one consistent mapping to random strings of equal character length. The operator stays fixed. The mapping is saved to `outputs/operand.json.mapping.json`. This defines a synthetic world; the renamed entities are not real-world factual claims.

### Operator-output corruption

```bash
python -m atomqa.perturbation.operator \
  --input examples/atomic_qa.json --output outputs/operator.json --attempts 3
```

The Perturber changes the answer while preserving the question and operand. A candidate is accepted only when Verification returns `fail`; if no candidate passes that criterion within `--attempts`, the command raises an error. Verification is a model judgment.

Completed subjects are saved to `outputs/operator.json.checkpoint.json`; the final file is written when the run succeeds. This command applies perturbation to individual QA records. The full Atomic-X encoder/decoder pipeline is not included.

## Prompts

All runtime prompts live in [`atomqa/prompts/`](../atomqa/prompts). Generation and perturbation share `atomic.py`, including the Operand Generation, Verification, and Perturber templates from the appendix. Python placeholder names are normalized where needed. The output-generation, system, and repair messages connect these templates to the QA format; the appendix does not provide a standalone output-generation template.

`classification.py` and `deduplication.py` provide taxonomy assignment and entity-equivalence prompts. [paper_prompts.json](../assets/prompts/paper_prompts.json) preserves all nine appendix prompts for reference, including tree construction and Atomic-X prompts.
