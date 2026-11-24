"""Fine-tune the path selector (Qwen3-VL-4B) with QLoRA on data from build_sft_data.py (phase 5).

The base model is loaded in 4-bit exactly as for the selector (vlm/backend.py: NF4, vision
encoder kept in bf16), stays frozen, and small LoRA adapters are trained on the language model
(rank 64, learning rate 1e-4, as DriveVLN; setting 11). Each example is the selector's own
prompt and picture followed by the target answer; only the answer is learned.

The data is in epoch order, so the adapter is saved at the end of every epoch
(<out>/epoch1, epoch2, ...), with the validation loss logged there. Pick the best epoch
afterwards by asking the validation frames with each adapter (run_selector.py --adapter) and
comparing them with scripts/evaluate.py --split val: early stopping by choosing the epoch.

Usage (from the repo root; needs a GPU with requirements-vlm.txt; meant for Polaris):
    python scripts/train_selector.py --data outputs/sft/nomatch-skip   # -> outputs/adapters/nomatch-skip
    python scripts/train_selector.py --data outputs/sft/nomatch-keep
    python scripts/train_selector.py --data outputs/sft/smoke --max-steps 4 --eval-limit 4   # smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.nav.sft import choice_span  # noqa: E402
from offroad.vlm.backend import DEFAULT_MODEL  # noqa: E402
from offroad.vlm.finetune import IGNORE, LORA_TARGET, answer_labels, epoch_end_steps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="a folder made by build_sft_data.py")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/adapters/<data folder name>")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8, help="examples per optimizer step (batch size 1)")
    parser.add_argument("--warmup", type=float, default=0.03, help="share of the steps for the learning-rate warm-up")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (smoke test)")
    parser.add_argument("--eval-limit", type=int, default=None, help="use only the first N validation examples")
    parser.add_argument("--workers", type=int, default=2, help="data-loading processes (0 on Windows)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Collator:
    """Turns one example record into model inputs with labels (batch size 1)."""

    def __init__(self, processor, data_dir: Path):
        self.processor = processor
        self.data_dir = data_dir
        tok = processor.tokenizer
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        from PIL import Image

        if len(batch) != 1:
            raise ValueError("use batch size 1 (with gradient accumulation)")
        ex = batch[0]
        image = Image.open(self.data_dir / ex["image"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": ex["prompt"]}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        enc = self.processor.tokenizer(ex["target"], add_special_tokens=False, return_offsets_mapping=True)
        span = choice_span(ex["target"]) if ex["mask_choice"] else None
        ids, labels = answer_labels(enc["input_ids"], enc["offset_mapping"], span, self.end_id)

        n_prompt = inputs["input_ids"].shape[1]
        out = dict(inputs)
        out["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([ids])], dim=1)
        out["attention_mask"] = torch.ones_like(out["input_ids"])
        if "mm_token_type_ids" in inputs:  # 0 = text token
            out["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"],
                                                  torch.zeros((1, len(ids)), dtype=inputs["mm_token_type_ids"].dtype)],
                                                 dim=1)
        out["labels"] = torch.tensor([[IGNORE] * n_prompt + labels])
        return out


def main() -> None:
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import SequentialSampler
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments)

    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    train = read_jsonl(args.data / "train.jsonl")
    val = read_jsonl(args.data / "val.jsonl")[:args.eval_limit]
    out = args.out or REPO_ROOT / "outputs" / "adapters" / args.data.name
    out.mkdir(parents=True, exist_ok=True)
    ends = epoch_end_steps(meta["train_examples_per_epoch"], args.grad_accum)
    print(f"{len(train)} training examples in {meta['epochs']} epochs (ends at optimizer steps {ends}), "
          f"{len(val)} validation examples", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["visual", "lm_head"])  # as vlm/backend.py
    model = AutoModelForImageTextToText.from_pretrained(args.model, quantization_config=quant, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class SaveEachEpoch(TrainerCallback):
        """Save the adapter (and evaluate) when an epoch of the data ends."""

        def on_step_end(self, targs, state, control, **kwargs):
            if state.global_step in ends[:-1]:
                epoch = ends.index(state.global_step) + 1
                model.save_pretrained(out / f"epoch{epoch}")
                control.should_evaluate = True
            return control

    class InOrderTrainer(Trainer):
        """Reads the data in its order (epoch by epoch), so a checkpoint per epoch means something."""

        def _get_train_sampler(self, *args, **kwargs):
            return SequentialSampler(self.train_dataset)

    targs = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        # warmup_steps: a float below 1 is a share of the steps (transformers 5 has no warmup_ratio).
        warmup_steps=args.warmup, num_train_epochs=1, max_steps=args.max_steps, bf16=True, logging_steps=10,
        save_strategy="no", eval_strategy="no", remove_unused_columns=False, dataloader_num_workers=args.workers,
        report_to="none", seed=args.seed, label_names=["labels"],
    )
    trainer = InOrderTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                             data_collator=Collator(processor, args.data), callbacks=[SaveEachEpoch()])
    started = time.perf_counter()
    trainer.train()
    last = out / f"epoch{meta['epochs']}" if args.max_steps < 0 else out / "smoke"
    model.save_pretrained(last)
    final_eval = trainer.evaluate()
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "data_meta": meta, "epoch_end_steps": ends,
           "minutes": round((time.perf_counter() - started) / 60, 1), "final_eval": final_eval,
           "log_history": trainer.state.log_history}
    (out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Saved the adapters and train_log.json to {out} ({log['minutes']} min); last: {last.name}")


if __name__ == "__main__":
    main()
