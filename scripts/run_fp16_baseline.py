#!/usr/bin/env python3
"""R1-W2 FP16 Baseline: AWQ INT4 vs FP16 accuracy on MMLU / ARC-Challenge.
Downloads models via ModelScope (domestic CDN) or HuggingFace, evaluates locally.
Runs on single 98GB RTX PRO 6000 Blackwell.
"""

import gc
import json
import logging
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/modelscope_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/fp16_baseline.log"),
    ],
)
log = logging.getLogger("fp16_baseline")

SEED = 42
N_SAMPLES = 100
DATA_PATH = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/fp16_eval_data.json")
MODELSCOPE_CACHE = "/root/autodl-tmp/modelscope_cache"
CLEANUP_AFTER_EVAL = True

MODELS = [
    {
        "name": "Qwen2.5-14B-Instruct",
        "awq_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "fp16_id": "Qwen/Qwen2.5-14B-Instruct",
        "can_fp16": True,
        "params_b": 14,
        "source": "modelscope",
    },
    {
        "name": "Qwen2.5-32B-Instruct",
        "awq_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "fp16_id": "Qwen/Qwen2.5-32B-Instruct",
        "can_fp16": True,
        "params_b": 32,
        "source": "modelscope",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "awq_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "fp16_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "can_fp16": True,
        "params_b": 8,
        "source": "modelscope",
    },
    {
        "name": "Llama-3.1-70B-Instruct",
        "awq_id": "LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "fp16_id": "LLM-Research/Meta-Llama-3.1-70B-Instruct",
        "can_fp16": False,
        "params_b": 70,
        "skip_reason": "FP16 requires ~140GB VRAM, exceeds single 98GB GPU",
        "source": "modelscope",
    },
    {
        "name": "Mistral-Large-Instruct-2407",
        "awq_id": "TechxGenus/Mistral-Large-Instruct-2407-AWQ",
        "fp16_id": "mistralai/Mistral-Large-Instruct-2407",
        "can_fp16": False,
        "params_b": 123,
        "skip_reason": "FP16 requires ~246GB VRAM, exceeds single 98GB GPU",
        "source": "huggingface",
    },
]

OUTPUT_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001")
CKPT_PATH = OUTPUT_DIR / "fp16_baseline_checkpoint.json"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_model(model_id, source="modelscope"):
    """Download model, return local path."""
    log.info(f"Downloading {model_id} via {source}...")
    t0 = time.time()
    if source == "modelscope":
        from modelscope import snapshot_download
        local_dir = snapshot_download(model_id, cache_dir=MODELSCOPE_CACHE)
    else:
        from huggingface_hub import snapshot_download as hf_download
        local_dir = hf_download(model_id)
    elapsed = time.time() - t0
    log.info(f"Download done in {elapsed:.1f}s -> {local_dir}")
    return local_dir


def format_mcq(question, choices, labels=None):
    if labels is None:
        labels = [chr(65 + i) for i in range(len(choices))]
    opts = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
    return (
        "Answer the following multiple-choice question. "
        "Reply with ONLY the letter of the correct answer.\n\n"
        f"Question: {question}\n{opts}\n\nAnswer:"
    )


def extract_answer(text):
    text = text.strip()
    m = re.match(r'^[(\[]?([A-Da-d])[)\].]?', text)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:answer|correct)\s*(?:is|:)\s*[(\[]?([A-Da-d])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return None


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)


def load_and_eval(model_path, mmlu_samples, arc_samples, model_name=""):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading from {model_path}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    log.info(f"Loaded in {load_time:.1f}s, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def evaluate(samples, dataset_name):
        correct = 0
        total = 0
        for i, s in enumerate(samples):
            prompt = format_mcq(s["question"], s["choices"])
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=5, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(response)
            gold = chr(65 + s["answer"])
            if pred == gold:
                correct += 1
            total += 1
            if i < 3:
                log.info(f"  [{model_name}] {dataset_name} #{i}: resp={response!r} pred={pred} gold={gold} {'OK' if pred==gold else 'WRONG'}")
            if (i + 1) % 25 == 0:
                log.info(f"  [{model_name}] {dataset_name}: {i+1}/{len(samples)}, acc={correct/total:.3f}")
        return correct / total if total > 0 else 0.0

    mmlu_acc = evaluate(mmlu_samples, "MMLU")
    arc_acc = evaluate(arc_samples, "ARC")
    log.info(f"Results for {model_name}: MMLU={mmlu_acc:.4f}, ARC={arc_acc:.4f}")

    del model
    del tokenizer
    release_gpu()

    return mmlu_acc, arc_acc


def load_checkpoint():
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"results": []}


def save_checkpoint(ckpt):
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CKPT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def save_output(results, summary=None):
    path = OUTPUT_DIR / "fp16_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "R1-W2 FP16 Baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "gpu": "1x NVIDIA RTX PRO 6000 Blackwell (98GB)",
        "results": results,
    }
    if summary:
        out["summary"] = summary
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def cleanup_model(model_path):
    if not CLEANUP_AFTER_EVAL or not model_path:
        return
    p = Path(model_path)
    if not p.exists():
        return
    size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    log.info(f"Cleaning up {model_path} ({size_gb:.1f}GB)")
    shutil.rmtree(model_path, ignore_errors=True)


def main():
    set_seed(SEED)
    log.info("R1-W2 FP16 Baseline V3 (answer format fix + all families)")
    log.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    mmlu = data["mmlu"]
    arc = data["arc"]
    log.info(f"Data: MMLU={len(mmlu)}, ARC={len(arc)}")

    ckpt = load_checkpoint()
    done_models = {r["model"] for r in ckpt.get("results", [])}
    final_results = list(ckpt.get("results", []))

    for cfg in MODELS:
        name = cfg["name"]

        if name in done_models:
            log.info(f"Skipping {name} (in checkpoint)")
            continue

        log.info(f"\n{'='*60}\n  {name} ({cfg['params_b']}B)\n{'='*60}")

        entry = {
            "model": name,
            "awq_model_id": cfg["awq_id"],
            "fp16_model_id": cfg["fp16_id"],
            "params_b": cfg["params_b"],
            "mmlu": {},
            "arc": {},
        }

        awq_path = None
        fp16_path = None

        # --- AWQ ---
        try:
            awq_path = download_model(cfg["awq_id"], source=cfg["source"])
            awq_mmlu, awq_arc = load_and_eval(awq_path, mmlu, arc, model_name=f"{name}-AWQ")
            entry["mmlu"]["awq_acc"] = round(awq_mmlu, 4)
            entry["arc"]["awq_acc"] = round(awq_arc, 4)
        except Exception as e:
            log.error(f"AWQ failed for {name}: {e}", exc_info=True)
            entry["mmlu"]["awq_acc"] = None
            entry["arc"]["awq_acc"] = None
            entry["awq_error"] = str(e)
            final_results.append(entry)
            ckpt["results"] = final_results
            save_checkpoint(ckpt)
            save_output(final_results)
            release_gpu()
            cleanup_model(awq_path)
            continue
        finally:
            cleanup_model(awq_path)

        # --- FP16 ---
        if cfg["can_fp16"]:
            try:
                fp16_path = download_model(cfg["fp16_id"], source=cfg["source"])
                fp16_mmlu, fp16_arc = load_and_eval(fp16_path, mmlu, arc, model_name=f"{name}-FP16")
                entry["mmlu"]["fp16_acc"] = round(fp16_mmlu, 4)
                entry["arc"]["fp16_acc"] = round(fp16_arc, 4)
                entry["mmlu"]["delta_pp"] = round((fp16_mmlu - awq_mmlu) * 100, 2)
                entry["arc"]["delta_pp"] = round((fp16_arc - awq_arc) * 100, 2)
                entry["gpu_config"] = "1x cuda:0"
            except Exception as e:
                log.error(f"FP16 failed for {name}: {e}", exc_info=True)
                entry["mmlu"]["fp16_acc"] = None
                entry["arc"]["fp16_acc"] = None
                entry["fp16_error"] = str(e)
                entry["gpu_config"] = "1x cuda:0"
            finally:
                cleanup_model(fp16_path)
        else:
            entry["mmlu"]["fp16_acc"] = "skipped"
            entry["mmlu"]["delta_pp"] = "skipped"
            entry["arc"]["fp16_acc"] = "skipped"
            entry["arc"]["delta_pp"] = "skipped"
            entry["skip_reason"] = cfg.get("skip_reason", "FP16 exceeds GPU memory")
            entry["gpu_config"] = "1x cuda:0 (AWQ only)"

        final_results.append(entry)
        ckpt["results"] = final_results
        save_checkpoint(ckpt)
        save_output(final_results)
        log.info(f"[checkpoint] {len(final_results)}/{len(MODELS)} models done")

    # --- Summary ---
    deltas = []
    for r in final_results:
        for bench in ("mmlu", "arc"):
            d = r[bench].get("delta_pp")
            if isinstance(d, (int, float)):
                deltas.append(abs(d))

    summary = {
        "n_compared": len([r for r in final_results
                          if isinstance(r["mmlu"].get("delta_pp"), (int, float))]),
        "n_awq_only": sum(1 for c in MODELS if not c["can_fp16"]),
        "models_evaluated": [m["name"] for m in MODELS],
    }
    if deltas:
        summary["mean_delta_pp"] = round(sum(deltas) / len(deltas), 2)
        summary["max_delta_pp"] = round(max(deltas), 2)
        summary["conclusion"] = (
            f"AWQ degradation {'<2pp' if max(deltas) < 2 else '>2pp'} "
            f"(mean {sum(deltas)/len(deltas):.1f}pp, max {max(deltas):.1f}pp)"
        )

    path = save_output(final_results, summary)
    log.info(f"\nDone. Results: {path}")
    log.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
