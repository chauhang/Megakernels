"""Voice -> megakernel-accelerated Llama-3.2-1B reply on GB10.

A decoupled, robust voice demo:
  * ASR frontend: IBM Granite-Speech-4.1-2b-plus (2026-06, native in transformers,
    no trust_remote_code, no NeMo) transcribes the speech.
  * LLM: the compiled megakernel (mk_llama) decodes the Llama-3.2-1B reply.

No kernel changes, no vendored code. Swap ASR_MODEL to try other native ASR
frontends (e.g. UsefulSensors/moonshine-base, which needs no torchaudio).

Deps: Granite-Speech needs `torchaudio` + `soundfile`. On the GB10/Spark torch
2.12+cu130 build (ahead of published torchaudio) install the ABI-compatible
2.11 with: pip install torchaudio==2.11.0 --no-deps --index-url \
https://download.pytorch.org/whl/cu130

Run from repo root with THUNDERKITTENS_ROOT / MEGAKERNELS_ROOT set (see README):
    python megakernels/scripts/voice_mk_demo.py
"""

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
from megakernels.generators import MK_Generator
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import assign_to_sms, tensorize_instructions

DEV = "cuda:0"
ASR_MODEL = "ibm-granite/granite-speech-4.1-2b-plus"
LLM = "meta-llama/Llama-3.2-1B-Instruct"
MK_DIR = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
NTOK = 96
GRANITE_SYSTEM = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)


@torch.inference_mode()
def main():
    torch.cuda.set_device(DEV)

    # Newer native ASR frontend (Granite-Speech-4.1, audio-LLM used for ASR).
    aproc = AutoProcessor.from_pretrained(ASR_MODEL)
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL, dtype=torch.bfloat16).to(DEV).eval()
    atok = aproc.tokenizer

    # Megakernel Llama-1B decoder (the compiled mk_llama).
    model = LlamaForCausalLM.from_pretrained(
        LLM, device=DEV,
        extra_config=ExtraModelConfig(interleave_rope=True, max_len_override=16384),
    )
    sched = make_schedule_builder("latency").build(model)
    tensorize_instructions(sched.globs, assign_to_sms("rr", schedule=sched))
    mk_gen = MK_Generator(model, make_mk_interpreter("latency", MK_DIR), sched)
    tok = AutoTokenizer.from_pretrained(LLM)

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    ).cast_column("audio", Audio(decode=False))
    audio, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
    audio = audio.astype(np.float32)
    print(f"[audio] {len(audio) / sr:.1f}s @ {sr}Hz")

    # 1) Speech -> text (Granite-Speech).
    chat = [
        {"role": "system", "content": GRANITE_SYSTEM},
        {"role": "user", "content": "<|audio|> can you transcribe the speech into a written format?"},
    ]
    prompt_text = atok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    torch.cuda.synchronize(); t0 = time.time()
    ai = aproc(prompt_text, audio, device=DEV, return_tensors="pt").to(DEV)
    aout = asr.generate(**ai, max_new_tokens=200, do_sample=False, num_beams=1)
    transcript = atok.decode(
        aout[0, ai["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    torch.cuda.synchronize()
    print(f"[ASR · Granite-Speech-4.1] {transcript!r}  ({(time.time() - t0) * 1000:.0f} ms)")

    # 2) Text prompt -> prefill -> megakernel decode.
    msgs = [{"role": "user", "content": f'I said: "{transcript}" Reply in one short sentence.'}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    ids = enc["input_ids"].to(DEV)
    plen = ids.shape[-1]
    pf = model(BatchState(input_ids=ids, position_ids=torch.arange(plen, device=DEV)))
    out = torch.zeros(1, NTOK, device=DEV, dtype=torch.long)
    out[:, 0] = pf.output_ids[:, -1:]

    torch.cuda.synchronize(); t0 = time.time()
    mk_gen.generate(out, plen, NTOK - 1)
    torch.cuda.synchronize(); dt = time.time() - t0

    ids_out = out[0].tolist()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot in ids_out:
        ids_out = ids_out[: ids_out.index(eot)]
    print(f"[reply · megakernel] {tok.decode(ids_out, skip_special_tokens=True).strip()!r}")
    print(f"[decode] {(NTOK - 1) / dt:.1f} tok/s on GB10 (mk_llama, batch-1)")


if __name__ == "__main__":
    main()
