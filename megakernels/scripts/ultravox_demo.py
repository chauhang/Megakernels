"""Ultravox voice demo: speech -> megakernel-accelerated Llama-3.2-1B on GB10.

Ultravox's audio tower (Whisper) + projector turn speech into embeddings that are
spliced into the text prompt; the repo's Llama prefills on those embeddings and the
compiled megakernel (mk_llama) decodes the response. No kernel changes are needed:
Ultravox's decoder *is* Llama-3.2-1B, exactly what mk_llama already runs.

Run from repo root with THUNDERKITTENS_ROOT / MEGAKERNELS_ROOT set (see README):
    python megakernels/scripts/ultravox_demo.py
"""

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoModel, AutoProcessor

from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
from megakernels.generators import MK_Generator
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import assign_to_sms, tensorize_instructions

DEV = "cuda:0"
UV = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
LLM = "meta-llama/Llama-3.2-1B-Instruct"
MK_DIR = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
NTOK = 80


@torch.inference_mode()
def main():
    torch.cuda.set_device(DEV)

    # Ultravox audio stack (audio_tower + projector) + processor.
    proc = AutoProcessor.from_pretrained(UV, trust_remote_code=True)
    uv = AutoModel.from_pretrained(UV, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()

    # Repo Llama (the megakernel's decoder) + compiled-kernel schedule.
    model = LlamaForCausalLM.from_pretrained(
        LLM, device=DEV,
        extra_config=ExtraModelConfig(interleave_rope=True, max_len_override=16384),
    )
    sched = make_schedule_builder("latency").build(model)
    tensorize_instructions(sched.globs, assign_to_sms("rr", schedule=sched))
    mk_gen = MK_Generator(model, make_mk_interpreter("latency", MK_DIR), sched)
    tok = proc.tokenizer

    # Audio sample (decode bytes with soundfile to avoid the torchcodec dep).
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    ).cast_column("audio", Audio(decode=False))
    audio, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
    audio = audio.astype(np.float32)
    print(f"[audio] {len(audio) / sr:.1f}s @ {sr}Hz")
    print(f"[reference transcript] {ds[0]['text']!r}\n")

    # Speech + text prompt -> merged input embeddings (Ultravox merge, inline).
    turns = [{"role": "user", "content": f"{proc.audio_placeholder} Transcribe the audio."}]
    text = proc.tokenizer.apply_chat_template(turns, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=audio, sampling_rate=sr, return_tensors="pt").to(DEV)

    embeds = uv.get_input_embeddings()(inp["input_ids"])
    # NB: run the Whisper encoder with no attention mask. For a single, unpadded
    # clip there is nothing to mask, and both mask paths in the custom encoder
    # (audio_len, and the streaming/latency mask) are incompatible with
    # transformers 5.12's SDPA attention. Null the streaming mask + omit audio_len.
    uv.audio_tower.audio_streaming_mask = None
    audio_hidden = uv.audio_tower.forward(
        inp["audio_values"].to(uv.audio_tower.dtype)
    ).last_hidden_state.to(embeds.dtype)
    audio_embeds = uv.multi_modal_projector.forward(audio_hidden)
    start = int(inp["audio_token_start_idx"][0])
    alen = int(inp["audio_token_len"][0])
    embeds[0, start : start + alen] = audio_embeds[0][:alen]

    # Prefill on the merged embeddings (fills the KV cache the megakernel reads).
    plen = inp["input_ids"].shape[-1]
    pos = torch.arange(plen, device=DEV)
    prefill = model(BatchState(input_ids=inp["input_ids"], position_ids=pos, hidden_states=embeds))
    out = torch.zeros(1, NTOK, device=DEV, dtype=torch.long)
    out[:, 0] = prefill.output_ids[:, -1:]

    # Megakernel decode.
    torch.cuda.synchronize()
    t0 = time.time()
    mk_gen.generate(out, plen, NTOK - 1)
    torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"[megakernel response] {tok.decode(out[0], skip_special_tokens=True)!r}")
    print(f"[decode] {(NTOK - 1) / dt:.1f} tok/s on GB10 (mk_llama, batch-1)")


if __name__ == "__main__":
    main()
