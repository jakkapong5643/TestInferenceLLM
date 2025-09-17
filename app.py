import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

st.set_page_config(page_title="Student Loan Chatbot (Typhoon T1 3B)", page_icon="", layout="wide")
st.title("กยศ. Chatbot — Typhoon T1 3B (PEFT optional)")

with st.sidebar:
    st.subheader("Settings")
    base_id = st.text_input(
        "Base model",
        value="scb10x/llama3.2-typhoon2-t1-3b-research-preview",
        help="Hugging Face repo id ของ base model"
    )
    use_adapter = st.checkbox("LoRA Adapter", value=True)
    adapter_dir = st.text_input(
        "Adapter directory",
        value=".\typhoon-t1-3b-qlora\adapter_safety\final"
    )
    system_msg = st.text_area(
        "System message",
        value=(
            "คุณคือผู้ช่วย AI ของกองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.) "
            "ตอบคำถามด้วยความสุภาพ กระชับ และอ้างอิงจากข้อมูลหรือบริบทที่มีอยู่เท่านั้น "
            "พร้อมอธิบายเหตุผลประกอบว่าทำไมถึงตอบเช่นนั้น "
            "ตอบให้ตรงกับคำถาม แต่หากข้อมูลไม่เพียงพอ ให้แจ้งผู้ใช้ตามตรงว่าไม่สามารถให้คำตอบได้ "
            "อธิบายคำตอบที่เกี่ยวข้องกับคำถามแบบละเอียด"
        ),
        height=140,
    )
    max_new_tokens = st.slider("Max new tokens", 2048, 4096, 512)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.01)
    do_sample = temperature > 0.0

    st.markdown("---")
    if st.button("ล้างบทสนทนา"):
        st.session_state["messages"] = []

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_id: str, adapter_dir: str | None, use_adapter: bool):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",       
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model = base
    if use_adapter and adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(base, adapter_dir)
        try:
            model = model.merge_and_unload() 
        except Exception:
            pass

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(
    base_id=base_id,
    adapter_dir=adapter_dir,
    use_adapter=use_adapter
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับกยศ.")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    messages = [{"role": "system", "content": system_msg}]
    history_tail = st.session_state["messages"][-8:] 
    messages.extend(history_tail)

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        prompt = (
            f"<|system|>\n{system_msg}\n"
            + "".join([f"<|{m['role']}|>\n{m['content']}\n" for m in history_tail])
            + "<|assistant|>\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with st.chat_message("assistant"):
        with st.spinner("กำลังตอบ…"):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if not text:
                full = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<|assistant|>" in full:
                    text = full.split("<|assistant|>")[-1].strip()
                else:
                    text = full.strip()

            st.markdown(text)
            st.session_state["messages"].append({"role": "assistant", "content": text})

st.caption(
    "โมเดล: **{}**{} • FP16/BF16 (ไม่ใช้ 4-bit)".format(
        base_id, " + LoRA" if use_adapter else ""
    )
)

