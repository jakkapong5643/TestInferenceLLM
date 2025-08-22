import os
import torch
import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from transformers.utils import is_bitsandbytes_available
from peft import PeftModel

# ================== ตั้งค่าเริ่มต้น (แก้ได้จาก Sidebar) ==================
DEFAULT_BASE_ID = "scb10x/llama3.2-typhoon2-t1-3b-research-preview"
DEFAULT_ADAPTER = "./adapter_safetylfinal"  # ใส่ path โฟลเดอร์ adapter หรือ HF repo id ถ้ามี

SYSTEM_MSG = (
    "คุณคือผู้ช่วย AI ของกองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.) "
    "ตอบคำถามด้วยความสุภาพ กระชับ และอ้างอิงจากข้อมูลหรือบริบทที่มีอยู่เท่านั้น "
    "พร้อมอธิบายเหตุผลประกอบว่าทำไมถึงตอบเช่นนั้น "
    "ตอบให้ตรงกับคำถาม แต่หากข้อมูลไม่เพียงพอ ให้แจ้งผู้ใช้ตามตรงว่าไม่สามารถให้คำตอบได้ "
    "อธิบายคำตอบที่เกี่ยวข้องกับคำถามอย่างละเอียด"
)

# ================== Sidebar ==================
st.set_page_config(page_title="Student Loan Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Student Loan Chatbot (Streamlit)")

with st.sidebar:
    st.header("⚙️ Settings")
    base_id = st.text_input("Base model id", value=DEFAULT_BASE_ID)
    adapter_dir = st.text_input("Adapter path or HF repo id (optional)", value=DEFAULT_ADAPTER)
    use_4bit = st.checkbox("Use 4-bit quantization (bitsandbytes)", value=True)
    max_new_tokens = st.slider("Max new tokens", 64, 4096, 1024, step=64)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, step=0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.05, step=0.01)

# ================== Cache โหลดโมเดล ==================
@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_id: str, adapter_dir: str, want_4bit: bool):
    # 1) เลือก dtype ให้เหมาะกับเครื่อง
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        # CPU/MPS/XPU ที่ไม่มี bnb -> ใช้ float32 เพื่อความเสถียร
        dtype = torch.float32

    # 2) ตรวจความพร้อมของ bitsandbytes อย่างปลอดภัย
    def _bnb_ok():
        if not want_4bit:
            return False
        try:
            import bitsandbytes as bnb  # noqa
        except Exception:
            return False
        # เปิดเฉพาะเมื่อมี backend จริง ๆ
        if torch.cuda.is_available():
            return True
        # (ถ้าคุณตั้งใจจะใช้ MPS/HPU/XPU/NPU/CPU+IPEX ต้องติดตั้ง backend ให้ครบก่อน ค่อยเปิดเอง)
        return False

    enable_4bit = _bnb_ok()
    quant = None
    if enable_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) พยายามโหลดด้วย bnb ถ้าเปิดไว้; ถ้าพัง ให้ fallback อัตโนมัติแบบไม่ใช้ bnb
    def _load_base(try_quant: bool):
        kw = dict(
            pretrained_model_name_or_path=base_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # บาง env ไม่รองรับ sdpa บางเวอร์ชัน ลองใส่แล้วค่อย fallback
        try:
            if try_quant and quant is not None:
                kw["quantization_config"] = quant
            base = AutoModelForCausalLM.from_pretrained(attn_implementation="sdpa", **kw)
            return base
        except Exception:
            if try_quant and quant is not None:
                # ลองใหม่แบบตัด quantization ออก
                pass
            # ลองอีกครั้งแบบไม่ระบุ sdpa/quantization เลย
            kw.pop("quantization_config", None)
            if "attn_implementation" in kw:
                kw.pop("attn_implementation", None)
            base = AutoModelForCausalLM.from_pretrained(**kw)
            return base

    base = _load_base(try_quant=enable_4bit)

    model = base
    if adapter_dir.strip():
        try:
            model = PeftModel.from_pretrained(base, adapter_dir.strip())
        except Exception as e:
            st.warning(f"โหลด Adapter ไม่สำเร็จ ({e}) -> ใช้ base model แทน")

    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass

    if not enable_4bit:
        st.info("🧩 bitsandbytes ถูกปิดอัตโนมัติ (ไม่พบ backend ที่รองรับ) — กำลังใช้โหลดโมเดลแบบไม่ quantized")

    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(base_id, adapter_dir, use_4bit)

# ================== ฟังก์ชันตอบคำถาม (เรียบง่ายและเร็ว) ==================
def generate_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question}
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(temperature > 0.0),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=tokenizer.pad_token_id,
        )

    # ตัดส่วน prompt ออก เหลือเฉพาะคำตอบ
    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # กันเคสที่ template คั่นด้วย <|assistant|>
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

# ================== จัดการประวัติแชท ==================
if "history" not in st.session_state:
    st.session_state.history = []

# ปุ่มล้างแชท
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
with col2:
    st.caption("Tip: ปรับค่าใน Sidebar ถ้าตอบช้า/ยาวเกินไป")

# แสดงประวัติ
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ช่องพิมพ์แชท
user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับการกู้ยืม (กยศ.) ที่นี่...")
if user_input:
    # แสดงคำถาม
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # สร้างคำตอบ
    with st.chat_message("assistant"):
        with st.spinner("กำลังคิดคำตอบ..."):
            answer = generate_answer(user_input)
            st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})

