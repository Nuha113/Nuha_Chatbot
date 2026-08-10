import spaces  # لازم يكون أول استيراد بالملف (متطلب ZeroGPU)
import os
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# ============================================
# 1. الإعدادات الأساسية
# ============================================
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

client = InferenceClient(token=HF_TOKEN)
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# ============================================
# 2. قراءة قاعدة المعرفة وتجهيز الـembeddings
# ============================================
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

knowledge_chunks = load_knowledge_base('nuha_knowledge_base.txt')
knowledge_embeddings = embedder.encode(knowledge_chunks)


# ============================================
# 3. البحث + التوليد (نفس منطق RAG اللي بنيناه)
# ============================================
def retrieve_relevant_info(query, top_k=6):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_chunks[i] for i in top_indices]


def generate_answer(query):
    relevant_info = retrieve_relevant_info(query)
    context = "\n".join(relevant_info)

    system_msg = f"""أنتِ مساعدة تجاوبين بالعربي فقط، نيابة عن نهى، بجملة أو جملتين قصار بس.
استخدمي المعلومات التالية فقط، ولا تضيفي معلومات من عندك:

{context}

لو المعلومات ما تكفي، قولي بس: "هذا سؤال أفضل تسألونه نهى مباشرة" وتوقفي فورًا."""

    response = client.chat_completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query}
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content


def respond(message, history):
    if not message.strip():
        return history, ""
    try:
        answer = generate_answer(message)
    except Exception as e:
        answer = f"حصل خطأ: {str(e)}"
    history = history + [(message, answer)]
    return history, ""


# ============================================
# 4. الهوية البصرية (نفس الألوان والعناصر من التصميم الأصلي)
# ============================================
CUSTOM_CSS = """
:root{
  --bg-deep:#120F24; --surface:#251F42; --line:#3A3163;
  --gold:#FFFFCC; --gold-soft:#FFF6A8; --teal:#4FC1B0;
  --text:#F4F1FA; --text-muted:#ADA3C9;
}
.gradio-container{ background:#191530 !important; font-family:'IBM Plex Sans Arabic', sans-serif !important; }
#header-html{
  background:linear-gradient(180deg, var(--surface) 0%, var(--bg-deep) 100%);
  border-radius:18px; padding:18px 20px; margin-bottom:10px;
  border:1px solid var(--line);
}
#chatbox{ background:var(--bg-deep) !important; border-color:var(--line) !important; }
#msginput textarea{ background:var(--bg-deep) !important; color:var(--text) !important; border-color:var(--line) !important; }
#sendbtn{ background:var(--gold) !important; color:var(--bg-deep) !important; font-weight:700 !important; }
"""

HEADER_HTML = """
<div id="header-html" style="display:flex; align-items:center; gap:16px; direction:rtl;">
  <div style="width:54px;height:54px;border-radius:50%;background:linear-gradient(145deg,#FFF6A8,#FFFFCC);
              display:flex;align-items:center;justify-content:center;font-weight:800;font-size:22px;color:#120F24;">ن</div>
  <div>
    <h2 style="margin:0 0 4px;color:#F4F1FA;font-weight:800;">تعرّفي على نهى</h2>
    <p style="margin:0;color:#ADA3C9;font-size:13px;">اسأليني عن شخصيتي، اهتماماتي، أو مشاريعي</p>
    <span style="display:inline-block;margin-top:6px;padding:3px 10px;border-radius:999px;
                  background:rgba(255,255,204,0.12);border:1px solid rgba(255,255,204,0.4);
                  font-size:11px;color:#FFF6A8;">★ ENFJ-A · البطل</span>
  </div>
</div>
"""

SUGGESTED = ["من هي نهى؟", "وش مشاريعها؟", "وش نمط شخصيتها؟", "وش تحب تتعلم؟"]

# ============================================
# 5. بناء الواجهة
# ============================================
with gr.Blocks(title="تعرّفي على نهى") as demo:
    gr.HTML(HEADER_HTML)
    chatbot = gr.Chatbot(elem_id="chatbox", height=380, rtl=True, show_label=False)
    msg = gr.Textbox(placeholder="اكتبي سؤالك هنا...", elem_id="msginput", show_label=False, text_align="right")

    with gr.Row():
        send = gr.Button("➤ إرسال", elem_id="sendbtn")

    with gr.Row():
        for q in SUGGESTED:
            gr.Button(q, size="sm").click(lambda q=q: q, None, msg).then(respond, [msg, chatbot], [chatbot, msg])

    send.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])

demo.launch(css=CUSTOM_CSS)
