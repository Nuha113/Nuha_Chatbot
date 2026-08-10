import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
 
app = Flask(__name__)
CORS(app)
 
# ============================================
# 1. تجهيز الإعدادات (تصير مرة وحدة عند تشغيل السيرفر)
# ============================================
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
 
client = InferenceClient(token=HF_TOKEN)
 
 
def embed_texts(texts):
    """يحسب embeddings عبر HF API بدل تحميل موديل محلي (يوفر ذاكرة كبيرة)"""
    result = client.feature_extraction(texts, model=EMBED_MODEL)
    return np.array(result)
 
 
# ============================================
# 2. قراءة قاعدة المعرفة وتجهيز الـembeddings
# ============================================
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]
 
knowledge_chunks = load_knowledge_base('nuha_knowledge_base.txt')
knowledge_embeddings = embed_texts(knowledge_chunks)
 
 
# ============================================
# 3. البحث عن أقرب معلومات للسؤال
# ============================================
def retrieve_relevant_info(query, top_k=6):
    query_embedding = embed_texts([query])
    similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_chunks[i] for i in top_indices]
 
 
# ============================================
# 4. توليد الجواب عبر Hugging Face Inference API
#    (بدل تحميل الموديل محليًا — أسرع وأدق، بدون قيود quantization)
# ============================================
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
 
 
# ============================================
# 5. مسارات السيرفر (Routes)
# ============================================
@app.route('/')
def index():
    return send_file('chatbot_interface.html')
 
 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message', '')
    if not query.strip():
        return jsonify({'answer': 'اكتبي سؤال أول 🙂'})
    try:
        answer = generate_answer(query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'حصل خطأ: {str(e)}'}), 500
 
 
# ============================================
# 6. التشغيل — لازم على منفذ 7860 و host="0.0.0.0" لـ Hugging Face Spaces
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
