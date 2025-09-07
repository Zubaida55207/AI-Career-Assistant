import os
import re
import difflib
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# STEP 1: Load local files
# ===============================
def load_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return f"{filename} not found."

bio_text = load_file("Bio.txt")
projects_text = load_file("Projects.txt")
goals_text = load_file("Career Goals.txt")
linkedin_text = load_file("linkedin.txt")

docs = {
    "Bio": bio_text,
    "Projects": projects_text,
    "Goals": goals_text,
    "LinkedIn": linkedin_text
}

# ===============================
# STEP 2: Retrieval with TF-IDF
# ===============================
sections = list(docs.keys())
contents = list(docs.values())

def clean_text(s: str) -> str:
    s = re.sub(r"(?i)i am your ai career assistant[\.\s]*", " ", s)
    s = re.sub(r'\r', '\n', s)
    s = re.sub(r'\n\s*\n+', '\n\n', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

contents = [clean_text(c) for c in contents]
vectorizer = TfidfVectorizer().fit(contents)
vectors = vectorizer.transform(contents)

def retrieve_answer(query, top_k=1):
    user_vec = vectorizer.transform([query])
    sims = cosine_similarity(user_vec, vectors).flatten()
    idx = sims.argsort()[::-1][:top_k]
    if sims[idx[0]] <= 0:
        return None
    section = sections[idx[0]]
    return f"**Answer based on {section}:**\n\n{contents[idx[0]]}"

# ===============================
# STEP 3: Recruiter FAQs (fuzzy matching)
# ===============================
recruiter_questions = {
    "why should we hire you": (
        "✅ You should consider hiring me because I bring a strong mix of **technical expertise and dedication**:\n"
        "- Proven skills in **Machine Learning, Data Science, and Full-Stack Development**\n"
        "- Hands-on experience with projects like **Sales Prediction Hybrid Model**\n"
        "- Strong foundation in **Python, PHP, Java, and ML tools (Scikit-learn, XGBoost, RandomForest)**\n"
        "- Committed to building **safe, impactful AI solutions**\n"
        "- Excellent **research, teamwork, and communication skills**\n\n"
        "I believe I can add real value to your team and grow with your organization 🚀."
    ),
    "tell me about yourself": (
        "👩‍💻 I am Zubaida Bibi, a passionate Computer Science graduate with expertise in **Machine Learning and Full-Stack Development**. "
        "I’ve built projects like a **Real-time Sales Prediction Hybrid Model** and I’m actively preparing for an **MS/MPhil abroad with scholarship**. "
        "I enjoy solving real-world problems with AI and aspire to contribute to impactful, safe, and innovative AI projects."
    ),
    "what are your strengths": (
        "💪 My strengths include:\n"
        "- Quick learner, especially with new technologies\n"
        "- Strong analytical and problem-solving skills\n"
        "- Ability to work independently and in teams\n"
        "- Good research and presentation skills\n"
    ),
    "what are your weaknesses": (
        "⚡ I sometimes get deeply focused on one task for too long, but I’ve been working on improving my time management "
        "by breaking projects into smaller milestones and setting deadlines."
    ),
    "what are your career goals": (
        "🎯 My career goal is to pursue an **MS/MPhil abroad (UK/USA)** with a scholarship, and then grow into a role as a **Machine Learning Engineer or Data Scientist**. "
        "Ultimately, I want to contribute to impactful, safe AI projects that solve real-world problems."
    ),
    "are you open to relocation or remote work": (
        "🌍 Yes! I am open to **relocation abroad (UK/USA)** for higher studies and job opportunities, and also to fully **remote ML/Data Science roles**."
    ),
    "when can you start": (
        "📅 I am available to start **immediately** for remote roles, and open to relocation opportunities depending on visa and sponsorship timelines."
    )
}

def match_recruiter_question(msg):
    for q, ans in recruiter_questions.items():
        if difflib.SequenceMatcher(None, msg, q).ratio() > 0.65:
            return ans
    return None

# ===============================
# STEP 4: Chatbot Logic
# ===============================
def chatbot_response(message, history):
    msg = message.lower().strip()

    # Greetings & casual replies
    if msg in ["hi", "hello", "hey"]:
        return "Hello 👋! I’m Zubaida’s Career Assistant. You can ask about her Bio, Projects, Skills, Goals, or LinkedIn."
    elif msg in ["ok", "okay", "thanks", "thank you"]:
        return "😊 Glad to help! You can ask me about Zubaida’s career, projects, or skills anytime."

    # Recruiter questions
    recruiter_ans = match_recruiter_question(msg)
    if recruiter_ans:
        return recruiter_ans

    # Direct Q&A from files
    if "bio" in msg:
        return bio_text
    elif "project" in msg:
        return projects_text
    elif "goal" in msg:
        return goals_text
    elif "linkedin" in msg:
        return linkedin_text
    elif "skill" in msg or "skills" in msg:
        return (
            "🛠️ Zubaida’s Skills:\n"
            "- **Programming & ML**: Python, Scikit-learn, XGBoost, RandomForest\n"
            "- **Web Development**: PHP, Laravel, JavaScript, HTML/CSS\n"
            "- **Tools & Design**: Gradio, GitHub, Graphic Design\n"
            "- **Soft Skills**: Research writing, presentation, teamwork\n"
        )

    # Retrieval from local files
    retrieved = retrieve_answer(msg)
    if retrieved:
        return retrieved

    # ❌ Default fallback
    return "❌ I only answer about Zubaida’s Bio, Projects, Skills, Goals, LinkedIn, or recruiter-style interview questions."

# ===============================
# STEP 5: Gradio UI
# ===============================
demo = gr.ChatInterface(
    chatbot_response,
    title="🤖 Zubaida – AI Career Assistant",
    description="Ask me about my Bio, Projects, Skills, Goals, LinkedIn, or typical recruiter interview questions!",
    textbox=gr.Textbox(
        placeholder="💡 Try asking: Why should we hire you? | Tell me about yourself | What are your strengths?"
    )
)

if __name__ == "__main__":
    demo.launch()




   
