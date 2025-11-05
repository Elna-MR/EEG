import os
import pyperclip
import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
from dotenv import load_dotenv
import re
import traceback
import time

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain.chains import RetrievalQA

from openai import OpenAI

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"

RAG_DB_PATH = r"C:\Users\saeed\Dropbox\Applied Companies\Bristol Myer\LDB"
PROJECT_KEYWORDS = [
    "gilead", "gabi", "audio search", "field force", "msl", "veeva", "smart search",
    "bedrock", "rwe", "marketing content", "penguin", "geniq", "gpro"
]

YOUR_BACKGROUND = """()"""
COMPANY_CONTEXT = """(company context here)"""

def format_answer(answer):
    abbreviations = ['e.g.', 'i.e.', 'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'U.S.A.', 'vs.', 'etc.']
    for abbr in abbreviations:
        answer = answer.replace(abbr, abbr.replace('.', '[DOT]'))
    answer = re.sub(r'(?<=[.!?])\s+', '\n\n', answer)
    answer = answer.replace('[DOT]', '.')
    return answer.strip()

def show_answer_gui(answer):
    formatted = format_answer(answer)
    pyperclip.copy(formatted)
    root = tk.Tk()
    root.title("Interview Assistant Answer")
    root.geometry("900x1000")

    txt = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, font=("Consolas", 16), bg="#22262e", fg="#f6f6f7", spacing3=14, padx=18, pady=18
    )
    txt.insert(tk.END, formatted)
    txt.configure(state='disabled')
    txt.pack(padx=24, pady=20, fill=tk.BOTH, expand=True)
    tk.Label(
        root, text="✅ Answer copied to clipboard (Click middle mouse button to close)", 
        fg="#a5eb6a", bg="#22262e", font=("Arial", 12)
    ).pack(pady=(0, 14))

    def on_middle_click(event):
        root.destroy()

    # Bind middle mouse button (button 2 on Windows, button 3 on X11/Linux)
    txt.bind("<Button-2>", on_middle_click)   # Windows, Mac (often)
    txt.bind("<Button-3>", on_middle_click)   # X11/Linux (also right click sometimes, but many mice use button 2)

    root.mainloop()

def choose_audio_device():
    print("\nAvailable audio devices:")
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{idx}: {name}")
    return int(input("Enter device index for VB-Audio Cable (usually not 0): "))

def is_project_related(question):
    q = question.lower()
    return any(kw in q for kw in PROJECT_KEYWORDS)

def answer_from_rag(question):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory=RAG_DB_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = LangChainOpenAI(model_name=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0.4)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff"
    )
    answer = qa_chain({"query": question})['result']
    return answer

client = OpenAI(api_key=OPENAI_API_KEY)

def answer_from_gpt(question, background, company_info):
    prompt = f"""
You are Dr. Saeed Arasteh, an expert in biostatistics, AI/ML, and oncology drug development, interviewing for Director, Statistical Methodology and Innovation at Bristol Myers Squibb.

Here is the interview question or comment:
"{question}"

Answer directly, as you would in a senior-level interview:
- Focus only on the question. Respond naturally, in 3-4 sentences.
- If it’s technical or strategic, share specific approaches or insights from your experience.
- If the interviewer asks for an example or more detail, add 1-2 clear sentences from a relevant project.
- Avoid repeating the question. Do not include any extra introduction or background—just answer as yourself.

Begin your answer now.
    """

    # ... (OpenAI call here)

    tries = 3
    while tries > 0:
        try:
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.6,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            traceback.print_exc()
            if "Rate limit" in str(e) or "429" in str(e):
                print("Rate limited. Sleeping 10 seconds...")
                time.sleep(10)
                tries -= 1
            else:
                break
    return "Sorry, an error occurred with the AI API."

def main():
    device_index = choose_audio_device()
    recognizer = sr.Recognizer()
    print("\n[INFO] Agent is now listening for questions. Say 'quit' or 'exit' to stop.\n")

    with sr.Microphone(device_index=device_index) as source:
        while True:
            print("🎤 Ready for the next interview question...")
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=25)
                try:
                    question = recognizer.recognize_google(audio)
                    question_clean = question.lower().strip()
                    print(f"📝 Transcribed: {question_clean}")
                except Exception as e:
                    print(f"❌ Speech Recognition failed: {e}")
                    continue

                if any(x in question_clean for x in ["quit", "exit", "stop", "enough"]):
                    print("🛑 Voice command received: exiting agent.")
                    break

                if is_project_related(question_clean):
                    print("🔍 Project-related question detected. Using RAG...")
                    answer = answer_from_rag(question)
                else:
                    print("💡 General/Non-project question. Using GPT-4o...")
                    answer = answer_from_gpt(question, YOUR_BACKGROUND, COMPANY_CONTEXT)

                show_answer_gui(answer)
            except KeyboardInterrupt:
                print("\n[User Interrupt] Stopping agent.")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()
