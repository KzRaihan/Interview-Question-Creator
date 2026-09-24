# ===============================
#  Define the All Prompt
# ===============================

# Define a Prompt to generate Question
from langchain_core.prompts import PromptTemplate

# Prompt template (Key Note: Specific your tone what are exactly you want to be Create)
prompt_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

---------
{text}
---------

Create questions that will prepared the coders or programmers for their test.
Make sure not to lose any important information.

QUESTIONS:

"""


# Prompt template 2 (here, inputs: existing question, chunk of documents)
refine_template = ("""
You are an expert at creating practice questions based on coding material and documentation.

Generate interview questions based strictly on the provided context.

Question requirements:
- Questions must be contextual and derived from the provided text.
- Prefer scenario-based, application-oriented, and reasoning-based questions.
- Avoid simple definition-recall questions.
- Avoid questions that can be answered without understanding the provided context.
- Do not invent information outside the provided context.
- Do not include explanations, headings, answers, or refinement notes.

Generate:
- 5 conceptual/contextual questions
- 3 scenario-based questions
- 2 reasoning/application questions

Return ONLY the questions as a numbered list.

Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context on Creative Questions or Contextual Questions, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:


"""
)



# Define the prompt that handles the context documents
system_prompt = (
""" 
You are a technical assistant. Give the shortest correct answer that fully addresses the question. Use bullet points only when listing multiple distinct items; otherwise use plain sentences. No introductory phrases like "Great question!" — start directly with the answer.

Answer the user's question using only the provided context. Respond in 1-3 sentences. If the context doesn't contain the answer, say so in one sentence — do not guess or use outside knowledge. Do not repeat the question back.
Context:\n{context}    

"""
)

