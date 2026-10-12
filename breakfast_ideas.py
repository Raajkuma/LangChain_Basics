from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

criteria = (
    "under 400 calories, at least 10 g protein or 5 g fiber, "
    "no added sugar, built mostly from whole foods"
)
prompt = ChatPromptTemplate.from_template(
    "Give me exactly 5 healthy breakfast ideas.\n"
    "'Healthy' here means: {criteria}.\n"
    "Format rules: a numbered list from 1 to 5, one line per idea, "
    "no preamble, no closing remarks, nothing else."
)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7, google_api_key=os.environ.get("GEMINI_API_KEY"))

chain = prompt | llm | StrOutputParser()
text = chain.invoke({"criteria": criteria})   # returns a plain str

print("---"* 10 + "  Normal version  " + "---"* 10)
print("Top 5 healthy breakfast ideas (normal version):\n")
print(text)
print("---"* 20)

# Streamed version
print("\n"+"---"* 10 + "  Streamed version  " + "---"* 10)
print("Top 5 healthy breakfast ideas (streamed version):\n")
for chunk in chain.stream({"criteria": criteria}):
    print(chunk, end="", flush=True)
print("\n"+"---"* 20)
