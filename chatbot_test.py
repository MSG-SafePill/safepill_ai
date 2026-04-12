import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv # 추가

# .env 파일에 저장된 변수들을 불러옵니다.
load_dotenv()

# 이제 os.environ.get을 통해 안전하게 키를 가져옵니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 이후 코드(TextLoader 등)는 그대로 두시면 됩니다!

# 1. 외부 파일에서 데이터 읽어오기 (2번 요구사항 해결)
# 프로젝트 폴더에 'medical_knowledge.txt' 파일을 만들고 내용을 적어두세요!
file_path = "medical_knowledge.txt"

# 만약 파일이 없으면 에러가 나니까, 예시 파일을 자동으로 하나 생성해둡니다.
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("여기에 실제 의학 지식이나 약물 정보를 길게 적어두면 AI가 읽고 답변합니다.")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) # 좀 더 길게 잘라도 됩니다.
texts = text_splitter.split_documents(documents)

# 2. Vector DB 저장
print(f"[{file_path}]에서 지식을 읽어와 DB를 구축하는 중...")
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

# 3. 챗봇 엔진 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_template("""
너는 전문 약사야. 오직 아래의 정보만을 바탕으로 대답해줘.

{context}

질문: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vectordb.as_retriever(), document_chain)

# 4. 사용자 질문 직접 입력받기 (1번 요구사항 해결)
print("\n✅ AI 약사가 준비되었습니다! (종료하려면 '나가기' 입력)")

while True:
    user_input = input("\n나: ") # 여기서 직접 질문을 입력합니다.

    if user_input == "나가기":
        print("상담을 종료합니다. 건강하세요!")
        break

    print("AI 약사 답변 중...")
    result = qa_chain.invoke({"input": user_input})
    print(f"💡 AI 약사: {result['answer']}")