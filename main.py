import sys
from config import config

sys.path.insert(1, config.get_langchain_path())

from langchain.agents import initialize_agent, load_tools
from langchain.chains import LLMChain, SimpleSequentialChain, qa_with_sources
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

CHAT_MODEL = "gpt-3.5-turbo"

def main():
    openai_api_key = config.get_openai_api_key()
    google_api_key = config.get_google_api_key()
    google_cse_id = config.get_google_custom_search_engine_id()
    
    # chat_sample(openai_api_key, "ドラムを始めたいのですが騒音が心配です、どのように対策すればよいでしょうか。")
    # chain_prompt_sample(openai_api_key, "キラキラ未来")
    # data_and_source_sample(openai_api_key, "HTML5 ゲーム開発に用いられる主なゲームエンジンはなに？日本語で回答してください。")
    agent_sample(openai_api_key, google_api_key, google_cse_id, "ジョジョの奇妙な冒険の第6部の空条承太郎の年齢の最大公約数は？日本語で回答してください。")

def chat_sample(openai_api_key: str, prompt: str):
    llm = ChatOpenAI(
        temperature=0.9,
        openai_api_key=openai_api_key,
        model_name=CHAT_MODEL
    )

    response = llm(prompt)
    print(response)


def chain_prompt_sample(openai_api_key: str, title: str):
    lyric_prompt = """あなたは作詞家です。楽曲のタイトルが与えられた場合、そのタイトルの歌詞を書くのがあなたの仕事です。

タイトル:{title}
歌詞:"""

    review_prompt = """あなたは音楽評論家です。 楽曲の歌詞が与えられた場合、その歌詞のレビューを書くのがあなたの仕事です。

歌詞:
{lyric}
レビュー:"""

    lyric_template = PromptTemplate(
        input_variables=["title"],
        template=lyric_prompt
    )
    review_template = PromptTemplate(
        input_variables=["lyric"],
        template=review_prompt
    )

    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key,
                 max_tokens=1024)
    lyric_chain = LLMChain(llm=llm, prompt=lyric_template)
    review_chain = LLMChain(llm=llm, prompt=review_template)
    overall_chain = SimpleSequentialChain(
        chains=[lyric_chain, review_chain],
        verbose=True
    )

    response = overall_chain.run(title)
    print(response)


def data_and_source_sample(api_key: str, prompt: str):
    sources = [
        "data/1_introduction_1.txt",
        "data/2_introduction_2.txt",
        "data/3_engines_1.txt",
        "data/4_engines_2.txt",
        "data/5_specification_1.txt",
        "data/6_specification_2.txt",
        "data/7_specification_3.txt",
        "data/8_specification_4.txt",
    ]
    texts = []

    for i in range(len(sources)):
        with open(sources[i]) as f:
            texts.append(f.read())

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    docsearch = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=[{"source": sources[i]} for i in range(len(texts))]
    )

    docs = docsearch.similarity_search(prompt)
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    chain = qa_with_sources.load_qa_with_sources_chain(llm, chain_type="map_reduce")
    response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
    
    print(response)


def agent_sample(openai_api_key: str, google_api_key: str, google_cse_id: str, prompt: str):
    llm = OpenAI(
        temperature=0.2,
        openai_api_key=openai_api_key
    )
    tools = load_tools(["google-search", "llm-math"], llm=llm,
                       google_api_key=google_api_key, google_cse_id=google_cse_id)
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent="zero-shot-react-description",
        verbose=True,
        return_intermediate_steps=True
    )

    # 質問応答の実行
    response = agent({"input": prompt})


if __name__ == "__main__":
    main()
