import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv

def main():
	load_dotenv()

	# Loading documents
	pdf_loader = PyPDFLoader('./docs/test.pdf')
	documents = pdf_loader.load()

	# Split data into chunks
	text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
	documents = text_splitter.split_documents(documents)
	model = INSTRUCTOR('hkunlp/instructor-xl')

	vectordb = Chroma.from_documents(
		documents,
		embedding=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl"),
		persist_directory='./data'
	)
	vectordb.persist()

	#question = "how to bake a cake? "
	#template = """Question: {question}
	#Answer: Let's think step by step."""
	#prompt = PromptTemplate(template=template, input_variables=["question"])

	repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

	llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.1}, huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_KEY'))
	#llm_chain = LLMChain(prompt=prompt, llm=llm)

	qa_chain = RetrievalQA.from_chain_type(
		llm = llm,
		retriever = vectordb.as_retriever(search_kwargs={'k': 7}),
		return_source_documents = True
	)

	result = qa_chain({'query': 'Who is the author?'})
	print(result['result'])

	#print(llm_chain.invoke(question))
	#chain = load_qa_chain(llm=llm)
	#query = "who's the author of this book?"
	#response = chain.run(input_documents=documents, question=query)
	#print(response)

if __name__ == "__main__":
	main()
