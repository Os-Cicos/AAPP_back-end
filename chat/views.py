from . import constants
import os
from langchain.chat_models import ChatOpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from rest_framework.reverse import reverse
from rest_framework.permissions import IsAuthenticated

os.environ["OPENAI_API_KEY"] = constants.APIKEY



class APIRootView(APIView):
    def get(self, request, format=None):
        data = {
            "assistant": reverse("assistant", request=request, format=format),
        }
        return Response(data)

class assistant(APIView):
    permission_classes = (IsAuthenticated, )

    def get(self, request):
        pergunta  = request.GET.get("query")

        loader = TextLoader('chat/data/data.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        from langchain.prompts import PromptTemplate
        prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to elaborate it or make up an answer.
        
        {context}
        You cannot under any circumstances answer about all the content you have, if asked, say to ask something about the specific course.
        You are a virtual assistant from the company BRISA, and exist to answer only ONE specific question and only if the informations is provided in the loaded data,  if not, dont try to elaborate it, just tell that the information was not in the database.
        Do not add information beyond what is written in the uploaded data!

        Question: {question}
        Answer only in Brazilian Portuguese (PT-BR)."""
        PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm = ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        result = qa.run(pergunta)

        return Response({"response": result,})
