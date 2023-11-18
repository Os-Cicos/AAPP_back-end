from . import constants
import os, json
from langchain.chat_models import ChatOpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from rest_framework.reverse import reverse
import pyttsx3

os.environ["OPENAI_API_KEY"] = constants.APIKEY



class APIRootView(APIView):
    def get(self, request, format=None):
        data = {
            "assistant": reverse("assistant", request=request, format=format),
        }
        return Response(data)

class assistant(APIView):

    def post(self, request):
        data = json.loads(request.body)
        pergunta = data.get("query")

        loader = DirectoryLoader('chat/data/')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        from langchain.prompts import PromptTemplate
        prompt_template = """Answer the question based only on the following context. If you don't know the answer, just say that you don't know, don't try to elaborate.
        
        {context}
        Answer questions only if the information is inside the documents, if you don't find, dont try to elaborate it, just tell that the information was not in the database. 
        Don't answer anything other than what you have in the database. Just copy and paste what is in the texts!
        You are a virtual assistant from the company BRISA, Called "Ciçin", your pronoun is masculine.
        Após responder, sempre pergunte se o usuário tem mais alguma dúvida.
        Seja direto e responda apenas apartir dos dados carregados, você não pode responder nada além do que há nos dados.

        Question: {question}
        Your answer has to be an output less than 600 characters lenght.
        Answer only in Brazilian Portuguese (PT-BR)."""
        PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm = ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        result = qa.run(pergunta)
        
        # TEXTO PARA ÁUDIO

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('rate', 180)  
        for indice, vozes in enumerate(voices): 
            print(indice, vozes.name)
        voz = 3 # Preencher de acordo com o print dos indices das vozes
        engine.setProperty('voice', voices[voz].id)
        engine.save_to_file(result, 'result.mp3')
        engine.runAndWait()
        engine.stop()

        return Response({"response": result})
        
