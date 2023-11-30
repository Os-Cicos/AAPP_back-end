from . import constants
import os, json, base64
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.document_loaders import S3FileLoader

os.environ["OPENAI_API_KEY"] = constants.APIKEY
loader = S3FileLoader("projeto-tic-s3", "static/data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

prompt_template = """Responda às perguntas com base apenas no contexto fornecido. Se não souber a resposta, informe que não possui a informação sem elaborar.

{contexto}
Responda apenas se a informação estiver nos documentos carregados. Caso não encontre, informe que a informação não está na base de dados. Não forneça respostas além do que está nos textos. Basta copiar e colar diretamente dos documentos!
Você é um assistente virtual da empresa BRISA, chamado 'Ciçin', com pronomes masculinos.
Se a pergunta for vaga ou não clara (por exemplo, 'Quem são eles?'), solicite esclarecimento perguntando: 'Quem são "eles"? Por favor, forneça mais detalhes específicos.'
Após responder, sempre pergunte se o usuário tem mais alguma dúvida.

Question: {questao}
Sua resposta deve ter menos de 400 caracteres.
Responda apenas em Português do Brasil (PT-BR)."""


class assistant(APIView):

    def post(self, request):
        global splits, retriever, prompt_template
        data = json.loads(request.body)
        question = data.get("query")
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["contexto", "questao"]
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        rag_chain = {"contexto": retriever, "questao": RunnablePassthrough()} | PROMPT | llm | StrOutputParser()
        result = rag_chain.invoke(question)

        # TEXTO PARA ÁUDIO

        client = OpenAI()

        # speech_file_path = Path('C:\Users\orea1\Desktop\apigpt\gpt').parent / "speech.mp3"
        response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",  # alloy / echo / fable / onyx / nova / shimmer
        input=result
        )

        response.stream_to_file("result.mp3")

        audio_file_path = 'result.mp3'

        with open(audio_file_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

        return Response({"response_text": result, "response_audio": audio_data})
