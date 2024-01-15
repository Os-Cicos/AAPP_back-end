from . import constants
import os, json, base64
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import DirectoryLoader, S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import whisper, torch

os.environ["OPENAI_API_KEY"] = constants.APIKEY

class Initializer:
    def __init__(self):
        # self.loader = DirectoryLoader('chat/data/')
        self.loader = S3FileLoader("projeto-tic-s3", "static/final.pdf")
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.documents)
        # self.embeddings = OpenAIEmbeddings()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

        self.prompt_template = """Responda às perguntas com base apenas no contexto fornecido. Se não souber a resposta, diga que não possui a informação, sem elaborar.

        {contexto}
        Responda apenas se houver informação da pergunta nos documentos carregados. Caso não encontre, diga que a informação não está na base de dados. Não forneça respostas além do que está nos textos.
        Você é um assistente virtual da empresa BRISA.
        Se a pergunta for vaga ou não clara solicite esclarecimento. Pedindo por mais detalhes específicos.
        Após responder, sempre pergunte se o usuário tem mais alguma dúvida.

        Question: {questao}
        Sua resposta deve ter menos de 800 caracteres.
        Responda apenas em Português do Brasil (PT-BR)."""
        self.PROMPT = PromptTemplate(
                    template=self.prompt_template, input_variables=["contexto", "questao"]
                )
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature= 0.0)
        self.rag_chain = {"contexto": self.retriever, "questao": RunnablePassthrough()} | self.PROMPT | self.llm | StrOutputParser()

initializer = Initializer()

def text_to_audio(result):
    client = OpenAI()

    response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",  # alloy / echo / fable / onyx / nova / shimmer
    input=result
    )

    response.stream_to_file("result.mp3")

    audio_file_path = 'result.mp3'

    with open(audio_file_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

    return audio_data



class assistant(APIView):

    def post(self, request):
        data = json.loads(request.body)
        question = data.get("query")
        use_audio = data.get("use_audio") 
        result = initializer.rag_chain.invoke(question)

        if use_audio:
            audio_data = text_to_audio(result)
            return Response({"response_text": result, "response_audio": audio_data})
        else:
            return Response({"response_text": result})
        
class transcribe(APIView):
    
    def post(self, request):
        data = json.loads(request.body)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print()
        print('Using device:', device)
        print()

        
        def record():
            audio_record = data.get("audio_record")
            audio = base64.b64decode(audio_record.split(',')[1])
            file_name = 'request_audio.wav'
            with open(file_name, 'wb') as f:
                f.write(audio)
            return file_name
        record_file = record()
        model = whisper.load_model("small")
        result_whisper = model.transcribe(record_file, fp16=False, language= "pt")

        # Informações adicionais
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        transcription = result_whisper["text"]
        return Response({"response_whisper": transcription})
