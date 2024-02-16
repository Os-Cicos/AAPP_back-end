import glob
from . import constants
import os, json, base64, whisper, torch, datetime
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from django.utils import timezone
from django.shortcuts import get_object_or_404
from .models import User
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from langchain.document_loaders import S3FileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["OPENAI_API_KEY"] = constants.APIKEY

class Loader(APIView):

    rag_chain = None
    ids = None
    vectorstore = None

    folder_path = "chat/data/"
    file_path = glob.glob(os.path.join(folder_path, "*"))
    directories = [(i, path) for i, path in enumerate(file_path)]

    # Código para diretório na nuvem

    # directories = [
    #    ("projeto-tic-s3", "static/Python.pdf"),
    #    ("projeto-tic-s3", "static/Lógica.pdf"),
    # ]

    # def get(self, request):
    #    files = [{"index": i, "name": os.path.splitext(os.path.basename(path))[0]} for i, (_, path) in enumerate(self.directories)]
    #    return Response(files, status=status.HTTP_200_OK)
    
    def get(self, request):
        files = [{"index": i, "name": os.path.splitext(os.path.basename(path))[0]} for i, path in self.directories]
        return Response(files, status=status.HTTP_200_OK)
    

    def post(self, request):
        data = json.loads(request.body)
        index = data.get("index") 

        if index < 0 or index >= len(self.directories):
            return Response({"error": "Índice inválido"}, status.HTTP_404_NOT_FOUND)
        
        _, file_path = self.directories[index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        directory_path = os.path.join("chat/data", file_name + ".pdf")
        loader = PyPDFLoader(directory_path)   
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        if Loader.ids is not None:
            for id in Loader.ids:
                Loader.vectorstore.delete(ids=[id])
        Loader.ids = [str(i) for i in range(1, len(splits) + 1)]
        Loader.vectorstore = Chroma.from_documents(splits, embedding=embeddings, ids=Loader.ids)
        retriever = Loader.vectorstore.as_retriever()
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You are a virtual assistant from the company BRISA, maintain a humanized conversation, making the necessary greetings.
        Always provide as much information and characters as possible.
        Please answer only in Brazilian Portuguese (PT-BR).
        {contexto}
        Question: {questao}
        
        Helpful Answer:"""

        PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["contexto", "questao"]
                )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature= 0.3, max_tokens=3000)
        Loader.rag_chain = {"contexto": retriever, "questao": RunnablePassthrough()} | PROMPT | llm | StrOutputParser()

        return Response({"message": "Dados carregados com sucesso!"})
    
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



class Assistant(APIView):

    # permission_classes = (IsAuthenticated)

    def post(self, request, idUser= None):
        
        max_count = 5
        reset_time = datetime.timedelta(minutes=1)

        idUser = request.GET.get('idUser')
        if idUser is not None: 
            user, created = User.objects.get_or_create(idUser=idUser)
            now = timezone.now()
            if created:
                user.count = 0
                user.start_time = timezone.now()
            if user.count >= max_count and now - user.start_time < reset_time:
                return Response({"error": "Limite de requisições atingido"}, status.HTTP_429_TOO_MANY_REQUESTS)
            elif now - user.start_time > reset_time:
                user.count = 0
            
            user.count += 1
            user.start_time = now
            user.save()

            data = json.loads(request.body)
            question = data.get("query")
            use_audio = data.get("use_audio") 
            
            if Loader.rag_chain is None:
                return Response({"error": "Nenhum dado foi carregado ainda.", "status": f"Requisição {user.count} de {max_count}"}, status.HTTP_400_BAD_REQUEST)
            result = Loader.rag_chain.invoke(question)

            if use_audio:
                audio_data = text_to_audio(result)
                return Response({"response_text": result, "status": f"Requisição {user.count} de {max_count}", "response_audio": audio_data})
            else:
                return Response({"response_text": result, "status": f"Requisição {user.count} de {max_count}"})
        else:
            return Response({"error": "idUser é necessário"}, status.HTTP_400_BAD_REQUEST)
        
class Transcribe(APIView):
    
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
