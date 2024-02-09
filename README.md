# Instalação/Execução do Back-End do Assistente.

## 1 - Primeiros Passos

Após o download  do código:

Obs.: Usar a versão 3.10.10 do python de preferência, mas outras versões 3.10.x ou inferior podem ser usadas!

- Criação do ambiente virtual

``` python -m venv myenv ```

- Ativação do ambiente virtual

``` myenv\Scripts\activate ```

- Instalação dos requirements

``` pip install -r requirements.txt ```

## 2 - Modificações no código

- Na views.py do diretório chat, caso queiram usar arquivos locais, usar:

``` loader = DirectoryLoader('diretório') ```

- Caso queiram continuar usando o AWS, é necessário digitar o comando no terminal:

``` aws configure ```

e colocar as keys que estão no diretório gpt/settings.py

- Atualização da APIKEY, no diretório chat/constants.py , colocar a APIKEY recebida.

## 3 - Execução do app

- Lembre de iniciar o ambiente virtual para iniciar o app.

- Para iniciar o server, use:

``` python manage.py runserver ```

## 4 - Teste com o Postman

- Recomendamos ver no vídeo para ficar mais intuitivo.

- No Endpoint ```/api/create_user/``` 2 parâmetros devem ser passados para criar um usuário no banco de dados do projeto e ser possível gerar o Token JWT, "username" e "password". Também pode ser criado um superuser no Django de sua preferência.

- No Endpoint ```/api/token/``` o user criado deve ser passado como parâmetro para o token JWT ser gerado, tanto o de acesso quanto o de refresh.

- No Endpoint ```/api/token/refresh/``` pode ser gerado outro token de acesso a partir do token refresh criado anteriormente.

- No Endpoint ```/api/loader/``` deve ser passado o parâmetro "index", com um inteiro indicando a string da tupla que deve ser carregada, onde a mesmo contém o diretório AWS.

- No EndPoint ```/api/assistant/``` 2 parâmetros são passados, "query" que é a pergunta ao professor, e "use_audio" que informa se é necessário o uso do text to speech. Também é necessário inserir o Token de acesso JWT no Header da requisição para ser validado. Após 3 requisições com o mesmo token, o uso é bloqueado. (Pode ser alterado diretamente no código views.py, dentro da Classe "Assistant".)

- No EndPoint ```/api/transcribe/``` temos o Speech to text, caso queira rodar o back-end juntamente do front-end, não é necessário alterar nada no código.

OBS.: Caso queira testar apenas o back-end, igual mostrado no vídeo, é necessário alterar a seguinte linha no código:

``` audio = base64.b64decode(audio_record.split(',')[1]) ```

por

``` audio = base64.b64decode(audio_record) ```

Obs.: Caso ocorra erro na execução do whisper utilizando o Endpoint Transcribe, instale o FFMPEG no computador e adicione ele na variável de ambiente PATH, após isso, reinicie o sistema."

https://ffmpeg.org/download.html 
