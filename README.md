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

- Migração dos modelos

``` python manage.py makemigrations ```

``` python manage.py migrate ```

## 2 - Modificações no código

- Na views.py do diretório chat, caso queiram usar arquivos locais, usar:

``` loader = DirectoryLoader('diretório') ```

- Caso queiram continuar usando o AWS, é necessário digitar o comando no terminal:

``` aws configure ```

e colocar as keys que estão no diretório gpt/settings.py

- Atualização da APIKEY, no diretório chat/constants.py , colocar a APIKEY recebida.

- O tempo de reset de requisições por id pode ser feito diretamente no arquivo views.py, dentro da classe Assistant, na variável ```reset_time```.

- A quantidade máxima de requisições por intervalo de tempo (definido anteriormente) pode ser alterada no mesmo local, na variável ```max_count```.

## 3 - Execução do app

- Lembre de iniciar o ambiente virtual para iniciar o app.

- Para iniciar o server, use:

``` python manage.py runserver ```

## 4 - Teste com o Postman

- Recomendamos ver no vídeo para ficar mais intuitivo.

- No Endpoint ```/api/token/``` o user criado deve ser passado como parâmetro para o token JWT ser gerado, tanto o de acesso quanto o de refresh.

- No Endpoint ```/api/token/refresh/``` pode ser gerado outro token de acesso a partir do token refresh criado anteriormente.

- No Endpoint ```/api/loader/``` com o método POST deve ser passado o parâmetro "index", com um inteiro indicando a string da tupla que deve ser carregada, onde a mesmo contém o diretório AWS. Com o método GET é recebido um index para cada arquivo no armazenamento.

- No EndPoint ```/api/assistant/?idUser=ID_do_usuario_brisa``` 3 parâmetros são passados, "query" que é a pergunta ao professor, e "use_audio" que informa se é necessário o uso do text to speech. Enquanto esses dois parâmetros vão no corpo do JSON, deve ser inserido o número de usuário no URL para ser criado no banco de dados. Caso seja ativada a verificação JWT, também será necessário inserir o Token de acesso JWT no Header da requisição para ser validado.

- No EndPoint ```/api/transcribe/``` temos o Speech to text, caso queira rodar o back-end juntamente do front-end, não é necessário alterar nada no código.

OBS.: Caso queira testar apenas o back-end, igual mostrado no vídeo, é necessário alterar a seguinte linha no código:

``` audio = base64.b64decode(audio_record.split(',')[1]) ```

por

``` audio = base64.b64decode(audio_record) ```

Obs.: Caso ocorra erro na execução do whisper utilizando o Endpoint Transcribe, instale o FFMPEG no computador e adicione ele na variável de ambiente PATH, após isso, reinicie o sistema."

https://ffmpeg.org/download.html 
