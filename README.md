# Instalação/Execução do Back-End do Assistente.

## 1 - Primeiros Passos

Após o download  do código:

Obs.: Usar a versão 3.10.10 do python!

- Criação do ambiente virtual

``` python -m venv myenv ```

- Ativação do ambiente virtual

``` myenv\Scripts\activate ```

- Instalação dos requirements

``` pip install -r requirements.txt ```

## 2 - Modificações no código

- Na views.py do diretório chat, caso queiram usar arquivos locais, usar:

``` loader = DirectoryLoader('diretório') ```

- Caso queiram continuar usando o AWS, é necessário a inserção das keys que estão no diretório gpt/settings.py, conforme pedido no terminal.

- Atualização da APIKEY, no diretório chat/constants.py , colocar a APIKEY recebida.

## 3 - Execução do app

- Lembre de iniciar o ambiente virtual para iniciar o app.

- Para iniciar o server, use:

``` python manage.py runserver ```

## 4 - Teste com o Postman

- Recomendamos ver no vídeo para ficar mais intuitivo.

- No EndPoint "Assistant" , 2 parâmetros são passados, "query" que é a pergunta ao professor, e "use_audio" que informa se é o envio do text to speech.

- No EndPoint "Transcribe" temos o Speech to text, caso queira rodar o back-end juntamente do front-end, não é necessário alterar nada no código.

OBS.: Caso queira testar apenas o back-end, igual mostrado no vídeo, é necessário alterar a seguinte linha no código:

``` audio = base64.b64decode(audio_record.split(',')[1]) ```

por

``` audio = base64.b64decode(audio_record) ```

