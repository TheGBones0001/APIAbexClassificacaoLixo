1. Configurar um Ambiente Virtual pelo terminal

Criar um ambiente virtual: Fazer só uma vez
python -m venv venv

Ativar o ambiente virtual:

Windows:
venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

Desativar:
deactivate

2. Instale os pacotes necessários usando o comando abaixo:
pip install numpy tensorflow matplotlib flask