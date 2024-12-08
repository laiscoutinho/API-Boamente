import re
import httpx
from typing import Dict
from unidecode import unidecode
from string import punctuation

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from .classifier import BERTClassifier, get_bert

# import requests
# from starlette.requests import Request
# from starlette.routing import request_response

app = FastAPI()

class ClassificationRequest(BaseModel):
    text: str
    identificador: str
    datetime: str

class ClassificationResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

def preProText(text):
    text = text.lower()
    text = re.sub('@[^\s]+', '', text)
    text = unidecode(text)
    text = re.sub('<[^<]+?>', '', text)
    text = ''.join(c for c in text if not c.isdigit())
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join(c for c in text if c not in punctuation)
    return text

def verTermos(text):
    termos = [
        "suicida", "suicidio", "me matar", "meu bilhete suicida",
        "minha carta suicida", "acabar com a minha vida", "nunca acordar",
        "não consigo continuar", "não vale a pena viver", "pronto para pular",
        "dormir para sempre", "quero morrer", "estar morto",
        "melhor sem mim", "vou me matar", "plano de suicídio", 
        "cansado de viver", "morrer sozinho"
    ]
    return any(term in text for term in termos)

# POST -> envia dados
@app.post("/classifica", response_model=ClassificationResponse)
async def classifica(rqt: ClassificationRequest, model: BERTClassifier = Depends(get_bert)):
    texto = preProText(rqt.text)
    identificador = rqt.identificador
    datetime = rqt.datetime

	# POST servidor
    url = 'https://boamente.minhadespesa.com.br/api/predicoes/store'
    token = 'wocUKkW9GNLxetcJLfirFdPsTfiBkv4eH4pfG7k2Lu8'

    if verTermos(texto):
        try:
            sentiment, confidence, probabilities = model.predict(texto)
            probabilidade = round(float(confidence), 5)
            possibilidade = int(sentiment)
        except Exception as e:
            raise RuntimeError(f"Erro ao realizar a predição: {e}")

    else:
        probabilidade = 0.0
        possibilidade = 0
        sentiment, probabilities = "Neutral", {}

    payload = {
        'token': token,
        'identificador': identificador,
        'probabilidade': probabilidade, 
        'possibilidade': possibilidade,
        'data_criacao': datetime
    }

    try:
        async with httpx.AsyncClient() as client:
            resposta = await client.post(url, data=payload)
            resposta.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Erro ao enviar os dados: {e}")

    return ClassificationResponse(
        sentiment=sentiment,
        confidence=probabilidade,
        probabilities=probabilities
    )
