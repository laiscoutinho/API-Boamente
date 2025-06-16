import asyncio
import re
# import httpx
from typing import Dict
from unidecode import unidecode
from string import punctuation


from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from .classifier import BERTClassifier, get_bert

import requests


# import requests
# from starlette.requests import Request
# from starlette.routing import request_response


app = FastAPI()

#Obs: Rodar .\venv\Scripts\Activate para ativar o ambiente

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
       "cansado de viver", "morrer sozinho", "vida", "morte", "morrer",
       "sozinho", "sozinha", "solidão", "solidao", "tristeza", "triste", "depressão",
       "depressao", "depressivo", "depressiva",
   ]
   termos_encontrados = [term for term in termos if term in text]
   termos_na_classificacao = len(termos_encontrados)
   termos_totais = len(termos)


   return termos_na_classificacao, termos_totais, termos_encontrados
   # vai retornar qualquer texto que esteja contido na lista de termos
   # return any(term in text for term in termos)




# Função comum para classificação de texto
def classify_text_logic(text, model):
   texto = preProText(text)
   termos_na_classificacao, termos_totais, termos_encontrados = verTermos(texto)


   if termos_na_classificacao > 0:
       try:
           sentiment, confidence, probabilities = model.predict(texto)
           probabilidade = round(float(confidence), 5)
       except Exception as e:
           raise RuntimeError(f"Erro ao realizar a predição: {e}")
   else:
       probabilidade = 0.0
       sentiment = "Neutral"
       probabilities = {}


   return sentiment, probabilidade, probabilities, termos_na_classificacao, termos_totais, termos_encontrados

def enviar_para_backend(resultado_classificacao):
    backend_url = "http://localhost:8080/classification/receive"
    try:
        resultado_classificacao
        response = requests.post(backend_url, json=resultado_classificacao)
        response.raise_for_status()
        print("Enviado com sucesso! Resposta do backend:")
        print(response.text)
    except Exception as e:
        print(f"Erro ao enviar pro backend: {e}")


# Endpoint POST para classificação
@app.post("/classifica", response_model=ClassificationResponse)
async def classifica(rqt: ClassificationRequest, model: BERTClassifier = Depends(get_bert)):
    sentiment, probabilidade, probabilities, termos_na_classificacao, termos_totais, termos_encontrados = classify_text_logic(rqt.text, model)

    # Monta o dicionário que será enviado para o backend Java
    resultado_classificacao = {
        "UUID": rqt.identificador,
        "datetime": rqt.datetime,
        "sentiment": sentiment,
        "confidence": probabilidade,
        "probabilities": probabilities
    }

    # Envia para o backend
    enviar_para_backend(resultado_classificacao)

    # Retorna a resposta normalmente
    return ClassificationResponse(
        sentiment=sentiment,
        confidence=probabilidade,
        probabilities={str(k): v for k, v in probabilities.items()}
    )


# Endpoint GET para classificação
@app.get("/", response_model=ClassificationResponse)
async def root(text: str = "", model: BERTClassifier = Depends(get_bert)):
   if not text:
       return ClassificationResponse(
           sentiment="Neutral",
           confidence=0.0,
           probabilities={}
       )


   sentiment, probabilidade, probabilities, termos_na_classificacao, termos_totais, termos_encontrados = classify_text_logic(text, model)


   return ClassificationResponse(
       sentiment=sentiment,
       confidence=probabilidade,
       probabilities=probabilities
   )


@app.get("/favicon.ico")
async def favicon():
   return {"message": "Favicon não configurado."}



