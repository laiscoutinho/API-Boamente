import httpx

url = "http://127.0.0.1:8000/classifica"
payload = {
    "text": "quero morrer",
    "identificador": "teste123",
    "datetime": "2024-12-09T10:00:00"
}

try:
    response = httpx.post(url, json=payload)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Erro: {response.status_code} - {response.text}")
except httpx.RequestError as exc:
    print(f"Erro de conexão: {exc}")
