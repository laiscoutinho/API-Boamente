import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F

# le o arquivo para carregar config peronalizadas do modelo
with open("config.json") as json_file:
	config = json.load(json_file)

class BERTClassifier():

	# carregando o tokenizer e o modelo pre treinado
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained(
			"laisdev/SentimentAnalysisBoamente")
		self.model = AutoModelForSequenceClassification.from_pretrained(
			"laisdev/SentimentAnalysisBoamente")

	# function para prever a classe de um texto
	# recebe a string e retorna a classe probabilidade e confiança 
	def predict(self, text):
		# converte textos para tokens
		tokens = self.tokenizer(text, max_length = config["MAX_TOKENS_LEN"], 
			padding = True, return_tensors = "pt")

		# calcula as probabilidades de cada classe usando softmax
		with torch.no_grad():
			probabilities = F.softmax(self.model(**tokens)['logits'], dim=1)

		# determinação da clsse prevista
		confidence, predicted_class = torch.max(probabilities, dim=1)
		predicted_class = predicted_class.cpu().item()
		probabilities = probabilities.flatten().cpu().numpy().tolist()

		# retorna nome da classe, confiança na previsao, dicionario com as probabilidades de todas as classes
		return (
			config["CLASS_NAMES"][predicted_class],
			confidence,
			dict(zip(config["CLASS_NAMES"], probabilities)),
		)

# cria instancia da classe
bert = BERTClassifier()

# retorna instancia criada
def get_bert():
	return bert