import json
import numpy as np
import os
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Скачиваем необходимые ресурсы NLTK
nltk.download('punkt')

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifierModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(IntentClassifierModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

class IntentClassifier:
    def __init__(self):
        # Используем русскоязычную модель BERT
        self.model_name = 'DeepPavlov/rubert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Расширенный словарь для лемматизации русских слов
        self.lemma_dict = {
            # Глаголы для управления светом
            "включи": "включить", 
            "зажги": "зажечь",
            "выключи": "выключить", 
            "погаси": "погасить",
            "потуши": "потушить",
            "выруби": "вырубить",
            "активируй": "активировать",
            "деактивируй": "деактивировать",
            "отключи": "отключить",
            "подсвети": "подсветить",
            "освети": "осветить",
            "убери": "убрать",
            "запусти": "запустить",
            "останови": "остановить",
            
            # Глаголы для управления температурой
            "прибавь": "прибавить",
            "убавь": "убавить", 
            "сделай": "сделать",
            "подогрей": "подогреть",
            "охлади": "охладить",
            "повысь": "повысить",
            "понизь": "понизить",
            "увеличь": "увеличить",
            "уменьши": "уменьшить",
            "подними": "поднять",
            "снизь": "снизить",
            "согрей": "согреть",
            
            # Глаголы для управления музыкой
            "поставь": "поставить",
            "запусти": "запустить",
            "включи": "включить",
            "выключи": "выключить",
            "останови": "остановить",
            "заглуши": "заглушить",
            "врубай": "врубить",
            "перестань": "перестать",
            "прекрати": "прекратить",
            "заверши": "завершить",
            "выруби": "вырубить",
            "отключи": "отключить",
            "включай": "включать",
            "играй": "играть",
            "сыграй": "сыграть",
            
            # Существительные
            "свет": "свет",
            "лампа": "лампа",
            "лампу": "лампа",
            "лампой": "лампа",
            "освещение": "освещение",
            "светильник": "светильник",
            "торшер": "торшер",
            "люстра": "люстра",
            "люстру": "люстра",
            "температура": "температура",
            "температуру": "температура",
            "музыка": "музыка",
            "музыку": "музыка",
            "песня": "песня",
            "песню": "песня",
            "плеер": "плеер",
            "звук": "звук",
            "мелодия": "мелодия",
            "мелодию": "мелодия",
            
            # Прилагательные и наречия
            "светлее": "светлый",
            "темнее": "темный",
            "ярче": "яркий",
            "теплее": "теплый",
            "холоднее": "холодный",
            "жарче": "жаркий",
            "прохладнее": "прохладный",
            "громче": "громкий",
            "тише": "тихий"
        }
        
    def preprocess_text(self, text):
        """Улучшенная предобработка текста с расширенной лемматизацией"""
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаление пунктуации и замена на пробелы
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Токенизация
        tokens = word_tokenize(text)
        
        # Лемматизация с расширенным словарем
        lemmatized_tokens = []
        for token in tokens:
            lemmatized_tokens.append(self.lemma_dict.get(token, token))
        
        return " ".join(lemmatized_tokens)
        
    def load_data(self, file_path):
        """Загрузка данных из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for intent in data['intents']:
            intent_name = intent['intent']
            for example in intent['examples']:
                texts.append(self.preprocess_text(example))
                labels.append(intent_name)
        
        # Кодируем метки в числовой формат
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        return texts, encoded_labels
    
    def train(self, file_path, batch_size=16, epochs=10):
        """Обучение модели с увеличенным количеством эпох"""
        print(f"Устройство для обучения: {self.device}")
        texts, encoded_labels = self.load_data(file_path)
        
        print(f"Данные для обучения: {len(texts)} примеров, {len(self.label_encoder.classes_)} интентов")
        
        # Создаем базовую модель BERT
        base_model = AutoModel.from_pretrained(self.model_name)
        
        # Создаем классификатор на основе BERT
        self.model = IntentClassifierModel(base_model, len(self.label_encoder.classes_))
        self.model.to(self.device)
        
        # Создаем датасет и загрузчик данных
        dataset = IntentDataset(texts, encoded_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Инициализируем оптимизатор и функцию потерь
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        # Цикл обучения
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_accuracy = correct / total
            print(f"Эпоха {epoch+1}/{epochs} - Потери: {total_loss/len(dataloader):.4f}, Точность: {epoch_accuracy:.4f}")
        
        # Оцениваем модель на обучающих данных
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Итоговая точность на тренировочных данных: {accuracy:.4f}")
        print("\nОтчет по классификации:")
        
        # Преобразуем числовые метки обратно в текстовые для отчета
        label_names = self.label_encoder.inverse_transform(np.unique(all_labels))
        print(classification_report(
            self.label_encoder.inverse_transform(all_labels),
            self.label_encoder.inverse_transform(all_predictions),
            target_names=label_names
        ))
        
    def predict(self, text):
        """Предсказание интента для нового текста"""
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена")
        
        processed_text = self.preprocess_text(text)
        
        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, dim=1)
            intent_idx = predicted_class.item()
            confidence_val = confidence.item()
        
        intent = self.label_encoder.inverse_transform([intent_idx])[0]
        
        return {
            "intent": intent,
            "confidence": confidence_val
        }
    
    def save_model(self, file_path):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Невозможно сохранить необученную модель")
        
        # Создаем папку, если не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Сохраняем состояние модели и энкодер меток
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'model_name': self.model_name
        }, file_path)
        
        print(f"Модель сохранена в: {file_path}")
    
    def load_model(self, file_path):
        """Загрузка сохраненной модели"""
        if not os.path.exists(file_path):
            raise ValueError(f"Файл модели не найден: {file_path}")
        
        # Загружаем состояние модели
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # Загружаем энкодер меток
        self.label_encoder = checkpoint['label_encoder']
        
        # Создаем и загружаем модель
        base_model = AutoModel.from_pretrained(checkpoint.get('model_name', self.model_name))
        self.model = IntentClassifierModel(base_model, len(self.label_encoder.classes_))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        print(f"Модель загружена из: {file_path}")

if __name__ == "__main__":
    # Создаем и обучаем классификатор
    classifier = IntentClassifier()
    data_path = os.path.join('data', 'intents.json')
    
    # Обучаем модель (может занять некоторое время)
    print("Начинаем обучение модели...")
    classifier.train(data_path, epochs=10)
    
    # Сохраняем модель
    model_path = os.path.join('data', 'intent_model.pt')
    classifier.save_model(model_path)
    
    # Тестируем на новых примерах
    test_phrases = [
        "включи свет в комнате",
        "мне очень холодно",
        "хочу послушать что-нибудь",
        "выключи пожалуйста лампу",
        "сделай жарче в комнате",
        "зажги лампу",
        "выруби свет",
        "мне жарко",
        "задвинь шторы",
        "выключи лампу"
    ]
    
    print("\nТестирование модели на новых фразах:")
    for phrase in test_phrases:
        result = classifier.predict(phrase)
        print(f"Фраза: '{phrase}'")
        print(f"Распознанный интент: {result['intent']} (уверенность: {result['confidence']:.2f})")
        print("-" * 50) 