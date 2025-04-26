import os
import json
from intent_classifier import IntentClassifier

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    print("Модуль speech_recognition не установлен. Будет использоваться текстовый ввод.")
    print("Для установки выполните: pip install SpeechRecognition")
    SPEECH_AVAILABLE = False

class SmartHomeAssistant:
    def __init__(self, model_path=None):
        self.classifier = IntentClassifier()
        
        # Список поддерживаемых интентов, по которым мы будем фильтровать результаты
        self.supported_intents = [
            "light_on", "light_off", 
            "temperature_up", "temperature_down", 
            "music_on", "music_off",
            "door_lock", "door_unlock",
            "sensor_check", "sensor_reset",
            "unknown"
        ]
        
        # Загружаем модель, если она существует
        if model_path and os.path.exists(model_path):
            self.classifier.load_model(model_path)
        else:
            # Иначе обучаем модель
            data_path = os.path.join('data', 'intents.json')
            self.classifier.train(data_path, epochs=10)
            
            # И сохраняем её
            if model_path:
                self.classifier.save_model(model_path)
                
        # Загружаем действия для каждого интента
        self.actions = {
            "light_on": self.turn_light_on,
            "light_off": self.turn_light_off,
            "temperature_up": self.increase_temperature,
            "temperature_down": self.decrease_temperature,
            "music_on": self.play_music,
            "music_off": self.stop_music,
            "door_lock": self.lock_door,
            "door_unlock": self.unlock_door,
            "sensor_check": self.check_sensors,
            "sensor_reset": self.reset_sensors,
            "unknown": self.handle_unknown
        }
    
    # Симуляции действий умного дома
    def turn_light_on(self):
        print("🔆 Включаю свет!")
        
    def turn_light_off(self):
        print("🌑 Выключаю свет!")
        
    def increase_temperature(self):
        print("🔥 Увеличиваю температуру!")
        
    def decrease_temperature(self):
        print("❄️ Уменьшаю температуру!")
        
    def play_music(self):
        print("🎵 Включаю музыку!")
        
    def stop_music(self):
        print("🔇 Выключаю музыку!")
        
    def lock_door(self):
        print("🔒 Закрываю дверь на замок!")
        
    def unlock_door(self):
        print("🔓 Открываю дверь!")
        
    def check_sensors(self):
        print("📊 Проверяю датчики...")
        print("  ✅ Датчик движения: никого нет")
        print("  ✅ Датчик температуры: 22°C")
        print("  ✅ Датчик влажности: 45%")
        print("  ✅ Датчик дыма: норма")
        print("  ✅ Датчик протечки: сухо")
        
    def reset_sensors(self):
        print("🔄 Перезагружаю все датчики...")
        print("  ✅ Датчики успешно перезагружены")
        
    def handle_unknown(self):
        print("🤔 Извините, я не умею выполнять эту команду.")
    
    def recognize_speech(self):
        """Распознавание речи с помощью микрофона"""
        if not SPEECH_AVAILABLE:
            return None
            
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Слушаю...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            print(f"Распознано: {text}")
            return text
        except sr.UnknownValueError:
            print("Не удалось распознать речь")
            return None
        except sr.RequestError:
            print("Ошибка сервиса распознавания речи")
            return None
    
    def process_command(self, text):
        """Обработка команды пользователя"""
        if not text:
            return
            
        # Получаем предсказание интента
        prediction = self.classifier.predict(text)
        intent = prediction["intent"]
        confidence = prediction["confidence"]
        
        print(f"Распознанный интент: {intent} (уверенность: {confidence:.2f})")
        
        # Проверяем, поддерживается ли интент
        if intent not in self.supported_intents:
            print(f"⚠️ Интент '{intent}' не поддерживается")
            intent = "unknown"
            confidence = 0.0  # Сбрасываем уверенность для неподдерживаемых интентов
        
        # Теперь модель на основе BERT должна давать высокую уверенность > 0.5
        if confidence > 0.5 and intent in self.actions:
            self.actions[intent]()
        else:
            # Если уверенность низкая, считаем, что это неизвестная команда
            print("Извините, я не уверен, что вы хотите сделать.")
            if "unknown" in self.actions:
                self.actions["unknown"]()
    
    def run_text_mode(self):
        """Запуск в текстовом режиме"""
        print("=== Ассистент умного дома (текстовый режим) ===")
        print("Введите команду или 'выход' для завершения")
        print("Поддерживаемые команды:")
        print("  - Управление светом (включить/выключить свет)")
        print("  - Управление температурой (теплее/холоднее)")
        print("  - Управление музыкой (включить/выключить)")
        print("  - Управление дверями (открыть/закрыть)")
        print("  - Проверка датчиков (проверить/сбросить)")
        
        while True:
            text = input("\nВаша команда: ")
            if text.lower() in ["выход", "exit", "quit"]:
                break
                
            self.process_command(text)
    
    def run_voice_mode(self):
        """Запуск в голосовом режиме"""
        if not SPEECH_AVAILABLE:
            print("Голосовой режим недоступен. Переключаюсь на текстовый режим.")
            self.run_text_mode()
            return
            
        print("=== Голосовой ассистент умного дома ===")
        print("Скажите команду или 'выход' для завершения")
        
        while True:
            # Распознаем речь
            text = self.recognize_speech()
            
            if not text:
                continue
                
            if "выход" in text.lower():
                break
                
            self.process_command(text)
            
if __name__ == "__main__":
    model_path = os.path.join('data', 'intent_model.pt')  # Новое расширение файла для PyTorch
    assistant = SmartHomeAssistant(model_path)
    
    # Проверяем доступность голосового режима
    if SPEECH_AVAILABLE:
        print("Доступен голосовой режим!")
        mode = input("Выберите режим (1 - текст, 2 - голос): ")
        if mode == "2":
            assistant.run_voice_mode()
        else:
            assistant.run_text_mode()
    else:
        assistant.run_text_mode() 