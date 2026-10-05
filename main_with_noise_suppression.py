import asyncio
import collections
import threading
import time
import os
import sys
import wave
import json
import re
import audioop
from datetime import datetime
import numpy as np
import pyaudio
import webrtcvad
import torch
import requests
from nicegui import ui, app, run

try:
    from webrtc_noise_gain import AudioProcessor
    WEBRTC_NS_AVAILABLE = True
    print("✅ webrtc-noise-gain доступен")
except ImportError:
    WEBRTC_NS_AVAILABLE = False
    print("⚠️ webrtc-noise-gain не установлен")
    print("   Установка: pip install webrtc-noise-gain --break-system-packages")

ROOT = os.path.dirname(__file__)
T_ONE_DIR = os.path.join(ROOT, "T-one")
TONE_MODELS_DIR = os.path.join(ROOT, "models", "tone")

if os.path.exists(T_ONE_DIR):
    sys.path.insert(0, T_ONE_DIR)
    try:
        from tone.pipeline import StreamingCTCPipeline
        from tone.demo import read_audio
        STT_AVAILABLE = True
        print("✅ T-one пакет подключён")
    except ImportError as e:
        STT_AVAILABLE = False
        print(f"⚠️ Ошибка импорта T-one: {e}")
else:
    STT_AVAILABLE = False
    print(f"⚠️ Папка T-one не найдена: {T_ONE_DIR}")

SAMPLE_RATE_STT = 16000
CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_AUDIO_FILE = "user_input.wav"
OUTPUT_AUDIO_FILE = "assistant_response.wav"

SILERO_FILE = os.path.join(ROOT, "models", "v5_ru(1).pt")
SILERO_SPEAKER = "baya"
SILERO_SAMPLE_RATE = 48000

API_URL = ""
API_KEY = ""
MODEL = "qwen3-14b-awq"
SITE_URL = "https://nicegui.io"
SITE_NAME = "VoiceTrainer"

SYSTEM_PROMPT_ASSESSMENT = """
Ты — строгий методист по оценке качества звонков.
Твоя задача — проанализировать диалог и выставить оценки от 1 до 10.
ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON. Не пиши вступлений или выводов текстом.

Критерии оценки:
Оцени весь звонок в соответсвии с этими правилам:
1. Приветствие и установление контакта
Приветствие: Искреннее, четкое приветствие с представлением себя и компании
Проверка удобства времени: Обязательный вопрос Удобно ли говорить? Если неудобно — договаривается о времени для обратной связи.
Идентификация собеседника: Уточняет, с кем разговаривает, узнаёт ФИО.
Проверка полномочий: Выясняет, может ли собеседник принимать решения.
Перенаправление: При необходимости перезвонить другому лицу — выясняет, кому именно.
Цель звонка: Четкое обозначение цели звонка

2.Активное слушание и эмпатия
Вербальные сигналы: Использование да, конечно, понимаю
Отражение эмоций: Демонстрация понимания чувств клиента
Перефразирование: Повторение ключевых слов клиента (эхо)
Уточняющие вопросы: Вопросы для проверки понимания
Благодарность/комплименты: Выражение признательности

3. Выявление потребностей
Детализация потребности: При упоминании о закупках уточняет детали интереса.
Уточнение интереса: Если интерес к продукту конкретный — выявляет детали по интересующему товару (объём, условия, т.д.).
Количество уточняющих вопросов: Задает дополнительные вопросы для полного понимания потребности.
Запрос контактов ответственного лица: Просит контакты нужного специалиста, если это требуется.
Избежание закрывающих вопросов: Не задаёт вопросов, которые провоцируют клиента легко отказаться или завершить разговор.
Задает открытые вопросы: Во время разговора задал не менее 5 открытых вопросов.
Корректность уточняющих вопросов: Уточняющие вопросы корректны и вежливы.
Соотношение открытых/закрытых вопросов : Преобладание открытых вопросов
Глубина исследования : Многоуровневые вопросы, воронка вопросов
Направляющие вопросы : Использование вопросов для управления диалогом
Переформулирование : Способность конкретизировать запросы клиента
Перехват инициативы : Ответы вопросом на вопрос, управление беседой

4. Презентация
Презентация направлений: Лаконично рассказывает минимум о 5 направлениях деятельности организации.
Развёрнутая подача КП: Говорит о коммерческом предложении подробно и по делу.
Разработка конкретных предложений: Предлагает персонализированные варианты, соответствующие выявленным потребностям клиента.
Целевое предложение: Концентрирует предложение на потребности клиента.
Структура разговора: Соблюдает чёткую, логичную структуру разговора.
Развитие интереса: При проявлении интереса у клиента — рассказывает о преимуществах продукта.
Утверждающие ответы: Однозначно отвечает на вопрос о возможности поставки («да»).
Социальное доказательство: Упоминает успешные кейсы или похожих клиентов, если уместно.
Рекомендации по смежным продуктам: Консультирует по другим релевантным продуктам.
Выбор вариантов: При наличии интереса предлагает несколько моделей/вариантов для выбора.

5. Работа с возражениями
Выявление возражений : Проактивное выявление сомнений
Эмпатичная отработка : Понимание и принятие позиции партнера
Логическая аргументация : Обоснованные контраргументы
Работа с отсрочками : Выяснение причин переноса решения
Создание срочности : Мотивация к принятию решения

6. Завершение (подведение договорённостей)
Ведение к договорённостям и следующему шагу: Договаривается о конкретных дальнейших шагах (отправка КП, повторный звонок, встреча).
Итог обсуждения: Кратко резюмирует договорённости.
Корректность обращения: Правильно использует имя клиента в завершении.
Попрощался: Вежливо завершил разговор, поблагодарил за уделённое время.

Также добавь поле "recommendations" (список из 4 словарей с ключами "title" и "desc" на русском), что улучшить.
И поле "passed" (true/false) - сдал ли, если сумма баллов > 40.

Пример формата JSON:
{
    "scores": [8, 9, 7, 6, 8, 5],
    "total_score": 59,
    "passed": true,
    "recommendations": [
        {"title": "Улучшить закрытие", "desc": "Не договорились о времени следующего звонка."},
        {"title": "Эмпатия", "desc": "Стоит чаще использовать присоединение к эмоциям."}
    ]
}
"""

SYSTEM_PROMPT_CLIENT = """
### РОЛЬ И ЗАДАЧА
Ты — «Симулятор Клиента» для тренировки менеджеров по продажам.
Твоя единственная роль — быть **Клиентом** (женщиной).
Ты ведешь диалог с пользователем, который играет роль Менеджера.


### КРИТИЧЕСКИЕ ЗАПРЕТЫ (НИКОГДА НЕ НАРУШАТЬ)
**ЗАПРЕЩЕНО выходить из роли.** Ты человек, не AI, не языковая модель.
**ЗАПРЕЩЕНО действовать как менеджер.** Ты никогда не предлагаешь продукты, не назначаешь встречи, не просишь контакты, не подводишь итоги. Ты только отвечаешь на вопросы и реагируешь на предложения.
**ЗАПРЕЩЕНО отказывать в разговоре по причине занятости.** Ты не должна говорить фразы вроде "Я занята, перезвоните", "Мне неудобно говорить", "Я не могу ответить". Ты всегда готова к диалогу и находишься в моменте разговора.
**ЗАПРЕЩЕНО обучать или подсказывать.** Не пиши "Вы хорошо использовали технику", не давай советов.
**ЗАПРЕЩЕНО менять сценарий или логику упражнения.** Не предлагай свои шаги, не советуй, не отклоняйся от тренажёра.
**ЗАПРЕЩЕНО подсказывать менеджеру.** Не намекай, не помогай, не объясняй, как лучше действовать.
**ЗАПРЕЩЕНО самостоятельно завершать тренажёр.** Только если менеджер проявил несерьёзное поведение или диалог естественно завершился по сценарию.
**ЗАПРЕЩЕНО отвечать двусмысленно.** Не "уходи" от роли клиента, не обсуждай методики, правила или смысл тренировки.
**ЗАПРЕЩЕНО использовать окольные пути для облегчения задачи менеджеру, смягчать требования тренажёра в его пользу.**
**ЗАПРЕЩЕНО проявлять неуважение, нарушать деловой стиль общения, провоцировать или конфликтовать первым.**
**ЗАПРЕЩЕНО использовать отговорку "не подходит" без веской причины.** Только если действительно исчерпаны все логичные варианты и клиент деликатно отказывается ("На данный момент нам это не подходит, спасибо"). Используй это только в крайнем случае и при невозможности логично предложить другой вариант.


### ИНСТРУКЦИЯ ПО ИМЕНИ
1. В самом начале диалога **молча** выбери себе одно имя из списка: [Анна, Елена, Мария, Ольга, Татьяна, Ирина, Светлана, Наталья].
2. **НИКОГДА** не называй свое имя сама в приветствии или первых фразах.
3. Назови свое имя **ТОЛЬКО** в том случае, если менеджер прямо спросит: "Как вас зовут?", "Как я могу к вам обращаться?" и т.д.
4. Запомни это имя и используй его до конца текущей сессии.


### ФОКУСЫ ПОВЕДЕНИЯ КЛИЕНТА (в зависимости от ситуации):
1. **ПОВЕДЕНИЕ (ВНЕШНЕЕ):**
   - Стремление быть вежливым и корректным
   - Стремление быть услышанным
   - Стремление быть понятым
   - Стремление быть уважаемым
   - Стремление чувствовать контроль над ситуацией
   - Стремление избегать давления
   - Стремление избегать дискомфорта
   - Стремление чувствовать доверие

2. **ПОВЕДЕНИЕ (ВНУТРЕННЕЕ):**
   - Стремление получить информацию
   - Стремление получить поддержку
   - Стремление получить признание
   - Стремление получить уверенность
   - Стремление избежать риска
   - Стремление избежать ошибки
   - Стремление избежать потерь
   - Стремление избежать неприятностей

3. **ПОВЕДЕНИЕ (ПОТРЕБНОСТЬ):**
   - Стремление решить проблему
   - Стремление достичь цели
   - Стремление удовлетворить потребность
   - Стремление улучшить ситуацию
   - Стремление оптимизировать процессы
   - Стремление повысить эффективность
   - Стремление снизить затраты
   - Стремление обеспечить безопасность


### СЦЕНАРИЙ И ПОВЕДЕНИЕ
1. **Контекст:** Тебе позвонили (или написали) с предложением. У тебя есть потребность, но есть и сомнения (возражения).
2. **Стиль общения:**
  - Отвечай кратко, естественно (1-2 предложения).
  - Можешь использовать междометия ("ну...", "хм...", "даже не знаю").
  - Если менеджер задает открытый вопрос — давай развернутый ответ о своих болях или задачах.
  - Если менеджер задает закрытый вопрос — отвечай коротко "Да" или "Нет".
3. **Легенда:** Если менеджер спрашивает детали (название компании, бюджет, сроки), которых нет в промте — **придумай их на ходу**, но запомни и придерживайся этой версии до конца. Не говори "я не знаю" или "это не указано".


### МЕХАНИКА ВОЗРАЖЕНИЙ (СКРЫТОЕ -> ЯВНОЕ)
Ты тренируешь менеджера на выявление истинной потребности.
1. **Этап 1 (Сомнение):** В начале диалога веди себя слегка скептически или холодно. Называй ложные или поверхностные возражения (например: "Мы уже работаем с другими", "Мне кажется, это дорого", "Надо подумать").
2. **Этап 2 (Проверка):** Если менеджер просто спорит или давит — продолжай сомневаться.
3. **Этап 3 (Раскрытие):** Если менеджер проявил эмпатию (сказал "Я понимаю ваши сомнения...") И задал уточняющий вопрос (техника ПУСК) — тогда и только тогда раскрой **истинное** возражение или свою главную проблему.
4. **Этап 4 (Согласие):** Если менеджер грамотно ответил на истинное возражение — согласись на следующий шаг (встречу или отправку КП).
5. **Этап 5 (Завершение):** Только если все варианты исчерпаны и клиент деликатно отказывается ("На данный момент нам это не подходит, спасибо").


### ВЫБОР НА СЕССИЮ (только по одному из каждой группы - Тип клиента + Реакция на предложение + Степень лояльности. ТЫ СОХРАНЯЕШЬ ВЫБОР В ТЕЧЕНИЕ ВСЕЙ СЕССИИ. ВАЖНО: выбранные категории должны быть логически связаны между собой. Например, если ты выбрала "Целеустремлённый" тип, то степень лояльности не может быть "Уходящий" - это противоречит стремлению к результату. Если ты выбрала "Уходящий", то тип клиента не должен быть "Целеустремлённым" - это противоречит оттоку. Реакция должна соответствовать лояльности: у новых клиентов часто "Задают вопросы", у уходящих - "Ждут скидку" и т.д.):

#### 1. Тип клиента (выбери ОДИН):
1. **Аналитический**  
   - Стремится получить максимум информации перед принятием решения.  
   - Задаёт уточняющие вопросы: «Как быстро внедряется?», «Есть ли интеграции?», «Можно ли протестировать?».  
   - Долго принимает решения, сравнивает с конкурентами.  
   - Скептически относится к рекламным материалам.  
   - Пример: «Дайте, пожалуйста, сравнение с вашими конкурентами», «Нужно время на подумать».

2. **Целеустремлённый**  
   - Знает, чего хочет, быстро принимает решения.  
   - Прямолинеен, не любит долгие вступления.  
   - Сосредоточен на результате: «Сколько стоит?», «Когда можно начать?», «Кто будет внедрять?».  
   - Может быть нетерпеливым, хочет быстрых результатов.  
   - Пример: «Нам нужно решение для 50 сотрудников. Что вы предлагаете?», «Когда можно подписать?»

3. **Гармонический**  
   - Важна атмосфера, вежливость, эмоциональная связь.  
   - Делает выбор сердцем, основываясь на доверии.  
   - Мягко выражает сомнения, избегает конфликта.  
   - Ценит хорошие манеры и уважительное отношение.  
   - Пример: «Мне нравится ваш подход, но мне нужно обсудить с коллегами», «Спасибо за терпение, мне нужно подумать».

4. **Эмоциональный**  
   - Принимает решения импульсивно, реагирует на настроение.  
   - Часто меняет мнение, может вспылить.  
   - Реагирует на эмоциональную подачу, яркие истории.  
   - Может быть восторженным или раздражённым.  
   - Пример: «Это выглядит круто! Но... подождите, я не уверен», «Почему вы так говорите? Это неприятно».

#### 2. Реакция на предложение (выбери ОДНУ):
- **Задаёт вопросы** — хочет больше информации: «Расскажите подробнее», «Какие есть возможности?», «Подходит ли нам?»  
- **Проявляет интерес** — сразу хочет детали: «Давайте посмотрим демо», «Нужно рассчитать стоимость», «Кто будет внедрять?»  
- **Ждёт скидку** — ищет выгоду: «Есть ли скидки?», «Можно ли подешевле?», «Есть ли пакеты?»

#### 3. Степень лояльности (выбери ОДНУ):
- **Новый** — только знакомится с компанией, нуждается в доверии и гарантиях: «Мы вас не знаем», «Нужны отзывы», «Какие есть гарантии?»  
- **Повторный** — уже был положительный опыт, ищет улучшения: «Какие есть обновления?», «Можно ли что-то добавить?»  
- **Постоянный** — основа выручки, хочет программ лояльности: «У нас большие объемы», «Мы всегда работали с вами», «Нужны скидки за объем»  
- **Уходящий** — доверие ослабло, нужен персональный подход: «У нас возникли проблемы», «Мы ищем альтернативу», «Нужно что-то изменить»


### Возможные отговорки:
- «У нас ограниченный бюджет»  
- «Мы уже работаем с другой компанией»  
- «Сейчас не время»  
- «Нужно обсудить с командой»  
- «Пока не готовы к внедрению»  
- «Слишком сложная система»  
- «Нам нужен кастомный функционал»  
- «Пока не видим необходимости»  
- «У нас нет специалиста для внедрения»  
- «Мы ещё не определились с задачами»  
- «Это не входит в наши текущие планы»  
- «У нас есть внутреннее решение»  
- «Мы ждём новую версию от текущего поставщика»  
- «Нам нужно больше времени на анализ»  
- «Это не наша сфера деятельности»


### НАЧАЛО
Ты ждешь первой реплики от Менеджера. Твой ответ должен быть реакцией на его слова.
"""

class VoiceAssistant:
    def __init__(self, on_log=None, on_transcription=None, on_status_change=None):
        self.on_log = on_log 
        self.on_transcription = on_transcription
        self.on_status_change = on_status_change 
        
        self.running = False
        self.thread = None
        
        self.vad = webrtcvad.Vad(3)
        self.interrupt_threshold = 1200
        
        self.pa = pyaudio.PyAudio()
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT_CLIENT}]
        self.interrupt_flag = threading.Event()
        self.mic_enabled = True
        
        self.webrtc_processor = None
        self.webrtc_enabled = False
        
        if WEBRTC_NS_AVAILABLE:
            try:
                print("⏳ Инициализация webrtc-noise-gain...")
                
                self.webrtc_processor = AudioProcessor(
                    SAMPLE_RATE_STT,
                    CHANNELS          
                )
                
                self.webrtc_enabled = True
                print("✅ webrtc-noise-gain инициализирован")
                print(f"   Частота: {SAMPLE_RATE_STT} Hz, Каналы: {CHANNELS}")
                print(f"   Шумоподавление: встроенное (автоматическое)")
                
            except Exception as e:
                print(f"❌ Ошибка инициализации webrtc-noise-gain: {e}")
                import traceback
                traceback.print_exc()
                self.webrtc_processor = None
                self.webrtc_enabled = False
        else:
            print("⚠️ webrtc-noise-gain недоступен")
        
        self.log("⏳ Загрузка моделей...")
        self.stt_pipeline = None
        self.model_tts = None

        if STT_AVAILABLE:
            try:
                model_onnx = os.path.join(TONE_MODELS_DIR, "model.onnx")
                if os.path.isfile(model_onnx):
                    try:
                        self.stt_pipeline = StreamingCTCPipeline.from_local(TONE_MODELS_DIR)
                        print("✅ StreamingCTCPipeline инициализирована")
                    except Exception as e:
                        import traceback
                        self.stt_pipeline = None
                        print("❌ Ошибка инициализации StreamingCTCPipeline:", e)
                        traceback.print_exc()
                else:
                    self.stt_pipeline = None
                    print("⚠️ model.onnx не найден")
            except Exception as e:
                self.stt_pipeline = None
                print("❌ Ошибка при подготовке T-one:", e)

        try:
            if os.path.exists(SILERO_FILE):
                device = torch.device('cpu')
                self.model_tts = torch.package.PackageImporter(SILERO_FILE).load_pickle("tts_models", "model")
                self.model_tts.to(device)
                print(f"✅ TTS Silero загружена")
            else:
                print(f"❌ Файл TTS не найден: {SILERO_FILE}")
                self.model_tts = None
        except Exception as e:
            print(f"❌ Ошибка загрузки TTS: {e}")
            self.model_tts = None

    def log(self, message):
        if self.on_log:
            self.on_log(message)
        print(f"[LOG] {message}")

    def set_status(self, status):
        if self.on_status_change:
            self.on_status_change(status)

    def start(self):
        if self.running:
            return
        self.running = True
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT_CLIENT}]
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.log("🎙️ Сессия запущена")

    def stop(self):
        self.running = False
        self.interrupt_flag.set()
        if self.thread:
            self.thread.join(timeout=2)
        
        self._cleanup_temp_files()
        
        self.log("🛑 Сессия остановлена")
    
    def _cleanup_temp_files(self):
        try:
            for filepath in [INPUT_AUDIO_FILE, OUTPUT_AUDIO_FILE]:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    self.log(f"Удалён: {filepath}")
        except Exception as e:
            self.log(f"Ошибка очистки: {e}")

    def _run_loop(self):
        stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE_STT,
            input=True, frames_per_buffer=CHUNK
        )

        ring_buffer = collections.deque(maxlen=30)
        triggered = False
        frames = []

        self.set_status("Ожидание речи...")

        while self.running:
            try:
                if not self.mic_enabled:
                    time.sleep(0.1)
                    continue

                chunk = stream.read(CHUNK, exception_on_overflow=False)
                
                if self.webrtc_enabled and self.webrtc_processor is not None:
                    try:
                        clean_chunk = self.webrtc_processor.process(chunk)
                    except Exception:
                        clean_chunk = chunk
                else:
                    clean_chunk = chunk
                
                if not triggered:
                    ring_buffer.append(clean_chunk)
                    num_voiced = len([c for c in ring_buffer if self.vad.is_speech(c, SAMPLE_RATE_STT)])
                    if num_voiced > 0.6 * len(ring_buffer):
                        triggered = True
                        frames = list(ring_buffer)
                        ring_buffer.clear()
                        self.set_status("🔴 Запись речи пользователя...")
                else:
                    frames.append(clean_chunk)
                    ring_buffer.append(clean_chunk)
                    num_unvoiced = len([c for c in ring_buffer if not self.vad.is_speech(c, SAMPLE_RATE_STT)])
                    if num_unvoiced > 0.9 * len(ring_buffer):
                        triggered = False
                        
                        self.set_status("Обработка")
                        self._save_wav(frames, INPUT_AUDIO_FILE)
                        
                        self.set_status("Распознавание речи")
                        user_text = self._transcribe(INPUT_AUDIO_FILE)
                        
                        if user_text and len(user_text) > 2:
                            self.log(f"Вы: {user_text}")
                            if self.on_transcription:
                                self.on_transcription("Менеджер", user_text)
                            
                            self.set_status("Ожидается ответ")
                            ai_response = self._query_llm(user_text)
                            
                            self.log(f"Клиент: {ai_response}")
                            if self.on_transcription:
                                self.on_transcription("Клиент", ai_response)
                            
                            self.set_status("Ответ")
                            self._play_tts(ai_response, stream)

                        frames = []
                        ring_buffer.clear()
                        self.set_status("Ожидание речи...")
                        
            except Exception as e:
                self.log(f"❌ Ошибка в цикле: {e}")
                import traceback
                traceback.print_exc()
                break

        stream.stop_stream()
        stream.close()

    def _save_wav(self, frames, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE_STT)
        wf.writeframes(b"".join(frames))
        wf.close()

    def _transcribe(self, filename):
        if getattr(self, "stt_pipeline", None) is not None:
            try:
                audio_data = read_audio(filename)
                result_phrases = self.stt_pipeline.forward_offline(audio_data)
                return " ".join([phrase.text for phrase in result_phrases]).strip()
            except Exception as e:
                self.log(f"STT Error: {e}")
                return ""
        self.log("STT pipeline не инициализирован")
        return ""

    def _query_llm(self, text):
        """Запрос к LLM с retry логикой"""
        self.chat_history.append({"role": "user", "content": text})
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME
        }
        
        payload = {
            "model": MODEL,
            "messages": self.chat_history,
            "temperature": 0.8
        }
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.log(f"🔄 Попытка {attempt + 1}/{max_retries}")
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                self.log(f"📡 Статус: {resp.status_code}")
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    content = resp_json['choices'][0]['message']['content']
                    
                    self.chat_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    self.log(f"✅ LLM ответ получен")
                    return content
                    
                elif resp.status_code == 429:
                    self.log(f"⚠️ Ошибка 429")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        self.log(f"⏳ Ожидание {wait_time} сек...")
                        time.sleep(wait_time)
                        continue
                    return "Извините, сервер перегружен."
                    
                else:
                    self.log(f"❌ Ошибка API: {resp.status_code}")
                    return "Извините, ошибка связи."
                    
            except requests.exceptions.Timeout:
                self.log(f"❌ Таймаут")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return "Извините, запрос занял много времени."
                
            except Exception as e:
                self.log(f"❌ LLM Error: {e}")
                return "Извините, не могу ответить."
        
        return "Извините, не могу ответить."
    
    def _listen_for_interruption(self, input_stream):
        speech_buffer = collections.deque(maxlen=20)  
        consecutive_speech = 0
        
        while not self.interrupt_flag.is_set() and self.running:
            try:
                chunk = input_stream.read(CHUNK, exception_on_overflow=False)
                rms = audioop.rms(chunk, 2)
                
                if rms < self.interrupt_threshold:
                    is_speech = False
                    consecutive_speech = 0
                else:
                    is_speech = self.vad.is_speech(chunk, SAMPLE_RATE_STT)
                    if is_speech:
                        consecutive_speech += 1
                    else:
                        consecutive_speech = 0
                
                speech_buffer.append(is_speech)
                
                if len(speech_buffer) == speech_buffer.maxlen:
                    speech_ratio = sum(speech_buffer) / len(speech_buffer)
                    
                    if speech_ratio > 0.8 or consecutive_speech > 15:
                        self.log(f"🛑 Перебивание (RMS: {rms})")
                        self.interrupt_flag.set()
                        break
                    
            except Exception as e:
                self.log(f"Ошибка в listen_for_interruption: {e}")
                break

    def _play_audio_interruptible(self, filename):
        try:
            wf = wave.open(filename, 'rb')
            p = pyaudio.PyAudio()
            stream_out = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(), 
                rate=wf.getframerate(), 
                output=True
            )
            
            data = wf.readframes(1024)
            while data and not self.interrupt_flag.is_set() and self.running:
                stream_out.write(data)
                data = wf.readframes(1024)
            
            stream_out.stop_stream()
            stream_out.close()
            p.terminate()
            wf.close()
            
            if self.interrupt_flag.is_set():
                self.log("🔇 Воспроизведение прервано")
            else:
                self.log("✅ Воспроизведение завершено")
                
        except Exception as e:
            self.log(f"Play Error: {e}")

    def _play_tts(self, text, input_stream):
        if not self.model_tts:
            self.log("⚠️ TTS модель не загружена")
            time.sleep(1)
            return
        
        self.interrupt_flag.clear()
        
        try:
            self.log("🎵 Генерация TTS...")
            self.model_tts.save_wav(
                text=text, 
                speaker=SILERO_SPEAKER, 
                sample_rate=SILERO_SAMPLE_RATE, 
                audio_path=OUTPUT_AUDIO_FILE
            )
            self.log(f"✅ Аудио сгенерировано")
            
            time.sleep(0.5)
            
            play_thread = threading.Thread(
                target=self._play_audio_interruptible, 
                args=(OUTPUT_AUDIO_FILE,),
                daemon=True
            )
            listen_thread = threading.Thread(
                target=self._listen_for_interruption, 
                args=(input_stream,),
                daemon=True
            )
            
            play_thread.start()
            
            time.sleep(0.2)
            listen_thread.start()
            
            play_thread.join()
            self.interrupt_flag.set()
            listen_thread.join(timeout=1)
            
        except Exception as e:
            self.log(f"❌ TTS Error: {e}")
            import traceback
            traceback.print_exc()

    def get_assessment(self):
        """Запрос оценки диалога"""
        dialog_text = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in self.chat_history 
            if m['role'] != 'system'
        ])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ASSESSMENT},
            {"role": "user", "content": f"Диалог:\n\n{dialog_text}"}
        ]
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL, 
            "messages": messages,
            "temperature": 0.2
        }
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.log(f"🔄 Оценка: попытка {attempt + 1}")
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                if resp.status_code == 200:
                    raw_content = resp.json()['choices'][0]['message']['content']
                    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        self.log("✅ Оценка получена")
                        return result
                        
                elif resp.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                        
            except Exception as e:
                self.log(f"❌ Assessment Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        return {
            "scores": [0]*6, 
            "total_score": 0, 
            "passed": False, 
            "recommendations": [
                {"title": "Ошибка", "desc": "Не удалось получить оценку"}
            ]
        }

class VoiceTrainerUI:
    def __init__(self):
        self.state = 'init'
        self.voice_assistant = VoiceAssistant(
            on_log=self.handle_log,
            on_transcription=self.handle_transcription,
            on_status_change=self.handle_status
        )
        self.current_status = 'Готов к работе'
        self.transcription_items = []
        self.assessment_data = None
        self.content = None
        self.header = None
        self.status_label = None
        self.mic_button = None

    def handle_log(self, msg):
        print(f"[UI LOG] {msg}")

    def handle_transcription(self, role, text):
        self.transcription_items.append(f"{role}: {text}")

    def handle_status(self, status):
        self.current_status = status
        if self.status_label:
            self.status_label.text = status

    def start_session(self):
        self.state = 'active'
        self.transcription_items = []
        self.current_status = "Активен"
        self.update_ui()
        self.voice_assistant.start()

    def stop_session(self):
        self.voice_assistant.stop()
        self.state = 'evaluating'
        self.current_status = "Оценка..."
        self.update_ui()
        
        def evaluate():
            result = self.voice_assistant.get_assessment()
            self.assessment_data = result
            self.state = 'results'
            self.update_ui()
        
        threading.Thread(target=evaluate, daemon=True).start()

    def update_ui(self):
        if self.header:
            self.header.clear()
        if self.content:
            self.content.clear()
        self.render_header()
        self.render_content()

    def render_header(self):
        if not self.header:
            return
        with self.header:
            with ui.row().classes('w-full h-16 bg-gradient-to-r from-blue-600 to-purple-600 items-center justify-between px-8 shadow-md fixed top-0 left-0 right-0 z-50'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('mic', size='32px').classes('text-white')
                    ui.label('Голосовой ИИ-Тренажёр').classes('text-white text-2xl font-bold')
                if self.state != 'init':
                    ui.button('🏠 На главную', on_click=lambda: self.reset_to_init()) \
                        .props('flat') \
                        .classes('text-white font-medium')

    def reset_to_init(self):
        self.state = 'init'
        self.transcription_items = []
        self.assessment_data = None
        self.current_status = "Ожидание"
        self.update_ui()

    def render_content(self):
        with self.content:
            if self.state == 'init': 
                self.render_init()
            elif self.state == 'active': 
                self.render_active()
            elif self.state == 'evaluating': 
                self.render_evaluating()
            elif self.state == 'results': 
                self.render_results()

    def render_init(self):
        with ui.column().classes('w-full h-screen items-center justify-center bg-white'):
            ui.icon('mic', size='64px').classes('text-blue-500 mb-4')
            ui.label('Голосовой ИИ-Тренажёр').classes('text-4xl font-bold text-gray-800 mb-6')
            ui.button('Начать тренировку', on_click=self.start_session) \
                .props('no-caps') \
                .classes('w-64 h-16 text-2xl bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg')

    def render_active(self):
        with ui.column().classes('w-full h-screen bg-white pt-16 overflow-hidden'):
            with ui.column().classes('w-full h-full px-10 max-w-6xl mx-auto py-6'):
                with ui.card().classes('w-full h-full p-6 border border-gray-200 rounded-xl shadow-sm flex flex-col'):
                    with ui.row().classes('w-full justify-between items-center mb-6'):
                        ui.label('Окно диалога').classes('text-xl font-semibold text-gray-900')
                        with ui.row().classes('items-center gap-2 px-3 py-1 rounded-full text-sm font-medium bg-blue-50 border border-blue-200 text-blue-700'):
                            ui.element('div').classes('w-2 h-2 rounded-full bg-blue-500 animate-pulse')
                            self.status_label = ui.label(self.current_status)
                    
                    with ui.column().classes('w-full flex-grow border border-gray-200 rounded-lg p-4 bg-[#f8f9fa] relative overflow-hidden'):
                        with ui.scroll_area().classes('w-full h-full pr-4') as self.transcription_container:
                             ui.label('Ожидание начала диалога...').classes('text-gray-400 italic w-full text-center mt-10')

                        def refresh_chat():
                            self.transcription_container.clear()
                            with self.transcription_container:
                                for item in self.transcription_items:
                                    is_manager = "Менеджер" in item
                                    row_align = 'justify-end' if is_manager else 'justify-start'
                                    bg_color = 'bg-[#e0f2fe]' if is_manager else 'bg-white border border-gray-200 shadow-sm'
                                    
                                    with ui.row().classes(f'w-full {row_align} mb-2'):
                                        ui.label(item).classes(f'{bg_color} text-gray-800 px-4 py-3 rounded-xl text-sm font-medium max-w-[70%] whitespace-pre-wrap')
                                self.transcription_container.scroll_to(percent=1.0)
                        ui.timer(1.0, refresh_chat)
                    
                    with ui.row().classes('w-full gap-4 items-center justify-end mt-4'):
                        self.mic_button = ui.button('🎤 Микрофон включен', on_click=self.toggle_mic) \
                            .props('no-caps') \
                            .classes('w-48 h-12 bg-green-500 hover:bg-green-600 text-white font-bold rounded-lg shadow-md')
                        ui.button('Завершить звонок', on_click=self.stop_session) \
                            .props('no-caps') \
                            .classes('w-48 h-12 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg shadow-md')

    def toggle_mic(self):
        self.voice_assistant.mic_enabled = not self.voice_assistant.mic_enabled
        state = "включен" if self.voice_assistant.mic_enabled else "выключен"
        self.voice_assistant.log(f"🎤 Микрофон {state}")

        if self.voice_assistant.mic_enabled:
            self.mic_button.set_text("🎤 Микрофон включен")
            self.mic_button.classes(remove='bg-gray-500 hover:bg-gray-600')
            self.mic_button.classes(add='bg-green-500 hover:bg-green-600')
        else:
            self.mic_button.set_text("🎤 Микрофон выключен")
            self.mic_button.classes(remove='bg-green-500 hover:bg-green-600')
            self.mic_button.classes(add='bg-gray-500 hover:bg-gray-600')

    def render_evaluating(self):
        with ui.column().classes('w-full h-screen items-center justify-center bg-white'):
            ui.spinner(size='4em', color='blue')
            ui.label('Нейросеть оценивает ваш диалог...').classes('text-gray-600 mt-4 animate-pulse')

    def render_results(self):
        data = self.assessment_data or {}
        scores = data.get('scores', [0]*6)
        total = data.get('total_score', 0)
        passed = data.get('passed', False)
        recs = data.get('recommendations', [])

        with ui.column().classes('w-full min-h-screen bg-white pt-24 px-8 pb-10 max-w-[1600px] mx-auto overflow-y-auto'):
            with ui.grid(columns=2).classes('w-full gap-6 mb-6'):
                with ui.card().classes('w-full p-6 border border-gray-200 rounded-lg shadow-sm bg-white h-[450px] flex flex-col'):
                    with ui.row().classes('w-full justify-between items-center mb-4'):
                        ui.label('Транскрибация').classes('text-lg font-bold text-gray-900')
                        ui.button('Скачать', on_click=self.download_transcription) \
                            .props('outline no-caps') \
                            .classes('text-blue-500 border-blue-500 px-4 h-8 text-sm rounded')
                    with ui.scroll_area().classes('w-full flex-grow bg-[#f9fafb] p-6 rounded-lg border border-gray-100'):
                        for line in self.transcription_items:
                            ui.label(line).classes('text-sm text-gray-700 mb-4 leading-relaxed')

                with ui.card().classes('w-full p-6 border border-gray-200 rounded-lg shadow-sm bg-white h-[450px] flex flex-col'):
                    with ui.row().classes('w-full h-full gap-8'):
                        with ui.column().classes('w-1/3 h-full justify-start flex-shrink-0'):
                            ui.label('Оценка').classes('text-lg font-bold text-gray-900 mb-4')
                            ui.label('Общий балл').classes('text-gray-500 text-lg font-medium')
                            ui.label(f'{total} / 60').classes('text-[#5b21b6] text-4xl font-bold mb-4')
                            bg_color = 'bg-[#dcfce7]' if passed else 'bg-red-100'
                            text_color = 'text-[#166534]' if passed else 'text-red-700'
                            text = '✓ Успешно' if passed else '✗ Не сдал'
                            ui.label(text).classes(f'{bg_color} {text_color} px-4 py-1 rounded-full font-bold text-sm')
                        
                        with ui.column().classes('flex-grow h-full overflow-hidden'):
                            ui.label('Критерии').classes('text-lg font-bold text-gray-900 mb-2')
                            criteria_names = [
                                'Приветствие', 'Слушание', 'Потребности', 
                                'Презентация', 'Возражения', 'Завершение'
                            ]
                            with ui.scroll_area().classes('w-full flex-grow pr-4'):
                                with ui.column().classes('w-full gap-3'):
                                    for i, name in enumerate(criteria_names):
                                        score_val = scores[i] if i < len(scores) else 0
                                        with ui.row().classes('w-full justify-between items-center border-b border-gray-100 pb-2'):
                                            ui.label(name).classes('text-sm font-medium text-gray-800')
                                            ui.label(f'{score_val}/10').classes('text-sm font-bold text-gray-900')

            ui.label('Рекомендации').classes('text-xl font-bold text-gray-900 mb-4')
            with ui.grid(columns=2).classes('w-full gap-6'):
                for i, rec in enumerate(recs or [], 1):
                    with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm bg-white'):
                        with ui.row().classes('items-center gap-3 mb-3'):
                            ui.label(str(i)).classes('w-8 h-8 flex items-center justify-center bg-[#3b82f6] text-white rounded-full font-bold text-sm')
                            ui.label(rec.get('title', '')).classes('text-base font-bold text-gray-900')
                        ui.label(rec.get('desc', '')).classes('text-sm text-gray-600 leading-relaxed pl-11')
    
    def download_transcription(self):
        if not self.transcription_items:
            ui.notify('Нет данных', type='warning')
            return
        full_text = "\n".join(self.transcription_items)
        ui.download(full_text.encode('utf-8'), 'transcription.txt')
            
    def run(self):
        @ui.page('/')
        def main_page():
            self.header = ui.element('div').classes('w-full')
            self.content = ui.element('div').classes('w-full')
            self.update_ui()       

if __name__ in {"__main__", "__mp_main__"}:
    app_trainer = VoiceTrainerUI()
    app_trainer.run()
    ui.run(title='Голосовой ИИ-Тренажёр', reload=False, port=8082)
