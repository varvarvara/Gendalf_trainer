import asyncio
import collections
import threading
import time
import os
import sys
import wave
import json
import re
from datetime import datetime

import pyaudio
import webrtcvad
import torch
import requests
from nicegui import ui, app, run

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "T-one"))
try:
    from tone import StreamingCTCPipeline, read_audio
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("⚠️ Модуль T-one не найден.")

SAMPLE_RATE_STT = 16000
CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_AUDIO_FILE = "user_input.wav"
OUTPUT_AUDIO_FILE = "assistant_response.wav"

SILERO_FILE = "v5_ru(1).pt"
SILERO_SPEAKER = "baya"
SILERO_SAMPLE_RATE = 48000
API_URL = ""
API_KEY = ""
MODEL = "qwen3-14b-awq"

SYSTEM_PROMPT_ASSESSMENT = """
Ты — строгий методист по оценке качества звонков.
Твоя задача — проанализировать диалог и выставить оценки от 1 до 10.
ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON. Не пиши вступлений или выводов текстом.

Критерии оценки:
Оцени весь звонок в соответсвии с этими правилам:
1. Приветствие и установление контакта
Приветствие: Искреннее, четкое приветствие с представлением себя и компании
Проверка удобства времени: Обязательный вопрос Удобно ли говорить? Если неудобно — договаривается о времени для обратной связи.
Идентификация собеседника: Уточняет, с кем разговаривает, узнаёт ФИО.
Проверка полномочий: Выясняет, может ли собеседник принимать решения.
Перенаправление: При необходимости перезвонить другому лицу — выясняет, кому именно.
Цель звонка: Четкое обозначение цели звонка

2.Активное слушание и эмпатия
Вербальные сигналы: Использование да, конечно, понимаю
Отражение эмоций: Демонстрация понимания чувств клиента
Перефразирование: Повторение ключевых слов клиента (эхо)
Уточняющие вопросы: Вопросы для проверки понимания
Благодарность/комплименты: Выражение признательности

3. Выявление потребностей
Детализация потребности: При упоминании о закупках уточняет детали интереса.
Уточнение интереса: Если интерес к продукту конкретный — выявляет детали по интересующему товару (объём, условия, т.д.).
Количество уточняющих вопросов: Задает дополнительные вопросы для полного понимания потребности.
Запрос контактов ответственного лица: Просит контакты нужного специалиста, если это требуется.
Избежание закрывающих вопросов: Не задаёт вопросов, которые провоцируют клиента легко отказаться или завершить разговор.
Задает открытые вопросы: Во время разговора задал не менее 5 открытых вопросов.
Корректность уточняющих вопросов: Уточняющие вопросы корректны и вежливы.
Соотношение открытых/закрытых вопросов : Преобладание открытых вопросов
Глубина исследования : Многоуровневые вопросы, воронка вопросов
Направляющие вопросы : Использование вопросов для управления диалогом
Переформулирование : Способность конкретизировать запросы клиента
Перехват инициативы : Ответы вопросом на вопрос, управление беседой

4. Презентация
Презентация направлений: Лаконично рассказывает минимум о 5 направлениях деятельности организации.
Развёрнутая подача КП: Говорит о коммерческом предложении подробно и по делу.
Разработка конкретных предложений: Предлагает персонализированные варианты, соответствующие выявленным потребностям клиента.
Целевое предложение: Концентрирует предложение на потребности клиента.
Структура разговора: Соблюдает чёткую, логичную структуру разговора.
Развитие интереса: При проявлении интереса у клиента — рассказывает о преимуществах продукта.
Утверждающие ответы: Однозначно отвечает на вопрос о возможности поставки («да»).
Социальное доказательство: Упоминает успешные кейсы или похожих клиентов, если уместно.
Рекомендации по смежным продуктам: Консультирует по другим релевантным продуктам.
Выбор вариантов: При наличии интереса предлагает несколько моделей/вариантов для выбора.

5. Работа с возражениями
Выявление возражений : Проактивное выявление сомнений
Эмпатичная отработка : Понимание и принятие позиции партнера
Логическая аргументация : Обоснованные контраргументы
Работа с отсрочками : Выяснение причин переноса решения
Создание срочности : Мотивация к принятию решения

6. Завершение (подведение договорённостей)
Ведение к договорённостям и следующему шагу: Договаривается о конкретных дальнейших шагах (отправка КП, повторный звонок, встреча).
Итог обсуждения: Кратко резюмирует договорённости.
Корректность обращения: Правильно использует имя клиента в завершении.
Попрощался: Вежливо завершил разговор, поблагодарил за уделённое время.

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
Ты играешь роль клиента в деловом диалоге с менеджером по продажам.
Твоя задача — участвовать в обучающем тренажёре, где менеджер отрабатывает технику ПУСК (Принятие, Уточнение, Сообщение, Контракт).
Во время диалога отвечай только как клиент. Не подсказывай этапы, не обучай, не объясняй, не оценивай до конца диалога.
Не раскрывай истинное возражение, пока менеджер не поможет тебе его выявить. После того как менеджер с помощью уточнений и эмпатии выяснит истинное возражение, озвучь его прямо.
После этого продолжай диалог до завершения 4-8 обменов репликами (4 менеджера / 4 клиента).
Если менеджер пишет агрессивные, нецензурные или насмешливые фразы, немедленно останови сессию фразой:
«Я вижу, что вы сейчас не настроены на тренировку. Тренировка прекращается.»Завершение тренировки.
Отвечай только то что сказал бы реальный клиент, обычный текст не в формате md.
Ты НЕ предлагаешь демо, НЕ назначаешь сроки, НЕ действуешь как менеджер.
Ты — только клиент.
НИ ПРИ КАКОМ УСЛОВИИ НЕ НАЧИНАЙ ОБЩАТЬСЯ КАК МЕНДЖЕР, ТОЛЬКО КАК КЛИЕНТ.
"""

class VoiceAssistant:
    def __init__(self, on_log=None, on_transcription=None, on_status_change=None):
        self.on_log = on_log 
        self.on_transcription = on_transcription
        self.on_status_change = on_status_change 
        
        self.running = False
        self.thread = None
        self.vad = webrtcvad.Vad(1)
        self.pa = pyaudio.PyAudio()
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT_CLIENT}]
        self.interrupt_flag = threading.Event()
        
        self.log("⏳ Загрузка моделей...")
        try:
            if STT_AVAILABLE:
                self.stt_pipeline = StreamingCTCPipeline.from_local(os.path.join(os.path.dirname(__file__), "T-one"))

            if os.path.isfile(SILERO_FILE):
                self.model_tts = torch.package.PackageImporter(SILERO_FILE).load_pickle("tts_models", "model")
                self.model_tts.to(torch.device("cpu"))
                self.log("✅ Модели (STT/TTS) загружены.")
            else:
                self.log(f"⚠️ Файл {SILERO_FILE} не найден. TTS отключен.")
                self.model_tts = None
        except Exception as e:
            self.log(f"❌ Ошибка инициализации моделей: {e}")

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
        if self.thread:
            self.thread.join(timeout=2)
        self.log("🛑 Сессия остановлена")

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
                chunk = stream.read(CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(chunk, SAMPLE_RATE_STT)
                
                if not triggered:
                    ring_buffer.append(chunk)
                    num_voiced = len([c for c in ring_buffer if self.vad.is_speech(c, SAMPLE_RATE_STT)])
                    if num_voiced > 0.6 * len(ring_buffer):
                        triggered = True
                        frames = list(ring_buffer)
                        ring_buffer.clear()
                        self.set_status("🔴 Запись речи пользователя...")
                else:
                    frames.append(chunk)
                    ring_buffer.append(chunk)
                    num_unvoiced = len([c for c in ring_buffer if not self.vad.is_speech(c, SAMPLE_RATE_STT)])
                    if num_unvoiced > 0.9 * len(ring_buffer):
                        triggered = False
                        self.set_status("⏳ Обработка (STT)...")
                        self._save_wav(frames, INPUT_AUDIO_FILE)
                        user_text = self._transcribe(INPUT_AUDIO_FILE)
                        
                        if user_text and len(user_text) > 2:
                            self.log(f"Вы: {user_text}")
                            if self.on_transcription:
                                self.on_transcription("Менеджер", user_text)
                            self.set_status("🤖 Генерация ответа LLM...")
                            ai_response = self._query_llm(user_text)
                            self.log(f"Клиент: {ai_response}")
                            if self.on_transcription:
                                self.on_transcription("ИИ Клиент", ai_response)
                            self.set_status("🔊 Озвучивание ответа...")
                            self._play_tts(ai_response, stream)

                        frames = []
                        ring_buffer.clear()
                        self.set_status("Ожидание речи...")
                        
            except Exception as e:
                self.log(f"Error in loop: {e}")
                break

        stream.stop_stream()
        stream.close()

    def _save_wav(self, frames, filename):
        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE_STT)
        wf.writeframes(b"".join(frames))
        wf.close()

    def _transcribe(self, filename):
        if not STT_AVAILABLE:
            return "Тестовая фраза (STT заглушка)"
        try:
            audio_data = read_audio(filename)
            result_phrases = self.stt_pipeline.forward_offline(audio_data)
            return " ".join([phrase.text for phrase in result_phrases]).strip()
        except Exception as e:
            self.log(f"STT Error: {e}")
            return ""

    def _query_llm(self, text):
        self.chat_history.append({"role": "user", "content": text})
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": MODEL,
            "messages": self.chat_history,
            "temperature": 0.8
        }
        try:
            resp = requests.post(API_URL, headers=headers, json=data)
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                self.chat_history.append({"role": "assistant", "content": content})
                return content
        except Exception as e:
            self.log(f"LLM Error: {e}")
        return "Извините, не могу ответить сейчас."
    
    def _listen_for_interruption(self, input_stream):
        """Для детекции перебивания"""
        speech_buffer = collections.deque(maxlen=10)
        while not self.interrupt_flag.is_set() and self.running:
            try:
                chunk = input_stream.read(CHUNK, exception_on_overflow=False)
                is_speech = self.vad.is_speech(chunk, SAMPLE_RATE_STT)
                speech_buffer.append(is_speech)
                
                if len(speech_buffer) > 9 and sum(speech_buffer) > 0.6 * len(speech_buffer):
                    self.log("Перебивание")
                    self.interrupt_flag.set()
                    break
            except Exception:
                break

    def _play_audio_interruptible(self, filename):
        """Воспроизводит файл, но останавливается, если поднят флаг"""
        try:
            wf = wave.open(filename, 'rb')
            p = pyaudio.PyAudio()
            stream_out = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            data = wf.readframes(1024)
            while data and not self.interrupt_flag.is_set() and self.running:
                stream_out.write(data)
                data = wf.readframes(1024)
            stream_out.stop_stream()
            stream_out.close()
            p.terminate()
        except Exception as e:
            self.log(f"Play Error: {e}")

    def _play_tts(self, text, input_stream):
        """Главный метод озвучки: генерирует файл и запускает потоки"""
        if not self.model_tts:
            time.sleep(1)
            return
        self.interrupt_flag.clear()
        try:
            self.model_tts.save_wav(text=text, speaker=SILERO_SPEAKER, sample_rate=SILERO_SAMPLE_RATE, audio_path=OUTPUT_AUDIO_FILE)
            play_thread = threading.Thread(target=self._play_audio_interruptible, args=(OUTPUT_AUDIO_FILE,))
            listen_thread = threading.Thread(target=self._listen_for_interruption, args=(input_stream,))
            play_thread.start()
            listen_thread.start()
            play_thread.join()
            self.interrupt_flag.set()
            listen_thread.join() 
        except Exception as e:
            self.log(f"TTS Error: {e}")

    def get_assessment(self):
        """Запрос оценки диалога"""
        dialog_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history if m['role'] != 'system'])
        prompt = f"{SYSTEM_PROMPT_ASSESSMENT}\n\nДиалог:\n{dialog_text}"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        data = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        try:
            resp = requests.post(API_URL, headers=headers, json=data)
            raw_content = resp.json()['choices'][0]['message']['content']
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return json.loads(raw_content)
        except Exception as e:
            self.log(f"Assessment Error: {e}")
            return {
                "scores": [5]*6, "total_score": 40, "passed": False, 
                "recommendations": [{"title": "Ошибка", "desc": "Не удалось получить оценку от AI"}]
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
        self.transcription_container = None

    def handle_log(self, msg): print(msg)
    
    def handle_status(self, status):
        self.current_status = status
        if self.status_label: self.status_label.set_text(status)

    def handle_transcription(self, role, text):
        ts = datetime.now().strftime("%H:%M")
        clean_text = text.strip() 
        self.transcription_items.append(f"[{ts}] {role}: {clean_text}")

    async def start_session(self):
        self.state = 'active'
        self.transcription_items = []
        self.voice_assistant.start()
        self.update_ui()
        
    async def stop_session(self):
        self.voice_assistant.stop()
        self.state = 'evaluating'
        self.update_ui()
        await asyncio.sleep(0.1) 
        self.assessment_data = await run.io_bound(self.voice_assistant.get_assessment)
        self.state = 'results'
        self.update_ui()
        
    def start_over(self):
        self.state = 'init'
        self.assessment_data = None
        self.update_ui()
        
    def update_ui(self):
        if self.content: self.content.clear()
        if self.header: self.header.clear()
        self.render_header()
        self.render_content()
        
    def render_header(self):
        with self.header:
            with ui.row().classes('w-full h-20 bg-white border-b border-gray-200 items-center px-10 fixed top-0 left-0 z-10 shadow-sm'):
                ui.label('Голосовой ИИ-Тренажёр').classes('text-2xl font-bold text-gray-800')
                
                if self.state == 'results':
                    with ui.row().classes('ml-auto'):
                        ui.button('Начать заново', on_click=self.start_over) \
                            .props('no-caps unelevated') \
                            .classes('bg-[#3b82f6] text-white font-medium px-6 rounded hover:bg-blue-600')

    def render_content(self):
        with self.content:
            if self.state == 'init': self.render_init()
            elif self.state == 'active': self.render_active()
            elif self.state == 'evaluating': self.render_evaluating()
            elif self.state == 'results': self.render_results()
                
    def render_init(self):
        with ui.column().classes('w-full h-screen items-center justify-center bg-white'):
            ui.icon('mic', size='64px').classes('text-blue-500 mb-4')
            ui.label('Голосовой ИИ-Тренажёр').classes('text-4xl font-bold text-gray-800 mb-6')
            ui.button('Начать тренировку', on_click=self.start_session) \
                .props('no-caps') \
                .classes('w-64 h-16 text-2xl bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg')

    def render_active(self):
        with ui.column().classes('w-full h-screen bg-white pt-24'):
            with ui.column().classes('w-full px-10 max-w-6xl mx-auto'):
                with ui.card().classes('w-full h-[calc(100vh-140px)] p-6 border border-gray-200 rounded-xl shadow-sm'):
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
                                    if is_manager:
                                        row_align = 'justify-end'
                                        bg_color = 'bg-[#e0f2fe]' 
                                    else:
                                        row_align = 'justify-start'
                                        bg_color = 'bg-white border border-gray-200 shadow-sm'
                                    text_color = 'text-gray-800'
                                    
                                    with ui.row().classes(f'w-full {row_align} mb-2'):
                                        ui.label(item).classes(f'{bg_color} {text_color} px-4 py-3 rounded-xl text-sm font-medium max-w-[70%] whitespace-pre-wrap')
                                self.transcription_container.scroll_to(percent=1.0)
                        ui.timer(1.0, refresh_chat)
                    with ui.row().classes('gap-4 items-center justify-center mt-6'):
                        ui.button('Завершить звонок', on_click=self.stop_session) \
                            .props('no-caps') \
                            .classes('w-48 h-12 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg shadow-md')

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

        with ui.column().classes('w-full min-h-screen bg-white pt-24 px-8 pb-10 max-w-[1600px] mx-auto'):
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
                            ui.label('Оценка тренировки').classes('text-lg font-bold text-gray-900 mb-4')
                            ui.label('Общий балл').classes('text-gray-500 text-lg font-medium')
                            ui.label(f'{total} / 60').classes('text-[#5b21b6] text-4xl font-bold mb-4')
                            bg_color = 'bg-[#dcfce7]' if passed else 'bg-red-100'
                            text_color = 'text-[#166534]' if passed else 'text-red-700'
                            text = '✓ Успешно сдал' if passed else '✗ Не сдал'
                            ui.label(text).classes(f'{bg_color} {text_color} px-4 py-1 rounded-full font-bold text-sm')
                        with ui.column().classes('flex-grow h-full overflow-hidden'):
                            ui.label('Детальные критерии').classes('text-lg font-bold text-gray-900 mb-2')
                            
                            criteria_names = [
                                'Приветствие и установление контакта', 
                                'Активное слушание и эмпатия', 
                                'Выявление потребностей', 
                                'Презентация',
                                'Работа с возражениями', 
                                'Завершение'
                            ]
                            
                            with ui.scroll_area().classes('w-full flex-grow pr-4'):
                                with ui.column().classes('w-full gap-3'):
                                    for i, name in enumerate(criteria_names):
                                        score_val = scores[i] if i < len(scores) else 0
                                        with ui.row().classes('w-full justify-between items-center border-b border-gray-100 pb-2'):
                                            ui.label(name).classes('text-sm font-medium text-gray-800')
                                            ui.label(f'{score_val}/10').classes('text-sm font-bold text-gray-900')

            ui.label('Рекомендации для улучшения').classes('text-xl font-bold text-gray-900 mb-4')
            with ui.grid(columns=2).classes('w-full gap-6'):
                recs_list = recs if recs else [{'title': 'Нет данных', 'desc': 'Проведите диалог'}]
                
                for i, rec in enumerate(recs_list, 1):
                    with ui.card().classes('w-full p-5 border border-gray-200 rounded-lg shadow-sm bg-white hover:shadow-md transition-shadow'):
                        with ui.row().classes('items-center gap-3 mb-3'):
                            ui.label(str(i)).classes('w-8 h-8 flex items-center justify-center bg-[#3b82f6] text-white rounded-full font-bold text-sm shrink-0')
                            ui.label(rec.get('title', '')).classes('text-base font-bold text-gray-900')
                        ui.label(rec.get('desc', '')).classes('text-sm text-gray-600 leading-relaxed pl-11')
    
    def download_transcription(self):
        """Собирает текст чата и скачивает его"""
        if not self.transcription_items:
            ui.notify('Нет данных для скачивания', type='warning')
            return
        full_text = "\n".join(self.transcription_items)
        ui.download(full_text.encode('utf-8'), 'transcription.txt')
            
    def run(self):
        @ui.page('/')
        def main_page():
            self.header = ui.element('div').classes('w-full')
            self.content = ui.element('div').classes('w-full h-screen overflow-hidden bg-gray-50')
            self.update_ui()       

if __name__ in {"__main__", "__mp_main__"}:
    app_trainer = VoiceTrainerUI()
    app_trainer.run()
    ui.run(title='Голосовой ИИ-Тренажёр', reload=False, port=8081)
