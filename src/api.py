from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="IT Ticket Routing API",
    description="Сервис для автоматической маршрутизации обращений в IT-поддержку",
    version="1.0.0"
)

MODEL_PATH = os.path.join("models", "ticket_classifier_final.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

class TicketRequest(BaseModel):
    subject: str
    description: str

class TicketResponse(BaseModel):
    queue: str
    model_version: str = "v1.0"

# --- СТИЛЬНЫЙ ВЕБ-ИНТЕРФЕЙС С ИМИТАЦИЕЙ ЧАТА ---
html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Ticket Router | IT Support</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-dark: #0f172a;
            --primary-blue: #1e3a8a;
            --accent-blue: #3b82f6;
            --accent-light: #eff6ff;
            --text-main: #334155;
            --text-muted: #64748b;
            --bg-color: #f1f5f9;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .main-card {
            background: #ffffff;
            width: 100%;
            max-width: 600px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 650px;
        }

        /* Верхняя синяя шапка */
        .header-section {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 25px 30px;
            text-align: center;
            color: white;
            flex-shrink: 0;
        }

        .header-icon {
            background: rgba(255, 255, 255, 0.2);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto 10px auto;
            backdrop-filter: blur(10px);
        }

        .header-icon svg { width: 30px; height: 30px; fill: white; }
        
        .header-section h1 { font-size: 22px; font-weight: 700; margin-bottom: 5px; }
        .header-section p { font-size: 13px; color: #bfdbfe; line-height: 1.4; }

        /* Форма */
        .form-section { 
            padding: 25px 30px; 
            overflow-y: auto;
            flex-grow: 1;
        }

        .input-group { margin-bottom: 15px; }
        
        .input-group label {
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 600;
            color: var(--primary-blue);
            margin-bottom: 6px;
        }

        .input-group label svg {
            width: 14px;
            height: 14px;
            margin-right: 6px;
            fill: var(--accent-blue);
        }

        .input-group input, .input-group textarea {
            width: 100%;
            padding: 12px 14px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            color: var(--text-main);
            transition: all 0.3s ease;
            background: #f8fafc;
        }

        .input-group input:focus, .input-group textarea:focus {
            outline: none;
            border-color: var(--accent-blue);
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
        }

        .input-group textarea { height: 100px; resize: vertical; }

        .submit-btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }
        
        .submit-btn:disabled { opacity: 0.7; cursor: not-allowed; transform: none; }

        /* Блок результата (ошибки) */
        #result-box {
            margin-top: 15px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            display: none;
            font-size: 14px;
            font-weight: 500;
            animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        .result-error { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }

        /* --- ИМИТАЦИЯ ЧАТА --- */
        .chat-section {
            display: none;
            flex-direction: column;
            flex-grow: 1;
            background: #f8fafc;
            animation: fadeIn 0.5s ease;
        }

        .chat-header {
            background: #ffffff;
            padding: 15px 20px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }

        .chat-header .avatar {
            width: 40px;
            height: 40px;
            background: #dbeafe;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-right: 12px;
            color: #1e3a8a;
            font-weight: bold;
        }

        .chat-header .info h3 { font-size: 15px; color: #1e293b; margin: 0; }
        .chat-header .info p { font-size: 12px; color: #10b981; margin: 0; display: flex; align-items: center; gap: 4px; }
        .chat-header .info p::before { content: ''; display: block; width: 6px; height: 6px; background: #10b981; border-radius: 50%; }

        .chat-messages {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.4;
            animation: slideUp 0.3s ease;
        }

        .message.bot {
            background: #ffffff;
            color: #334155;
            border: 1px solid #e2e8f0;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }

        .message.user {
            background: #3b82f6;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }

        .chat-input-area {
            padding: 15px 20px;
            background: #ffffff;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 10px;
        }

        .chat-input-area input {
            flex-grow: 1;
            padding: 12px 15px;
            border: 1px solid #cbd5e1;
            border-radius: 20px;
            outline: none;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }

        .chat-input-area input:focus { border-color: var(--accent-blue); }

        .chat-input-area button {
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 50%;
            width: 42px;
            height: 42px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-input-area button:hover { background: var(--primary-blue); }

    </style>
</head>
<body>

    <div class="main-card">
        <!-- ШАПКА -->
        <div class="header-section">
            <div class="header-icon">
                <svg viewBox="0 0 24 24"><path d="M22 10V6c0-1.1-.9-2-2-2H4c-1.1 0-1.99.9-1.99 2v4c1.1 0 1.99.9 1.99 2s-.89 2-2 2v4c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2v-4c-1.1 0-2-.9-2-2s.9-2 2-2zm-2-1.46c-1.19.69-2 1.99-2 3.46s.81 2.77 2 3.46V18H4v-2.54c1.19-.69 2-1.99 2-3.46 0-1.48-.8-2.77-1.99-3.46L4 6h16v2.54zM11 15h2v2h-2zm0-8h2v6h-2z"/></svg>
            </div>
            <h1>Smart Ticket Router</h1>
            <p>Опишите вашу проблему, и наш ИИ автоматически направит её в нужный отдел поддержки.</p>
        </div>

        <!-- ФОРМА -->
        <div class="form-section" id="form-section">
            <div class="input-group">
                <label>
                    <svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    Тема обращения (English)
                </label>
                <input type="text" id="subject" placeholder="Example: Payment failed">
            </div>

            <div class="input-group">
                <label>
                    <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>
                    Описание проблемы (English)
                </label>
                <textarea id="description" placeholder="Example: I tried to pay for my subscription, but the transaction was declined. I need help urgently."></textarea>
            </div>

            <button class="submit-btn" id="submit-btn" onclick="predict()">
                <span id="btn-text">Маршрутизировать тикет</span>
                <div class="loader" id="loader" style="display:none; border: 3px solid #f3f3f3; border-top: 3px solid white; border-radius: 50%; width: 18px; height: 18px; animation: spin 1s linear infinite;"></div>
                <svg id="send-icon" viewBox="0 0 24 24" width="20" height="20" fill="white"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
            </button>

            <div id="result-box" class="result-error"></div>
        </div>

        <!-- ЧАТ (Скрыт по умолчанию) -->
        <div class="chat-section" id="chat-section">
            <div class="chat-header">
                <div class="avatar" id="chat-avatar">TS</div>
                <div class="info">
                    <h3 id="chat-dept-name">Отдел поддержки</h3>
                    <p>В сети</p>
                </div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <!-- Сюда будут добавляться сообщения -->
            </div>

            <div class="chat-input-area">
                <input type="text" id="chat-input" placeholder="Напишите сообщение..." onkeypress="handleChatEnter(event)">
                <button onclick="sendChatMessage()">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="white"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
        </div>
    </div>

    <style>
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>

    <script>
        async function predict() {
            const subject = document.getElementById("subject").value;
            const description = document.getElementById("description").value;
            const resultBox = document.getElementById("result-box");
            
            const btn = document.getElementById("submit-btn");
            const btnText = document.getElementById("btn-text");
            const loader = document.getElementById("loader");
            const sendIcon = document.getElementById("send-icon");

            if (!subject.trim() && !description.trim()) {
                resultBox.style.display = "block";
                resultBox.innerText = "Ошибка: Пожалуйста, заполните хотя бы одно поле.";
                return;
            }

            // Загрузка
            btn.disabled = true;
            btnText.innerText = "Анализируем текст...";
            loader.style.display = "block";
            sendIcon.style.display = "none";
            resultBox.style.display = "none";

            try {
                const response = await fetch('/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ subject: subject, description: description })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Скрываем форму, показываем чат
                    document.getElementById("form-section").style.display = "none";
                    document.getElementById("chat-section").style.display = "flex";
                    
                    // Обновляем шапку чата
                    const deptName = data.queue;
                    document.getElementById("chat-dept-name").innerText = deptName;
                    
                    // Делаем аватарку из первых букв отдела
                    const initials = deptName.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
                    document.getElementById("chat-avatar").innerText = initials;

                    // Добавляем приветственное сообщение
                    addMessage("bot", `Здравствуйте! На связи отдел "${deptName}". Опишите более подробно вашу проблему, и мы обязательно поможем.`);

                } else {
                    const error = await response.json();
                    resultBox.style.display = "block";
                    resultBox.innerText = "Ошибка: " + (error.detail || "Неизвестная ошибка");
                }
            } catch (e) {
                resultBox.style.display = "block";
                resultBox.innerText = "Ошибка соединения с сервером.";
            } finally {
                btn.disabled = false;
                btnText.innerText = "Маршрутизировать тикет";
                loader.style.display = "none";
                sendIcon.style.display = "block";
            }
        }

        // --- ЛОГИКА ЧАТА ---
        function addMessage(sender, text) {
            const messagesContainer = document.getElementById("chat-messages");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${sender}`;
            msgDiv.innerText = text;
            messagesContainer.appendChild(msgDiv);
            
            // Прокрутка вниз
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function handleChatEnter(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        }

        function sendChatMessage() {
            const input = document.getElementById("chat-input");
            const text = input.value.trim();
            if (!text) return;

            // Добавляем сообщение пользователя
            addMessage("user", text);
            input.value = "";

            // Имитация ответа бота через 1.5 секунды
            setTimeout(() => {
                addMessage("bot", "Спасибо за информацию. Я передал ваши детали специалисту. Ожидайте ответа, это займет не более 15 минут.");
            }, 1500);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def main():
    return html_content

@app.post("/v1/predict", response_model=TicketResponse)
async def predict_queue(ticket: TicketRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере")
    
    if not ticket.subject.strip() and not ticket.description.strip():
        raise HTTPException(status_code=422, detail="Текст обращения не может быть пустым")

    combined_text = f"{ticket.subject} {ticket.description}"
    prediction = model.predict([combined_text])[0]
    
    return TicketResponse(queue=prediction)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}