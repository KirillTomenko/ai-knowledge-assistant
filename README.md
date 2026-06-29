# 🤖 AI Knowledge Assistant

<p align="center">
  <img src="docs/assets/banner.png" alt="AI Knowledge Assistant" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/aiogram-3.x-blue?logo=telegram&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-0.5-orange"/>
  <img src="https://img.shields.io/badge/LangChain-0.3-yellow"/>
  <img src="https://img.shields.io/badge/GPT--4o-OpenAI-412991?logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tests-49%20passed-brightgreen?logo=pytest"/>
</p>

**RAG-система для корпоративного поиска по документам.** Загрузи PDF/DOCX/TXT — и сотрудники смогут задавать вопросы на естественном языке через Telegram и получать точные ответы со ссылками на источники.

---
## Бизнес-проблема

Во многих компаниях сотрудники регулярно обращаются к внутренним регламентам, инструкциям, политикам и другим документам. Поиск нужной информации занимает время, а ответы коллег или специалистов HR/IT часто повторяются.

В результате:

- сотрудники тратят время на поиск информации;
- HR и служба поддержки отвечают на одни и те же вопросы;
- возрастает риск использования устаревших версий документов;
- снижается скорость адаптации новых сотрудников.

Проект решает эту проблему с помощью RAG (Retrieval-Augmented Generation): сотрудник задаёт вопрос в Telegram, система находит релевантные фрагменты документов, формирует ответ и обязательно указывает источник (документ и страницу).
---

## Польза для бизнеса

Сервис позволяет:

- сократить время поиска информации по внутренним документам;
- уменьшить нагрузку на HR и внутреннюю поддержку;
- получать ответы с указанием источника, а не "галлюцинации" модели;
- быстрее адаптировать новых сотрудников;
- использовать корпоративную базу знаний через привычный Telegram.
---
## ✨ Возможности

### 👨‍💼 Администратор (FastAPI Admin API)

- 📄 Загрузка документов (PDF, DOCX, DOC, TXT) размером до 200 MB / 500 страниц
- ⚙️ Фоновая индексация документов с отслеживанием статуса (`pending → processing → done`)
- 🗂️ Управление коллекциями ChromaDB
- 🤖 Проверка качества ответов через `POST /api/v1/ask` в Swagger UI
- 🔍 Расширенный `/health` с проверкой ChromaDB и LLM

### 👤 Пользователь (Telegram Bot)

- 💬 Ответы на вопросы по корпоративным документам (`/ask` или обычное сообщение)
- 📖 Использование истории последних диалогов для сохранения контекста
- 📝 Персональные заметки (`/note`, `/notes`, `/notes <поиск>`)
- 📊 Просмотр статистики использования (`/stats`)
- 🧹 Очистка истории общения (`/clear`)
- 🔗 Каждый ответ содержит ссылку на источник: документ и страницу
---

### 🧠 RAG Pipeline

- 🔎 Семантический поиск по корпоративным документам
- 📚 Retrieval-Augmented Generation (RAG)
- 📄 Ответы только на основе найденного контекста
- 📌 Ссылки на документ и страницу
- 🧩 Автоматическое разбиение документов на чанки
- 🗃️ Векторное хранение в ChromaDB
---



## 📋 Содержание

- [Архитектура](#-архитектура)
- [Возможности](#-возможности)
- [Стек технологий](#-стек-технологий)
- [Быстрый старт](#-быстрый-старт)
- [Структура проекта](#-структура-проекта)
- [API документация](#-api-документация)
- [Конфигурация прокси](#-конфигурация-прокси)
- [Тесты](#-тесты)
- [Схема базы данных](#-схема-базы-данных)

---

## 🏗 Архитектура

```
┌──────────────────────────────────────────────────────────────────┐
│                      INGESTION FLOW                              │
│                                                                  │
│  PDF/DOCX/TXT → DocumentParser → RecursiveCharacterTextSplitter  │
│       → OpenAI Embeddings → ChromaDB (persistent)               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       QUERY FLOW                                 │
│                                                                  │
│  Telegram User → aiogram Bot → RAGService                        │
│                                    │                             │
│                          condense_question()   ← chat history    │
│                                    │           (Supabase)        │
│                          ChromaService.similarity_search()       │
│                                    │                             │
│                          build_prompt(context + history)         │
│                                    │                             │
│                          GPT-4o (via ProxyAPI)                   │
│                                    │                             │
│                          RAGResult(answer + sources)             │
│                                    │                             │
│                          save_pair() → Supabase                  │
└──────────────────────────────────────────────────────────────────┘
```

---
## 🛠 Стек технологий

| Компонент | Технология | Версия |
|-----------|-----------|--------|
| Язык | Python | 3.12 |
| LLM | GPT-4o (via ProxyAPI) | — |
| Embeddings | text-embedding-3-small | — |
| RAG Framework | LangChain | 0.3.x |
| Vector Store | ChromaDB | 0.5.x |
| Backend | FastAPI | 0.115 |
| Bot | aiogram | 3.x |
| База данных | Supabase (PostgreSQL) | — |
| Тесты | pytest + pytest-asyncio | 8.x |
| Инфраструктура | Docker + Compose | — |

---

## 🚀 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/KirillTomenko/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Настроить окружение

```bash
cp .env.example .env
# Открой .env и заполни переменные
```

Ключевые переменные:
```env
OPENAI_API_KEY=sk-...               # ключ ProxyAPI
OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
TELEGRAM_BOT_TOKEN=123:AAA...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
HTTP_PROXY=socks5://host.docker.internal:3067   # Karing
```

### 3. Создать таблицы в Supabase

Выполни содержимое `sql/schema.sql` в Supabase SQL Editor.

### 4. Запустить через Docker Compose

```bash
docker compose up -d --build
```

Проверить статус:
```bash
docker compose ps
docker compose logs -f bot
curl http://localhost:8000/health
```

### 5. Загрузить документ

```bash
curl -u admin:your_password \
  -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@handbook.pdf"
```

Проверить статус индексации:
```bash
curl -u admin:your_password \
  http://localhost:8000/api/v1/documents/<doc_id>/status
```

### 6. Протестировать RAG

Через Swagger UI: **http://localhost:8000/docs** → `POST /api/v1/ask`

Или через Telegram: найди бота и напиши _«Какой режим работы в компании?»_

---

## 💻 Локальная разработка

В этом режиме FastAPI, Telegram-бот и Alembic запускаются локально, а ChromaDB используется как отдельный сервис в Docker. Такой подход удобен для разработки и отладки, так как позволяет быстро перезапускать приложение без пересборки контейнеров.

### Требования

Перед запуском убедитесь, что установлены:

- Python 3.12+
- Git
- Docker (только для запуска ChromaDB)
- доступ к базе данных Supabase PostgreSQL
- ключ OpenAI / ProxyAPI
- Telegram Bot Token

---

### 1. Клонировать репозиторий

```bash
git clone https://github.com/KirillTomenko/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

---

### 2. Создать виртуальное окружение

Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

---

### 4. Настроить переменные окружения

Создайте файл `.env` на основе шаблона:

```bash
cp .env.example .env
```

Заполните необходимые параметры:

- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `DATABASE_URL`
- параметры подключения к ChromaDB
- при необходимости — настройки прокси

---

### 5. Запустить ChromaDB

```bash
docker run -p 8000:8000 chromadb/chroma
```

---

### 6. Применить миграции базы данных

```bash
alembic upgrade head
```

Будут созданы необходимые таблицы:

- `users`
- `chats`
- `notes`
- `documents`

---

### 7. Запустить FastAPI

```bash
uvicorn app.main:app --reload --port 8001
```

После запуска будут доступны:

- API: `http://localhost:8001`
- Swagger UI: `http://localhost:8001/docs`

---

### 8. Запустить Telegram-бота

Откройте второй терминал и выполните:

```bash
python -m app.telegram.bot
```

---

### 9. Загрузить первый документ

Откройте Swagger UI и вызовите:

```
POST /documents/upload
```

После загрузки документ автоматически:

1. разбирается на текст;
2. разбивается на чанки;
3. преобразуется в эмбеддинги;
4. индексируется в ChromaDB.

---

### 10. Проверить работу RAG

В Telegram отправьте сообщение:

```
/ask Сколько дней отпуска положено сотруднику?
```

Пример ответа:

```
Каждому сотруднику предоставляется 28 календарных дней оплачиваемого отпуска.

Источник:
employee_handbook.pdf, стр. 2
```

---

### Использование прокси (Windows)

Если для доступа к OpenAI API или Telegram используется SOCKS5-прокси (например, Karing), укажите в файле `.env`:

```env
HTTPS_PROXY=socks5://127.0.0.1:3067
TELEGRAM_PROXY_URL=socks5://127.0.0.1:3067
```

---



## 📁 Структура проекта

```
ai-knowledge-assistant/
├── docker/
│   ├── api.Dockerfile
│   └── bot.Dockerfile
├── requirements/
│   ├── api.txt          # зависимости API
│   ├── bot.txt          # зависимости бота
│   └── test.txt         # зависимости тестов
├── sql/
│   └── schema.sql       # схема таблиц Supabase
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + /health + /ask
│   │   ├── auth.py          # HTTP Basic Auth (require_admin)
│   │   ├── routers/
│   │   │   ├── documents.py # upload, status, list, delete
│   │   │   └── collections.py
│   │   └── schemas/
│   │       └── documents.py # Pydantic models (Response, Health...)
│   ├── bot/
│   │   ├── main.py          # aiogram + proxy setup
│   │   ├── handlers/
│   │   │   ├── ask.py       # RAG pipeline handler
│   │   │   ├── notes.py     # /note, /notes, /note_delete, /clear_notes
│   │   │   └── start.py     # /start, /help, /clear, /stats
│   │   └── middlewares/
│   │       └── logging.py
│   ├── core/
│   │   └── config.py        # pydantic-settings (все переменные)
│   ├── rag/
│   │   └── parser.py        # PDF/DOCX/TXT → Document chunks
│   ├── services/
│   │   ├── chroma_service.py  # ChromaDB: индексация, поиск, удаление
│   │   └── rag_service.py     # RAG pipeline: condense → retrieve → generate
│   └── db/
│       ├── supabase_client.py    # sync + async клиенты
│       ├── models.py             # Pydantic Row/Create/Update модели
│       └── repositories/
│           ├── base.py           # BaseRepository, RepositoryError
│           ├── documents.py      # CRUD documents (9 методов)
│           ├── history.py        # CRUD dialog_history (8 методов)
│           └── notes.py          # CRUD user_notes (8 методов)
├── tests/
│   ├── conftest.py          # общие фикстуры (моки Supabase, Chroma, FastAPI)
│   ├── unit/
│   │   ├── test_documents_repository.py
│   │   ├── test_history_repository.py
│   │   ├── test_notes_repository.py
│   │   ├── test_chroma_service.py
│   │   └── test_rag_service.py
│   └── integration/
│       └── test_api_documents.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml           # pytest config + coverage
├── README.md
├── CASE_HH.md
└── RAG_INTERVIEW_CHEATSHEET.md
```

---

## 📖 API документация

После запуска: **http://localhost:8000/docs**

| Метод | Путь | Auth | Описание |
|-------|------|------|----------|
| `GET` | `/health` | — | Статус всех сервисов |
| `POST` | `/api/v1/documents/upload` | ✅ | Загрузить документ |
| `GET` | `/api/v1/documents/` | ✅ | Список документов |
| `GET` | `/api/v1/documents/{id}/status` | ✅ | Статус индексации |
| `DELETE` | `/api/v1/documents/{id}` | ✅ | Удалить документ |
| `GET` | `/api/v1/collections/` | ✅ | Коллекции ChromaDB |
| `POST` | `/api/v1/ask` | ✅ | RAG-запрос (тест) |

✅ = требует HTTP Basic Auth (admin / пароль из `.env`)

---

## 🔧 Конфигурация прокси

Для работы из России через Karing (SOCKS5, порт 3067):

```env
HTTP_PROXY=socks5://host.docker.internal:3067
HTTPS_PROXY=socks5://host.docker.internal:3067
```

**aiogram бот** использует `aiohttp-socks`:
```python
connector = ProxyConnector.from_url(settings.http_proxy)
session = AiohttpSession(connector=connector)
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN, session=session)
```

**OpenAI SDK** — прокси через `httpx.AsyncClient`:
```python
client = ChatOpenAI(
    http_async_client=httpx.AsyncClient(proxy=settings.https_proxy)
)
```

---

## 🧪 Тесты

```bash
# Установить зависимости
pip install -r requirements/test.txt

# Запустить все тесты
pytest

# С покрытием
pytest --cov=src --cov-report=term-missing

# Только unit-тесты (быстро, без Docker)
pytest tests/unit/ -v

# Только integration
pytest tests/integration/ -v
```

**49 тестов** покрывают:
- `DocumentsRepository` — 14 тестов (CRUD, статусы, ошибки)
- `HistoryRepository` — 11 тестов (save_pair, get_as_langchain_pairs, delete)
- `NotesRepository` — 13 тестов (CRUD, search, ownership)
- `ChromaService` — 13 тестов (батчинг, поиск, фильтры, удаление)
- `RAGService` — 11 тестов (condense, pipeline, mmr/similarity, streaming)
- `FastAPI endpoints` — 17 тестов (auth, upload, status, delete, health)

---

## 🗄 Схема базы данных

```sql
-- Метаданные документов и статус индексации
documents (
  id UUID PK, filename TEXT, file_type TEXT,
  status TEXT,   -- pending | processing | done | error
  chunk_count INT, chroma_ids JSONB, error_msg TEXT,
  created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
)

-- История диалогов (для контекста в RAG)
dialog_history (
  id UUID PK, user_id BIGINT, username TEXT,
  role TEXT,    -- user | assistant
  content TEXT, metadata JSONB, created_at TIMESTAMPTZ
)

-- Персональные заметки пользователей
user_notes (
  id UUID PK, user_id BIGINT, title TEXT,
  content TEXT, source_document TEXT, created_at TIMESTAMPTZ
)
```

---

## 🤝 Лицензия

MIT © [Kirill Tomenko](https://github.com/KirillTomenko)
