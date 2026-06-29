-- =============================================================
-- AI Knowledge Assistant — Supabase Schema
-- Выполни в Supabase SQL Editor
-- =============================================================

-- Расширение для UUID
create extension if not exists "uuid-ossp";

-- ── Документы (метаданные, статус индексации) ─────────────────
create table if not exists documents (
    id          uuid primary key default uuid_generate_v4(),
    filename    text not null,
    file_type   text not null check (file_type in ('pdf', 'docx', 'doc', 'txt')),
    file_size   bigint,
    status      text not null default 'pending'
                    check (status in ('pending', 'processing', 'done', 'error')),
    chunk_count integer default 0,
    error_msg   text,
    chroma_ids  jsonb,          -- массив chroma document IDs
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Автообновление updated_at
create or replace function update_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger documents_updated_at
    before update on documents
    for each row execute function update_updated_at();

-- ── История диалогов ─────────────────────────────────────────
create table if not exists dialog_history (
    id          uuid primary key default uuid_generate_v4(),
    user_id     bigint not null,       -- Telegram user ID
    username    text,
    role        text not null check (role in ('user', 'assistant')),
    content     text not null,
    metadata    jsonb default '{}',   -- source_docs, tokens, etc.
    created_at  timestamptz not null default now()
);

create index idx_dialog_history_user_id on dialog_history(user_id);
create index idx_dialog_history_created_at on dialog_history(created_at desc);

-- ── Персональные заметки ──────────────────────────────────────
create table if not exists user_notes (
    id              uuid primary key default uuid_generate_v4(),
    user_id         bigint not null,   -- Telegram user ID
    title           text,
    content         text not null,
    source_document text,              -- из какого документа (если сохранялось из ответа)
    created_at      timestamptz not null default now()
);

create index idx_user_notes_user_id on user_notes(user_id);

-- ── RLS (Row Level Security) — опционально для Supabase ───────
-- Раскомментируй если хочешь защиту на уровне строк
-- alter table dialog_history enable row level security;
-- alter table user_notes enable row level security;
