-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.

-- One row per candidate; the whole record (scores, brief, drafts, sent log) lives in `data`.
create table if not exists public.candidates (
  id         text primary key,
  data       jsonb not null,
  updated_at timestamptz not null default now()
);

-- Row Level Security on with no policies: the public/anon key can't read candidate data.
-- The app uses the secret key, which bypasses RLS.
alter table public.candidates enable row level security;

-- Private bucket for the uploaded resume files.
insert into storage.buckets (id, name, public)
values ('resumes', 'resumes', false)
on conflict (id) do nothing;
