import type { Citation } from "./api";

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  duration_ms?: number;
  interrupted?: boolean;
}

export interface ChatSession {
  id: string;
  doc_id: string;
  doc_name: string;
  messages: StoredMessage[];
  created_at: string;
  updated_at: string;
}

const KEY = "documind_chats";

export function loadSessions(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function saveSessions(sessions: ChatSession[]): void {
  localStorage.setItem(KEY, JSON.stringify(sessions));
}

export function createSession(doc_id: string, doc_name: string): ChatSession {
  return {
    id: crypto.randomUUID(),
    doc_id,
    doc_name,
    messages: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

export function upsertSession(
  sessions: ChatSession[],
  updated: ChatSession
): ChatSession[] {
  const idx = sessions.findIndex((s) => s.id === updated.id);
  const next = idx >= 0
    ? sessions.map((s, i) => (i === idx ? updated : s))
    : [updated, ...sessions];
  saveSessions(next);
  return next;
}

export function deleteSession(
  sessions: ChatSession[],
  id: string
): ChatSession[] {
  const next = sessions.filter((s) => s.id !== id);
  saveSessions(next);
  return next;
}
