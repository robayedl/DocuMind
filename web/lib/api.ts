const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Doc {
  doc_id: string;
  filename: string;
  uploaded_at: string;
  indexed: boolean;
  index_time_s?: number;
}

export interface UploadResponse {
  doc_id: string;
  filename: string;
  stored_path: string;
}

export interface IndexResponse {
  doc_id: string;
  chunks_indexed: number;
  collection: string;
}

export interface Citation {
  ref: string;
  page: number;
  chunk_id: number;
  source: string;
  text?: string;
}

export interface ChatParams {
  doc_id: string;
  question: string;
  session_id?: string;
}

export async function listDocs(): Promise<Doc[]> {
  const res = await fetch(`${API_URL}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function uploadDoc(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function deleteDoc(docId: string): Promise<void> {
  const res = await fetch(`${API_URL}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete document");
}

export function indexDocStream(
  docId: string,
  onStatus: (msg: string) => void,
  onDone: (chunks: number, index_time_s: number) => void,
  onError: (err: string) => void,
  signal?: AbortSignal
): void {
  fetch(`${API_URL}/documents/${docId}/index/stream`, { method: "POST", signal })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const match = part.match(/^event: (\w+)\ndata: ([\s\S]*)$/);
          if (!match) continue;
          const [, event, data] = match;
          if (event === "status") onStatus(data);
          else if (event === "done") {
            try {
              const { chunks, index_time_s } = JSON.parse(data);
              onDone(chunks, index_time_s);
            } catch { onDone(0, 0); }
          }
          else if (event === "error") onError(data);
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(String(err));
    });
}

export function getDocFileUrl(docId: string): string {
  return `${API_URL}/documents/${docId}/file`;
}

export function chat(
  params: ChatParams,
  onStatus: (msg: string) => void,
  onToken: (token: string) => void,
  onCitations: (citations: Citation[]) => void,
  onDone: (duration_ms: number) => void,
  onError: (err: string) => void,
  signal?: AbortSignal
): void {
  const startTime = Date.now();
  fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";

        for (const part of parts) {
          const match = part.match(/^event: (\w+)\ndata: ([\s\S]*)$/);
          if (!match) continue;
          const [, event, data] = match;
          if (event === "status") onStatus(data);
          else if (event === "token") onToken(data);
          else if (event === "citations") {
            try { onCitations(JSON.parse(data)); } catch { /* ignore */ }
          }
          else if (event === "done") onDone(Date.now() - startTime);
          else if (event === "error") onError(data);
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(String(err));
    });
}
