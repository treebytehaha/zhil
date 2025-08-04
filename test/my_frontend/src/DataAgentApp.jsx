import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FiPlus,
  FiTrash2,
  FiSend,
  FiUpload,
  FiPaperclip,
  FiFileText,
  FiX,
  FiRefreshCw,
} from "react-icons/fi";

/* ------------------------------------------------------------------ */

export default function DataAgentApp() {
  /* ---------- state ---------- */
  const [sessions, setSessions] = useState([]);          // [{id,name}]
  const [currentSession, setCurrentSession] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [docs, setDocs] = useState([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [showDocs, setShowDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const chatBottomRef = useRef(null);

  /* ---------- axios ---------- */
  const api = axios.create({ /* baseURL 可配置 */ });

  /* ---------- init ---------- */
  useEffect(() => { loadSessions(); }, []);
  useEffect(() => {
    if (!currentSession) { setMessages([]); setDocs([]); return; }
    loadMessages();  loadDocs();
  }, [currentSession]);
  useEffect(() => { chatBottomRef.current?.scrollIntoView({ behavior: "smooth" }); },
            [messages, pending]);

  /* ---------- sessions ---------- */
  async function loadSessions() {
    try {
      const res = await api.get("/api/chat/sessions");
      const list = res.data.sessions || [];            // [{id,name}]
      setSessions(list);
      if (!currentSession && list.length) setCurrentSession(list[0].id);
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
  }

  async function createSession() {
    try {
      const { data } = await api.post("/api/chat/sessions"); // {session_id,name}
      await loadSessions();
      setCurrentSession(data.session_id);
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
  }

  async function deleteSession(id) {
    if (!confirm("确定删除该会话及其所有内容？")) return;
    try {
      await api.delete(`/api/chat/sessions/${id}`);
      if (id === currentSession) setCurrentSession(null);
      await loadSessions();
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
  }

  async function renameSession(id, name) {
    // 本地立即更新
    setSessions(s => s.map(v => v.id === id ? { ...v, name } : v));
    try {
      await api.put(`/api/chat/sessions/${id}/name`, { name });
    } catch {
      /* 后端失败时可回滚或提示，这里简单忽略 */
    }
  }

  /* ---------- messages & docs ---------- */
  async function loadMessages() {
    try {
      const { data } = await api.get(
        `/api/chat/sessions/${currentSession}/messages?offset=0&limit=500`
      );
      setMessages(data.messages || []);
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
  }

  async function loadDocs() {
    try {
      const { data } = await api.get(`/api/chat/sessions/${currentSession}/docs`);
      setDocs(data.docs || []);
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
  }

  /* ---------- 文件上传 ---------- */
  const onPickFiles = (e) => setFiles(Array.from(e.target.files || []));
  async function uploadDocs() {
    if (!currentSession || files.length === 0) return;
    setUploading(true); setError("");
    try {
      for (const f of files) {
        const form = new FormData();  form.append("file", f);
        await api.post(`/api/chat/sessions/${currentSession}/docs`, form, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      }
      setFiles([]); await loadDocs();
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
    finally { setUploading(false); }
  }

  /* ---------- 发送消息 ---------- */
  async function sendMessage() {
    if (!currentSession) return setError("请先创建或选择会话");
    const text = input.trim(); if (!text) return;
    setError(""); setPending(true);
    const localUserMsg = { role: "user", content: text };
    setMessages(prev => [...prev, localUserMsg]);
    setInput("");

    try {
      await api.post(`/api/chat/sessions/${currentSession}/messages`, localUserMsg);
      const { data } = await api.post(
        `/api/chat/sessions/${currentSession}/analyze`, { command: text }
      );
      const assistantMsg = {
        role: "assistant",
        content: data.result ?? "",
        refs: data.refs || null,
      };
      await api.post(`/api/chat/sessions/${currentSession}/messages`, assistantMsg);
      await loadMessages();
    } catch (e) { setError(e?.response?.data?.detail || e.message); }
    finally { setPending(false); }
  }

  /* ------------------------------------------------------------------ */

  const current = sessions.find(s => s.id === currentSession);

  return (
    <div className="h-screen w-screen overflow-hidden flex">

      {/* ------------ 左侧栏 ------------ */}
      <aside className="hidden md:flex md:w-72 lg:w-80 flex-col border-r border-gray-200 bg-white">
        <div className="px-4 py-4 border-b">
          <div className="text-lg font-semibold">会话</div>
          <button onClick={createSession}
            className="mt-3 inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-700">
            <FiPlus /> 新建会话
          </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scroll px-2 py-2 space-y-1">
          {sessions.length === 0 && (
            <div className="text-sm text-gray-500 px-2">暂无会话，点击“新建会话”。</div>
          )}

          {sessions.map(({ id, name }) => {
            const active = id === currentSession;
            const editing = id === editingId;

            return (
              <div key={id}
                className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm cursor-pointer ${
                  active ? "bg-brand-50 text-brand-700"
                         : "hover:bg-gray-50 text-gray-700"
                }`}
                onClick={() => setCurrentSession(id)}>

                {/* 名称 or 输入框 */}
                {editing ? (
                  <input
                    autoFocus
                    value={name}
                    onChange={e => renameSession(id, e.target.value)}
                    onBlur={() => setEditingId(null)}
                    onKeyDown={e => e.key === "Enter" && setEditingId(null)}
                    className="w-full bg-transparent outline-none truncate"
                  />
                ) : (
                  <div
                    onDoubleClick={e => { e.stopPropagation(); setEditingId(id);} }
                    className="truncate"
                  >
                    {name}
                  </div>
                )}

                {/* 删除 */}
                <button
                  className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-600"
                  onClick={e => { e.stopPropagation(); deleteSession(id); }}
                  title="删除会话">
                  <FiTrash2 />
                </button>
              </div>
            );
          })}
        </div>

        <div className="px-4 py-3 text-xs text-gray-400 border-t">
          数据智能对话台 © {new Date().getFullYear()}
        </div>
      </aside>

      {/* ------------ 主体 ------------ */}
      <main className="flex-1 flex flex-col">
        {/* 顶部栏 */}
        <div className="h-14 flex items-center justify-between border-b bg-white px-4">
          <div className="flex items-center gap-3">
            <div className="font-semibold">
              {current ? current.name : "请选择或创建会话"}
            </div>
            <button className="text-gray-500 hover:text-gray-700"
              onClick={() => currentSession && loadMessages()}
              title="刷新消息"><FiRefreshCw /></button>
          </div>

          <button
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border ${
              showDocs ? "bg-gray-900 text-white" : "bg-white text-gray-700 hover:bg-gray-50"
            }`}
            onClick={() => setShowDocs(v => !v)}
          >
            <FiPaperclip /> 文档
          </button>
        </div>

        {/* 聊天区 */}
        <div className="flex-1 overflow-y-auto custom-scroll px-4 lg:px-8 py-6 bg-[linear-gradient(to_bottom,#f8fafc,transparent)]">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">这里将显示对话内容…</div>
          )}

          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((m, idx) => (
              <ChatBubble key={idx} role={m.role} content={m.content} refs={m.refs} />
            ))}

            {pending && (
              <div className="flex gap-2 items-center text-gray-500 text-sm">
                <span className="animate-pulse">助手思考中…</span>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>
        </div>

        {/* 底部输入 */}
        <div className="border-t bg-white p-3">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2">
              {/* 选文件 */}
              <label
                className="shrink-0 inline-flex items-center justify-center w-10 h-10 rounded-lg border hover:bg-gray-50 cursor-pointer"
                title="选择文件">
                <input type="file" className="hidden" multiple
                  onChange={onPickFiles} accept=".xlsx,.xls,.csv,.pdf,.txt" />
                <FiUpload />
              </label>

              {/* 文本框 */}
              <textarea rows={1} value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (!pending) sendMessage();
                  }
                }}
                placeholder="发送消息（Enter 发送，Shift+Enter 换行）"
                className="flex-1 resize-none rounded-xl border px-4 py-3 outline-none focus:ring-2 focus:ring-brand-600 custom-scroll"
              />

              {/* 发送 */}
              <button
                className={`shrink-0 inline-flex items-center justify-center w-12 h-12 rounded-xl ${
                  pending ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                          : "bg-brand-600 text-white hover:bg-brand-700"
                }`}
                onClick={sendMessage} disabled={pending} title="发送">
                <FiSend />
              </button>
            </div>

            {/* 待上传文件列表 */}
            <AnimatePresence>
              {files.length > 0 && (
                <motion.div initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 rounded-lg border bg-gray-50 p-3 text-sm">

                  <div className="font-medium mb-2">待上传文件</div>
                  <ul className="space-y-1 max-h-40 overflow-y-auto custom-scroll">
                    {files.map((f, i) => (
                      <li key={i} className="flex items-center gap-2">
                        <FiFileText className="text-gray-500" />
                        <span className="truncate">{f.name}</span>
                      </li>
                    ))}
                  </ul>

                  <div className="mt-3 flex gap-2">
                    <button onClick={() => setFiles([])}
                      className="px-3 py-2 rounded-lg border hover:bg-white">清空</button>
                    <button onClick={uploadDocs} disabled={uploading}
                      className={`px-3 py-2 rounded-lg ${
                        uploading ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                                   : "bg-gray-900 text-white hover:bg-black"
                      }`}>
                      {uploading ? "上传中…" : "上传"}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

            <div className="mt-3 text-center text-xs text-gray-400">
              模型会根据最近上传的 Excel / 文档进行统计分析。
            </div>
          </div>
        </div>
      </main>

      {/* ------------ 文档抽屉 ------------ */}
      <AnimatePresence>
        {showDocs && (
          <motion.aside initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 30 }}
            className="absolute right-0 top-0 h-full w-[360px] bg-white border-l shadow-xl z-40 flex flex-col">

            <div className="h-14 border-b flex items-center justify-between px-4">
              <div className="font-semibold">会话文档</div>
              <button className="text-gray-500 hover:text-gray-700"
                onClick={() => setShowDocs(false)}>
                <FiX />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scroll p-4 space-y-2">
              {docs.length === 0 && ( <div className="text-sm text-gray-500">暂无文档。</div> )}
              {docs.map(d => (
                <a key={d.doc_id}
                   className="group block rounded-lg border px-3 py-2 hover:bg-gray-50"
                   href={`/api/chat/sessions/${currentSession}/docs/${d.doc_id}`}
                   target="_blank" rel="noreferrer">
                  <div className="flex items-center gap-2">
                    <FiFileText className="text-gray-500" />
                    <span className="truncate text-sm">{d.filename}</span>
                  </div>
                </a>
              ))}
            </div>

            <div className="p-4 border-t">
              <label className="w-full inline-flex items-center gap-2 justify-center px-3 py-2 rounded-lg border hover:bg-gray-50 cursor-pointer">
                <input type="file" multiple className="hidden"
                  onChange={onPickFiles} accept=".xlsx,.xls,.csv,.pdf,.txt" />
                <FiUpload /> 选择文件
              </label>
              <button onClick={uploadDocs}
                disabled={uploading || files.length === 0}
                className={`mt-2 w-full px-3 py-2 rounded-lg ${
                  uploading || files.length === 0
                    ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                    : "bg-brand-600 text-white hover:bg-brand-700"
                }`}>
                {uploading ? "上传中…" : "上传文件"}
              </button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ----------------- 附件与气泡 ----------------- */

function ChatBubble({ role, content, refs }) {
  const isUser = role === "user";
  return (
    <div className={`flex items-start gap-3 ${isUser ? "justify-end" : ""}`}>
      {!isUser && <Avatar fallback="AI" className="bg-brand-600 text-white" />}
      <div
        className={`prose prose-sm max-w-none rounded-2xl px-4 py-3 border ${
          isUser
            ? "bg-brand-600 text-white border-brand-600 prose-invert"
            : "bg-white text-gray-800"
        }`} style={{ overflowWrap: "anywhere" }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>

        {!isUser && Array.isArray(refs) && refs.length > 0 && (
          <div className="mt-3 not-prose space-y-2">
            {refs.map((r, i) => (
              <details key={i} className="text-xs border rounded-md p-2">
                <summary className="cursor-pointer select-none">
                  参考结果 #{i + 1}{r.tool && ` · ${r.tool}`}
                </summary>
                <pre className="mt-2 whitespace-pre-wrap break-all text-[11px]">
                  {JSON.stringify(r.payload, null, 2)}
                </pre>
              </details>
            ))}
          </div>
        )}
      </div>
      {isUser && <Avatar fallback="U" className="bg-gray-800 text-white" />}
    </div>
  );
}

function Avatar({ fallback, className = "", title = "" }) {
  return (
    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold ${className}`}
         title={title}>
      {fallback}
    </div>
  );
}
