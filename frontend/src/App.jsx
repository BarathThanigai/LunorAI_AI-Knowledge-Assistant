import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

const getFileExtension = (filename) => {
  const lastDot = filename.lastIndexOf(".");
  return lastDot >= 0 ? filename.slice(lastDot).toLowerCase() : "";
};

const getFileTypeLabel = (filename) => {
  const extension = getFileExtension(filename);

  const labels = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
    ".md": "MD",
  };

  return labels[extension] || "DOC";
};

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/documents`);

      if (!response.ok) {
        throw new Error("Failed to load documents.");
      }

      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message || "Failed to load documents.");
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const askQuestion = async () => {
    if (!question.trim() || loading) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to generate answer."
        );
      }

      setAnswer(data.answer || "");
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message || "Failed to generate answer.");
    } finally {
      setLoading(false);
    }
  };

  const handleQuestionKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  };

  const uploadDocument = async (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    const extension = getFileExtension(file.name);

    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setError(
        "Unsupported file type. Please upload a PDF, DOCX, TXT, or Markdown file."
      );
      event.target.value = "";
      return;
    }

    setUploading(true);
    setError("");
    setAnswer("");
    setSources([]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_BASE}/api/documents/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Document upload failed."
        );
      }

      await fetchDocuments();
    } catch (err) {
      setError(err.message || "Document upload failed.");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const deleteDocument = async (documentId) => {
    setError("");

    try {
      const response = await fetch(
        `${API_BASE}/api/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to delete document."
        );
      }

      await fetchDocuments();

      setAnswer("");
      setSources([]);
    } catch (err) {
      setError(err.message || "Failed to delete document.");
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">✦</div>

          <div>
            <h1>LunorAI</h1>
            <span>Mini AI Knowledge Assistant</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Knowledge Base
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="eyebrow">YOUR KNOWLEDGE, SEARCHABLE</p>

          <h2>
            Ask your documents.
            <br />
            <span>Get grounded answers.</span>
          </h2>

          <p className="hero-description">
            Upload your documents and ask questions using a
            retrieval-augmented AI assistant.
          </p>
        </section>

        <section className="ask-card">
          <div className="question-label">
            Ask LunorAI
          </div>

          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            onKeyDown={handleQuestionKeyDown}
            placeholder="What would you like to know?"
            rows={4}
          />

          <div className="ask-footer">
            <span>Press Enter to ask</span>

            <button
              className="primary-button"
              onClick={askQuestion}
              disabled={!question.trim() || loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Thinking...
                </>
              ) : (
                <>
                  Ask LunorAI
                  <span>→</span>
                </>
              )}
            </button>
          </div>
        </section>

        {error && (
          <div className="error-message">
            <span>!</span>
            {error}
          </div>
        )}

        {answer && (
          <section className="answer-section">
            <div className="section-heading">
              <div>
                <p className="section-eyebrow">RESPONSE</p>
                <h3>Answer</h3>
              </div>
            </div>

            <div className="answer-card">
              <div className="answer-icon">✦</div>

              <div className="answer-text">
                {answer}
              </div>
            </div>

            {sources.length > 0 && (
              <div className="sources-section">
                <div className="sources-heading">
                  <h3>Sources</h3>
                  <span>{sources.length} retrieved</span>
                </div>

                <div className="sources-grid">
                  {sources.map((source, index) => (
                    <div
                      className="source-card"
                      key={`${source.document}-${source.page}-${index}`}
                    >
                      <div className="source-top">
                        <div className="pdf-icon">
                          {getFileTypeLabel(source.document)}
                        </div>

                        <span className="source-number">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                      </div>

                      <h4>{source.document}</h4>

                      <div className="source-meta">
                        {source.page !== null &&
                        source.page !== undefined ? (
                          <span>
                            Page {source.page}
                          </span>
                        ) : (
                          <span>
                            Document
                          </span>
                        )}

                        <span>
                          Score{" "}
                          {Number(source.score).toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        <section className="knowledge-section">
          <div className="section-heading knowledge-heading">
            <div>
              <p className="section-eyebrow">
                DOCUMENTS
              </p>

              <h3>Knowledge Base</h3>
            </div>

            <span className="document-count">
              {documents.length}{" "}
              {documents.length === 1
                ? "document"
                : "documents"}
            </span>
          </div>

          <div
            className="upload-zone"
            onClick={() =>
              !uploading && fileInputRef.current?.click()
            }
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={uploadDocument}
              disabled={uploading}
              hidden
            />

            <div className="upload-icon">
              ↑
            </div>

            <div>
              <strong>
                {uploading
                  ? "Processing document..."
                  : "Upload a Document"}
              </strong>

              <p>
                {uploading
                  ? "Extracting text and building embeddings"
                  : "PDF, DOCX, TXT, and Markdown supported"}
              </p>
            </div>
          </div>

          {documents.length > 0 && (
            <div className="document-list">
              {documents.map((document) => (
                <div
                  className="document-row"
                  key={document.id}
                >
                  <div className="document-info">
                    <div className="document-icon">
                      {getFileTypeLabel(document.filename)}
                    </div>

                    <div>
                      <strong>
                        {document.filename}
                      </strong>

                      <span>
                        Indexed in knowledge base
                      </span>
                    </div>
                  </div>

                  <button
                    className="delete-button"
                    onClick={() =>
                      deleteDocument(document.id)
                    }
                    title="Delete document"
                  >
                    🗑
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      <footer>
        <span>LunorAI</span>
        <span>Local RAG · FAISS · NVIDIA AI</span>
      </footer>
    </div>
  );
}

export default App;
