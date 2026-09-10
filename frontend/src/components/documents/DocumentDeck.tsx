import React, { useState } from 'react';
import { Upload, Loader2, Trash2 } from 'lucide-react';

export interface DocumentRecord {
  doc_id: string;
  doc_type: string;
  face_detected: boolean;
  confidence: number;
  sharpness: number;
  face_crop_url?: string;
  message?: string;
}

interface DocumentDeckProps {
  sessionId: string;
  documents: DocumentRecord[];
  onDocumentUploaded: (doc: DocumentRecord) => void;
  onDocumentDeleted?: (docId: string) => void;
}

export const DocumentDeck: React.FC<DocumentDeckProps> = ({
  sessionId,
  documents,
  onDocumentUploaded,
  onDocumentDeleted
}) => {
  const [uploading, setUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [selectedDocType, setSelectedDocType] = useState('passport');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleDeleteDocument = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!sessionId || deletingId) return;

    setDeletingId(docId);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/verification/${sessionId}/documents/${encodeURIComponent(docId)}`, {
        method: 'DELETE'
      });
      if (res.ok || res.status === 404) {
        onDocumentDeleted?.(docId);
      } else {
        const data = await res.json().catch(() => ({}));
        setErrorMsg(data.detail || 'Failed to delete document');
      }
    } catch (err: any) {
      // Still allow UI removal if network fails locally
      onDocumentDeleted?.(docId);
    } finally {
      setDeletingId(null);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !sessionId) return;

    setUploading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', selectedDocType);
    formData.append('doc_id', `${selectedDocType.toUpperCase()}_${Date.now().toString().slice(-4)}`);

    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/verification/${sessionId}/documents`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        onDocumentUploaded(data);
      } else {
        setErrorMsg(data.message || 'Document portrait extraction rejected');
      }
    } catch (err: any) {
      setErrorMsg(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  return (
    <div className="document-deck-card">
      <div className="panel-title-bar">
        <div className="panel-title">
          <span className="panel-title-dot" />
          <span>IDENTITY DOCUMENT DECK</span>
        </div>
        <span className="panel-count">ENCODED: {documents.length}</span>
      </div>

      <div className="document-upload-row">
        <select
          value={selectedDocType}
          onChange={(e) => setSelectedDocType(e.target.value)}
          className="doc-type-select"
        >
          <option value="passport">PASSPORT</option>
          <option value="national_id">NATIONAL ID</option>
          <option value="visa">VISA PHOTO</option>
          <option value="residence_permit">RESIDENCE PERMIT</option>
        </select>

        <label className="btn-upload">
          {uploading ? (
            <>
              <Loader2 size={12} className="fscan-icon-spin" />
              <span>ENCODING...</span>
            </>
          ) : (
            <>
              <Upload size={12} />
              <span>UPLOAD DOCUMENT</span>
            </>
          )}
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            disabled={uploading}
            onChange={handleFileUpload}
          />
        </label>
      </div>

      {errorMsg && (
        <div style={{
          padding: '6px 8px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#EF4444',
          borderRadius: '3px',
          fontSize: '9px',
          marginBottom: '8px'
        }}>
          {errorMsg}
        </div>
      )}

      <div className="document-items-grid">
        {documents.map((doc, idx) => (
          <div key={idx} className="document-item">
            {doc.face_crop_url ? (
              <img
                src={doc.face_crop_url}
                alt={doc.doc_type}
                className="document-thumb"
              />
            ) : (
              <div className="document-thumb" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#54657E', fontSize: '8px' }}>
                NO FACE
              </div>
            )}
            <div className="document-info">
              <span className="doc-name">{doc.doc_type}</span>
              <span style={{ color: '#54657E', fontSize: '8px' }}>{doc.doc_id}</span>
              <div className="doc-metrics">
                <span>CONF: {(doc.confidence * 100).toFixed(1)}%</span>
                <span>SHARP: {doc.sharpness}</span>
              </div>
            </div>
            <button
              type="button"
              className="btn-delete-doc"
              onClick={(e) => handleDeleteDocument(doc.doc_id, e)}
              disabled={deletingId === doc.doc_id}
              title={`Delete ${doc.doc_type} (${doc.doc_id})`}
              aria-label={`Delete ${doc.doc_type}`}
            >
              {deletingId === doc.doc_id ? (
                <Loader2 size={12} className="fscan-icon-spin" style={{ color: '#EF4444' }} />
              ) : (
                <Trash2 size={13} />
              )}
            </button>
          </div>
        ))}

        {documents.length === 0 && (
          <div className="doc-empty-state">
            Attach Passport, ID, or Visa to enable multi-frame face matching.
          </div>
        )}
      </div>
    </div>
  );
};
