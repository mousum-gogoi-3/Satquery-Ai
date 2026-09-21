import React from 'react';

export default function Header({ backendConnected, onReset, hasImage }) {
  return (
    <header className="header" id="app-header">
      <div className="header-left">
        <div className="logo-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            <path d="M2 12h20" />
          </svg>
        </div>
        <div className="brand-info">
          <h1 className="brand-title">SatQuery AI</h1>
          <span className="brand-subtitle">Multimodal Satellite & Remote-Sensing Vision Agent</span>
        </div>
      </div>

      <div className="header-right">
        <div className={`status-pill ${backendConnected ? 'online' : 'offline'}`} id="backend-status-badge">
          <span className="status-dot"></span>
          <span className="status-text">
            {backendConnected ? 'Backend Connected' : 'Backend Offline'}
          </span>
        </div>

        {hasImage && (
          <button 
            type="button" 
            className="btn-reset" 
            onClick={onReset}
            id="reset-session-btn"
            title="Start new analysis with a new image"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            <span>New Image</span>
          </button>
        )}
      </div>
    </header>
  );
}
