import React from 'react';
import ResultImage from './ResultImage';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="chat-message user-message" id={`msg-${message.id}`}>
        <div className="message-bubble user-bubble">
          <p className="message-text">{message.text}</p>
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className={`chat-message assistant-message ${message.isError ? 'error-message' : ''}`} id={`msg-${message.id}`}>
      <div className="assistant-header">
        <div className="assistant-avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v4" />
            <path d="m4.93 4.93 2.83 2.83" />
            <path d="M2 12h4" />
            <path d="m4.93 19.07 2.83-2.83" />
            <path d="M12 22v-4" />
            <path d="m19.07 19.07-2.83-2.83" />
            <path d="M22 12h-4" />
            <path d="m19.07 4.93-2.83 2.83" />
          </svg>
        </div>
        <span className="assistant-name">SatQuery AI</span>
        
        {message.analysisType && (
          <span className={`meta-pill type-pill ${message.analysisType}`}>
            {message.analysisType === 'image_understanding' ? 'Scene Understanding' :
             message.analysisType === 'counting' ? 'Detection & Count' :
             message.analysisType === 'temporal_change' ? 'Change Analysis' :
             'Spatial Grounding'}
          </span>
        )}

        {message.confidence !== undefined && message.confidence !== null && (

          <span className="meta-pill confidence-pill">
            Confidence: {Math.round(message.confidence * 100)}%
          </span>
        )}

        {message.processingTime !== undefined && message.processingTime !== null && (
          <span className="meta-pill time-pill">
            {message.processingTime}s
          </span>
        )}
      </div>

      <div className="message-bubble assistant-bubble">
        <div className="assistant-answer">
          <p className="message-text">{message.answer || message.text}</p>
        </div>

        {/* Primary Evidence Image (Rendered directly in response card) */}
        {message.evidenceImagePath && (
          <ResultImage 
            imagePath={message.evidenceImagePath} 
            title="Visual Grounding Evidence"
            regions={message.evidenceRegions || []}
          />
        )}

        {/* Bi-Temporal Change Heatmap if applicable */}
        {message.changeHeatmapPath && (
          <ResultImage 
            imagePath={message.changeHeatmapPath} 
            title="Change Magnitude Heatmap"
          />
        )}
      </div>
    </div>
  );
}
