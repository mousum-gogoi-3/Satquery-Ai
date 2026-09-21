import React, { useState } from 'react';

const SINGLE_SUGGESTIONS = [
  "Explain this image",
  "Where are the buildings?",
  "How many buildings are there?",
  "Where is the water?",
  "Where is the vegetation?",
];

const TEMPORAL_SUGGESTIONS = [
  "What changed?",
  "Where did vegetation decrease?",
  "Show me the new buildings",
  "What changed between these images?",
  "Where did construction occur?",
];

export default function ChatInput({ 
  onSendMessage, 
  isLoading, 
  hasImage, 
  hasBothImages = false,
  disabledReason 
}) {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading || !hasImage) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const handleSuggestionClick = (query) => {
    if (isLoading || !hasImage) return;
    onSendMessage(query);
  };

  const currentSuggestions = hasBothImages ? TEMPORAL_SUGGESTIONS : SINGLE_SUGGESTIONS;

  return (
    <div className="chat-input-wrapper">
      {/* Quick suggestions */}
      {hasImage && !isLoading && (
        <div className="suggestion-pills" id="query-suggestions">
          <span className="suggestion-label">
            {hasBothImages ? 'Comparison suggestions:' : 'Suggested queries:'}
          </span>
          {currentSuggestions.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className="suggestion-pill"
              onClick={() => handleSuggestionClick(s)}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Loading state indicator */}
      {isLoading && (
        <div className="analyzing-indicator" id="analyzing-state">
          <div className="pulse-dot"></div>
          <span>Analyzing satellite image with SatQuery AI...</span>
        </div>
      )}

      {/* Input bar */}
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-text-input"
          placeholder={
            hasImage 
              ? "Ask something about this satellite image... (e.g. 'Where is the building?')" 
              : "Please upload a satellite image first to ask queries..."
          }
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={!hasImage || isLoading}
          id="query-input"
          autoFocus={hasImage}
        />
        <button
          type="submit"
          className="btn-send"
          disabled={!hasImage || isLoading || !inputText.trim()}
          id="send-query-btn"
          title="Send query"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  );
}
