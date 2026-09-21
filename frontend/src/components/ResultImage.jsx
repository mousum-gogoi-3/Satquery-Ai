import React, { useState } from 'react';
import { getImageUrl } from '../api';

export default function ResultImage({ 
  imagePath, 
  title = "Visual Grounding Evidence", 
  regions = [],
  dimensions = null 
}) {
  const [hasError, setHasError] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  const fullUrl = getImageUrl(imagePath);

  if (!imagePath) return null;

  return (
    <div className="result-evidence-container" id="evidence-image-container">
      <div className="evidence-header">
        <div className="evidence-title-group">
          <span className="evidence-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
            {title}
          </span>
          {regions && regions.length > 0 && (
            <span className="evidence-count-badge">
              {regions.length} detected region{regions.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <button 
          type="button" 
          className="evidence-toggle-btn"
          onClick={() => setIsZoomed(!isZoomed)}
          title={isZoomed ? "Shrink view" : "Enlarge view inside page"}
        >
          {isZoomed ? 'Shrink' : 'Enlarge'}
        </button>
      </div>

      <div className={`evidence-image-wrapper ${isZoomed ? 'zoomed' : ''}`}>
        {hasError ? (
          <div className="image-error-fallback">
            <span>Unable to load visual evidence image from server</span>
          </div>
        ) : (
          <img 
            src={fullUrl} 
            alt={title} 
            className="evidence-image"
            onError={() => setHasError(true)}
            loading="lazy"
          />
        )}
      </div>

      {/* Detected regions list if available */}
      {regions && regions.length > 0 && (
        <div className="evidence-regions-list">
          {regions.slice(0, 6).map((reg, idx) => (
            <span key={idx} className="region-chip">
              <span className="region-label">{reg.label || 'Feature'}</span>
              <span className="region-conf">
                {reg.confidence ? `${Math.round(reg.confidence * 100)}%` : ''}
              </span>
            </span>
          ))}
          {regions.length > 6 && (
            <span className="region-chip more">+{regions.length - 6} more</span>
          )}
        </div>
      )}
    </div>
  );
}
