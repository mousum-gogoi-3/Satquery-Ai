import React, { useRef, useState } from 'react';
import { getImageUrl } from '../api';

/**
 * Reusable slot card for Before or After satellite imagery.
 * Supports upload, preview, replace, and remove.
 */
function SlotCard({
  slotKey,
  label,
  sublabel,
  image,
  isUploading,
  onUpload,
  onRemove,
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0 && onUpload) {
      onUpload(files[0]);
    }
    // Clear input value so selecting same file triggers change
    e.target.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0 && onUpload) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const triggerSelect = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // State A: Image loaded in this slot
  if (image) {
    const previewSrc = image.thumbnail_url 
      ? getImageUrl(image.thumbnail_url) 
      : image.previewUrl;

    const meta = image.metadata;
    const bandsCount = image.bands ? image.bands.length : (meta ? meta.band_count : null);

    return (
      <div className="upload-slot-wrapper">
        <div className="slot-badge-bar">
          <span className="slot-badge">{label}</span>
          {sublabel && <span className="slot-sublabel">{sublabel}</span>}
        </div>

        <div className="image-preview-card" id={`image-preview-${slotKey}`}>
          <div className="preview-image-container">
            <img 
              src={previewSrc} 
              alt={`${label} Satellite Raster`} 
              className="preview-image"
            />
          </div>

          <div className="preview-details">
            <div className="preview-header">
              <span className="preview-tag">{label} IMAGE</span>
              <div className="preview-actions">
                <button 
                  type="button" 
                  className="btn-change-image"
                  onClick={triggerSelect}
                  disabled={isUploading}
                  title={`Replace ${label} image`}
                >
                  {isUploading ? 'Uploading...' : 'Replace'}
                </button>
                {onRemove && (
                  <button 
                    type="button" 
                    className="btn-remove-image"
                    onClick={onRemove}
                    disabled={isUploading}
                    title={`Remove ${label} image`}
                  >
                    Remove
                  </button>
                )}
              </div>
            </div>

            <h3 className="preview-filename" title={image.filename}>
              {image.filename}
            </h3>

            <div className="preview-meta-chips">
              {meta && (
                <span className="meta-chip">
                  <strong>{meta.width}</strong> × <strong>{meta.height}</strong> px
                </span>
              )}
              {bandsCount && (
                <span className="meta-chip">
                  <strong>{bandsCount}</strong> band{bandsCount > 1 ? 's' : ''}
                </span>
              )}
              {meta && meta.crs && (
                <span className="meta-chip crs">
                  {meta.crs}
                </span>
              )}
            </div>
          </div>

          <input 
            ref={fileInputRef}
            type="file" 
            accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.jp2"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id={`file-input-${slotKey}`}
          />
        </div>
      </div>
    );
  }

  // State B: Empty dropzone for this slot
  return (
    <div className="upload-slot-wrapper">
      <div className="slot-badge-bar">
        <span className="slot-badge">{label}</span>
        {sublabel && <span className="slot-sublabel">{sublabel}</span>}
      </div>

      <div 
        className={`upload-dropzone slot-dropzone ${isDragOver ? 'dragover' : ''} ${isUploading ? 'uploading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerSelect}
        id={`upload-dropzone-${slotKey}`}
        role="button"
        tabIndex={0}
      >
        <input 
          ref={fileInputRef}
          type="file" 
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.jp2"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id={`raster-file-input-${slotKey}`}
        />

        <div className="dropzone-content">
          <div className="upload-icon-circle">
            {isUploading ? (
              <div className="spinner"></div>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            )}
          </div>

          <h3 className="upload-title">
            {isUploading ? `Ingesting ${label}...` : `Upload ${label} Image`}
          </h3>

          <p className="upload-subtitle">
            Drag and drop or <span className="browse-link">browse</span>
          </p>

          <p className="upload-formats">
            GeoTIFF, PNG, JPEG, JP2
          </p>
        </div>
      </div>
    </div>
  );
}

export default function ImageUploader({ 
  imageBefore,
  imageAfter,
  currentImage,
  onImageUploaded,
  onUploadBefore,
  onUploadAfter,
  onRemoveBefore,
  onRemoveAfter,
  isUploading = false,
  isUploadingBefore = false,
  isUploadingAfter = false,
  uploadError 
}) {
  const effectiveBefore = imageBefore || currentImage || null;
  const effectiveAfter = imageAfter || null;
  const handleBeforeUpload = onUploadBefore || onImageUploaded;
  const handleAfterUpload = onUploadAfter;

  return (
    <div className="uploader-section" id="two-image-uploader">
      <div className="uploader-grid">
        <SlotCard
          slotKey="before"
          label="BEFORE"
          sublabel="Earlier Observation / Image 1"
          image={effectiveBefore}
          isUploading={isUploadingBefore || (isUploading && !effectiveBefore)}
          onUpload={handleBeforeUpload}
          onRemove={onRemoveBefore}
        />

        <SlotCard
          slotKey="after"
          label="AFTER"
          sublabel="Later Observation / Image 2 (for Change Detection)"
          image={effectiveAfter}
          isUploading={isUploadingAfter}
          onUpload={handleAfterUpload}
          onRemove={onRemoveAfter}
        />
      </div>

      {uploadError && (
        <div className="alert-box error-alert" id="upload-error-alert">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{uploadError}</span>
        </div>
      )}
    </div>
  );
}
