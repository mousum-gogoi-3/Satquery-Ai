import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { 
  checkHealth, 
  uploadImage, 
  queryGrounded, 
  queryTemporalChange, 
  isTemporalChangeQuery 
} from './api';
import './App.css';

export default function App() {
  const [backendConnected, setBackendConnected] = useState(false);
  const [imageBefore, setImageBefore] = useState(null);
  const [imageAfter, setImageAfter] = useState(null);
  const [isUploadingBefore, setIsUploadingBefore] = useState(false);
  const [isUploadingAfter, setIsUploadingAfter] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Poll backend health status on mount & every 15 seconds
  useEffect(() => {
    let isMounted = true;
    const verifyHealth = async () => {
      const res = await checkHealth();
      if (isMounted) {
        setBackendConnected(res.ok);
      }
    };

    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Auto-scroll chat to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle uploading "Before" (Image 1)
  const handleUploadBefore = async (file) => {
    setIsUploadingBefore(true);
    setUploadError(null);

    try {
      const uploadResult = await uploadImage(file);
      setImageBefore(uploadResult);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          answer: `Before image "${uploadResult.filename}" uploaded successfully. You can now ask questions about this scene, or upload an After image for change analysis.`,
        }
      ]);
    } catch (err) {
      setUploadError(err.message || 'Failed to upload Before image.');
    } finally {
      setIsUploadingBefore(false);
    }
  };

  // Handle uploading "After" (Image 2)
  const handleUploadAfter = async (file) => {
    setIsUploadingAfter(true);
    setUploadError(null);

    try {
      const uploadResult = await uploadImage(file);
      setImageAfter(uploadResult);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          answer: `After image "${uploadResult.filename}" uploaded successfully. Both images are now active for bi-temporal change detection or individual inspection.`,
        }
      ]);
    } catch (err) {
      setUploadError(err.message || 'Failed to upload After image.');
    } finally {
      setIsUploadingAfter(false);
    }
  };

  // Remove individual slot images
  const handleRemoveBefore = () => {
    setImageBefore(null);
  };

  const handleRemoveAfter = () => {
    setImageAfter(null);
  };

  // Reset entire session
  const handleReset = () => {
    setImageBefore(null);
    setImageAfter(null);
    setMessages([]);
    setUploadError(null);
  };

  // Handle sending a natural-language query to SatQuery backend
  const handleSendMessage = async (queryText) => {
    const hasImage = Boolean(imageBefore || imageAfter);
    if (!hasImage || isLoading || !queryText.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      text: queryText,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const hasBoth = Boolean(imageBefore && imageAfter);
      const isChange = isTemporalChangeQuery(queryText);

      // ----------------------------------------------------------------------
      // CASE 1: Two images uploaded AND query is a temporal change query
      // ----------------------------------------------------------------------
      if (hasBoth && isChange) {
        const data = await queryTemporalChange(
          imageBefore.file_id,
          imageAfter.file_id,
          queryText
        );

        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          answer: data.summary || 'Bi-temporal change analysis complete.',
          analysisType: 'temporal_change',
          evidenceImagePath: data.evidence_image_path || null,
          changeHeatmapPath: data.change_heatmap_path || null,
          evidenceRegions: data.change_regions || [],
          confidence: data.dominant_change_type ? 0.95 : 0.85,
          processingTime: data.processing_time_sec !== undefined ? data.processing_time_sec : null,
        };
        setMessages((prev) => [...prev, assistantMessage]);
        return;
      }

      // ----------------------------------------------------------------------
      // CASE 2: Two images uploaded, but user asked a normal single-image query
      // ----------------------------------------------------------------------
      if (hasBoth && !isChange) {
        const qLower = queryText.toLowerCase();
        const specifiesBefore = /\b(before|first|image\s*1|image\s*a)\b/.test(qLower);
        const specifiesAfter = /\b(after|second|image\s*2|image\s*b)\b/.test(qLower);

        let targetFileId = null;
        if (specifiesBefore && !specifiesAfter) {
          targetFileId = imageBefore.file_id;
        } else if (specifiesAfter && !specifiesBefore) {
          targetFileId = imageAfter.file_id;
        } else {
          // Cannot determine which image the user means
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              role: 'assistant',
              answer: "You have two images loaded (Before and After). Please specify which image you'd like to inspect (e.g. 'in the before image' or 'in the second image'), or ask a change detection question (e.g. 'What changed?').",
            }
          ]);
          return;
        }

        const data = await queryGrounded(targetFileId, queryText);
        const answerText = data.answer || data.summary || 'Analysis complete.';
        const evidencePath = data.evidence_image_path || null;
        const regions = data.evidence_regions || [];
        const confidence = data.overall_confidence !== undefined ? data.overall_confidence : data.confidence;
        const processingTime = data.processing_time_sec !== undefined ? data.processing_time_sec : data.processing_time;

        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          answer: answerText,
          analysisType: data.analysis_type || 'grounding',
          evidenceImagePath: evidencePath,
          evidenceRegions: regions,
          confidence: confidence,
          processingTime: processingTime,
        };
        setMessages((prev) => [...prev, assistantMessage]);
        return;
      }

      // ----------------------------------------------------------------------
      // CASE 3: Only ONE image is uploaded (either Before or After)
      // ----------------------------------------------------------------------
      const targetImage = imageBefore || imageAfter;
      const data = await queryGrounded(targetImage.file_id, queryText);

      const answerText = data.answer || data.summary || 'Analysis complete.';
      const evidencePath = data.evidence_image_path || null;
      const regions = data.evidence_regions || [];
      const confidence = data.overall_confidence !== undefined ? data.overall_confidence : data.confidence;
      const processingTime = data.processing_time_sec !== undefined ? data.processing_time_sec : data.processing_time;

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        answer: answerText,
        analysisType: data.analysis_type || 'grounding',
        evidenceImagePath: evidencePath,
        evidenceRegions: regions,
        confidence: confidence,
        processingTime: processingTime,
      };
      setMessages((prev) => [...prev, assistantMessage]);

    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        isError: true,
        answer: `Error: ${err.message || 'Failed to process query on SatQuery backend.'}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const hasAnyImage = Boolean(imageBefore || imageAfter);
  const hasBothImages = Boolean(imageBefore && imageAfter);

  return (
    <div className="app-layout">
      <Header 
        backendConnected={backendConnected} 
        onReset={handleReset}
        hasImage={hasAnyImage}
      />

      <main className="main-content">
        <div className="content-container">
          {/* Top Panel: Satellite Image Uploader & Preview (Two slots) */}
          <section className="image-section">
            <ImageUploader 
              imageBefore={imageBefore}
              imageAfter={imageAfter}
              onUploadBefore={handleUploadBefore}
              onUploadAfter={handleUploadAfter}
              onRemoveBefore={handleRemoveBefore}
              onRemoveAfter={handleRemoveAfter}
              isUploadingBefore={isUploadingBefore}
              isUploadingAfter={isUploadingAfter}
              uploadError={uploadError}
            />
          </section>

          {/* Bottom Panel: Chat and Results Feed */}
          {hasAnyImage && (
            <section className="chat-section">
              <div className="messages-feed" id="messages-feed">
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}
                <div ref={messagesEndRef} />
              </div>

              <ChatInput 
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                hasImage={hasAnyImage}
                hasBothImages={hasBothImages}
              />
            </section>
          )}

          {/* Placeholder state when no image is loaded yet */}
          {!hasAnyImage && !isUploadingBefore && !isUploadingAfter && (
            <div className="empty-guide-card">
              <div className="guide-step">
                <div className="step-num">1</div>
                <div>
                  <h4>Upload Satellite Scene(s)</h4>
                  <p>Upload a single image for VQA, captioning, or grounding — or two observations (Before & After) for change detection.</p>
                </div>
              </div>
              <div className="guide-step">
                <div className="step-num">2</div>
                <div>
                  <h4>Ask Natural-Language Queries</h4>
                  <p>Inquire about features ("Where is the building?"), counts ("How many buildings?"), or comparisons ("What changed?").</p>
                </div>
              </div>
              <div className="guide-step">
                <div className="step-num">3</div>
                <div>
                  <h4>Inspect Visual Evidence Inline</h4>
                  <p>Answers, spatial bounding boxes, and bi-temporal difference heatmaps appear directly in your chat cards.</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
