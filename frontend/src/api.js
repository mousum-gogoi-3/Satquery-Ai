/**
 * SatQuery AI - API Client Module
 * Connects frontend UI to FastAPI backend engine.
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Resolves an image path or URL returned by backend into a valid browser image source URL.
 * Never open evidence in a new tab; this helper ensures correct <img> loading.
 *
 * @param {string} imagePath - The path or URL from the API response
 * @returns {string|null} - Fully-qualified browser image URL
 */
export function getImageUrl(imagePath) {
  if (!imagePath) return null;

  // Already a full external URL
  if (imagePath.startsWith('http://') || imagePath.startsWith('https://') || imagePath.startsWith('blob:') || imagePath.startsWith('data:')) {
    return imagePath;
  }

  // Ensure leading slash
  const cleanPath = imagePath.startsWith('/') ? imagePath : `/${imagePath}`;
  return `${API_BASE_URL}${cleanPath}`;
}

/**
 * Check backend engine health status
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: 'Unable to connect to SatQuery AI backend.' };
  }
}

/**
 * Upload a satellite / remote sensing image to the backend
 * @param {File} file - The image file selected by the user
 */
export async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/raster/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = `Upload failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          detail = typeof errorData.detail === 'string' 
            ? errorData.detail 
            : errorData.detail.message || JSON.stringify(errorData.detail);
        }
      } catch (_) {}
      throw new Error(detail);
    }

    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('Unable to connect to SatQuery AI backend. Please verify the server is running.');
    }
    throw err;
  }
}

/**
 * Perform spatial grounding analysis (semantic answer + visual evidence bounding box image)
 * @param {string} fileId - The backend file_id of the uploaded raster
 * @param {string} query - The natural-language query/question
 */
export async function queryGrounded(fileId, query) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/analysis/grounded`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id: fileId,
        query: query.trim(),
      }),
    });

    if (!res.ok) {
      let detail = `Analysis failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          detail = typeof errorData.detail === 'string' 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
        }
      } catch (_) {}
      throw new Error(detail);
    }

    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('Unable to connect to SatQuery AI backend.');
    }
    throw err;
  }
}

/**
 * Perform standard VQA question-answering
 * @param {string} fileId - The backend file_id
 * @param {string} question - Question text
 */
export async function queryVQA(fileId, question) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/analysis/vqa`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id: fileId,
        question: question.trim(),
      }),
    });

    if (!res.ok) {
      let detail = `Query failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) detail = errorData.detail;
      } catch (_) {}
      throw new Error(detail);
    }

    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('Unable to connect to SatQuery AI backend.');
    }
    throw err;
  }
}

/**
 * Perform bi-temporal change analysis between two satellite images (Phase 4)
 * @param {string} fileIdBefore - File ID of the earlier 'Before' satellite raster
 * @param {string} fileIdAfter - File ID of the later 'After' satellite raster
 * @param {string} query - Natural-language query regarding changes
 */
export async function queryTemporalChange(fileIdBefore, fileIdAfter, query = 'What changed?') {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/analysis/change`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id_before: fileIdBefore,
        file_id_after: fileIdAfter,
        query: (query || 'What changed?').trim(),
      }),
    });

    if (!res.ok) {
      let detail = `Change analysis failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          detail = typeof errorData.detail === 'string' 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
        }
      } catch (_) {}
      throw new Error(detail);
    }

    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('Unable to connect to SatQuery AI backend.');
    }
    throw err;
  }
}

/**
 * Minimal intent detector to check if a query indicates bi-temporal change/comparison.
 * Aligns with backend router change patterns without duplicating complex logic.
 * @param {string} query - The natural language query
 * @returns {boolean}
 */
export function isTemporalChangeQuery(query) {
  if (!query) return false;
  const q = query.trim();
  const patterns = [
    /\bwhat\s+(has\s+)?changed\b/i,
    /\bchanges?(\s+between)?\b/i,
    /\bcompare\b/i,
    /\bbefore\s+and\s+after\b/i,
    /\btemporal\b/i,
    /\b(vegetation|water)\s+(loss|decrease|gain|increase|expansion|reduction|shrink)\b/i,
    /\bwhere\s+did\s+(vegetation|water|construction|change)\b/i,
    /\b(deforest|deforestation|flooding)\b/i,
    /\bnew\s+(buildings?|constructions?|structures?|roads?)\b/i,
    /\b(construction|change)\s+occur(red)?\b/i,
    /\bdifferences?(\s+between)?\b/i,
  ];
  return patterns.some((p) => p.test(q));
}

