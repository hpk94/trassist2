/* Minimal API client mapping backend endpoints */
(function(global){
  function getApiBase(){
    return global.API_BASE || global.location.origin;
  }

  async function handleJson(response){
    if (!response.ok) {
      let detail = '';
      try { const j = await response.json(); detail = j.detail || j.error || JSON.stringify(j); } catch {}
      throw new Error(`HTTP ${response.status} ${response.statusText}${detail ? ` - ${detail}` : ''}`);
    }
    return response.json();
  }

  function ApiClient(){
    const base = getApiBase();
    return {
      health: async () => handleJson(await fetch(`${base}/health`)),
      upload: async (file) => {
        const form = new FormData();
        form.append('file', file);
        return handleJson(await fetch(`${base}/upload`, { method: 'POST', body: form }));
      },
      getAnalysis: async (id) => handleJson(await fetch(`${base}/analysis/${id}`)),
      reanalyze: async (id) => handleJson(await fetch(`${base}/analysis/${id}/reanalyze`, { method: 'POST' })),
      listAnalyses: async () => handleJson(await fetch(`${base}/analyses`)),
      deleteAnalysis: async (id) => handleJson(await fetch(`${base}/analysis/${id}`, { method: 'DELETE' })),
      listUploads: async () => handleJson(await fetch(`${base}/uploads`)),
    };
  }

  global.TrassistApi = { ApiClient, getApiBase };
})(window);


