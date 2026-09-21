// Minimal static server for local preview: node src/serve.mjs [port]
import { createServer } from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const PORT = Number(process.argv[2] || process.env.PORT || 4173);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

createServer(async (req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const candidates = url.endsWith('/')
    ? [path.join(ROOT, url, 'index.html')]
    : [path.join(ROOT, url), path.join(ROOT, url, 'index.html')];

  for (const file of candidates) {
    if (!path.resolve(file).startsWith(ROOT)) break;
    try {
      const body = await fs.readFile(file);
      res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' });
      return res.end(body);
    } catch { /* try the next candidate */ }
  }

  try {
    const body = await fs.readFile(path.join(ROOT, '404.html'));
    res.writeHead(404, { 'content-type': TYPES['.html'] });
    return res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('Not found');
  }
}).listen(PORT, () => console.log(`serving dist/ on http://localhost:${PORT}`));
