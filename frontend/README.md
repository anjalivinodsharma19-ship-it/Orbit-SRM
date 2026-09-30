# OrbitSRM frontend

Responsive React/Vite dashboard for the **OrbitSRM** satellite super-resolution API in
`../backend`.

## Brand assets

The official logo is `logo.png` in this folder (the 1975×796 transparent PNG supplied by
the project owner). It is used **unmodified** — never cropped, stretched, recolored, or
regenerated:

* **Sidebar lockup** — `src/App.jsx` imports it (`import logoUrl from '../logo.png'`), so
  Vite serves it from `/logo.png` in development and from a content-hashed
  `/assets/logo-<hash>.png` in the production build. The same import supplies the browser
  tab icon in `src/main.jsx`, which fills the `<link id="favicon" rel="icon">` placeholder
  declared in `index.html`.
* **Contrast plate** — the artwork is a dark-green and lime wordmark, which is not legible
  directly on the dark sidebar, so `src/styles/theme.css` places it on the theme's
  off-white surface (`.brand-logo-wrap`). The base light styles in `src/styles.css` need no
  plate. The plate changes the surrounding surface only; the logo files themselves are
  untouched.

There is deliberately no separate favicon file: creating one would require cropping the
wordmark out of the supplied logo.

## Run locally

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Set `VITE_API_URL` to the backend origin (default `http://127.0.0.1:8000`). The backend CORS configuration must allow the frontend origin (`http://localhost:5173` by default). Build with `npm run build`.

The dashboard reads jobs, health, and models from the API. Uploaded image metadata is remembered in this browser because the backend does not expose an imagery listing route. It is stored under the `orbitsrm.images` key; the previous `satquery.images` key is still read once for existing browsers and removed after the next upload, so no remembered list is lost. PNG/JPEG are uploadable as preview-only files; only GeoTIFF can start geospatial processing. API documentation opens the backend's real `/docs` route.
