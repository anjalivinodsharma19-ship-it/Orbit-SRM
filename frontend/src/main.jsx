import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import logoUrl from '../logo.png';
import './styles.css';
import './styles/theme.css';
// The supplied OrbitSRM logo is the tab icon. It is referenced through Vite's asset
// pipeline so the same file resolves in dev (/logo.png) and in the production build
// (/assets/logo-<hash>.png) without copying or cropping the artwork.
const favicon=document.getElementById('favicon');
if(favicon)favicon.href=logoUrl;
createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
