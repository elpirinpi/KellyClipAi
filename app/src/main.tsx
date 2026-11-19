import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import './styles.css';

type ErrorBoundaryProps = { children: React.ReactNode };
type ErrorBoundaryState = { hasError: boolean };

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: unknown): void {
    console.error('Error no controlado en la interfaz de kellyClipAi:', error);
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <main className="error-boundary-page">
          <section className="error-boundary-card">
            <h1>Algo salió mal</h1>
            <p>La aplicación encontró un error inesperado. Tus proyectos y archivos siguen guardados en este dispositivo.</p>
            <div className="error-boundary-actions">
              <button type="button" className="create-button" onClick={() => window.location.reload()}>Recargar</button>
              <button type="button" className="subtle-button" onClick={() => window.location.assign('/')}>Volver al inicio</button>
            </div>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </BrowserRouter>
  </React.StrictMode>,
);
