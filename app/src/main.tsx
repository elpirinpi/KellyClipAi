import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { API_BASE } from './context/constants';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import './styles.css';

type ErrorBoundaryProps = { children: React.ReactNode };
type ErrorBoundaryState = {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  copied: boolean;
};

// v1.4.3: el ErrorBoundary ya no se traga el error. Antes solo hacía
// console.error — invisible en la app empaquetada (WebView2 con
// debug=False) — y la pantalla "Algo salió mal" no mostraba nada,
// así que un crash era indiagnosticable (eso pasó con el crash del
// clip manual del 22-11-2026). Ahora la pantalla muestra el mensaje
// y el stack (colapsado), con un botón para copiarlo, y el error se
// envía al backend (POST /api/client-errors) para que quede en
// desktop.log. El envío es fire-and-forget: si falla, la pantalla
// sigue mostrando el error y nada se rompe.
async function reportClientError(error: unknown, errorInfo: React.ErrorInfo | null): Promise<void> {
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack ?? '' : '';
  try {
    await fetch(`${API_BASE}/api/client-errors`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        stack,
        component_stack: errorInfo?.componentStack ?? '',
        url: window.location.href,
        user_agent: navigator.userAgent,
      }),
      keepalive: true,
    });
  } catch {
    // El reporte nunca debe bloquear la UI ni romper si falla.
  }
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null, errorInfo: null, copied: false };

  static getDerivedStateFromError(error: unknown): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }

  componentDidCatch(error: unknown, errorInfo: React.ErrorInfo): void {
    console.error('Error no controlado en la interfaz de kellyClipAi:', error);
    this.setState({ errorInfo });
    void reportClientError(error, errorInfo);
  }

  private errorText(): string {
    const { error, errorInfo } = this.state;
    const parts = [error ? `${error.name}: ${error.message}` : 'Error desconocido'];
    if (error?.stack) parts.push(error.stack);
    if (errorInfo?.componentStack) parts.push(`Componentes:${errorInfo.componentStack}`);
    return parts.join('\n\n');
  }

  private handleCopy = async (): Promise<void> => {
    const text = this.errorText();
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Respaldo para contextos sin portapapeles asíncrono.
      try {
        const area = document.createElement('textarea');
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        area.remove();
      } catch {
        return;
      }
    }
    this.setState({ copied: true });
  };

  render(): React.ReactNode {
    if (this.state.hasError) {
      const { error, errorInfo, copied } = this.state;
      const stack = error?.stack ?? '';
      const componentStack = errorInfo?.componentStack ?? '';
      const detail = [stack, componentStack ? `Componentes:${componentStack}` : ''].filter(Boolean).join('\n\n');
      return (
        <main className="error-boundary-page">
          <section className="error-boundary-card">
            <h1>Algo salió mal</h1>
            <p>La aplicación encontró un error inesperado. Tus proyectos y archivos siguen guardados en este dispositivo.</p>
            <div className="error-boundary-detail" role="alert">
              <strong>{error ? `${error.name}: ${error.message}` : 'Error desconocido'}</strong>
            </div>
            {detail && (
              <details className="error-boundary-details">
                <summary>Detalle técnico</summary>
                <pre className="error-boundary-stack">{detail}</pre>
              </details>
            )}
            <div className="error-boundary-actions">
              <button type="button" className="create-button" onClick={() => window.location.reload()}>Recargar</button>
              <button type="button" className="subtle-button" onClick={() => void this.handleCopy()}>{copied ? 'Copiado' : 'Copiar error'}</button>
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
