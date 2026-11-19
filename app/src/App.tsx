import { AppShell } from './components/AppShell';
import { AppProvider } from './context/AppContext';

// Raíz de composición (ítem 23 de la Fase 2): el estado y la lógica
// viven en los hooks de dominio y en AppProvider; el armazón (cabecera,
// banner de actualización, rutas, modales y pie) en AppShell.
export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}
