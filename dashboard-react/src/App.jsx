import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import DatasetDetail from './pages/DatasetDetail'
import Compare from './pages/Compare'
import RunExplorer from './pages/RunExplorer'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="dataset/:name" element={<DatasetDetail />} />
        <Route path="compare" element={<Compare />} />
        <Route path="run/:dataset/:strategy/:model" element={<RunExplorer />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
