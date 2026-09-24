import { Routes, Route } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { UploadDataset } from './pages/UploadDataset'
import { DatasetAnalysis } from './pages/DatasetAnalysis'
import { AgentExecution } from './pages/AgentExecution'
import { Experiments } from './pages/Experiments'
import { ModelComparison } from './pages/ModelComparison'
import { FinalReport } from './pages/FinalReport'
import { Layout } from './components/Layout'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<UploadDataset />} />
        <Route path="/dataset/:datasetId" element={<DatasetAnalysis />} />
        <Route path="/agent/:jobId" element={<AgentExecution />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/experiments/:jobId/compare" element={<ModelComparison />} />
        <Route path="/report/:jobId" element={<FinalReport />} />
      </Routes>
    </Layout>
  )
}

export default App