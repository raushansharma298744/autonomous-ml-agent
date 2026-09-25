import { useState } from 'react'
import { Upload, FileText, ArrowRight, CheckCircle, XCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import clsx from 'clsx'

export function UploadDataset() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [problemType, setProblemType] = useState('auto')
  const [isDragActive, setIsDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true)
    } else if (e.type === 'dragleave') {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a dataset file')
      return
    }
    if (!targetColumn) {
      setError('Please specify the target column')
      return
    }

    setUploading(true)
    setError('')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('target_column', targetColumn)
    formData.append('problem_type', problemType)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${apiUrl}/api/v1/datasets/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      if (data.job_id) {
        navigate(`/agent/${data.job_id}`)
      } else {
        setError(data.detail || 'Upload failed')
      }
    } catch (err) {
      setError('Failed to upload dataset')
    } finally {
      setUploading(false)
    }
  }

  const cardClass = "bg-white rounded-xl shadow-sm border border-gray-200"
  const inputClass = "w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
  const btnPrimaryClass = "px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
  const btnSecondaryClass = "px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Upload Dataset</h1>
        <p className="text-gray-600 mt-1">Upload a CSV file and specify the target column to start an autonomous ML workflow</p>
      </div>

      <form onSubmit={handleSubmit} className={`${cardClass} p-6 space-y-6`}>
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={clsx(
            'border-2 border-dashed rounded-xl p-8 text-center transition-colors',
            isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'
          )}>
          <input
            type="file"
            id="file-upload"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            disabled={uploading}
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            {file ? (
              <div className="flex items-center justify-center gap-3 text-green-600">
                <CheckCircle className="w-8 h-8" />
                <div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                <Upload className="w-12 h-12 text-gray-400" />
                <div>
                  <p className="text-lg font-medium text-gray-900">Drag & drop your CSV file here</p>
                  <p className="text-gray-500">or click to browse</p>
                </div>
                <p className="text-xs text-gray-400">Supports CSV files up to 100MB</p>
              </div>
            )}
          </label>
        </div>

        {error && (
          <div className="p-4 rounded-lg bg-red-50 border border-red-200 flex items-center gap-3 text-red-700">
            <XCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="target_column" className="block text-sm font-medium text-gray-700 mb-1">
              Target Column <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="target_column"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              placeholder="e.g., target, label, price, survived"
              className={inputClass}
              disabled={uploading}
              required
            />
            <p className="mt-1 text-sm text-gray-500">The column you want to predict</p>
          </div>

          <div>
            <label htmlFor="problem_type" className="block text-sm font-medium text-gray-700 mb-1">
              Problem Type
            </label>
            <select
              id="problem_type"
              value={problemType}
              onChange={(e) => setProblemType(e.target.value)}
              className={inputClass}
              disabled={uploading}
            >
              <option value="auto">Auto-detect</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
            <p className="mt-1 text-sm text-gray-500">Auto-detect based on target column</p>
          </div>
        </div>

        <div className="flex justify-end gap-4 pt-4 border-t border-gray-200">
          <button type="submit" className={btnPrimaryClass} disabled={uploading || !file}>
            {uploading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Uploading...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                Start Autonomous ML Workflow
                <ArrowRight className="w-4 h-4" />
              </span>
            )}
          </button>
        </div>
      </form>

      <div className={`${cardClass} p-6`}>
        <h3 className="font-semibold text-gray-900 mb-4">Sample Datasets</h3>
        <p className="text-gray-600 mb-4">Don't have a dataset? Try one of our built-in samples:</p>
        <div className="flex flex-wrap gap-3">
          <button className={btnSecondaryClass}>Titanic (Classification)</button>
          <button className={btnSecondaryClass}>Housing Prices (Regression)</button>
          <button className={btnSecondaryClass}>Customer Churn (Classification)</button>
          <button className={btnSecondaryClass}>Wine Quality (Regression)</button>
        </div>
      </div>
    </div>
  )
}