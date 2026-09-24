import { Link } from 'react-router-dom'
import { Database, FlaskConical, ArrowRight, Filter, Download, Eye, TrendingUp, Clock } from 'lucide-react'
import clsx from 'clsx'

const experiments = [
  { id: 'exp_001', jobId: 'job_001', dataset: 'titanic.csv', model: 'XGBoost', status: 'completed', score: 0.91, metric: 'F1', time: '2 hours ago', duration: '45s' },
  { id: 'exp_002', jobId: 'job_001', dataset: 'titanic.csv', model: 'RandomForest', status: 'completed', score: 0.89, metric: 'F1', time: '2 hours ago', duration: '38s' },
  { id: 'exp_003', jobId: 'job_001', dataset: 'titanic.csv', model: 'LogisticRegression', status: 'completed', score: 0.82, metric: 'F1', time: '2 hours ago', duration: '12s' },
  { id: 'exp_004', jobId: 'job_002', dataset: 'housing.csv', model: 'XGBoost', status: 'completed', score: 0.87, metric: 'R²', time: '1 day ago', duration: '52s' },
  { id: 'exp_005', jobId: 'job_002', dataset: 'housing.csv', model: 'RandomForest', status: 'completed', score: 0.84, metric: 'R²', time: '1 day ago', duration: '41s' },
  { id: 'exp_006', jobId: 'job_003', dataset: 'churn.csv', model: 'XGBoost', status: 'failed', score: null, metric: 'F1', time: '3 days ago', duration: '—' },
]

const badgeClass = (status) => clsx(
  'px-2 py-1 text-xs font-medium rounded-full',
  status === 'completed' && 'bg-green-100 text-green-700',
  status === 'running' && 'bg-blue-100 text-blue-700',
  status === 'failed' && 'bg-red-100 text-red-700',
  status === 'pending' && 'bg-gray-100 text-gray-700'
)

const btnSecondaryClass = "px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
const btnSecondaryDisabled = "px-4 py-2 text-sm bg-gray-100 text-gray-400 rounded-lg cursor-not-allowed"

const cardClass = "bg-white rounded-xl shadow-sm border border-gray-200"

export function Experiments() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Experiments</h1>
          <p className="text-gray-600 mt-1">Track and compare all your ML experiments</p>
        </div>
        <div className="flex gap-3">
          <button className={`${btnSecondaryClass} flex items-center gap-2`}>
            <Filter className="w-4 h-4" />
            Filters
          </button>
          <button className={`${btnSecondaryClass} flex items-center gap-2`}>
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      <div className={cardClass}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-200">
                <th className="p-4 font-medium">Experiment</th>
                <th className="p-4 font-medium">Dataset</th>
                <th className="p-4 font-medium">Model</th>
                <th className="p-4 font-medium">Status</th>
                <th className="p-4 font-medium">Primary Metric</th>
                <th className="p-4 font-medium">Duration</th>
                <th className="p-4 font-medium">Created</th>
                <th className="p-4 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="p-4">
                    <Link to={`/experiments/${exp.jobId}/compare`} className="font-mono text-sm text-primary-600 hover:underline">
                      {exp.id}
                    </Link>
                  </td>
                  <td className="p-4 text-sm text-gray-900">{exp.dataset}</td>
                  <td className="p-4 text-sm text-gray-900">{exp.model}</td>
                  <td className="p-4">
                    <span className={badgeClass(exp.status)}>
                      {exp.status}
                    </span>
                  </td>
                  <td className="p-4">
                    {exp.score !== null ? (
                      <span className="font-mono font-medium">{exp.score} ({exp.metric})</span>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="p-4 text-sm text-gray-500 font-mono">{exp.duration}</td>
                  <td className="p-4 text-sm text-gray-500">{exp.time}</td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <Link to={`/experiments/${exp.jobId}/compare`} className="p-2 rounded-lg hover:bg-gray-100 transition-colors" title="Compare">
                        <Eye className="w-4 h-4 text-gray-500" />
                      </Link>
                      <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors" title="Details">
                        <TrendingUp className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="p-4 border-t border-gray-200 flex items-center justify-between">
          <p className="text-sm text-gray-500">Showing 6 of 6 experiments</p>
          <div className="flex gap-2">
            <button className={btnSecondaryDisabled} disabled>Previous</button>
            <button className={btnSecondaryDisabled} disabled>Next</button>
          </div>
        </div>
      </div>
    </div>
  )
}