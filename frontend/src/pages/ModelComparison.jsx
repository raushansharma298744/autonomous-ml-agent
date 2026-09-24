import { useParams } from 'react-router-dom'
import { ArrowLeft, TrendingUp, Download, ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'

const models = [
  { name: 'XGBoost (tuned)', accuracy: 0.91, precision: 0.89, recall: 0.78, f1: 0.83, roc_auc: 0.94, time: '45s', status: 'best' },
  { name: 'RandomForest (tuned)', accuracy: 0.89, precision: 0.87, recall: 0.75, f1: 0.81, roc_auc: 0.92, time: '38s', status: '' },
  { name: 'XGBoost (baseline)', accuracy: 0.87, precision: 0.85, recall: 0.68, f1: 0.76, roc_auc: 0.90, time: '12s', status: '' },
  { name: 'RandomForest (baseline)', accuracy: 0.85, precision: 0.83, recall: 0.65, f1: 0.73, roc_auc: 0.88, time: '10s', status: '' },
  { name: 'LogisticRegression', accuracy: 0.82, precision: 0.80, recall: 0.62, f1: 0.70, roc_auc: 0.85, time: '5s', status: '' },
]

const badgeClass = "px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-700"
const btnSecondaryClass = "px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
const cardClass = "bg-white rounded-xl shadow-sm border border-gray-200"

export function ModelComparison() {
  const { jobId } = useParams()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/experiments" className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Model Comparison</h1>
            <p className="text-gray-600">Job: {jobId}</p>
          </div>
        </div>
        <button className={btnSecondaryClass}>
          <Download className="w-4 h-4" />
          Export Comparison
        </button>
      </div>

      <div className={`${cardClass} overflow-hidden`}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-200 bg-gray-50">
                <th className="p-4 font-medium">Model</th>
                <th className="p-4 font-medium text-center">Accuracy</th>
                <th className="p-4 font-medium text-center">Precision</th>
                <th className="p-4 font-medium text-center">Recall</th>
                <th className="p-4 font-medium text-center">F1 Score</th>
                <th className="p-4 font-medium text-center">ROC-AUC</th>
                <th className="p-4 font-medium text-center">Train Time</th>
                <th className="p-4 font-medium"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {models.map((model, index) => (
                <tr key={model.name} className={clsx('hover:bg-gray-50 transition-colors', model.status === 'best' && 'bg-green-50')}>
                  <td className="p-4">
                    <div className="flex items-center gap-3">
                      {model.status === 'best' && <TrendingUp className="w-5 h-5 text-green-600" />}
                      <div>
                        <p className={clsx('font-medium', model.status === 'best' && 'text-green-700')}>{model.name}</p>
                        {model.status === 'best' && <span className={badgeClass}>Best Model</span>}
                      </div>
                    </div>
                  </td>
                  <td className="p-4 text-center font-mono font-medium">{model.accuracy.toFixed(2)}</td>
                  <td className="p-4 text-center font-mono">{model.precision.toFixed(2)}</td>
                  <td className="p-4 text-center font-mono">{model.recall.toFixed(2)}</td>
                  <td className="p-4 text-center font-mono font-semibold">{model.f1.toFixed(2)}</td>
                  <td className="p-4 text-center font-mono">{model.roc_auc.toFixed(2)}</td>
                  <td className="p-4 text-center text-gray-500 font-mono">{model.time}</td>
                  <td className="p-4 text-center">
                    <Link to={`/experiments/${jobId}/${index}`} className="text-primary-600 hover:underline text-sm">
                      Details
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`${cardClass} p-6`}>
          <h3 className="font-semibold text-gray-900 mb-4">Metric Comparison Chart</h3>
          <div className="h-80 flex items-center justify-center text-gray-400">
            Bar chart placeholder - would show grouped bar chart comparing metrics across models
          </div>
        </div>

        <div className={`${cardClass} p-6`}>
          <h3 className="font-semibold text-gray-900 mb-4">Training Time vs Performance</h3>
          <div className="h-80 flex items-center justify-center text-gray-400">
            Scatter plot placeholder - would show training time vs F1 score
          </div>
        </div>
      </div>

      <div className={`${cardClass} p-6`}>
        <h3 className="font-semibold text-gray-900 mb-4">Hyperparameter Comparison (Top 2 Models)</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">XGBoost (Best)</h4>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between"><dt className="text-gray-500">n_estimators</dt><dd className="font-medium">200</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">learning_rate</dt><dd className="font-medium">0.1</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">max_depth</dt><dd className="font-medium">6</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">subsample</dt><dd className="font-medium">0.9</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">colsample_bytree</dt><dd className="font-medium">0.8</dd></div>
            </dl>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-3">RandomForest</h4>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between"><dt className="text-gray-500">n_estimators</dt><dd className="font-medium">150</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">max_depth</dt><dd className="font-medium">10</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">min_samples_split</dt><dd className="font-medium">5</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">min_samples_leaf</dt><dd className="font-medium">2</dd></div>
              <div className="flex justify-between"><dt className="text-gray-500">max_features</dt><dd className="font-medium">sqrt</dd></div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  )
}