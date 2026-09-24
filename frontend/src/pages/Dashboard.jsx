import { Bot, Database, FlaskConical, TrendingUp, Clock, CheckCircle, AlertCircle, Upload, FileText } from 'lucide-react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'

const stats = [
  { name: 'Datasets', value: '0', icon: Database, color: 'text-blue-600 bg-blue-100' },
  { name: 'Experiments', value: '0', icon: FlaskConical, color: 'text-green-600 bg-green-100' },
  { name: 'Best Model Score', value: 'N/A', icon: TrendingUp, color: 'text-purple-600 bg-purple-100' },
  { name: 'Active Jobs', value: '0', icon: Clock, color: 'text-orange-600 bg-orange-100' },
]

const recentJobs = [
  { id: 'job_001', dataset: 'titanic.csv', status: 'completed', score: '0.87', time: '2 hours ago' },
  { id: 'job_002', dataset: 'housing.csv', status: 'running', score: '—', time: '5 min ago' },
  { id: 'job_003', dataset: 'churn.csv', status: 'failed', score: '—', time: '1 day ago' },
]

export function Dashboard() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Monitor your autonomous ML workflows</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div key={stat.name} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">{stat.name}</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stat.value}</p>
              </div>
              <div className={clsx('p-3 rounded-xl', stat.color)}>
                <stat.icon className="w-6 h-6" aria-hidden="true" />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Recent Jobs</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {recentJobs.map((job) => (
              <Link
                key={job.id}
                to={`/agent/${job.id}`}
                className="flex items-center justify-between p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
                    <Database className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{job.dataset}</p>
                    <p className="text-sm text-gray-500">{job.time}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <span className={clsx(
                    'px-2 py-1 text-xs font-medium rounded-full',
                    job.status === 'completed' && 'bg-green-100 text-green-700',
                    job.status === 'running' && 'bg-blue-100 text-blue-700',
                    job.status === 'failed' && 'bg-red-100 text-red-700'
                  )}>
                    {job.status}
                  </span>
                  <span className="font-mono text-sm text-gray-900">{job.score}</span>
                </div>
              </Link>
            ))}
          </div>
          <div className="p-4 border-t border-gray-200">
            <Link to="/upload" className="block w-full text-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
              Start New Experiment
            </Link>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Quick Actions</h2>
          </div>
          <div className="p-6 space-y-4">
            <Link to="/upload" className="flex items-center gap-4 p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors">
              <div className="p-3 rounded-xl bg-primary-100 text-primary-600">
                <Upload className="w-6 h-6" />
              </div>
              <div>
                <p className="font-medium text-gray-900">Upload Dataset</p>
                <p className="text-sm text-gray-500">Upload a CSV file to start an autonomous ML workflow</p>
              </div>
            </Link>
            <Link to="/experiments" className="flex items-center gap-4 p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors">
              <div className="p-3 rounded-xl bg-green-100 text-green-600">
                <FlaskConical className="w-6 h-6" />
              </div>
              <div>
                <p className="font-medium text-gray-900">View Experiments</p>
                <p className="text-sm text-gray-500">Compare models and track experiment history</p>
              </div>
            </Link>
            <Link to="/reports" className="flex items-center gap-4 p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors">
              <div className="p-3 rounded-xl bg-purple-100 text-purple-600">
                <FileText className="w-6 h-6" />
              </div>
              <div>
                <p className="font-medium text-gray-900">View Reports</p>
                <p className="text-sm text-gray-500">Detailed analysis and model performance reports</p>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}