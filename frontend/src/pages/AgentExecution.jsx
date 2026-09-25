import { useParams, Link } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import { ArrowLeft, RotateCw, CheckCircle, AlertCircle, Loader2, ChevronRight, FileText, TrendingUp, Brain, Settings, Search, Zap, Trophy } from 'lucide-react'
import clsx from 'clsx'

const stepConfig = {
  dataset_analysis: { name: 'Dataset Analysis', icon: FileText, color: 'blue' },
  planner: { name: 'Planner', icon: Brain, color: 'purple' },
  preprocessing: { name: 'Preprocessing', icon: Settings, color: 'orange' },
  eda: { name: 'EDA', icon: Search, color: 'green' },
  feature_engineering: { name: 'Feature Engineering', icon: Zap, color: 'pink' },
  model_selection: { name: 'Model Selection', icon: Trophy, color: 'indigo' },
  tuning: { name: 'Hyperparameter Tuning', icon: Settings, color: 'amber' },
  critic: { name: 'Critic Review', icon: AlertCircle, color: 'red' },
  improvement_router: { name: 'Decision Router', icon: ChevronRight, color: 'gray' },
  generate_final_report: { name: 'Final Report', icon: FileText, color: 'emerald' },
}

const improvementSteps = [
  'apply_class_weight', 'apply_resampling', 'apply_smote',
  'adjust_threshold', 'engineer_features', 'select_different_model',
  'run_hyperparameter_tuning'
]

export function AgentExecution() {
  const { jobId } = useParams()
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [currentNode, setCurrentNode] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState([])
  const [stats, setStats] = useState(null)
  const ws = useRef(null)
  const logContainerRef = useRef(null)

  useEffect(() => {
    if (!jobId) return
    const wsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/^http/, 'ws');
    ws.current = new WebSocket(`${wsUrl}/api/v1/agent/ws/${jobId}`)
    
    ws.current.onopen = () => {
      console.log('WebSocket connected')
      setIsRunning(true)
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      fetch(`${apiUrl}/api/v1/agent/jobs/${jobId}/run`, { method: 'POST' })
    }

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'step_update') {
        setCurrentStep(data.currentStep)
        setTotalSteps(data.totalSteps || 0)
        setCurrentNode(data.currentNode || '')
        if (data.currentNode === 'completed' || data.currentStep >= (data.totalSteps || 9)) {
          setIsRunning(false)
        }
      } else if (data.type === 'log') {
        setLogs(prev => [...prev, data.log])
      } else if (data.type === 'stats_update') {
        setStats(data.stats)
      }
    }

    ws.current.onclose = () => {
      console.log('WebSocket disconnected')
      setIsRunning(false)
    }

    return () => {
      if (ws.current) ws.current.close()
    }
  }, [jobId])

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const getStepStatus = (nodeName) => {
    if (currentNode === 'completed') return 'completed'
    if (currentNode === nodeName) return 'running'
    // consider completed if we have passed this node in sequence
    const order = Object.keys(stepConfig)
    const currentIndex = order.indexOf(currentNode)
    const nodeIndex = order.indexOf(nodeName)
    if (currentIndex >= 0 && nodeIndex >= 0 && nodeIndex < currentIndex) return 'completed'
    return 'pending'
  }

  const getStepIcon = (status, StepIcon) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'running':
        return <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
      default:
        return <StepIcon className="w-5 h-5 text-gray-400" />
    }
  }

  const getColorClass = (color) => {
    const map = {
      blue: 'bg-blue-100 text-blue-700',
      purple: 'bg-purple-100 text-purple-700',
      orange: 'bg-orange-100 text-orange-700',
      green: 'bg-green-100 text-green-700',
      pink: 'bg-pink-100 text-pink-700',
      indigo: 'bg-indigo-100 text-indigo-700',
      amber: 'bg-amber-100 text-amber-700',
      red: 'bg-red-100 text-red-700',
      gray: 'bg-gray-100 text-gray-700',
      emerald: 'bg-emerald-100 text-emerald-700',
    }
    return map[color] || 'bg-gray-100 text-gray-700'
  }

  const mainSteps = Object.entries(stepConfig).map(([key, config]) => ({ key, ...config }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Autonomous ML Agent</h1>
            <p className="text-gray-600">Job: {jobId || 'unknown'}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className={clsx('px-3 py-1 rounded-full text-sm font-medium', 
            isRunning ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700')}>
            {isRunning ? 'Running' : 'Finished'}
          </span>
          <button 
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
            onClick={() => window.location.reload()}
          >
            <RotateCw className="w-4 h-4" /> Restart
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200">
            <div className="p-4 border-b border-gray-200">
              <h2 className="font-semibold text-gray-900">Autonomous Workflow</h2>
              <p className="text-sm text-gray-500">Iteration: {stats?.iteration || '0/0'}</p>
            </div>
            <div className="p-4 space-y-3">
              {mainSteps.map((step) => {
                const status = getStepStatus(step.key)
                const StepIcon = step.icon
                return (
                  <div key={step.key} className="flex items-center gap-4">
                    <div className="flex items-center gap-3 relative">
                      <div className={clsx('flex items-center justify-center w-10 h-10 rounded-full relative z-10',
                        status === 'completed' ? 'bg-green-100' :
                        status === 'running' ? getColorClass(step.color) : 'bg-gray-100')}>
                        {getStepIcon(status, StepIcon)}
                      </div>
                    </div>
                    <div className="flex-1">
                      <p className={clsx('font-medium', 
                        status === 'completed' ? 'text-gray-900' : 
                        status === 'running' ? getColorClass(step.color).replace('bg-','text-').replace('100','700') : 'text-gray-500')}>
                        {step.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {status === 'completed' ? 'Completed' : status === 'running' ? 'In progress...' : 'Pending'}
                      </p>
                    </div>
                    {status === 'running' && (
                      <span className="px-2 py-1 text-xs rounded-full bg-primary-100 text-primary-700 animate-pulse">Active</span>
                    )}
                  </div>
                )
              })}
              
              {/* Improvement loop indicator */}
              {stats && parseInt(stats.iteration?.split('/')[0] || '0') > 0 && (
                <div className="pt-4 border-t border-gray-200">
                  <p className="text-sm font-medium text-gray-700 mb-2">Improvement Loop (Iteration {stats.iteration?.split('/')[0]})</p>
                  <div className="flex flex-wrap gap-2">
                    {improvementSteps.map(imp => (
                      <span key={imp} className="px-2 py-1 text-xs rounded bg-gray-100 text-gray-600">
                        {imp.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">Agent Log</h2>
              <button className="px-2 py-1 text-xs text-gray-600 hover:text-gray-900" onClick={() => setLogs([])}>Clear</button>
            </div>
            <div ref={logContainerRef} className="p-4 max-h-96 overflow-y-auto scrollbar-thin space-y-3">
              {logs.map((log, index) => (
                <div key={index} className={clsx('flex items-start gap-3 text-sm',
                  log.level === 'success' ? 'text-green-700' :
                  log.level === 'warning' ? 'text-yellow-700' :
                  log.level === 'error' ? 'text-red-700' : 'text-gray-700')}>
                  <span className="text-gray-400 font-mono whitespace-nowrap">{log.time}</span>
                  <span className="font-medium text-gray-900 whitespace-nowrap">[{log.agent}]</span>
                  <span>{log.message}</span>
                </div>
              ))}
              {logs.length === 0 && (
                <p className="text-gray-500 text-center py-8">Waiting for agent logs...</p>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Current Experiment</h3>
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-500">Model</p>
                <p className="font-medium">{stats?.model || 'Waiting...'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Iteration</p>
                <p className="font-medium">{stats?.iteration || '0/0'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Best Score</p>
                <p className="font-medium text-green-600">{stats?.best_score || '0.0000'}</p>
              </div>
              <div className="pt-3 border-t border-gray-200">
                <p className="text-sm text-gray-500">Problem Type</p>
                <p className="font-medium">{stats?.problem_type || 'Classification'}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              {stats?.problem_type !== 'Regression' && (
                <>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <p className="text-2xl font-bold text-gray-900">{stats?.accuracy ?? '-'}</p>
                    <p className="text-xs text-gray-500">Accuracy</p>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <p className="text-2xl font-bold text-gray-900">{stats?.precision ?? '-'}</p>
                    <p className="text-xs text-gray-500">Precision</p>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <p className="text-2xl font-bold text-gray-900">{stats?.recall ?? '-'}</p>
                    <p className="text-xs text-gray-500">Recall</p>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-50">
                    <p className="text-2xl font-bold text-gray-900">{stats?.f1 ?? '-'}</p>
                    <p className="text-xs text-gray-500">F1 Score</p>
                  </div>
                </>
              )}
              {stats?.problem_type === 'Regression' && stats?.r2 && stats.r2 !== '-' && (
                <div className="p-3 rounded-lg bg-gray-50">
                  <p className="text-2xl font-bold text-gray-900">{stats.r2}</p>
                  <p className="text-xs text-gray-500">R² Score</p>
                </div>
              )}
              {stats?.problem_type === 'Regression' && stats?.rmse && stats.rmse !== '-' && (
                <div className="p-3 rounded-lg bg-gray-50">
                  <p className="text-2xl font-bold text-gray-900">{stats.rmse}</p>
                  <p className="text-xs text-gray-500">RMSE</p>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Critic Feedback</h3>
            {stats && stats.critic_feedback ? (
              <div className="p-3 rounded-lg bg-yellow-50 border border-yellow-200">
                <p className="text-sm text-yellow-800">
                  {stats.critic_feedback}
                </p>
              </div>
            ) : (
              <p className="text-sm text-gray-500">Waiting for critic analysis...</p>
            )}
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <Link to={`/report/${jobId}`} className={clsx("px-4 py-2 bg-primary-600 text-white rounded-lg text-center block hover:bg-primary-700 transition-colors", isRunning && "opacity-50 pointer-events-none")}>
              View Final Report
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}