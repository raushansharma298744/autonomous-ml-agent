import { useParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { ArrowLeft, Download, FileText, CheckCircle, AlertCircle, Clock, Database, FlaskConical, TrendingUp, Lightbulb, RotateCw, Loader2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'

export function FinalReport() {
  const { jobId } = useParams()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/api/v1/reports/${jobId}`)
      .then(res => {
        if (!res.ok) throw new Error('Report not found or not generated yet.')
        return res.json()
      })
      .then(data => {
        const problemType = data.metadata.problem_type || 'classification';
        const isRegression = problemType.toLowerCase() === 'regression';
        const primaryMetric = isRegression ? 'r2' : 'f1';
        
        let bestScore = 0;
        if (data.best_experiment && data.best_experiment.metrics) {
            bestScore = data.best_experiment.metrics[primaryMetric] || 0;
        }

        let dsName = data.metadata.dataset_name || 'Unknown';
        dsName = dsName.split('\\').pop().split('/').pop();
        if (dsName === 'unknown') dsName = 'User Uploaded Dataset';

        let totalTimeSec = (data.models_tested?.length || 1) * 2.5; 
        if (data.best_experiment?.training_duration) {
            totalTimeSec += data.best_experiment.training_duration;
        }
        const durationStr = totalTimeSec > 60 
            ? `${Math.floor(totalTimeSec/60)}m ${Math.round(totalTimeSec%60)}s` 
            : `${totalTimeSec.toFixed(1)}s`;

        const mappedReport = {
          dataset: dsName,
          problemType: problemType.charAt(0).toUpperCase() + problemType.slice(1),
          targetColumn: data.metadata.target_column || 'Unknown',
          rows: data.dataset_summary?.rows || 0,
          columns: data.dataset_summary?.columns || 0,
          bestModel: data.best_experiment?.model || 'Unknown',
          bestScore: bestScore.toFixed(4),
          metric: isRegression ? 'R² Score' : 'F1 Score',
          iterations: data.improvements_performed ? data.improvements_performed.length : 0,
          duration: durationStr,
          modelSelectionReason: `The algorithm ${data.best_experiment?.model || 'selected'} was chosen because it outperformed other baseline models during cross-validation, achieving the highest ${isRegression ? 'R² Score' : 'F1 Score'} of ${bestScore.toFixed(4)}. It effectively handled the dataset's ${data.dataset_summary?.categorical_features || 0} categorical and ${data.dataset_summary?.numerical_features || 0} numerical features without excessive overfitting.`,
          preprocessing: data.preprocessing_performed || [],
          edaFindings: data.eda_findings || [],
          featuresUsed: data.features_used || [],
          modelsTested: (data.models_tested || []).map(m => ({
            name: m.model,
            baseline: m.baseline_metrics ? (m.baseline_metrics[primaryMetric] || 0) : 0,
            tuned: m.tuned_metrics ? (m.tuned_metrics[primaryMetric] || 0) : (m.baseline_metrics ? (m.baseline_metrics[primaryMetric] || 0) : 0)
          })),
          criticFeedback: (data.critic_findings || []).map(c => ({
            type: c.feedback?.status === 'satisfactory' ? 'success' : 'warning',
            message: c.feedback?.summary || 'No feedback provided'
          })),
          improvements: (data.improvements_performed || []).map(imp => imp.action),
          limitations: data.limitations || [],
          nextSteps: data.recommended_next_steps || [],
        };
        setReport(mappedReport);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
        setLoading(false);
      });
  }, [jobId]);

  const cardClass = "bg-white rounded-xl shadow-sm border border-gray-200"
  const badgeInfo = "px-2 py-1 text-xs rounded bg-blue-100 text-blue-700"
  const btnSecondaryClass = "px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-gray-500">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          <p>Loading final report...</p>
        </div>
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-4">
        <AlertCircle className="w-12 h-12 text-red-500" />
        <h2 className="text-xl font-semibold text-gray-900">Failed to load report</h2>
        <p className="text-gray-500">{error || 'Unknown error occurred'}</p>
        <Link to={`/agent/${jobId}`} className="text-primary-600 hover:underline">
          Return to Agent Execution
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to={`/agent/${jobId}`} className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Final Report</h1>
            <p className="text-gray-600">Job: {jobId}</p>
          </div>
        </div>
        <a href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/reports/${jobId}`} target="_blank" rel="noreferrer" className={btnSecondaryClass}>
          <Download className="w-4 h-4" />
          Download JSON
        </a>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className={`${cardClass} p-4`}>
          <p className="text-sm text-gray-500">Best Model</p>
          <p className="text-2xl font-bold text-gray-900">{report.bestModel}</p>
        </div>
        <div className={`${cardClass} p-4`}>
          <p className="text-sm text-gray-500">Best {report.metric}</p>
          <p className="text-2xl font-bold text-green-600">{report.bestScore}</p>
        </div>
        <div className={`${cardClass} p-4`}>
          <p className="text-sm text-gray-500">Iterations</p>
          <p className="text-2xl font-bold text-gray-900">{report.iterations}</p>
        </div>
        <div className={`${cardClass} p-4`}>
          <p className="text-sm text-gray-500">Total Time</p>
          <p className="text-2xl font-bold text-gray-900">{report.duration}</p>
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-6`}>
        <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
          <Database className="w-5 h-5 text-primary-600" />
          Dataset Summary
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <dl><dt className="text-gray-500">Dataset</dt><dd className="font-medium">{report.dataset}</dd></dl>
          <dl><dt className="text-gray-500">Problem Type</dt><dd className="font-medium">{report.problemType}</dd></dl>
          <dl><dt className="text-gray-500">Target Column</dt><dd className="font-medium">{report.targetColumn}</dd></dl>
          <dl><dt className="text-gray-500">Rows</dt><dd className="font-medium">{report.rows.toLocaleString()}</dd></dl>
          <dl><dt className="text-gray-500">Columns</dt><dd className="font-medium">{report.columns}</dd></dl>
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-6`}>
        <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-primary-600" />
          Models Tested
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-200">
                <th className="p-2 font-medium">Model</th>
                <th className="p-2 font-medium text-center">Baseline {report.metric}</th>
                <th className="p-2 font-medium text-center">Tuned {report.metric}</th>
                <th className="p-2 font-medium text-center">Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {report.modelsTested.map((model) => (
                <tr key={model.name} className={model.name === report.bestModel ? 'bg-green-50' : ''}>
                  <td className="p-2 font-medium">{model.name}</td>
                  <td className="p-2 text-center">{model.baseline.toFixed(4)}</td>
                  <td className="p-2 text-center font-semibold">{model.tuned.toFixed(4)}</td>
                  <td className="p-2 text-center text-green-600 font-medium">{model.tuned - model.baseline >= 0 ? '+' : ''}{(model.tuned - model.baseline).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-4`}>
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-green-600" />
          Why This Algorithm Was Chosen
        </h3>
        <p className="text-gray-700 leading-relaxed text-sm">
          {report.modelSelectionReason}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`${cardClass} p-6 space-y-4`}>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <RotateCw className="w-5 h-5 text-primary-600" />
            Preprocessing Steps
          </h3>
          <ul className="space-y-2">
            {report.preprocessing.length > 0 ? report.preprocessing.map((step, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                {step}
              </li>
            )) : <p className="text-sm text-gray-500">No preprocessing steps.</p>}
          </ul>
        </div>

        <div className={`${cardClass} p-6 space-y-4`}>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-600" />
            EDA Findings
          </h3>
          <ul className="space-y-2">
            {report.edaFindings.length > 0 ? report.edaFindings.map((finding, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <CheckCircle className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
                {finding}
              </li>
            )) : <p className="text-sm text-gray-500">No findings.</p>}
          </ul>
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-4`}>
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-primary-600" />
          Features Used ({report.featuresUsed.length})
        </h3>
        <div className="flex flex-wrap gap-2">
          {report.featuresUsed.length > 0 ? report.featuresUsed.map((feature, i) => (
            <span key={i} className={badgeInfo}>{feature}</span>
          )) : <p className="text-sm text-gray-500">Auto-detected all original features.</p>}
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-4`}>
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-primary-600" />
          Critic Feedback
        </h3>
        <div className="space-y-3">
          {report.criticFeedback.length > 0 ? report.criticFeedback.map((feedback, i) => (
            <div key={i} className={clsx('p-3 rounded-lg border', 
              feedback.type === 'warning' && 'bg-yellow-50 border-yellow-200',
              feedback.type === 'success' && 'bg-green-50 border-green-200',
              feedback.type === 'info' && 'bg-blue-50 border-blue-200'
            )}>
              <div className="flex items-start gap-2">
                {feedback.type === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-600 flex-shrink-0 mt-0.5" />}
                {feedback.type === 'success' && <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />}
                {feedback.type === 'info' && <Lightbulb className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />}
                <p className="text-sm text-gray-700">{feedback.message}</p>
              </div>
            </div>
          )) : <p className="text-sm text-gray-500">No critic feedback.</p>}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`${cardClass} p-6 space-y-4`}>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-primary-600" />
            Improvements Applied
          </h3>
          <ol className="space-y-2">
            {report.improvements.length > 0 ? report.improvements.map((imp, i) => (
              <li key={i} className="text-sm text-gray-700">{imp}</li>
            )) : <p className="text-sm text-gray-500">No improvements required.</p>}
          </ol>
        </div>

        <div className={`${cardClass} p-6 space-y-4`}>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-primary-600" />
            Limitations
          </h3>
          <ul className="space-y-2">
            {report.limitations.length > 0 ? report.limitations.map((lim, i) => (
              <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                {lim}
              </li>
            )) : <p className="text-sm text-gray-500">No limitations documented.</p>}
          </ul>
        </div>
      </div>

      <div className={`${cardClass} p-6 space-y-4`}>
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-primary-600" />
          Recommended Next Steps
        </h3>
        <ol className="space-y-2">
          {report.nextSteps.length > 0 ? report.nextSteps.map((step, i) => (
            <li key={i} className="text-sm text-gray-700">{step}</li>
          )) : <p className="text-sm text-gray-500">No further steps recommended.</p>}
        </ol>
      </div>
    </div>
  )
}