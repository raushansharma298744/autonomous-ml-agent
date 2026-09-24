import { useParams, Link } from 'react-router-dom'
import { Database, ArrowLeft, BarChart2, Table, Eye, Download } from 'lucide-react'

const cardClass = "bg-white rounded-xl shadow-sm border border-gray-200"
const btnSecondaryClass = "px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"

export function DatasetAnalysis() {
  const { datasetId } = useParams()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Dataset Analysis</h1>
            <p className="text-gray-600">Dataset: {datasetId}</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button className={btnSecondaryClass}>
            <Download className="w-4 h-4" />
            Download Report
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className={`${cardClass} p-6`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
              <Database className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-gray-900">Overview</h3>
          </div>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between"><dt className="text-gray-500">Rows</dt><dd className="font-medium">—</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Columns</dt><dd className="font-medium">—</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Missing Values</dt><dd className="font-medium">—</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Duplicates</dt><dd className="font-medium">—</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Problem Type</dt><dd className="font-medium">—</dd></div>
          </dl>
        </div>

        <div className={`${cardClass} p-6 md:col-span-2`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-green-100 text-green-600">
              <Table className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-gray-900">Column Information</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-200">
                  <th className="pb-2 font-medium">Column</th>
                  <th className="pb-2 font-medium">Type</th>
                  <th className="pb-2 font-medium">Missing</th>
                  <th className="pb-2 font-medium">Unique</th>
                  <th className="pb-2 font-medium">Sample Values</th>
                </tr>
              </thead>
              <tbody>
                <tr><td colSpan={5} className="py-8 text-center text-gray-400">No data available</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`${cardClass} p-6`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
              <BarChart2 className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-gray-900">Target Distribution</h3>
          </div>
          <div className="h-64 flex items-center justify-center text-gray-400">
            Chart placeholder
          </div>
        </div>

        <div className={`${cardClass} p-6`}>
          <h3 className="font-semibold text-gray-900 mb-4">Missing Values Heatmap</h3>
          <div className="h-64 flex items-center justify-center text-gray-400">
            Chart placeholder
          </div>
        </div>
      </div>
    </div>
  )
}