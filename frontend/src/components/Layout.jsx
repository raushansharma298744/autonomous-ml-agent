import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Upload, Database, Bot, FlaskConical, BarChart2, FileText, Settings, Play, History } from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Upload Dataset', href: '/upload', icon: Upload },
  { name: 'Experiments', href: '/experiments', icon: FlaskConical },
]

export function Layout({ children }) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50">
      <aside className="fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-200 hidden lg:block">
        <div className="flex flex-col h-full">
          <div className="flex items-center h-16 px-6 border-b border-gray-200">
            <div className="flex items-center gap-2">
              <Bot className="w-8 h-8 text-primary-600" />
              <span className="text-xl font-bold text-gray-900">AutoML Agent</span>
            </div>
          </div>
          <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  location.pathname === item.href
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                )}
              >
                <item.icon className="w-5 h-5" aria-hidden="true" />
                {item.name}
              </Link>
            ))}
          </nav>
          <div className="p-4 border-t border-gray-200">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Version 0.1.0</p>
          </div>
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-40 bg-white border-b border-gray-200 lg:hidden">
          <div className="flex items-center justify-between h-16 px-4">
            <div className="flex items-center gap-2">
              <Bot className="w-8 h-8 text-primary-600" />
              <span className="text-xl font-bold text-gray-900">AutoML Agent</span>
            </div>
          </div>
        </header>
        <main className="p-6 lg:p-8">
          {children}
        </main>
      </div>
    </div>
  )
}