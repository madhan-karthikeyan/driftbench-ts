import { useState, useEffect } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { BarChart3, Database, GitCompare, ChevronDown, Activity } from 'lucide-react'
import api from '../services/api'

export default function Layout() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const location = useLocation()

  useEffect(() => {
    api.getDatasets()
      .then(data => setDatasets(data.datasets || []))
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Navbar */}
      <nav className="bg-dark-800 border-b border-dark-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-lg">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold text-lg text-white">DriftBench-TS</span>
                <span className="hidden sm:inline text-slate-500 text-sm ml-2">ML Observability</span>
              </div>
            </Link>

            {/* Navigation */}
            <div className="flex items-center gap-1">
              <NavLink to="/" icon={<BarChart3 className="w-4 h-4" />} active={location.pathname === '/'}>
                Overview
              </NavLink>
              
              <div className="relative group">
                <button className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  location.pathname.startsWith('/dataset') 
                    ? 'bg-dark-700 text-white' 
                    : 'text-slate-400 hover:text-white hover:bg-dark-700'
                }`}>
                  <Database className="w-4 h-4" />
                  Datasets
                  <ChevronDown className="w-4 h-4" />
                </button>
                <div className="absolute left-0 mt-2 w-48 bg-dark-800 border border-dark-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                  {loading ? (
                    <div className="px-4 py-2 text-slate-500">Loading...</div>
                  ) : (
                    datasets.map(ds => (
                      <Link
                        key={ds}
                        to={`/dataset/${ds}`}
                        className="block px-4 py-2 text-slate-300 hover:bg-dark-700 hover:text-white capitalize"
                      >
                        {ds}
                      </Link>
                    ))
                  )}
                </div>
              </div>

              <NavLink to="/compare" icon={<GitCompare className="w-4 h-4" />} active={location.pathname === '/compare'}>
                Compare
              </NavLink>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-dark-700 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
          DriftBench-TS • Time-Series Drift Detection Benchmarking
        </div>
      </footer>
    </div>
  )
}

function NavLink({ to, icon, children, active }) {
  return (
    <Link
      to={to}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
        active ? 'bg-dark-700 text-white' : 'text-slate-400 hover:text-white hover:bg-dark-700'
      }`}
    >
      {icon}
      <span className="hidden sm:inline">{children}</span>
    </Link>
  )
}
