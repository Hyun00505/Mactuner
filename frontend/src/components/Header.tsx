import React, { useState } from "react";
import { Link } from "react-router-dom";

export const Header: React.FC = () => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        {/* 로고 */}
        <Link to="/" className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center">
            <span className="text-blue-600 font-bold text-lg">🍎</span>
          </div>
          <h1 className="text-2xl font-bold">MacTuner</h1>
          <span className="text-sm text-blue-200 ml-2">v0.1.0</span>
        </Link>

        {/* 네비게이션 */}
        <nav className="hidden md:flex items-center space-x-8">
          <Link
            to="/"
            className="hover:text-blue-100 transition-colors"
          >
            대시보드
          </Link>
          <Link
            to="/model"
            className="hover:text-blue-100 transition-colors"
          >
            모델
          </Link>
          <Link
            to="/data"
            className="hover:text-blue-100 transition-colors"
          >
            데이터
          </Link>
          <Link
            to="/chat"
            className="hover:text-blue-100 transition-colors"
          >
            Chat
          </Link>
          <Link
            to="/editor"
            className="hover:text-blue-100 transition-colors"
          >
            에디터
          </Link>
          <Link
            to="/history"
            className="hover:text-blue-100 transition-colors"
          >
            히스토리
          </Link>
        </nav>

        {/* 오른쪽 메뉴 */}
        <div className="flex items-center space-x-4">
          <button className="px-4 py-2 bg-white text-blue-600 rounded-lg font-semibold hover:bg-blue-50 transition-colors">
            설정
          </button>
          <button
            className="md:hidden"
            onClick={() => setShowMenu(!showMenu)}
          >
            ☰
          </button>
        </div>
      </div>

      {/* 모바일 메뉴 */}
      {showMenu && (
        <nav className="md:hidden bg-blue-700 px-4 py-3 space-y-2">
          <Link to="/" className="block hover:bg-blue-600 px-3 py-2 rounded">
            대시보드
          </Link>
          <Link to="/model" className="block hover:bg-blue-600 px-3 py-2 rounded">
            모델
          </Link>
          <Link to="/data" className="block hover:bg-blue-600 px-3 py-2 rounded">
            데이터
          </Link>
          <Link to="/chat" className="block hover:bg-blue-600 px-3 py-2 rounded">
            Chat
          </Link>
          <Link to="/editor" className="block hover:bg-blue-600 px-3 py-2 rounded">
            에디터
          </Link>
          <Link to="/history" className="block hover:bg-blue-600 px-3 py-2 rounded">
            히스토리
          </Link>
        </nav>
      )}
    </header>
  );
};
