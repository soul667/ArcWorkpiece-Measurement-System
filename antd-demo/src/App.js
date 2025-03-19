import React, { useState, useEffect } from 'react';
import { ConfigProvider,theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { StyleProvider } from '@ant-design/cssinjs';
import AuthenticatedApp from './component/AuthenticatedApp';
import ErrorBoundary from './component/ErrorBoundary';
import './App.css';

const App = () => {
  // 获取保存的主题偏好，默认为系统偏好
  const [currentTheme, setCurrentTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) return savedTheme;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  // 监听系统主题变化
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => {
      const newTheme = e.matches ? 'dark' : 'light';
      if (!localStorage.getItem('theme')) {
        setCurrentTheme(newTheme);
      }
    };
    mediaQuery.addListener(handleChange);
    return () => mediaQuery.removeListener(handleChange);
  }, []);

  // 更新 data-theme 属性
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
  }, [currentTheme]);

  // 切换主题
  const toggleTheme = () => {
    setCurrentTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  // 全局 Antd 配置
  const antdConfig = {
    // 主题配置
    theme: {
      algorithm: [
        currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
        theme.compactAlgorithm
      ],
      token: {
        colorPrimary: currentTheme === 'dark' ? '#177ddc' : '#1a365d',
        borderRadius: 4,
      },
      components: {
        Button: {
          colorPrimary: '#1a365d',
          algorithm: true,
        },
        Input: {
          borderRadius: 4,
        },
        Card: {
          borderRadius: 8,
        },
      },
    },
    // 尺寸配置
    componentSize: 'middle',
    // 输入组件大小写
    form: {
      requiredMark: true,
      validateMessages: {
        required: '请输入${label}',
      },
    },
  };

  return (
    <ErrorBoundary>
      <ConfigProvider locale={zhCN} {...antdConfig}>
        <StyleProvider hashPriority="high">
          <AuthenticatedApp toggleTheme={toggleTheme} currentTheme={currentTheme} />
        </StyleProvider>
      </ConfigProvider>
    </ErrorBoundary>
  );
};

// 开发环境性能监控
if (process.env.NODE_ENV === 'development') {
  App.whyDidYouRender = {
    customName: 'App',
  };
}

export default App;
