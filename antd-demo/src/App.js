import React from 'react';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { StyleProvider } from '@ant-design/cssinjs';
import AuthenticatedApp from './component/AuthenticatedApp';
import ErrorBoundary from './component/ErrorBoundary';
import './App.css';

const App = () => {
  // 全局 Antd 配置
  const antdConfig = {
    // 主题配置
    theme: {
      token: {
        colorPrimary: '#1a365d',
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
          <AuthenticatedApp />
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
