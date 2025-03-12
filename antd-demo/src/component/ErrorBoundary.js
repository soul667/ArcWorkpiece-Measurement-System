import React from 'react';
import { Alert, Button } from 'antd';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error: error,
            errorInfo: errorInfo
        });
        console.error('React错误边界捕获到错误:', error, errorInfo);
    }

    handleReload = () => {
        window.location.reload();
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ 
                    padding: '24px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '16px'
                }}>
                    <Alert
                        message="应用程序出错"
                        description={
                            <div>
                                <p>抱歉，应用程序遇到了一个错误。</p>
                                <p>错误信息: {this.state.error && this.state.error.toString()}</p>
                                {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                                    <details style={{ whiteSpace: 'pre-wrap' }}>
                                        {this.state.errorInfo.componentStack}
                                    </details>
                                )}
                            </div>
                        }
                        type="error"
                        showIcon
                    />
                    <Button type="primary" onClick={this.handleReload}>
                        刷新页面
                    </Button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
