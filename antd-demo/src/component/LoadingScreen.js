import React from 'react';
import { Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const LoadingScreen = ({ tip = '加载中...' }) => {
    const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

    return (
        <div style={{
            height: '100vh',
            width: '100vw',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '16px',
            background: '#f0f2f5'
        }}>
            <div style={{
                background: 'white',
                padding: '32px 48px',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '16px'
            }}>
                <div style={{
                    fontSize: '20px',
                    color: '#1a365d',
                    fontWeight: 500,
                    marginBottom: '8px'
                }}>
                    圆弧测量系统
                </div>
                <Spin indicator={antIcon} tip={tip} />
            </div>
        </div>
    );
};

export default LoadingScreen;
