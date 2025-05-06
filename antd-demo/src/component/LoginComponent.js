import React, { useState } from 'react';
import { Form, Input, Button, message } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { auth } from '../utils/axios';
import settings from '../config';

const LoginComponent = ({ onLoginSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [form] = Form.useForm();

    const onFinish = async (values) => {
        setLoading(true);
        try {
            const { access_token } = await auth.login(
                values.username,
                values.password
            );
            
            // Store the token
            localStorage.setItem(settings.tokenKey, access_token);
            
            // Get user info
            const userInfo = await auth.getUserInfo();
            
            message.success(settings.messages.loginSuccess);
            
            // Notify parent component of successful login
            if (onLoginSuccess) {
                onLoginSuccess(access_token, userInfo.username);
            }
        } catch (error) {
            console.error('Login error:', error);
            message.error('登录失败：' + (error.response?.data?.detail || '请检查用户名和密码'));
            form.setFields([
                {
                    name: 'password',
                    errors: ['密码错误']
                }
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ 
            maxWidth: 360, 
            margin: '100px auto', 
            padding: 24,
            boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
            borderRadius: 8
            // backgroundColor: '#fff'
        }}>
            <h2 style={{ 
                textAlign: 'center', 
                marginBottom: 32,
                color: '#1a365d',
                fontSize: '24px',
                fontWeight: 500
            }}>
                圆弧测量系统
            </h2>
            <Form
                form={form}
                name="login"
                onFinish={onFinish}
                size="large"
                layout="vertical"
            >
                <Form.Item
                    name="username"
                    rules={[
                        { required: true, message: '请输入用户名！' }
                    ]}
                >
                    <Input 
                        prefix={<UserOutlined style={{ color: '#bfbfbf' }} />} 
                        placeholder="用户名" 
                    />
                </Form.Item>

                <Form.Item
                    name="password"
                    rules={[
                        { required: true, message: '请输入密码！' }
                    ]}
                >
                    <Input.Password 
                        prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
                        placeholder="密码"
                    />
                </Form.Item>

                <Form.Item style={{ marginBottom: 0 }}>
                    <Button 
                        type="primary" 
                        htmlType="submit" 
                        style={{ 
                            width: '100%',
                            height: '40px',
                            background: 'linear-gradient(135deg, #1a365d 0%, #004599 100%)',
                        }}
                        loading={loading}
                    >
                        登录
                    </Button>
                </Form.Item>
            </Form>
            <div style={{ 
                marginTop: 16, 
                textAlign: 'center',
                color: '#999',
                fontSize: '13px'
            }}>
                默认账号: admin123 密码: admin
            </div>
        </div>
    );
};

export default LoginComponent;
