import React, { useState, useEffect } from 'react';
import { message } from 'antd';
import IndustrialArcMeasurement from './IndustrialArcMeasurement';
import LoginComponent from './LoginComponent';
import LoadingScreen from './LoadingScreen';
import ErrorBoundary from './ErrorBoundary';
import { auth } from '../utils/axios';
import settings from '../config';

const AuthenticatedApp = ({ toggleTheme, currentTheme }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [loading, setLoading] = useState(true);
    const [username, setUsername] = useState('');

    // Check authentication status on component mount
    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        const token = localStorage.getItem(settings.tokenKey);
        if (token) {
            try {
                const userData = await auth.getUserInfo();
                setUsername(userData.username);
                setIsAuthenticated(true);
            } catch (error) {
                console.error('Authentication check failed:', error);
                handleLogout();
                message.error(settings.messages.sessionExpired);
            }
        }
        // Add a minimum loading time to prevent flashing
        setTimeout(() => setLoading(false), settings.ui.loadingDelay);
    };

    const handleLoginSuccess = (token, username) => {
        setIsAuthenticated(true);
        setUsername(username);
    };

    const handleLogout = () => {
        auth.logout();
        setIsAuthenticated(false);
        setUsername('');
    };

    if (loading) {
        return (
            <ErrorBoundary>
                <LoadingScreen tip="正在验证登录状态..." />
            </ErrorBoundary>
        );
    }

    return (
        <ErrorBoundary>
            {isAuthenticated ? (
                <IndustrialArcMeasurement 
                    username={username}
                    onLogout={handleLogout}
                    toggleTheme={toggleTheme}
                    currentTheme={currentTheme}
                />
            ) : (
                <LoginComponent onLoginSuccess={handleLoginSuccess} />
            )}
        </ErrorBoundary>
    );
};

// Error Boundary wrapper for development mode
if (process.env.NODE_ENV === 'development') {
    AuthenticatedApp.whyDidYouRender = {
        customName: 'AuthenticatedApp',
        logOnDifferentValues: true,
    };
}

export default AuthenticatedApp;
