import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './component/ErrorBoundary';
import reportWebVitals from './reportWebVitals';
import './index.css';
import settings from './config';

// Global error handlers
window.onerror = function(message, source, lineno, colno, error) {
    console.error('Global error:', {
        message,
        source,
        lineno,
        colno,
        error
    });
    return false;
};

window.onunhandledrejection = function(event) {
    console.error('Unhandled Promise rejection:', event.reason);
};

// Development tools setup
if (process.env.NODE_ENV === 'development') {
    const whyDidYouRender = require('@welldone-software/why-did-you-render');
    whyDidYouRender(React, {
        trackAllPureComponents: true,
        logOnDifferentValues: true,
        collapseGroups: true,
    });
}

// Initialize performance monitoring
const reportMetrics = (metric) => {
    if (process.env.NODE_ENV === 'development') {
        console.log(metric);
    } else {
        // Send metrics to analytics service in production
    }
};

// Mount application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <ErrorBoundary>
            <App />
        </ErrorBoundary>
    </React.StrictMode>
);

// Performance monitoring
reportWebVitals(reportMetrics);

// Log environment info in development
if (process.env.NODE_ENV === 'development') {
    console.log('Environment:', process.env.NODE_ENV);
    console.log('API URL:', settings.apiUrl);
    console.log('Version:', process.env.REACT_APP_VERSION);
}
