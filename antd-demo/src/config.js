const environment = process.env.NODE_ENV || 'development';

const config = {
    development: {
        apiUrl: 'http://localhost:9304',
        authPrefix: '/auth',
        tokenKey: 'arc_workpiece_token',
        defaultTimeout: 10000,
        refreshTokenInterval: 25 * 60 * 1000, // 25 minutes
    },
    production: {
        apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:9304',
        authPrefix: '/auth',
        tokenKey: 'arc_workpiece_token',
        defaultTimeout: 10000,
        refreshTokenInterval: 25 * 60 * 1000, // 25 minutes
    },
    test: {
        apiUrl: 'http://localhost:9304',
        authPrefix: '/auth',
        tokenKey: 'arc_workpiece_token_test',
        defaultTimeout: 2000,
        refreshTokenInterval: 1000, // 1 second for testing
    }
};

// Get environment specific configuration
const currentConfig = config[environment];

export const settings = {
    // API Settings
    apiUrl: currentConfig.apiUrl,
    authPrefix: currentConfig.authPrefix,
    tokenKey: currentConfig.tokenKey,
    defaultTimeout: currentConfig.defaultTimeout,
    refreshTokenInterval: currentConfig.refreshTokenInterval,

    // API Endpoints
    endpoints: {
        login: '/token',
        register: '/register',
        verify: '/verify',
        userInfo: '/me',
    },

    // Auth Settings
    auth: {
        tokenType: 'Bearer',
        headerKey: 'Authorization',
    },

    // UI Settings
    ui: {
        loginPath: '/login',
        homePath: '/',
        unauthorizedPath: '/unauthorized',
        loadingDelay: 300,
    },

    // Error Messages
    messages: {
        loginSuccess: '登录成功！',
        loginError: '登录失败：',
        logoutSuccess: '已退出登录',
        sessionExpired: '登录已过期，请重新登录',
        networkError: '网络连接失败',
        unauthorized: '没有访问权限',
        serverError: '服务器错误',
        invalidCredentials: '用户名或密码错误',
    },

    // Get full API URL for a given endpoint
    getApiUrl: (endpoint) => `${currentConfig.apiUrl}${currentConfig.authPrefix}${endpoint}`,

    // Get auth header
    getAuthHeader: (token) => ({
        'Authorization': `Bearer ${token}`
    }),

    // Validation rules
    validation: {
        username: {
            min: 3,
            max: 20,
            pattern: /^[a-zA-Z0-9_-]+$/,
        },
        password: {
            min: 8,
            max: 50,
            requireUppercase: true,
            requireLowercase: true,
            requireNumbers: true,
        }
    }
};

export default settings;
