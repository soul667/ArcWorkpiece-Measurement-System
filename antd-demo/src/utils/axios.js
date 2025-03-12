import axios from 'axios';
import { message } from 'antd';
import settings from '../config';

// Create axios instance with custom config
const instance = axios.create({
    baseURL: settings.apiUrl,
    timeout: settings.defaultTimeout,
});

// Request interceptor
instance.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem(settings.tokenKey);
        if (token) {
            config.headers[settings.auth.headerKey] = `${settings.auth.tokenType} ${token}`;
        }
        return config;
    },
    (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor
instance.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response) {
            switch (error.response.status) {
                case 401:
                    message.error(settings.messages.sessionExpired);
                    localStorage.removeItem(settings.tokenKey);
                    window.location.href = settings.ui.loginPath;
                    break;
                case 403:
                    message.error(settings.messages.unauthorized);
                    break;
                case 500:
                    message.error(settings.messages.serverError);
                    break;
                default:
                    if (error.response.data?.detail) {
                        message.error(error.response.data.detail);
                    } else {
                        message.error(`${settings.messages.loginError}${error.response.status}`);
                    }
            }
        } else if (error.request) {
            message.error(settings.messages.networkError);
        } else {
            console.error('Request configuration error:', error.message);
            message.error(error.message);
        }
        return Promise.reject(error);
    }
);

// Auth API methods
export const auth = {
    login: async (username, password) => {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const response = await instance.post(
            settings.getApiUrl(settings.endpoints.login),
            formData.toString(),
            {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            }
        );
        return response.data;
    },

    register: async (userData) => {
        const response = await instance.post(
            settings.getApiUrl(settings.endpoints.register),
            userData
        );
        return response.data;
    },

    verify: async () => {
        const response = await instance.get(
            settings.getApiUrl(settings.endpoints.verify)
        );
        return response.data;
    },

    getUserInfo: async () => {
        const response = await instance.get(
            settings.getApiUrl(settings.endpoints.userInfo)
        );
        return response.data;
    },

    logout: () => {
        localStorage.removeItem(settings.tokenKey);
        delete instance.defaults.headers.common[settings.auth.headerKey];
        message.success(settings.messages.logoutSuccess);
    }
};

export default instance;
