import React, { useEffect, useRef, useState } from 'react';
import {
  Layout,
  Menu,
  Card,
  Typography,
  Button,
  Space,
  Dropdown,
} from 'antd';
import { 
  UploadOutlined, 
  FilterOutlined, 
  SettingOutlined, 
  HistoryOutlined,
  RadarChartOutlined,
  UserOutlined,
  LogoutOutlined,
  LineChartOutlined,
  BulbOutlined,
  BulbFilled
} from '@ant-design/icons';
import FilterSection from './FilterSection';
import ParamsSettingComponent from './ParamsSettingComponent';
import UploadSection from './UploadSection';
import LineQualityViewer from './LineQualityViewer';
import HistorySection from './History';
import ArcFittingComponent from './ArcFittingComponent';
import PointCloudGeneratorComponent from './PointCloudGeneratorComponent';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

const IndustrialArcMeasurement = ({ username, onLogout, toggleTheme, currentTheme }) => {
  const [selectedKey, setSelectedKey] = useState('upload');
  const [collapsed, setCollapsed] = useState(false);
  const [logoHovered, setLogoHovered] = useState(false);

  return (
    <Layout style={{ height: '100vh', background: 'var(--background-color)' }}>
      <Header 
        style={{ 
          background: 'linear-gradient(135deg, hsl(277deg 81.07% 47.65%) 0%, hsl(176deg 74.42% 33.73%) 100%)',
          padding: '0 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: 'var(--box-shadow)',
          height: '42px',
          lineHeight: '42px'
        }}
      >
        <div 
          onMouseEnter={() => setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
          style={{ 
            display: 'flex', 
            alignItems: 'center',
            gap: '12px',
            cursor: 'pointer',
            padding: '0 8px',
            borderRadius: '4px',
            transition: 'all 0.2s ease-out',
            height: '32px',
            backgroundColor: logoHovered ? 'rgba(255,255,255,0.08)' : 'transparent',
            transform: logoHovered ? 'translateY(-1px)' : 'none'
        }}>
          <RadarChartOutlined style={{ fontSize: 20, color: '#fff', transform: 'rotate(-30deg)' }} />
          <Typography.Title 
            level={5}
            style={{ 
              margin: 0,
              color: '#fff',
              fontWeight: 400,
              letterSpacing: '0.3px',
              fontSize: '15px'
            }}
          >
            圆弧测量系统
          </Typography.Title>
        </div>

        {/* User Info and Logout */}
        <Space style={{ marginLeft: 'auto' }} size={4}>
          <Button
            type="text"
            icon={currentTheme === 'dark' ? <BulbOutlined /> : <BulbFilled />}
            onClick={toggleTheme}
            style={{ 
              color: '#fff',
              display: 'flex',
              alignItems: 'center'
            }}
          />
          <Dropdown menu={{
            items: [
              {
                key: '1',
                label: '退出登录',
                icon: <LogoutOutlined />,
                onClick: onLogout
              }
            ]
          }}>
            <Button 
              type="text" 
              icon={<UserOutlined />}
              style={{ 
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              {username}
            </Button>
          </Dropdown>
        </Space>
      </Header>
      
      <Layout>
        <Sider 
          width={180} 
          collapsible 
          collapsed={collapsed}
          onCollapse={(value) => setCollapsed(value)}
          style={{ 
            background: 'var(--component-background)',
            boxShadow: 'var(--box-shadow)',
            zIndex: 10,
            transition: 'all 0.2s ease-in-out'
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={({ key }) => setSelectedKey(key)}
            style={{ 
              height: '100%', 
              borderRight: 0
              // backgroundColor: 'transparent'
            }}
            // theme="light"
          >
            {/* <Menu.Item key="generate" icon={<RadarChartOutlined />}>点云生成</Menu.Item> */}
            <Menu.Item key="upload" icon={<UploadOutlined />}>点云上传</Menu.Item>
            <Menu.Item key="filter" icon={<FilterOutlined />}>点云预处理</Menu.Item>
            <Menu.Item key="params" icon={<SettingOutlined />}>参数设置</Menu.Item>
            <Menu.Item key="arc-fitting" icon={<LineChartOutlined />}>圆弧拟合</Menu.Item>
            <Menu.Item key="quality" icon={<LineChartOutlined />}>线质量分析</Menu.Item>
            <Menu.Item key="history" icon={<HistoryOutlined />}>历史记录</Menu.Item>
          </Menu>
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{padding: 5, margin: 0, minHeight: 480 }}>
            {/* {selectedKey === 'generate' && <PointCloudGeneratorComponent />} */}
            {selectedKey === 'upload' && <UploadSection />}
            {selectedKey === 'filter' && <FilterSection />}
            {selectedKey === 'params' && <ParamsSettingComponent />}
            {selectedKey === 'arc-fitting' && <ArcFittingComponent />}
            {selectedKey === 'quality' && <LineQualityViewer />}
            {selectedKey === 'history' && <HistorySection />}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default IndustrialArcMeasurement;
