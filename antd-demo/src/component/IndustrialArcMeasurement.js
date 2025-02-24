import React, { useEffect, useRef, useState } from 'react';
import {
  Layout,
  Menu,
  Card,
  Typography,
  Button,
  Form,
  InputNumber,
  Upload,
  Row,
  Col,
  List,
  message,
} from 'antd';
import { 
  UploadOutlined, 
  FilterOutlined, 
  SettingOutlined, 
  EyeOutlined, 
  HistoryOutlined,
  RadarChartOutlined 
} from '@ant-design/icons';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import FilterSection from './FilterSection';
import ParamsSettingComponent from './ParamsSettingComponent';
import UploadSection from './UploadSection';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

const IndustrialArcMeasurement = () => {
  const [selectedKey, setSelectedKey] = useState('upload');
  const [collapsed, setCollapsed] = useState(false);
  const [logoHovered, setLogoHovered] = useState(false);

  return (
    <Layout style={{ height: '100vh', background: '#f0f2f5' }}>
      <Header 
        style={{ 
          background: 'linear-gradient(135deg, #1a365d 0%, #004599 100%)',
          padding: '0 12px',
          display: 'flex',
          alignItems: 'center',
          boxShadow: '0 1px 2px rgba(0,0,0,0.08)',
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
      </Header>
      <Layout>
        <Sider 
          width={180} 
          collapsible 
          collapsed={collapsed}
          onCollapse={(value) => setCollapsed(value)}
          style={{ 
            background: '#fff',
            boxShadow: '2px 0 8px rgba(0,0,0,0.05)',
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
              borderRight: 0,
              backgroundColor: 'transparent'
            }}
            theme="light"
          >
            <Menu.Item key="upload" icon={<UploadOutlined />}>点云上传</Menu.Item>
            <Menu.Item key="filter" icon={<FilterOutlined />}>点云预处理</Menu.Item>
            <Menu.Item key="params" icon={<SettingOutlined />}>参数设置</Menu.Item>
            <Menu.Item key="display" icon={<EyeOutlined />}>点云显示</Menu.Item>
            <Menu.Item key="history" icon={<HistoryOutlined />}>历史记录</Menu.Item>
          </Menu>
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 480 }}>
            {selectedKey === 'upload' && <UploadSection />}
            {selectedKey === 'filter' && <FilterSection />}
            {selectedKey === 'params' && <ParamsSection />}
            {selectedKey === 'display' && <DisplaySection />}
            {selectedKey === 'history' && <HistorySection />}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

const ParamsSection = () => {
  return <ParamsSettingComponent />;
};

const DisplaySection = () => {
  const [showThree, setShowThree] = useState(false);
  const threeCanvasRef = useRef(null);

  useEffect(() => {
    if (showThree && threeCanvasRef.current) {
      const canvas = threeCanvasRef.current;
      const renderer = new THREE.WebGLRenderer({ canvas });
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        75,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
      );
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      camera.position.z = 5;

      const animate = () => {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };

      const loader = new PLYLoader();
      loader.load(
        '/get_ply',
        (geometry) => {
          const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.01 });
          const points = new THREE.Points(geometry, material);
          scene.add(points);
        },
        undefined,
        (error) => console.error(error)
      );
      animate();
    }
  }, [showThree]);

  return (
    <Card title="点云显示" bordered={false}>
      <Button onClick={() => setShowThree(!showThree)}>
        {showThree ? '隐藏点云' : '显示点云'}
      </Button>
      {showThree && (
        <canvas
          ref={threeCanvasRef}
          style={{
            width: '800px',
            height: '600px',
            border: '1px solid #d9d9d9',
            display: 'block',
            marginTop: 16,
          }}
        />
      )}
    </Card>
  );
};

const HistorySection = () => {
  const [history, setHistory] = useState([]);
  return (
    <Card title="历史记录" bordered={false}>
      <List
        dataSource={history}
        renderItem={(item) => <List.Item>{item}</List.Item>}
        locale={{ emptyText: '暂无历史记录' }}
      />
    </Card>
  );
};

export default IndustrialArcMeasurement;
